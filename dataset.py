"""CASIA-B multimodal gait dataset with Large Sample Training (LST) split.

Each sample consists of:
    - A variable-length set of silhouette frames (dynamic branch input).
    - A single Gait Energy Image / GEI (static branch input).

The LST protocol partitions subjects 001–074 for training and 075–124 for
testing.

Performance architecture:
    - Frames and GEIs are loaded from pre-packed .npy files (created by
      pack_npy.py), which is ~100× faster than reading individual PNGs.
    - All data is pre-loaded into RAM at init time (~1.6 GB for training).
    - During training, 30 frames are randomly sampled per sequence
      (standard GaitSet practice), providing fixed tensor shapes and
      implicit data augmentation.
"""

import glob
import os
import random
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# Type alias for a single dataset sample: (frames, gei, label)
SampleTensors = Tuple[torch.Tensor, torch.Tensor, int]


class GaitMultiModalDataset(Dataset):
    """PyTorch Dataset for the CASIA-B gait recognition benchmark.

    All data is loaded from .npy archives into RAM during initialisation.
    During training, a fixed number of frames are randomly sampled from
    each sequence (``_TRAIN_SET_SIZE``).

    Prerequisites:
        Run ``python pack_npy.py --train-only`` to create .npy files
        from the processed PNGs before using this dataset.

    Args:
        data_dir: Root directory containing per-subject folders.
        is_train: If ``True``, load subjects 001–074 (training split).
            If ``False``, load subjects 075–124 (gallery/probe split).
    """

    # LST protocol boundary (inclusive upper bound for training subjects)
    _TRAIN_SUBJECT_UPPER: int = 74

    # Pixel intensity range used for normalisation
    _PIXEL_MAX: float = 255.0

    # ┌──────────────────────────────────────────────────────────────────┐
    # │  RANDOM SET SAMPLING: During training, sample exactly this many │
    # │  frames per sequence.  30 frames captures ~2 full gait cycles.  │
    # │                                                                 │
    # │  Benefits:                                                      │
    # │  • Fixed tensor shape → no custom collate / zero-padding needed │
    # │  • Reduces per-sample compute by ~40%                           │
    # │  • Implicit data augmentation (different subset each epoch)     │
    # └──────────────────────────────────────────────────────────────────┘
    _TRAIN_SET_SIZE: int = 30

    def __init__(self, data_dir: str, is_train: bool = True) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.is_train = is_train

        # Online augmentation settings (applied only during training)
        self._flip_prob: float = 0.5
        self._crop_margin: int = 2

        # ------------------------------------------------------------------
        # 1. DISCOVER SUBJECTS & BUILD LABEL MAP
        # ------------------------------------------------------------------
        all_entries: List[str] = sorted(os.listdir(data_dir))
        subjects: List[str] = [
            s for s in all_entries if os.path.isdir(os.path.join(data_dir, s))
        ]

        if self.is_train:
            subjects = [s for s in subjects if int(s) <= self._TRAIN_SUBJECT_UPPER]
        else:
            subjects = [s for s in subjects if int(s) > self._TRAIN_SUBJECT_UPPER]

        self.subject_to_label: Dict[str, int] = {
            subj: idx for idx, subj in enumerate(subjects)
        }
        self.num_classes: int = len(subjects)

        # ------------------------------------------------------------------
        # 2. DISCOVER SEQUENCE PATHS (must have .npy files)
        # ------------------------------------------------------------------
        self._labels: List[int] = []
        seq_dirs: List[str] = []
        skipped = 0

        for subject in subjects:
            label: int = self.subject_to_label[subject]
            subject_dir: str = os.path.join(data_dir, subject)

            for condition in sorted(os.listdir(subject_dir)):
                cond_dir: str = os.path.join(subject_dir, condition)
                if not os.path.isdir(cond_dir):
                    continue

                for angle in sorted(os.listdir(cond_dir)):
                    angle_dir: str = os.path.join(cond_dir, angle)
                    if not os.path.isdir(angle_dir):
                        continue

                    # Require pre-packed .npy files (created by pack_npy.py)
                    frames_npy = os.path.join(angle_dir, "frames.npy")
                    gei_npy = os.path.join(angle_dir, "gei.npy")

                    if not os.path.exists(frames_npy) or not os.path.exists(gei_npy):
                        skipped += 1
                        continue

                    seq_dirs.append(angle_dir)
                    self._labels.append(label)

        if skipped > 0:
            print(
                f"[GaitDataset] WARNING: {skipped} sequences skipped (missing .npy). "
                f"Run: python pack_npy.py --train-only"
            )

        # ------------------------------------------------------------------
        # 3. PRE-LOAD ALL .npy FILES INTO RAM (ONE-TIME COST)
        # ------------------------------------------------------------------
        # ┌──────────────────────────────────────────────────────────────┐
        # │  np.load() on a .npy file is a single fread() — orders of  │
        # │  magnitude faster than 56× cv2.imread() per sequence.       │
        # │                                                             │
        # │  Memory budget for 74 training subjects:                    │
        # │    ~8,100 sequences × ~56 frames × 64×64 bytes ≈ 1.6 GB    │
        # │  Fits comfortably in 16 GB system RAM.                      │
        # └──────────────────────────────────────────────────────────────┘
        self._frames_cache: List[np.ndarray] = []  # each is (N_i, 64, 64) uint8
        self._gei_cache: List[np.ndarray] = []     # each is (64, 64) uint8

        print(f"[GaitDataset] Loading .npy archives into RAM...")

        for seq_dir in tqdm(seq_dirs, desc="Loading cache"):
            frames = np.load(os.path.join(seq_dir, "frames.npy"))  # (N, 64, 64)
            gei = np.load(os.path.join(seq_dir, "gei.npy"))        # (64, 64)
            self._frames_cache.append(frames)
            self._gei_cache.append(gei)

        # Report cache stats
        total_frames = sum(f.shape[0] for f in self._frames_cache)
        cache_bytes = (
            sum(f.nbytes for f in self._frames_cache)
            + sum(g.nbytes for g in self._gei_cache)
        )
        print(
            f"[GaitDataset] Split={'train' if is_train else 'test'} | "
            f"Subjects={self.num_classes} | Sequences={len(self._labels)} | "
            f"Frames={total_frames:,} | "
            f"RAM={cache_bytes / 1e9:.2f} GB"
        )

    def __len__(self) -> int:
        """Return the total number of gait sequences in the split."""
        return len(self._labels)

    def __getitem__(self, idx: int) -> SampleTensors:
        """Return a single gait sequence from the RAM cache.

        During training, exactly ``_TRAIN_SET_SIZE`` frames are randomly
        sampled.  During evaluation, all frames are returned.

        Args:
            idx: Index of the sample to fetch.

        Returns:
            A 3-tuple of:
                - **frames** – ``(N, 64, 64)`` silhouette tensor.
                - **gei** – ``(1, 64, 64)`` GEI tensor.
                - **label** – Integer subject identity (0-indexed).
        """
        label: int = self._labels[idx]

        # ── 1. FETCH FROM RAM CACHE (zero disk I/O) ──
        gei: np.ndarray = self._gei_cache[idx].copy()          # (64, 64)
        all_frames: np.ndarray = self._frames_cache[idx]       # (N_full, 64, 64)

        # ── 2. RANDOM SET SAMPLING (training only) ──
        if self.is_train:
            N_full: int = all_frames.shape[0]
            k: int = self._TRAIN_SET_SIZE

            if N_full >= k:
                indices = np.random.choice(N_full, k, replace=False)
            else:
                # Fewer frames than k → oversample with replacement
                indices = np.random.choice(N_full, k, replace=True)

            indices.sort()
            frames_array: np.ndarray = all_frames[indices].copy()
        else:
            frames_array = all_frames.copy()

        # ── 3. ONLINE DATA AUGMENTATION (training only) ──
        if self.is_train:
            # Random horizontal flip (consistent across all frames + GEI)
            if random.random() < self._flip_prob:
                frames_array = np.flip(frames_array, axis=-1).copy()
                gei = np.flip(gei, axis=-1).copy()

            # Random minor crop + resize back to original dims
            m: int = self._crop_margin
            orig_h, orig_w = frames_array.shape[-2], frames_array.shape[-1]
            top = random.randint(0, m)
            left = random.randint(0, m)
            bottom = random.randint(0, m)
            right = random.randint(0, m)

            if top + bottom > 0 or left + right > 0:
                cropped_frames = frames_array[
                    :, top : orig_h - bottom, left : orig_w - right
                ]
                cropped_gei = gei[top : orig_h - bottom, left : orig_w - right]

                resized_frames: List[np.ndarray] = [
                    cv2.resize(f, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                    for f in cropped_frames
                ]
                frames_array = np.stack(resized_frames)
                gei = cv2.resize(
                    cropped_gei, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
                )

        # ── 4. CONVERT TO PYTORCH TENSORS & NORMALIZE ──
        # GEI: add channel dim → (1, 64, 64)
        gei_tensor: torch.Tensor = (
            torch.from_numpy(gei).unsqueeze(0).float() / self._PIXEL_MAX
        )
        frames_tensor: torch.Tensor = (
            torch.from_numpy(frames_array).float() / self._PIXEL_MAX
        )

        return frames_tensor, gei_tensor, label


# ---------------------------------------------------------------------------
# Custom collate function
# ---------------------------------------------------------------------------
# During TRAINING all samples have exactly _TRAIN_SET_SIZE=30 frames, so
# torch.stack() works directly.  During EVALUATION frame counts may vary,
# so we zero-pad to the batch maximum.


def gait_collate_fn(
    batch: List[SampleTensors],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate gait samples into a batch, padding frames if necessary."""
    frames_list, gei_list, label_list = zip(*batch)

    max_frames: int = max(f.shape[0] for f in frames_list)
    all_same = all(f.shape[0] == max_frames for f in frames_list)

    if all_same:
        batch_frames = torch.stack(list(frames_list))
    else:
        padded: List[torch.Tensor] = []
        for f in frames_list:
            n, h, w = f.shape
            if n < max_frames:
                padding = torch.zeros(max_frames - n, h, w, dtype=f.dtype)
                f = torch.cat([f, padding], dim=0)
            padded.append(f)
        batch_frames = torch.stack(padded)

    batch_gei = torch.stack(list(gei_list))
    batch_labels = torch.tensor(list(label_list), dtype=torch.long)

    return batch_frames, batch_gei, batch_labels


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    print("=" * 60)
    print("  Dataset Smoke Test (with .npy cache)")
    print("=" * 60)

    t0 = time.time()
    train_dataset = GaitMultiModalDataset(
        data_dir=r"C:\Users\adity\Desktop\Gait- MultiModal Fusion\Processed_CASIAB",
        is_train=True,
    )
    cache_time = time.time() - t0
    print(f"\nCache load time: {cache_time:.1f}s")

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=gait_collate_fn,
    )

    print(f"Total sequences: {len(train_dataset)}")
    print(f"Classes:         {train_dataset.num_classes}")

    # Time a single batch fetch
    t0 = time.time()
    for frames, gei, labels in train_loader:
        batch_ms = (time.time() - t0) * 1000
        print(f"\nFirst batch:")
        print(f"  Frames: {frames.shape} (B, N={train_dataset._TRAIN_SET_SIZE}, H, W)")
        print(f"  GEI:    {gei.shape} (B, 1, H, W)")
        print(f"  Labels: {labels}")
        print(f"  Time:   {batch_ms:.1f} ms")
        break