"""Evaluation script for the Global-Local Multimodal Fusion gait network.

Implements the **official GaitSet Gallery/Probe evaluation protocol** on the
CASIA-B Large Sample Training (LST) test split (subjects 075–124).

Protocol:
    Gallery  : nm-01 through nm-04  (4 sequences per subject per angle)
    NM Probe : nm-05, nm-06
    BG Probe : bg-01, bg-02
    CL Probe : cl-01, cl-02

For every (probe, gallery) pair sharing the *same viewing angle*, we compute
the cosine similarity of their feature embeddings and report Rank-1 accuracy
— i.e., the fraction of probes whose nearest gallery neighbour belongs to the
correct subject identity.

Embedding extraction:
    The feature vector is extracted from the **fused representation before the
    final classification head**.  Specifically, we capture the flattened output
    of the attention-fusion module (128 × 16 × 16 = 32,768-D), then
    L2-normalise it for cosine distance matching.  The linear classifier is
    irrelevant for open-set retrieval.

Hardware constraints (Windows + RTX 4050 / 6 GB VRAM):
    - num_workers=0 on all DataLoaders (Windows process-spawning bottleneck).
    - Entire extraction wrapped in torch.no_grad() to prevent graph build-up.
    - Default batch_size=4 to keep VRAM usage under 3 GB.
    - All data pre-cached into system RAM at startup to avoid per-batch I/O.

Usage:
    python eval.py
    python eval.py --data-dir path/to/Processed_CASIAB
    python eval.py --weights results/fused_gait_model.pth --batch-size 4
"""

import argparse
import glob
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from model import GlobalLocalFusedNetwork


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

# Test split: subjects 075–124 (50 subjects)
TEST_SUBJECT_RANGE = range(75, 125)

# CASIA-B viewing angles (zero-padded 3-digit folder names)
ANGLES = ["000", "018", "036", "054", "072", "090", "108", "126", "144", "162", "180"]
ANGLE_LABELS = ["0°", "18°", "36°", "54°", "72°", "90°", "108°", "126°", "144°", "162°", "180°"]

# Gallery / Probe split per the official GaitSet protocol
GALLERY_CONDITIONS = ["nm-01", "nm-02", "nm-03", "nm-04"]
PROBE_SETS: Dict[str, List[str]] = {
    "NM": ["nm-05", "nm-06"],
    "BG": ["bg-01", "bg-02"],
    "CL": ["cl-01", "cl-02"],
}

# Frame sampling — must match training protocol
N_SAMPLE_FRAMES = 30
PIXEL_MAX = 255.0
IMG_SIZE = 64


# ═══════════════════════════════════════════════════════════════════════════
# Data loading utilities
# ═══════════════════════════════════════════════════════════════════════════

def load_sequence(seq_dir: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load a single gait sequence, preferring .npy files over raw PNGs.

    Returns:
        (frames, gei) tuple where frames is (N, 64, 64) uint8 and
        gei is (64, 64) uint8, or None if the sequence cannot be loaded.
    """
    # ── Fast path: pre-packed .npy files ──
    frames_npy = os.path.join(seq_dir, "frames.npy")
    gei_npy = os.path.join(seq_dir, "gei.npy")
    if os.path.exists(frames_npy) and os.path.exists(gei_npy):
        return np.load(frames_npy), np.load(gei_npy)

    # ── Fallback: read individual PNGs ──
    all_pngs = sorted(glob.glob(os.path.join(seq_dir, "*.png")))
    frame_pngs = [p for p in all_pngs if "GEI" not in os.path.basename(p)]
    gei_pngs = [p for p in all_pngs if "GEI" in os.path.basename(p)]

    if not frame_pngs or not gei_pngs:
        return None

    frames = []
    for p in frame_pngs:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            frames.append(img)
    if not frames:
        return None

    gei = cv2.imread(gei_pngs[0], cv2.IMREAD_GRAYSCALE)
    if gei is None:
        return None

    return np.stack(frames), gei


def sample_frames_deterministic(
    frames: np.ndarray, n: int = N_SAMPLE_FRAMES
) -> np.ndarray:
    """Deterministically sample *n* frames with a fixed seed for reproducibility.

    If the sequence has fewer than *n* frames, oversample with replacement
    (same strategy used during training, but deterministic here).
    """
    rng = np.random.RandomState(seed=42)
    n_total = frames.shape[0]
    if n_total >= n:
        indices = rng.choice(n_total, n, replace=False)
    else:
        indices = rng.choice(n_total, n, replace=True)
    indices.sort()
    return frames[indices]


# ═══════════════════════════════════════════════════════════════════════════
# Sequence record + Discovery
# ═══════════════════════════════════════════════════════════════════════════

class SequenceRecord:
    """Lightweight container: metadata + pre-cached data for one sequence."""

    __slots__ = ("subject", "condition", "angle", "frames", "gei")

    def __init__(
        self,
        subject: str,
        condition: str,
        angle: str,
        frames: np.ndarray,
        gei: np.ndarray,
    ):
        self.subject = subject
        self.condition = condition
        self.angle = angle
        self.frames = frames   # (N_SAMPLE_FRAMES, 64, 64) uint8, already sampled
        self.gei = gei         # (64, 64) uint8

    def __repr__(self) -> str:
        return f"Seq({self.subject}/{self.condition}/{self.angle})"


def discover_and_cache_sequences(
    data_dir: str,
) -> Tuple[List[SequenceRecord], Dict[str, List[SequenceRecord]]]:
    """Walk test subjects, load ALL data into RAM, partition into Gallery/Probes.

    This is a ONE-TIME cost at startup.  All subsequent embedding extraction
    operates purely from RAM — zero disk I/O during the GPU-bound phase.

    Returns:
        gallery_records: Gallery sequences with pre-cached frames/GEI.
        probe_dict:      {"NM": [...], "BG": [...], "CL": [...]}.
    """
    # Build a quick lookup: condition → probe category name
    condition_to_probe: Dict[str, str] = {}
    for probe_name, conds in PROBE_SETS.items():
        for c in conds:
            condition_to_probe[c] = probe_name

    # First pass: collect all (subject, condition, angle, seq_dir) tuples
    work_items: List[Tuple[str, str, str, str]] = []  # (subj, cond, angle, path)
    for subj_id in TEST_SUBJECT_RANGE:
        subject = f"{subj_id:03d}"
        subject_dir = os.path.join(data_dir, subject)
        if not os.path.isdir(subject_dir):
            continue

        for condition in sorted(os.listdir(subject_dir)):
            cond_dir = os.path.join(subject_dir, condition)
            if not os.path.isdir(cond_dir):
                continue

            is_gallery = condition in GALLERY_CONDITIONS
            is_probe = condition in condition_to_probe
            if not is_gallery and not is_probe:
                continue

            for angle in sorted(os.listdir(cond_dir)):
                angle_dir = os.path.join(cond_dir, angle)
                if not os.path.isdir(angle_dir):
                    continue
                work_items.append((subject, condition, angle, angle_dir))

    # Second pass: load everything into RAM with a progress bar
    gallery_records: List[SequenceRecord] = []
    probe_dict: Dict[str, List[SequenceRecord]] = {k: [] for k in PROBE_SETS}
    skipped = 0

    print(f"[Eval] Pre-caching {len(work_items)} sequences into RAM...")
    for subject, condition, angle, seq_dir in tqdm(work_items, desc="Loading data"):
        result = load_sequence(seq_dir)
        if result is None:
            skipped += 1
            continue

        frames_raw, gei_raw = result
        frames_sampled = sample_frames_deterministic(frames_raw, N_SAMPLE_FRAMES)

        rec = SequenceRecord(subject, condition, angle, frames_sampled, gei_raw)

        if condition in GALLERY_CONDITIONS:
            gallery_records.append(rec)
        if condition in condition_to_probe:
            probe_dict[condition_to_probe[condition]].append(rec)

    if skipped > 0:
        print(f"[Eval] WARNING: {skipped} sequences skipped (missing data).")

    # Report cache stats
    total_cached = len(gallery_records) + sum(len(v) for v in probe_dict.values())
    cache_bytes = total_cached * (N_SAMPLE_FRAMES * IMG_SIZE * IMG_SIZE + IMG_SIZE * IMG_SIZE)
    print(f"[Eval] Gallery sequences : {len(gallery_records)}")
    for pname, precs in probe_dict.items():
        print(f"[Eval] {pname} Probe sequences: {len(precs)}")
    print(f"[Eval] RAM usage (data)  : {cache_bytes / 1e6:.1f} MB")

    return gallery_records, probe_dict


# ═══════════════════════════════════════════════════════════════════════════
# Embedding extraction  (ENTIRE pipeline under torch.no_grad)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_embeddings(
    model: GlobalLocalFusedNetwork,
    records: List[SequenceRecord],
    device: torch.device,
    batch_size: int = 4,
) -> np.ndarray:
    """Extract L2-normalised embeddings for a list of pre-cached sequences.

    ┌──────────────────────────────────────────────────────────────────────┐
    │  CRITICAL: This function is decorated with @torch.no_grad() so     │
    │  that NO computational graph is built during the entire extraction  │
    │  pipeline.  Without this, VRAM usage explodes and causes OOM on    │
    │  a 6 GB GPU.                                                       │
    │                                                                     │
    │  All tensors are explicitly moved to the target device with         │
    │  .to(device) before the forward pass.                               │
    │                                                                     │
    │  Data is served from the pre-cached SequenceRecord objects (RAM),   │
    │  so there is ZERO disk I/O during this phase.                       │
    └──────────────────────────────────────────────────────────────────────┘

    Args:
        model:      The gait network, already on `device` in eval mode.
        records:    List of SequenceRecord with pre-cached frames/GEI.
        device:     torch.device ("cuda" or "cpu").
        batch_size: Mini-batch size (default 4 for 6 GB VRAM safety).

    Returns:
        Stacked embedding matrix of shape ``(len(records), embed_dim)``.
    """
    all_embeddings: List[np.ndarray] = []
    n_batches = (len(records) + batch_size - 1) // batch_size

    for start in tqdm(
        range(0, len(records), batch_size),
        total=n_batches,
        desc="  Extracting",
        leave=False,
    ):
        batch_records = records[start : start + batch_size]
        B = len(batch_records)

        # ── Assemble batch from RAM cache (numpy) ──
        frames_batch = np.stack([rec.frames for rec in batch_records])  # (B, 30, 64, 64)
        gei_batch = np.stack([rec.gei for rec in batch_records])       # (B, 64, 64)

        # ── Convert to float tensors and move to GPU ──
        frames_t = (
            torch.from_numpy(frames_batch).float() / PIXEL_MAX
        ).to(device)                                                   # (B, 30, 64, 64)

        gei_t = (
            torch.from_numpy(gei_batch).float().unsqueeze(1) / PIXEL_MAX
        ).to(device)                                                   # (B, 1, 64, 64)

        # ── Forward pass: branches → fusion (skip classifier head) ──
        feat_a = model.branch_a(frames_t)                              # (B, 128, 16, 16)
        feat_b = model.branch_b(gei_t)                                 # (B, 128, 16, 16)
        fused = model.fusion(feat_a, feat_b)                           # (B, 128, 16, 16)

        # Flatten and L2-normalise for cosine similarity
        embeddings = fused.view(B, -1)                                 # (B, 32768)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # ── Move result to CPU immediately to free VRAM ──
        all_embeddings.append(embeddings.cpu().numpy())

        # Explicitly free GPU tensors
        del frames_t, gei_t, feat_a, feat_b, fused, embeddings
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return np.concatenate(all_embeddings, axis=0)


# ═══════════════════════════════════════════════════════════════════════════
# Rank-1 accuracy computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_rank1_accuracy(
    gallery_records: List[SequenceRecord],
    gallery_embeddings: np.ndarray,
    probe_records: List[SequenceRecord],
    probe_embeddings: np.ndarray,
) -> Dict[str, float]:
    """Compute per-angle and overall Rank-1 recognition accuracy.

    For each probe, we find its nearest gallery neighbour (by cosine
    similarity) **among gallery sequences of the same viewing angle** and
    check whether the predicted subject matches the true subject.

    Returns:
        Dictionary mapping angle codes (e.g. "090") to Rank-1 accuracy
        (0–100 %), plus an "avg" key for the mean across all angles.
    """
    # Group gallery indices by angle
    gallery_by_angle: Dict[str, List[int]] = defaultdict(list)
    for i, rec in enumerate(gallery_records):
        gallery_by_angle[rec.angle].append(i)

    # Accumulate per-angle correct / total counts
    correct_by_angle: Dict[str, int] = defaultdict(int)
    total_by_angle: Dict[str, int] = defaultdict(int)

    for p_idx, p_rec in enumerate(probe_records):
        angle = p_rec.angle
        g_indices = gallery_by_angle.get(angle, [])
        if not g_indices:
            continue

        # Gallery embeddings for this angle
        g_embeds = gallery_embeddings[g_indices]         # (K, D)
        p_embed = probe_embeddings[p_idx : p_idx + 1]   # (1, D)

        # Cosine similarity (embeddings are already L2-normalised)
        similarities = (p_embed @ g_embeds.T).flatten()  # (K,)
        best_idx_in_subset = int(np.argmax(similarities))
        best_gallery_idx = g_indices[best_idx_in_subset]

        predicted_subject = gallery_records[best_gallery_idx].subject
        true_subject = p_rec.subject

        total_by_angle[angle] += 1
        if predicted_subject == true_subject:
            correct_by_angle[angle] += 1

    # Per-angle accuracies
    results: Dict[str, float] = {}
    for angle in ANGLES:
        total = total_by_angle.get(angle, 0)
        correct = correct_by_angle.get(angle, 0)
        results[angle] = (correct / total * 100.0) if total > 0 else 0.0

    # Overall average (equally weighted across angles)
    valid_accs = [results[a] for a in ANGLES if total_by_angle.get(a, 0) > 0]
    results["avg"] = float(np.mean(valid_accs)) if valid_accs else 0.0

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Result formatting
# ═══════════════════════════════════════════════════════════════════════════

def print_results_table(results: Dict[str, Dict[str, float]]) -> None:
    """Print a formatted table of Rank-1 accuracies to the terminal.

    Rows    = Probe categories (NM, BG, CL) + ALL
    Columns = Viewing angles (0° … 180°) + Average
    """
    col_w = 7
    header_angles = " | ".join(f"{a:>{col_w}}" for a in ANGLE_LABELS)
    header = f"| {'Probe':>6} | {header_angles} | {'  Avg':>{col_w}} |"
    sep_cell = "-" * (col_w + 1)
    sep = f"|{'-' * 8}|" + "|".join(sep_cell for _ in ANGLES) + f"|{sep_cell}|"

    print()
    print("=" * len(header))
    print("  RANK-1 IDENTIFICATION ACCURACY (%)  —  CASIA-B Test (Subjects 075–124)")
    print("  Gallery: nm-01 to nm-04  |  Matching: Cosine Similarity")
    print("=" * len(header))
    print()
    print(header)
    print(sep)

    # Data rows
    probe_order = ["NM", "BG", "CL"]
    for probe_name in probe_order:
        if probe_name not in results:
            continue
        r = results[probe_name]
        cells = " | ".join(f"{r.get(a, 0.0):>{col_w}.2f}" for a in ANGLES)
        avg = r.get("avg", 0.0)
        print(f"| {probe_name:>6} | {cells} | {avg:>{col_w}.2f} |")

    print(sep)

    # Overall average across all three probe types
    all_avgs = [results[p]["avg"] for p in probe_order if p in results]
    overall = float(np.mean(all_avgs)) if all_avgs else 0.0
    overall_cells = []
    for angle in ANGLES:
        vals = [results[p].get(angle, 0.0) for p in probe_order if p in results]
        overall_cells.append(f"{np.mean(vals):>{col_w}.2f}" if vals else f"{'—':>{col_w}}")
    overall_row = " | ".join(overall_cells)
    print(f"| {'ALL':>6} | {overall_row} | {overall:>{col_w}.2f} |")

    print(sep)
    print()


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the fused gait model on CASIA-B test subjects 075–124.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", type=str, default=r"Processed_CASIAB",
        help="Root directory containing per-subject folders (default: Processed_CASIAB).",
    )
    parser.add_argument(
        "--weights", type=str, default=r"results/fused_gait_model.pth",
        help="Path to trained model weights (default: results/fused_gait_model.pth).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size for embedding extraction (default: 4, safe for 6 GB VRAM).",
    )
    parser.add_argument(
        "--num-classes", type=int, default=74,
        help="Number of classes the model was trained on (default: 74).",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device override (default: auto-detect CUDA/CPU).",
    )
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("  Global-Local Multimodal Fusion — Evaluation Pipeline")
    print("  Protocol: GaitSet Gallery/Probe (CASIA-B LST)")
    print("=" * 70)

    # ── 1. DEVICE SETUP ───────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU:  {gpu_name}")
        print(f"  VRAM: {gpu_vram:.1f} GB")

    # ── 2. LOAD MODEL ─────────────────────────────────────────────────
    print(f"\nLoading model weights from: {args.weights}")
    if not os.path.exists(args.weights):
        print(f"  [ERROR] Weights file not found: {args.weights}")
        sys.exit(1)

    # Instantiate model and move to GPU FIRST, then load weights
    model = GlobalLocalFusedNetwork(num_classes=args.num_classes)
    model = model.to(device)   # ← Explicit .to(device)

    state_dict = torch.load(args.weights, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    # Switch to eval mode (disables Dropout, freezes BatchNorm running stats)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    embed_dim = model._FEAT_CHANNELS * model._FEAT_HEIGHT * model._FEAT_WIDTH
    print(f"  Model loaded: {total_params:,} parameters")
    print(f"  Embedding dimension: {embed_dim:,}")
    print(f"  Mode: eval (dropout OFF, batchnorm frozen)")

    # ── 3. DISCOVER & PRE-CACHE ALL TEST DATA INTO RAM ────────────────
    # ┌──────────────────────────────────────────────────────────────────┐
    # │  FIX #1: Load everything into RAM upfront, just like dataset.py │
    # │  does for training.  This eliminates ALL per-batch disk I/O     │
    # │  during the GPU-bound extraction phase.                         │
    # │                                                                  │
    # │  On Windows, per-batch cv2.imread() on hundreds of PNGs is      │
    # │  catastrophically slow due to filesystem overhead.  Pre-caching  │
    # │  reduces total wall time from 20+ minutes to ~10 seconds.       │
    # │                                                                  │
    # │  NOTE: num_workers=0 is implicit — we don't use DataLoader at   │
    # │  all, so there are no worker processes to spawn (which is the   │
    # │  safest approach on Windows).                                    │
    # └──────────────────────────────────────────────────────────────────┘
    print()
    t0 = time.time()
    gallery_records, probe_dict = discover_and_cache_sequences(args.data_dir)
    cache_time = time.time() - t0
    print(f"[Eval] Data cached in {cache_time:.1f}s")

    if not gallery_records:
        print("[ERROR] No gallery sequences found. Check your data directory.")
        sys.exit(1)

    for pname, precs in probe_dict.items():
        if not precs:
            print(f"[WARN] No {pname} probe sequences found.")

    # ── 4. EXTRACT GALLERY EMBEDDINGS ─────────────────────────────────
    # ┌──────────────────────────────────────────────────────────────────┐
    # │  FIX #2: @torch.no_grad() decorator on extract_embeddings()     │
    # │  ensures NO computational graph is built.  Without this, PyTorch│
    # │  retains every intermediate activation for autograd, which       │
    # │  quickly exhausts 6 GB VRAM.                                    │
    # │                                                                  │
    # │  FIX #3: All tensors are explicitly created on the CPU then     │
    # │  moved to the device with .to(device).  The model itself is     │
    # │  on device via model.to(device) above.                          │
    # └──────────────────────────────────────────────────────────────────┘
    print(f"\nExtracting gallery embeddings ({len(gallery_records)} sequences)...")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    gallery_embeddings = extract_embeddings(
        model, gallery_records, device, batch_size=args.batch_size
    )
    gallery_time = time.time() - t0
    print(f"  Done in {gallery_time:.1f}s — shape: {gallery_embeddings.shape}")

    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        print(f"  Peak VRAM usage: {peak_mb:.0f} MB")

    # ── 5. EXTRACT PROBE EMBEDDINGS & COMPUTE RANK-1 ──────────────────
    all_results: Dict[str, Dict[str, float]] = {}

    for probe_name in ["NM", "BG", "CL"]:
        probe_records = probe_dict[probe_name]
        if not probe_records:
            continue

        print(f"\nExtracting {probe_name} probe embeddings ({len(probe_records)} sequences)...")
        t0 = time.time()
        probe_embeddings = extract_embeddings(
            model, probe_records, device, batch_size=args.batch_size
        )
        probe_time = time.time() - t0
        print(f"  Done in {probe_time:.1f}s — shape: {probe_embeddings.shape}")

        print(f"  Computing Rank-1 accuracy for {probe_name}...")
        results = compute_rank1_accuracy(
            gallery_records, gallery_embeddings,
            probe_records, probe_embeddings,
        )
        all_results[probe_name] = results
        print(f"  {probe_name} Average Rank-1: {results['avg']:.2f}%")

    # ── 6. PRINT FINAL RESULTS TABLE ──────────────────────────────────
    print_results_table(all_results)

    # ── 7. SUMMARY ────────────────────────────────────────────────────
    total_probes = sum(len(probe_dict[p]) for p in PROBE_SETS)
    print("Evaluation complete.")
    print(f"  Total gallery sequences:  {len(gallery_records)}")
    print(f"  Total probe sequences:    {total_probes}")
    if all_results:
        overall_avg = float(np.mean([all_results[p]["avg"] for p in all_results]))
        print(f"  Overall Rank-1 accuracy:  {overall_avg:.2f}%")
    print()


if __name__ == "__main__":
    main()
