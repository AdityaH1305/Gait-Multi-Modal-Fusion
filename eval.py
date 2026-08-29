"""Evaluation script for the Global-Local Multimodal Fusion gait network.

Implements the **official CASIA-B cross-view Gallery/Probe protocol** on the
Large Sample Training (LST) test split (subjects 075-124).

Protocol:
    Gallery  : nm-01 through nm-04, averaged into ONE template per
               (subject, angle) pair  ->  50 subjects x 11 angles = 550 templates
    NM Probe : nm-05, nm-06
    BG Probe : bg-01, bg-02
    CL Probe : cl-01, cl-02

For every probe we compute Rank-1 accuracy against the gallery templates at
*each* of the 11 viewing angles, producing an 11x11 cross-view matrix indexed
``[gallery_angle, probe_angle]``.  Two summaries are reported:

    CROSS-VIEW (headline) : mean of the OFF-diagonal cells (gallery != probe
                            angle).  This is the number published in the
                            gait-recognition literature.
    SAME-VIEW  (reference) : mean of the diagonal (gallery == probe angle).
                            Much easier, and reported here only so the two
                            can be compared directly.

Embedding extraction:
    Features come from the model's trained embedding head - the split-head
    architecture projects fused features into a 256-D L2-normalised space.
    ``model.forward()`` is called and the logits discarded.

    ALL frames of each sequence are used.  Set Pooling (element-wise max over
    the frame axis) is invariant to sequence length, so there is no reason to
    subsample at test time.

Outputs:
    results/embeddings.npz  - gallery/probe/template embeddings + labels and
                              the computed cross-view matrices.  This file is
                              the single source of truth for plot_view_matrix.py
                              and compute_biometrics.py, so the model only ever
                              runs once.

Usage:
    python eval.py
    python eval.py --data-dir path/to/Processed_CASIAB
    python eval.py --weights results/fused_gait_model.pth --frame-budget 1024
    python eval.py --show-matrix          # print the full 11x11 matrices
"""

import argparse
import glob
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

from model import GlobalLocalFusedNetwork


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

# Test split: subjects 075-124 (50 subjects)
TEST_SUBJECT_RANGE = range(75, 125)

# CASIA-B viewing angles (zero-padded 3-digit folder names)
ANGLES = ["000", "018", "036", "054", "072", "090", "108", "126", "144", "162", "180"]
ANGLE_LABELS = ["0", "18", "36", "54", "72", "90", "108", "126", "144", "162", "180"]

# Gallery / Probe split per the official protocol
GALLERY_CONDITIONS = ["nm-01", "nm-02", "nm-03", "nm-04"]
PROBE_SETS: Dict[str, List[str]] = {
    "NM": ["nm-05", "nm-06"],
    "BG": ["bg-01", "bg-02"],
    "CL": ["cl-01", "cl-02"],
}
PROBE_ORDER = ["NM", "BG", "CL"]

PIXEL_MAX = 255.0
IMG_SIZE = 64

# Sequences shorter than this are flagged as degenerate (CASIA-B has a handful
# with only 2-3 usable frames).  They are still evaluated, just reported.
MIN_HEALTHY_FRAMES = 10

# Default number of (sequence x frame) images per forward pass.  The dynamic
# branch reshapes to (B*N, 1, 64, 64), so B*N - not B - is what drives VRAM.
# Measured on a 6 GB RTX 4050: a budget of 1024 peaks at ~5.0 GB, which works
# but leaves little headroom; 768 peaks near 3.7 GB at essentially the same
# speed. Raise it with --frame-budget if you have more VRAM.
DEFAULT_FRAME_BUDGET = 768


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
        self.frames = frames   # (N, 64, 64) uint8 - FULL sequence, not sampled
        self.gei = gei         # (64, 64) uint8

    @property
    def n_frames(self) -> int:
        return int(self.frames.shape[0])

    def __repr__(self) -> str:
        return f"Seq({self.subject}/{self.condition}/{self.angle}, N={self.n_frames})"


def discover_and_cache_sequences(
    data_dir: str,
    subject_range=TEST_SUBJECT_RANGE,
    verbose: bool = True,
) -> Tuple[List[SequenceRecord], Dict[str, List[SequenceRecord]]]:
    """Walk subjects, load ALL data into RAM, partition into Gallery/Probes.

    Every frame of every sequence is retained - Set Pooling does not care how
    many frames it is given, so subsampling at test time only discards signal.

    Args:
        data_dir:      Root directory containing per-subject folders.
        subject_range: Which subject IDs to load.  Defaults to the held-out
            test split (075-124).  ``train.py`` passes a smaller range to run
            the same gallery/probe protocol on validation identities.
        verbose:       Print cache statistics.  Disabled by the training loop,
            which calls this once and does not want the noise.

    Returns:
        gallery_records: Gallery sequences (nm-01..nm-04).
        probe_dict:      {"NM": [...], "BG": [...], "CL": [...]}.
    """
    # Build a quick lookup: condition -> probe category name
    condition_to_probe: Dict[str, str] = {}
    for probe_name, conds in PROBE_SETS.items():
        for c in conds:
            condition_to_probe[c] = probe_name

    # First pass: collect all (subject, condition, angle, seq_dir) tuples
    work_items: List[Tuple[str, str, str, str]] = []
    for subj_id in subject_range:
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
    degenerate: List[str] = []

    if verbose:
        print(f"[Eval] Pre-caching {len(work_items)} sequences into RAM (all frames)...")
    for subject, condition, angle, seq_dir in tqdm(
        work_items, desc="Loading data", disable=not verbose
    ):
        result = load_sequence(seq_dir)
        if result is None:
            skipped += 1
            continue

        frames_raw, gei_raw = result
        rec = SequenceRecord(subject, condition, angle, frames_raw, gei_raw)

        if rec.n_frames < MIN_HEALTHY_FRAMES:
            degenerate.append(f"{subject}/{condition}/{angle} (N={rec.n_frames})")

        if condition in GALLERY_CONDITIONS:
            gallery_records.append(rec)
        if condition in condition_to_probe:
            probe_dict[condition_to_probe[condition]].append(rec)

    if skipped > 0 and verbose:
        print(f"[Eval] WARNING: {skipped} sequences skipped (missing data).")

    if not verbose:
        return gallery_records, probe_dict

    # ── Frame statistics ──
    all_recs = gallery_records + [r for v in probe_dict.values() for r in v]
    counts = np.array([r.n_frames for r in all_recs])
    cache_bytes = sum(r.frames.nbytes + r.gei.nbytes for r in all_recs)

    print(f"[Eval] Gallery sequences : {len(gallery_records)}")
    for pname in PROBE_ORDER:
        print(f"[Eval] {pname} Probe sequences: {len(probe_dict[pname])}")
    print(
        f"[Eval] Frames/sequence   : min={counts.min()} median={int(np.median(counts))} "
        f"max={counts.max()} mean={counts.mean():.1f} | total={counts.sum():,}"
    )
    print(f"[Eval] RAM usage (data)  : {cache_bytes / 1e9:.2f} GB")

    if degenerate:
        print(
            f"[Eval] NOTE: {len(degenerate)} sequence(s) have < {MIN_HEALTHY_FRAMES} "
            f"frames and will produce weak embeddings. They are still evaluated."
        )
        for d in degenerate[:5]:
            print(f"         - {d}")
        if len(degenerate) > 5:
            print(f"         ... and {len(degenerate) - 5} more")

    return gallery_records, probe_dict


# ═══════════════════════════════════════════════════════════════════════════
# Embedding extraction  (ENTIRE pipeline under torch.no_grad)
# ═══════════════════════════════════════════════════════════════════════════

def _tile_pad(frames: np.ndarray, target_n: int) -> np.ndarray:
    """Pad a frame set up to ``target_n`` by CYCLING its own frames.

    ┌──────────────────────────────────────────────────────────────────────┐
    │  WHY TILING AND NOT ZERO-PADDING:                                   │
    │                                                                      │
    │  Set Pooling is an element-wise MAX over the frame axis, so adding  │
    │  a duplicate of a frame that is already present cannot change the   │
    │  result:  max(a, b, a) == max(a, b).  Tiling is therefore exactly   │
    │  equivalent to running the sequence at its true length.             │
    │                                                                      │
    │  Zero-padding is NOT safe.  An all-zero frame still produces        │
    │  non-zero activations after Conv -> BatchNorm -> ReLU (the BN shift │
    │  and conv bias are non-zero), and those activations compete in the  │
    │  max.  Short sequences would be silently contaminated.              │
    └──────────────────────────────────────────────────────────────────────┘
    """
    n = frames.shape[0]
    if n >= target_n:
        return frames
    return frames[np.arange(target_n) % n]


def _make_frame_budget_batches(
    n_frames_list: List[int],
    frame_budget: int,
) -> List[List[int]]:
    """Greedily pack sequence indices into batches under a total-frame budget.

    The dynamic branch flattens to ``(B*N, 1, 64, 64)`` before the first
    convolution, so VRAM scales with ``B * N`` rather than ``B``.  A fixed
    batch size would therefore swing wildly in memory (a batch of 165-frame
    sequences is 4x the cost of a batch of 42-frame ones).

    Indices are sorted by frame count first, so each batch contains sequences
    of similar length and almost no tiling is needed.

    Returns:
        A list of batches, each a list of indices into the original record list.
    """
    order = sorted(range(len(n_frames_list)), key=lambda i: n_frames_list[i])

    batches: List[List[int]] = []
    current: List[int] = []

    for idx in order:
        n = n_frames_list[idx]
        # Sorted ascending, so `n` is the new batch max by construction.
        if current and (len(current) + 1) * n > frame_budget:
            batches.append(current)
            current = [idx]
        else:
            current.append(idx)

    if current:
        batches.append(current)

    return batches


@torch.no_grad()
def extract_embeddings(
    model: GlobalLocalFusedNetwork,
    records: List[SequenceRecord],
    device: torch.device,
    frame_budget: int = DEFAULT_FRAME_BUDGET,
    verbose: bool = True,
) -> np.ndarray:
    """Extract L2-normalised embeddings for a list of pre-cached sequences.

    Decorated with ``@torch.no_grad()`` so no autograd graph is built.

    Results are written back at each record's ORIGINAL index, so the returned
    array lines up with ``records`` despite the internal length-sorted batching.

    Args:
        model:        The gait network, already on `device` in eval mode.
        records:      Sequences with full (unsampled) frame sets.
        device:       torch.device ("cuda" or "cpu").
        frame_budget: Max (sequences x frames) per forward pass.

    Returns:
        Embedding matrix of shape ``(len(records), embed_dim)``, float32.
    """
    n_frames_list = [rec.n_frames for rec in records]
    batches = _make_frame_budget_batches(n_frames_list, frame_budget)

    out = np.zeros((len(records), model.embed_dim), dtype=np.float32)

    for batch in tqdm(batches, desc="  Extracting", leave=False, disable=not verbose):
        target_n = max(n_frames_list[i] for i in batch)

        frames_batch = np.stack([_tile_pad(records[i].frames, target_n) for i in batch])
        gei_batch = np.stack([records[i].gei for i in batch])

        frames_t = (torch.from_numpy(frames_batch).float() / PIXEL_MAX).to(device)
        gei_t = (torch.from_numpy(gei_batch).float().unsqueeze(1) / PIXEL_MAX).to(device)

        _, embeddings = model(frames_t, gei_t)

        # Scatter back to original positions - restores input ordering.
        out[batch] = embeddings.cpu().numpy()

    return out


# ═══════════════════════════════════════════════════════════════════════════
# Gallery templates
# ═══════════════════════════════════════════════════════════════════════════

def build_gallery_templates(
    records: List[SequenceRecord],
    embeddings: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average the nm-01..nm-04 embeddings into ONE template per (subject, angle).

    Any single recording carries walk-specific noise - an odd stride, a
    segmentation glitch.  Matching against the best of four individual
    sequences is sensitive to that noise; matching against their mean averages
    it out.  This is standard practice in the gait literature.

    Returns:
        (template_emb, template_subject, template_angle) where template_emb is
        ``(M, D)`` L2-normalised and the label arrays are ``(M,)`` of str.
    """
    groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        groups[(rec.subject, rec.angle)].append(i)

    subjects: List[str] = []
    angles: List[str] = []
    vectors: List[np.ndarray] = []

    for (subject, angle) in sorted(groups.keys()):
        idxs = groups[(subject, angle)]
        v = embeddings[idxs].mean(axis=0)
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
        subjects.append(subject)
        angles.append(angle)
        vectors.append(v.astype(np.float32))

    return np.stack(vectors), np.array(subjects), np.array(angles)


# ═══════════════════════════════════════════════════════════════════════════
# Cross-view Rank-1 computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_crossview_matrix(
    template_emb: np.ndarray,
    template_subject: np.ndarray,
    template_angle: np.ndarray,
    probe_emb: np.ndarray,
    probe_subject: np.ndarray,
    probe_angle: np.ndarray,
) -> np.ndarray:
    """Compute the full 11x11 cross-view Rank-1 accuracy matrix.

    Every probe angle is matched against every gallery angle - including the
    identical view, which lands on the diagonal and is excluded from the
    headline average by :func:`summarise_matrix`.

    Returns:
        ``(11, 11)`` float array indexed ``[gallery_angle, probe_angle]``,
        in percent.  Cells with no data are NaN.
    """
    n = len(ANGLES)
    correct = np.zeros((n, n), dtype=np.float64)
    total = np.zeros((n, n), dtype=np.float64)

    tmpl_by_angle = {a: np.where(template_angle == a)[0] for a in ANGLES}
    probe_by_angle = {a: np.where(probe_angle == a)[0] for a in ANGLES}

    for j, p_angle in enumerate(ANGLES):          # column = probe angle
        p_idx = probe_by_angle[p_angle]
        if p_idx.size == 0:
            continue

        P = probe_emb[p_idx]                       # (n_probe, D)
        p_subj = probe_subject[p_idx]

        for i, g_angle in enumerate(ANGLES):       # row = gallery angle
            t_idx = tmpl_by_angle[g_angle]
            if t_idx.size == 0:
                continue

            # Cosine similarity - both sides are already L2-normalised.
            sims = P @ template_emb[t_idx].T        # (n_probe, n_templates)
            best = sims.argmax(axis=1)
            predicted = template_subject[t_idx][best]

            correct[i, j] = float((predicted == p_subj).sum())
            total[i, j] = float(p_idx.size)

    acc = np.full((n, n), np.nan, dtype=np.float64)
    valid = total > 0
    acc[valid] = correct[valid] / total[valid] * 100.0
    return acc


def summarise_matrix(acc: np.ndarray) -> Dict[str, object]:
    """Reduce an 11x11 cross-view matrix to the numbers that get published.

    Returns a dict with:
        crossview       - mean of OFF-diagonal cells (the headline number)
        sameview        - mean of the diagonal (the easy, non-standard number)
        per_probe_angle - per probe angle, the mean over all gallery angles
                          EXCLUDING the identical view (the row usually
                          printed in papers)
    """
    n = acc.shape[0]
    eye = np.eye(n, dtype=bool)

    off_diag = np.where(eye, np.nan, acc)

    with np.errstate(invalid="ignore"):
        return {
            "crossview": float(np.nanmean(off_diag)),
            "sameview": float(np.nanmean(acc[eye])),
            "per_probe_angle": np.nanmean(off_diag, axis=0),  # mean over gallery
        }


# ═══════════════════════════════════════════════════════════════════════════
# Sanity gate
# ═══════════════════════════════════════════════════════════════════════════

def sanity_report(
    probe_emb: np.ndarray,
    probe_subject: np.ndarray,
    template_emb: np.ndarray,
    template_subject: np.ndarray,
    label: str,
) -> bool:
    """Verify embeddings are unit-norm and actually carry identity information.

    Prints the mean genuine vs mean impostor cosine similarity.  If those two
    numbers are indistinguishable the embeddings are noise, and every metric
    computed downstream would be meaningless - so this is checked BEFORE
    anything depends on them.

    Returns:
        True if the embeddings look healthy, False if they look like noise.
    """
    norms = np.linalg.norm(probe_emb, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        print(
            f"  [FAIL] {label}: embeddings are not unit-norm "
            f"(min={norms.min():.4f} max={norms.max():.4f})"
        )
        return False

    sims = probe_emb @ template_emb.T
    genuine_mask = probe_subject[:, None] == template_subject[None, :]

    g_mean = float(sims[genuine_mask].mean())
    i_mean = float(sims[~genuine_mask].mean())
    separation = g_mean - i_mean

    print(
        f"  {label:<3} | genuine {g_mean:+.4f} | impostor {i_mean:+.4f} | "
        f"separation {separation:+.4f}"
    )

    if separation < 0.05:
        print(
            f"  [WARNING] {label}: genuine and impostor scores are nearly "
            f"identical. The embeddings carry almost no identity signal."
        )
        return False

    return True


# ═══════════════════════════════════════════════════════════════════════════
# Result formatting
# ═══════════════════════════════════════════════════════════════════════════

def print_crossview_table(summaries: Dict[str, Dict[str, object]]) -> None:
    """Print per-probe-angle cross-view Rank-1 accuracy (identical view excluded)."""
    col_w = 6
    header_angles = " ".join(f"{a:>{col_w}}" for a in ANGLE_LABELS)
    header = f"| {'Probe':>5} | {header_angles} | {'Mean':>{col_w}} |"
    width = len(header)

    print()
    print("=" * width)
    print("  RANK-1 CROSS-VIEW ACCURACY (%)  -  CASIA-B test subjects 075-124")
    print("  Gallery: nm-01..04 averaged per (subject, angle)")
    print("  Each column = probe angle, averaged over all 10 OTHER gallery angles")
    print("=" * width)
    print()
    print(header)
    print("|" + "-" * (width - 2) + "|")

    for probe_name in PROBE_ORDER:
        if probe_name not in summaries:
            continue
        s = summaries[probe_name]
        per_angle = s["per_probe_angle"]
        cells = " ".join(f"{v:>{col_w}.2f}" for v in per_angle)
        print(f"| {probe_name:>5} | {cells} | {s['crossview']:>{col_w}.2f} |")

    print("|" + "-" * (width - 2) + "|")

    # ALL row
    stacked = np.vstack([summaries[p]["per_probe_angle"] for p in PROBE_ORDER if p in summaries])
    all_cells = " ".join(f"{v:>{col_w}.2f}" for v in np.nanmean(stacked, axis=0))
    all_mean = float(np.mean([summaries[p]["crossview"] for p in PROBE_ORDER if p in summaries]))
    print(f"| {'ALL':>5} | {all_cells} | {all_mean:>{col_w}.2f} |")
    print("=" * width)


def print_protocol_comparison(summaries: Dict[str, Dict[str, object]]) -> None:
    """Contrast the standard cross-view number against the same-view diagonal."""
    print()
    print("=" * 74)
    print("  PROTOCOL COMPARISON")
    print("=" * 74)
    print()
    print(f"  {'Probe':<8} {'CROSS-VIEW':>14} {'SAME-VIEW':>14}   {'Difference':>12}")
    print(f"  {'':<8} {'(standard)':>14} {'(diagonal)':>14}")
    print("  " + "-" * 54)

    for probe_name in PROBE_ORDER:
        if probe_name not in summaries:
            continue
        s = summaries[probe_name]
        cv, sv = s["crossview"], s["sameview"]
        print(f"  {probe_name:<8} {cv:>13.2f}% {sv:>13.2f}%   {cv - sv:>+11.2f}")

    cv_all = float(np.mean([summaries[p]["crossview"] for p in PROBE_ORDER if p in summaries]))
    sv_all = float(np.mean([summaries[p]["sameview"] for p in PROBE_ORDER if p in summaries]))
    print("  " + "-" * 54)
    print(f"  {'ALL':<8} {cv_all:>13.2f}% {sv_all:>13.2f}%   {cv_all - sv_all:>+11.2f}")
    print()
    print("  CROSS-VIEW is the number to report. SAME-VIEW compares gallery and")
    print("  probe recorded from the identical camera angle - a much easier task,")
    print("  shown here only for reference.")
    print("=" * 74)


def print_full_matrix(name: str, acc: np.ndarray) -> None:
    """Print one full 11x11 matrix with the diagonal bracketed."""
    col_w = 6
    print()
    print(f"  {name} - rows = gallery angle, cols = probe angle  [diagonal in brackets]")
    print("        " + " ".join(f"{a:>{col_w}}" for a in ANGLE_LABELS))
    for i, row_label in enumerate(ANGLE_LABELS):
        cells = []
        for j in range(len(ANGLES)):
            v = acc[i, j]
            cells.append(f"[{v:>4.1f}]" if i == j else f"{v:>{col_w}.2f}")
        print(f"  {row_label:>5} " + " ".join(cells))


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the fused gait model on CASIA-B test subjects 075-124.",
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
        "--frame-budget", type=int, default=DEFAULT_FRAME_BUDGET,
        help=f"Max (sequences x frames) per forward pass (default: {DEFAULT_FRAME_BUDGET}).",
    )
    parser.add_argument(
        "--num-classes", type=int, default=74,
        help="Number of classes the model was trained on (default: 74).",
    )
    parser.add_argument(
        "--dump-embeddings", type=str, default=r"results/embeddings.npz",
        help="Where to save embeddings + matrices (default: results/embeddings.npz). "
             "Pass an empty string to skip.",
    )
    parser.add_argument(
        "--show-matrix", action="store_true",
        help="Print the full 11x11 cross-view matrix for each probe type.",
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

    print("=" * 74)
    print("  Global-Local Multimodal Fusion - Evaluation Pipeline")
    print("  Protocol: CASIA-B cross-view Gallery/Probe (LST split)")
    print("=" * 74)

    # ── 1. DEVICE SETUP ───────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    if device.type == "cuda":
        # Lets cuDNN pick the fastest algorithm for our (now stable) shapes.
        torch.backends.cudnn.benchmark = True
        print(f"  GPU:  {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── 2. LOAD MODEL ─────────────────────────────────────────────────
    print(f"\nLoading model weights from: {args.weights}")
    if not os.path.exists(args.weights):
        print(f"  [ERROR] Weights file not found: {args.weights}")
        sys.exit(1)

    # Checkpoints written by train.py are dicts carrying their own config, so
    # the head type and class count never have to be supplied by hand.  Raw
    # state_dicts (the Phase 1 checkpoint) are still accepted, falling back to
    # the CLI defaults.
    ckpt = torch.load(args.weights, map_location=device, weights_only=True)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        num_classes = int(ckpt.get("num_classes", args.num_classes))
        head = str(ckpt.get("head", "linear"))
        embed_dim = int(ckpt.get("embed_dim", 256))
        print(f"  Checkpoint config: head={head}, num_classes={num_classes}, "
              f"embed_dim={embed_dim}")
        if "epoch" in ckpt:
            print(f"  Saved at epoch {int(ckpt['epoch'])}"
                  + (f" (val Rank-1 {float(ckpt['val_rank1']):.2f}%)"
                     if "val_rank1" in ckpt else ""))
    else:
        state_dict = ckpt
        num_classes, head, embed_dim = args.num_classes, "linear", 256
        print(f"  Raw state_dict: assuming head=linear, num_classes={num_classes}")

    model = GlobalLocalFusedNetwork(
        num_classes=num_classes, embed_dim=embed_dim, head=head
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {total_params:,} parameters")
    print(f"  Embedding dimension: {model.embed_dim}")
    print(f"  Mode: eval (dropout OFF, batchnorm frozen)")
    print(f"  Frame budget: {args.frame_budget} images per forward pass")

    # ── 3. DISCOVER & PRE-CACHE ALL TEST DATA INTO RAM ────────────────
    print()
    t0 = time.time()
    gallery_records, probe_dict = discover_and_cache_sequences(args.data_dir)
    print(f"[Eval] Data cached in {time.time() - t0:.1f}s")

    if not gallery_records:
        print("[ERROR] No gallery sequences found. Check your data directory.")
        sys.exit(1)

    for pname, precs in probe_dict.items():
        if not precs:
            print(f"[WARN] No {pname} probe sequences found.")

    # ── 4. EXTRACT GALLERY EMBEDDINGS & BUILD TEMPLATES ───────────────
    print(f"\nExtracting gallery embeddings ({len(gallery_records)} sequences)...")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    gallery_embeddings = extract_embeddings(
        model, gallery_records, device, frame_budget=args.frame_budget
    )
    print(f"  Done in {time.time() - t0:.1f}s - shape: {gallery_embeddings.shape}")

    if device.type == "cuda":
        print(f"  Peak VRAM usage: {torch.cuda.max_memory_allocated() / 1e6:.0f} MB")

    template_emb, template_subject, template_angle = build_gallery_templates(
        gallery_records, gallery_embeddings
    )
    print(
        f"  Gallery templates: {template_emb.shape[0]} "
        f"({len(set(template_subject))} subjects x {len(set(template_angle))} angles)"
    )

    # ── 5. EXTRACT PROBE EMBEDDINGS ───────────────────────────────────
    probe_data: Dict[str, Dict[str, np.ndarray]] = {}

    for probe_name in PROBE_ORDER:
        probe_records = probe_dict[probe_name]
        if not probe_records:
            continue

        print(f"\nExtracting {probe_name} probe embeddings ({len(probe_records)} sequences)...")
        t0 = time.time()
        emb = extract_embeddings(
            model, probe_records, device, frame_budget=args.frame_budget
        )
        print(f"  Done in {time.time() - t0:.1f}s - shape: {emb.shape}")

        probe_data[probe_name] = {
            "emb": emb,
            "subject": np.array([r.subject for r in probe_records]),
            "angle": np.array([r.angle for r in probe_records]),
            "condition": np.array([r.condition for r in probe_records]),
        }

    # ── 6. SANITY GATE ────────────────────────────────────────────────
    print()
    print("=" * 74)
    print("  SANITY CHECK - mean cosine similarity, probes vs gallery templates")
    print("=" * 74)
    all_healthy = True
    for probe_name in PROBE_ORDER:
        if probe_name not in probe_data:
            continue
        d = probe_data[probe_name]
        healthy = sanity_report(
            d["emb"], d["subject"], template_emb, template_subject, probe_name
        )
        all_healthy = all_healthy and healthy

    if not all_healthy:
        print()
        print("  [!] At least one probe set failed the sanity check. Metrics below")
        print("      may be meaningless - investigate before reporting them.")
    print("=" * 74)

    # ── 7. CROSS-VIEW MATRICES ────────────────────────────────────────
    matrices: Dict[str, np.ndarray] = {}
    summaries: Dict[str, Dict[str, object]] = {}

    for probe_name in PROBE_ORDER:
        if probe_name not in probe_data:
            continue
        d = probe_data[probe_name]
        acc = compute_crossview_matrix(
            template_emb, template_subject, template_angle,
            d["emb"], d["subject"], d["angle"],
        )
        matrices[probe_name] = acc
        summaries[probe_name] = summarise_matrix(acc)

    # ── 8. REPORT ─────────────────────────────────────────────────────
    print_crossview_table(summaries)
    print_protocol_comparison(summaries)

    if args.show_matrix:
        print()
        print("=" * 74)
        print("  FULL CROSS-VIEW MATRICES")
        print("=" * 74)
        for probe_name in PROBE_ORDER:
            if probe_name in matrices:
                print_full_matrix(probe_name, matrices[probe_name])

    # ── 9. DUMP EMBEDDINGS FOR DOWNSTREAM SCRIPTS ─────────────────────
    if args.dump_embeddings:
        os.makedirs(os.path.dirname(args.dump_embeddings) or ".", exist_ok=True)

        payload: Dict[str, np.ndarray] = {
            "gallery_emb": gallery_embeddings,
            "gallery_subject": np.array([r.subject for r in gallery_records]),
            "gallery_angle": np.array([r.angle for r in gallery_records]),
            "gallery_condition": np.array([r.condition for r in gallery_records]),
            "template_emb": template_emb,
            "template_subject": template_subject,
            "template_angle": template_angle,
            "angles": np.array(ANGLES),
            "probe_types": np.array(PROBE_ORDER),
            # Provenance
            "weights_path": np.array(args.weights),
            "embed_dim": np.array(model.embed_dim),
            "created_utc": np.array(datetime.now(timezone.utc).isoformat()),
        }

        for probe_name, d in probe_data.items():
            payload[f"probe_{probe_name}_emb"] = d["emb"]
            payload[f"probe_{probe_name}_subject"] = d["subject"]
            payload[f"probe_{probe_name}_angle"] = d["angle"]
            payload[f"probe_{probe_name}_condition"] = d["condition"]
            payload[f"matrix_{probe_name}"] = matrices[probe_name]

        np.savez_compressed(args.dump_embeddings, **payload)
        size_mb = os.path.getsize(args.dump_embeddings) / 1e6
        print(f"\nEmbeddings + matrices saved -> {args.dump_embeddings} ({size_mb:.1f} MB)")
        print("  Consumed by: plot_view_matrix.py, compute_biometrics.py")

    # ── 10. SUMMARY ───────────────────────────────────────────────────
    total_probes = sum(len(probe_dict[p]) for p in PROBE_SETS)
    overall = float(np.mean([summaries[p]["crossview"] for p in summaries])) if summaries else 0.0
    print()
    print("Evaluation complete.")
    print(f"  Gallery sequences: {len(gallery_records)} -> {template_emb.shape[0]} templates")
    print(f"  Probe sequences:   {total_probes}")
    print(f"  Overall Rank-1 (cross-view): {overall:.2f}%")
    print()


if __name__ == "__main__":
    main()
