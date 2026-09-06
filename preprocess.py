"""CASIA-B silhouette preprocessing pipeline.

Reads raw silhouettes from GaitDatasetB-silh, normalises them WITHOUT
distorting body proportions, computes a Gait Energy Image (GEI) per sequence,
and writes the result either as .npy archives (fast) or as PNGs.

Why this was rewritten
----------------------
The previous version cropped each frame to its bounding box and resized that
crop to 64x64.  Measured across 720 frames spanning every subject, condition
and angle, the raw CASIA-B silhouettes are **already normalised**:

    - every raw image is already 64x64
    - the silhouette spans the full 64px height in 100% of frames
    - it is already horizontally centred (mean centroid offset 0.54 px)

So that crop-and-resize normalised nothing.  Since the height was already 64,
its only effect was to STRETCH THE WIDTH -- by a median of 2.67x, up to 4.57x.

The damage is that the stretch was not constant.  Within a single sequence the
bounding-box width swings from about 15 to 37 px as the arms and legs move, so
the horizontal scale changed by ~2.47x from frame to frame, synchronised with
the gait cycle.  Set Pooling then took an element-wise max over frames whose
geometry disagreed, and the GEI averaged bodies at inconsistent widths.

The fix is therefore to REMOVE the resize rather than to add normalisation.

Usage:
    python preprocess.py --compare                       # visual check first
    python preprocess.py --output-dir Processed_CASIAB_v2 --npy-only
    python preprocess.py --train-only                    # subjects 001-074 only
"""

import argparse
import glob
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from tqdm import tqdm


# --- CONFIGURATION ---
INPUT_DIR = "GaitDatasetB-silh"
OUTPUT_DIR = "Processed_CASIAB"
TARGET_SIZE = 64          # output is TARGET_SIZE x TARGET_SIZE

# LST protocol: training subjects are 001-074
TRAIN_SUBJECT_UPPER = 74


def normalize_silhouette(img: np.ndarray | None, size: int = TARGET_SIZE) -> np.ndarray | None:
    """Normalise a silhouette to ``size x size`` WITHOUT changing its aspect ratio.

    Steps:
        1. Crop to the silhouette's vertical extent and scale so its height is
           exactly ``size``, preserving aspect ratio.  On CASIA-B this is a
           no-op (height is already 64 in every frame measured), but it is done
           explicitly so a frame that differs is handled correctly rather than
           silently mangled.
        2. Find the horizontal centre of mass of the foreground pixels.
        3. Copy a ``size``-wide window centred on that centroid, zero-padding
           where the window runs past the edge.

    The centroid is used rather than the bounding-box centre because a swinging
    arm drags the bounding box around while the centre of mass stays anchored
    near the torso.

    Args:
        img: Grayscale silhouette, or ``None`` if the file could not be read.

    Returns:
        ``(size, size)`` uint8 image, or ``None`` if ``img`` was ``None``.
    """
    if img is None:
        return None

    ys, xs = np.where(img > 0)
    if len(ys) == 0:
        # Completely empty frame - return a blank image
        return np.zeros((size, size), dtype=np.uint8)

    # ── 1. Vertical extent -> scale height to `size`, aspect preserved ──
    top, bottom = int(ys.min()), int(ys.max())
    cropped = img[top : bottom + 1, :]

    h, w = cropped.shape
    if h != size:
        new_w = max(1, int(round(w * size / h)))
        cropped = cv2.resize(cropped, (new_w, size), interpolation=cv2.INTER_CUBIC)

    H, W = cropped.shape

    # ── 2. Horizontal centre of mass ──
    col_mass = cropped.sum(axis=0).astype(np.float64)
    total = col_mass.sum()
    if total > 0:
        cx = int(round(float((col_mass * np.arange(W)).sum() / total)))
    else:
        cx = W // 2

    # ── 3. Centred `size`-wide window, zero-padded ──
    out = np.zeros((size, size), dtype=cropped.dtype)
    left = cx - size // 2

    src_l = max(0, left)
    src_r = min(W, left + size)
    dst_l = src_l - left
    dst_r = dst_l + (src_r - src_l)

    if src_r > src_l:
        out[:, dst_l:dst_r] = cropped[:, src_l:src_r]

    return out


def crop_and_resize_legacy(img: np.ndarray | None, size: int = TARGET_SIZE) -> np.ndarray | None:
    """The ORIGINAL (distorting) normalisation, retained only for comparison.

    Crops to the bounding box and resizes to a square, which stretches the
    width by a gait-cycle-dependent factor.  Used by ``--compare`` to show the
    before/after difference; not used by the pipeline.
    """
    if img is None:
        return None

    ys, xs = np.where(img > 0)
    if len(ys) == 0:
        return np.zeros((size, size), dtype=np.uint8)

    cropped = img[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    return cv2.resize(cropped, (size, size), interpolation=cv2.INTER_CUBIC)


def generate_gei(processed_frames: list[np.ndarray]) -> np.ndarray | None:
    """Compute the Gait Energy Image by averaging all frames of a sequence.

    Args:
        processed_frames: List of ``(64, 64)`` uint8 arrays.

    Returns:
        ``(64, 64)`` uint8 GEI, or ``None`` if the list is empty.
    """
    if not processed_frames:
        return None

    sequence_tensor = np.stack(processed_frames).astype(np.float32)
    return np.mean(sequence_tensor, axis=0).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════
# Visual comparison (the go / no-go check before reprocessing everything)
# ═══════════════════════════════════════════════════════════════════════════

def make_comparison_figure(output_path: str, n_frames: int = 6) -> None:
    """Render old vs new normalisation for a few sequences.

    The GEI row is the informative one.  Because the old pipeline averaged
    bodies stretched by different amounts, its GEIs are blurred; the new ones
    should be visibly sharper.  If they are not, the hypothesis behind this
    change is wrong and it is not worth reprocessing the dataset.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sequences = [
        ("001", "nm-01", "090"),
        ("001", "nm-01", "000"),
        ("015", "bg-01", "054"),
    ]

    fig, axes = plt.subplots(
        len(sequences) * 2, n_frames + 2,
        figsize=(1.5 * (n_frames + 2), 3.1 * len(sequences)),
    )

    for si, (subj, cond, ang) in enumerate(sequences):
        seq_dir = os.path.join(INPUT_DIR, subj, cond, ang)
        paths = sorted(glob.glob(os.path.join(seq_dir, "*.png")))
        if not paths:
            continue

        raws = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in paths]
        raws = [r for r in raws if r is not None]

        old = [crop_and_resize_legacy(r) for r in raws]
        new = [normalize_silhouette(r) for r in raws]
        old_gei, new_gei = generate_gei(old), generate_gei(new)

        # Evenly spaced frames across the gait cycle, where the stretch varies most
        idx = np.linspace(0, len(raws) - 1, n_frames).astype(int)

        for row, (frames, gei, label) in enumerate(
            [(old, old_gei, "OLD"), (new, new_gei, "NEW")]
        ):
            r = si * 2 + row
            for c, i in enumerate(idx):
                axes[r, c].imshow(frames[i], cmap="gray", vmin=0, vmax=255)
                axes[r, c].axis("off")
                if row == 0:
                    xs = np.where(raws[i] > 0)[1]
                    width = xs.max() - xs.min() + 1 if len(xs) else 0
                    axes[r, c].set_title(f"w={width}px\nstretch {64 / max(width, 1):.1f}x",
                                         fontsize=7)

            axes[r, n_frames].axis("off")
            axes[r, n_frames + 1].imshow(gei, cmap="gray", vmin=0, vmax=255)
            axes[r, n_frames + 1].axis("off")
            axes[r, n_frames + 1].set_title(f"{label} GEI", fontsize=9, fontweight="bold")

            axes[r, 0].set_ylabel(label)
            axes[r, 0].axis("off")
            axes[r, 0].text(-0.35, 0.5, f"{subj}/{cond}/{ang}\n{label}",
                            transform=axes[r, 0].transAxes, fontsize=8,
                            va="center", ha="right",
                            fontweight="bold" if label == "NEW" else "normal",
                            color="#0b6e7f" if label == "NEW" else "#a5262a")

    fig.suptitle(
        "Silhouette normalisation: OLD (bbox stretched to square) vs NEW (aspect preserved)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0.02, 0, 1, 0.96])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    print(f"Comparison figure -> {output_path}")
    print("  Check the GEI column: the NEW GEIs should be visibly sharper.")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="CASIA-B preprocessing")
    parser.add_argument(
        "--train-only", action="store_true",
        help="Process only training subjects 001-074 (faster verification).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=OUTPUT_DIR,
        help="Where to write processed sequences.",
    )
    parser.add_argument(
        "--npy-only", action="store_true",
        help="Write frames.npy and gei.npy directly, skipping the per-frame PNGs. "
             "Much faster and smaller; pack_npy.py is then unnecessary.",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Render results/preprocessing_comparison.png and exit without processing.",
    )
    parser.add_argument(
        "--compare-output", type=str, default="results/preprocessing_comparison.png",
    )
    parser.add_argument(
        "--workers", type=int, default=min(8, (os.cpu_count() or 4)),
        help="Parallel worker threads. The bottleneck is reading ~1.1M small "
             "PNGs, which parallelises well.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip sequences already present in the output directory.",
    )
    args = parser.parse_args()

    if args.compare:
        make_comparison_figure(args.compare_output)
        return

    os.makedirs(args.output_dir, exist_ok=True)

    print("Starting CASIA-B pre-processing...")
    print(f"  Normalisation: aspect-preserving, centroid-aligned, {TARGET_SIZE}x{TARGET_SIZE}")
    print(f"  Output        : {args.output_dir}/  ({'*.npy only' if args.npy_only else 'PNGs + GEI'})")

    all_subjects = sorted(
        s for s in os.listdir(INPUT_DIR)
        if os.path.isdir(os.path.join(INPUT_DIR, s))
    )

    if args.train_only:
        subjects = [s for s in all_subjects if int(s) <= TRAIN_SUBJECT_UPPER]
        print(f"  --train-only mode: processing {len(subjects)} subjects (001-074)")
    else:
        subjects = all_subjects
        print(f"  Processing all {len(subjects)} subjects")

    # ------------------------------------------------------------------
    # Flat work list so tqdm can show a meaningful total
    # ------------------------------------------------------------------
    work_items: list[tuple[str, str, str, str]] = []
    for subject in subjects:
        subject_dir = os.path.join(INPUT_DIR, subject)
        for sequence in sorted(os.listdir(subject_dir)):
            seq_dir = os.path.join(subject_dir, sequence)
            if not os.path.isdir(seq_dir):
                continue
            for angle in sorted(os.listdir(seq_dir)):
                angle_dir = os.path.join(seq_dir, angle)
                if not os.path.isdir(angle_dir):
                    continue
                work_items.append((subject, sequence, angle, angle_dir))

    print(f"  Total sequences to process: {len(work_items)}")

    # ------------------------------------------------------------------
    # Resume: skip sequences that are already complete
    # ------------------------------------------------------------------
    if args.resume:
        before = len(work_items)
        remaining = []
        for item in work_items:
            subject, sequence, angle, _ = item
            d = os.path.join(args.output_dir, subject, sequence, angle)
            done = (
                os.path.exists(os.path.join(d, "frames.npy"))
                and os.path.exists(os.path.join(d, "gei.npy"))
            ) if args.npy_only else os.path.isdir(d) and bool(os.listdir(d))
            if not done:
                remaining.append(item)
        work_items = remaining
        print(f"  Resuming: {before - len(work_items)} already done, {len(work_items)} to go")

    if not work_items:
        print("\nNothing to do - all sequences already processed.")
        return

    # ------------------------------------------------------------------
    # Process.  Threads (not processes) because the bottleneck is
    # cv2.imread on ~1.1M small PNGs, and OpenCV releases the GIL during
    # decode -- so threading gives most of the speedup with none of the
    # Windows process-spawning overhead.
    # ------------------------------------------------------------------
    def process_one(item) -> int:
        subject, sequence, angle, angle_dir = item

        frame_paths = sorted(glob.glob(os.path.join(angle_dir, "*.png")))
        if not frame_paths:
            return 0

        processed_frames: list[np.ndarray] = []
        for path in frame_paths:
            out = normalize_silhouette(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
            if out is not None:
                processed_frames.append(out)

        if not processed_frames:
            return 0

        gei_image = generate_gei(processed_frames)

        save_dir = os.path.join(args.output_dir, subject, sequence, angle)
        os.makedirs(save_dir, exist_ok=True)

        if args.npy_only:
            # Written in exactly the layout dataset.py and eval.py expect.
            np.save(os.path.join(save_dir, "frames.npy"), np.stack(processed_frames))
            np.save(os.path.join(save_dir, "gei.npy"), gei_image)
        else:
            for i, frame in enumerate(processed_frames):
                cv2.imwrite(os.path.join(save_dir, f"frame_{i + 1:03d}.png"), frame)
            condition = sequence.split("-")[0]        # 'nm', 'bg' or 'cl'
            cv2.imwrite(
                os.path.join(save_dir, f"{subject}_{condition}_{angle}_GEI.png"),
                gei_image,
            )

        return len(processed_frames)

    print(f"  Workers: {args.workers}")

    counts: list[int] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for n in tqdm(
            pool.map(process_one, work_items),
            total=len(work_items),
            desc="Processing",
        ):
            counts.append(n)

    skipped = sum(1 for n in counts if n == 0)
    total_frames = sum(counts)

    print("\nPre-processing complete.")
    print(f"  Sequences processed : {len(work_items) - skipped}")
    print(f"  Sequences skipped   : {skipped}")
    print(f"  Frames written      : {total_frames:,}")
    print(f"  Output directory    : {args.output_dir}/")
    if not args.npy_only:
        print("  Next: python pack_npy.py")


if __name__ == "__main__":
    main()
