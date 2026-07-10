"""Convert existing Processed_CASIAB PNG folders into fast-loading .npy archives.

This is a ONE-TIME post-processing step that runs AFTER preprocess.py.
It reads all the individual frame PNGs in each sequence folder and packs
them into two numpy files:

    frames.npy  – (N, 64, 64) uint8 array of all silhouette frames
    gei.npy     – (64, 64)    uint8 array of the Gait Energy Image

Loading a single .npy file is ~100× faster than reading ~56 individual
PNGs with cv2.imread(), which eliminates the I/O bottleneck that was
causing the 2+ hour epoch hang.

Usage:
    # Convert only training subjects (fast):
    python pack_npy.py --train-only

    # Convert all subjects:
    python pack_npy.py
"""

import argparse
import glob
import os

import cv2
import numpy as np
from tqdm import tqdm

DATA_DIR = "Processed_CASIAB"
TRAIN_SUBJECT_UPPER = 74


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack processed PNGs into fast-loading .npy archives."
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Convert only training subjects 001–074.",
    )
    args = parser.parse_args()

    # Discover subject folders
    all_subjects = sorted(
        s for s in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, s))
    )

    if args.train_only:
        subjects = [s for s in all_subjects if int(s) <= TRAIN_SUBJECT_UPPER]
        print(f"Converting training subjects only: {len(subjects)} subjects")
    else:
        subjects = all_subjects
        print(f"Converting all subjects: {len(subjects)}")

    # Build flat work list for tqdm
    work_items: list[str] = []
    for subject in subjects:
        subject_dir = os.path.join(DATA_DIR, subject)
        for condition in sorted(os.listdir(subject_dir)):
            cond_dir = os.path.join(subject_dir, condition)
            if not os.path.isdir(cond_dir):
                continue
            for angle in sorted(os.listdir(cond_dir)):
                angle_dir = os.path.join(cond_dir, angle)
                if not os.path.isdir(angle_dir):
                    continue
                work_items.append(angle_dir)

    print(f"Sequences to convert: {len(work_items)}")

    converted = 0
    skipped = 0
    already_done = 0

    for seq_dir in tqdm(work_items, desc="Packing .npy"):
        frames_npy_path = os.path.join(seq_dir, "frames.npy")
        gei_npy_path = os.path.join(seq_dir, "gei.npy")

        # Skip if already converted
        if os.path.exists(frames_npy_path) and os.path.exists(gei_npy_path):
            already_done += 1
            continue

        # Find frame PNGs and GEI PNG
        all_pngs = sorted(glob.glob(os.path.join(seq_dir, "*.png")))
        frame_pngs = [f for f in all_pngs if "GEI" not in f]
        gei_pngs = [f for f in all_pngs if "GEI" in f]

        if not frame_pngs or not gei_pngs:
            skipped += 1
            continue

        # Load and stack all frames
        frames = []
        for path in frame_pngs:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                frames.append(img)

        if not frames:
            skipped += 1
            continue

        frames_array = np.stack(frames)  # (N, 64, 64) uint8

        # Load GEI
        gei = cv2.imread(gei_pngs[0], cv2.IMREAD_GRAYSCALE)  # (64, 64) uint8

        # Save as .npy (compact, fast to load)
        np.save(frames_npy_path, frames_array)
        np.save(gei_npy_path, gei)

        converted += 1

    print(f"\nDone!")
    print(f"  Converted:    {converted}")
    print(f"  Already done: {already_done}")
    print(f"  Skipped:      {skipped}")

    # Show size comparison for one sequence
    if work_items:
        sample_dir = work_items[0]
        png_size = sum(
            os.path.getsize(f)
            for f in glob.glob(os.path.join(sample_dir, "*.png"))
        )
        npy_size = sum(
            os.path.getsize(os.path.join(sample_dir, f))
            for f in ["frames.npy", "gei.npy"]
            if os.path.exists(os.path.join(sample_dir, f))
        )
        print(f"\nSize comparison ({os.path.basename(sample_dir)}):")
        print(f"  PNGs total:  {png_size / 1024:.1f} KB")
        print(f"  NPY total:   {npy_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
