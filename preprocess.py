"""CASIA-B silhouette preprocessing pipeline.

Reads raw silhouette images from GaitDatasetB-silh, applies bounding-box
cropping and resizing to 64×64, computes a Gait Energy Image (GEI) per
sequence, and saves EVERY processed frame plus the GEI.

Usage:
    # Process only training subjects (001–074) for fast verification:
    python preprocess.py --train-only

    # Process all 124 subjects:
    python preprocess.py
"""

import argparse
import glob
import os

import cv2
import numpy as np
from tqdm import tqdm


# --- CONFIGURATION ---
INPUT_DIR = "GaitDatasetB-silh"
OUTPUT_DIR = "Processed_CASIAB"
TARGET_SIZE = (64, 64)

# LST protocol: training subjects are 001–074
TRAIN_SUBJECT_UPPER = 74


def crop_and_resize(image_path: str) -> np.ndarray | None:
    """Load a raw silhouette, crop to its bounding box, and resize to TARGET_SIZE.

    Args:
        image_path: Path to a grayscale silhouette PNG.

    Returns:
        Resized (64, 64) uint8 image, or None if the file cannot be read.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # Bounding box extraction (non-zero pixels)
    y_coords, x_coords = np.where(img > 0)

    if len(y_coords) == 0 or len(x_coords) == 0:
        # Completely empty frame – return a blank image
        return np.zeros(TARGET_SIZE, dtype=np.uint8)

    y_min, y_max = np.min(y_coords), np.max(y_coords)
    x_min, x_max = np.min(x_coords), np.max(x_coords)

    cropped_img = img[y_min : y_max + 1, x_min : x_max + 1]

    # Cubic interpolation resize to the canonical 64×64
    resized_img = cv2.resize(cropped_img, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)
    return resized_img


def generate_gei(processed_frames: list[np.ndarray]) -> np.ndarray | None:
    """Compute the Gait Energy Image by averaging all frames.

    Args:
        processed_frames: List of (64, 64) uint8 arrays.

    Returns:
        (64, 64) uint8 GEI, or None if the list is empty.
    """
    if not processed_frames:
        return None

    # Stack → (N, 64, 64) and average across the temporal axis
    sequence_tensor = np.stack(processed_frames).astype(np.float32)
    gei = np.mean(sequence_tensor, axis=0)
    return gei.astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="CASIA-B Preprocessing")
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Process only training subjects 001–074 (faster verification).",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Starting CASIA-B Pre-processing...")

    # Enumerate subject folders
    all_subjects = sorted(
        s for s in os.listdir(INPUT_DIR)
        if os.path.isdir(os.path.join(INPUT_DIR, s))
    )

    if args.train_only:
        subjects = [s for s in all_subjects if int(s) <= TRAIN_SUBJECT_UPPER]
        print(f"  → --train-only mode: processing {len(subjects)} subjects (001–074)")
    else:
        subjects = all_subjects
        print(f"  → Processing all {len(subjects)} subjects")

    # ------------------------------------------------------------------
    # Build a flat list of (subject, sequence, angle, angle_dir) tuples
    # so tqdm can show a meaningful total.
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

    print(f"  → Total sequences to process: {len(work_items)}")

    # ------------------------------------------------------------------
    # Process each sequence
    # ------------------------------------------------------------------
    skipped = 0

    for subject, sequence, angle, angle_dir in tqdm(work_items, desc="Processing"):
        frame_paths = sorted(glob.glob(os.path.join(angle_dir, "*.png")))
        if not frame_paths:
            skipped += 1
            continue

        # Process every raw frame
        processed_frames: list[np.ndarray] = []
        for path in frame_paths:
            processed_img = crop_and_resize(path)
            if processed_img is not None:
                processed_frames.append(processed_img)

        if not processed_frames:
            skipped += 1
            continue

        # Compute GEI from ALL frames
        gei_image = generate_gei(processed_frames)

        # Create output directory for this sequence
        save_dir = os.path.join(OUTPUT_DIR, subject, sequence, angle)
        os.makedirs(save_dir, exist_ok=True)

        # ┌──────────────────────────────────────────────────────────┐
        # │  FIX #1: SAVE **ALL** PROCESSED FRAMES (not just one)   │
        # │  Old code:  cv2.imwrite(..., processed_frames[0])       │
        # │  New code:  save every frame as frame_001.png, etc.     │
        # └──────────────────────────────────────────────────────────┘
        for i, frame in enumerate(processed_frames):
            frame_filename = f"frame_{i + 1:03d}.png"
            cv2.imwrite(os.path.join(save_dir, frame_filename), frame)

        # Save the GEI (naming convention matches the dataset loader's glob)
        condition = sequence.split("-")[0]  # 'nm', 'bg', or 'cl'
        gei_filename = f"{subject}_{condition}_{angle}_GEI.png"
        cv2.imwrite(os.path.join(save_dir, gei_filename), gei_image)

    print(f"\nPre-processing complete!")
    print(f"  → Sequences processed: {len(work_items) - skipped}")
    print(f"  → Sequences skipped (empty): {skipped}")
    print(f"  → Output directory: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()