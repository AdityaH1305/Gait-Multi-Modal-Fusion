"""Render the measured cross-view Rank-1 accuracy matrices as heatmaps.

Reads ``results/embeddings.npz`` (produced by ``eval.py``) and plots one
11x11 heatmap per probe condition - NM, BG and CL - on a shared colour scale.

Rows are gallery viewing angles, columns are probe viewing angles.  The
diagonal (gallery angle == probe angle) is same-view matching, a much easier
task than the cross-view protocol, so it is outlined and excluded from the
headline mean printed beneath each panel.

Usage:
    python eval.py                 # must be run first - produces the .npz
    python plot_view_matrix.py
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


PROBE_ORDER = ["NM", "BG", "CL"]
PROBE_TITLES = {
    "NM": "NM - Normal Walking",
    "BG": "BG - Carrying a Bag",
    "CL": "CL - Wearing a Coat",
}


def crossview_mean(acc: np.ndarray) -> float:
    """Mean of the off-diagonal cells - the standard reported number."""
    eye = np.eye(acc.shape[0], dtype=bool)
    return float(np.nanmean(np.where(eye, np.nan, acc)))


def sameview_mean(acc: np.ndarray) -> float:
    """Mean of the diagonal - same-view matching, shown for reference only."""
    eye = np.eye(acc.shape[0], dtype=bool)
    return float(np.nanmean(acc[eye]))


def draw_panel(ax, acc: np.ndarray, angle_labels, title: str) -> None:
    """Draw one annotated 11x11 heatmap onto ``ax``."""
    n = acc.shape[0]
    im = ax.imshow(acc, cmap="YlGnBu", vmin=0.0, vmax=100.0, aspect="equal")

    # Annotate every cell, flipping text colour so it stays legible on the
    # dark end of the colormap.
    for i in range(n):
        for j in range(n):
            v = acc[i, j]
            if np.isnan(v):
                continue
            ax.text(
                j, i, f"{v:.0f}",
                ha="center", va="center",
                fontsize=7.5,
                color="white" if v > 55 else "#1a1a1a",
                fontweight="bold" if i == j else "normal",
            )

    # Outline the diagonal - excluded from the cross-view mean.
    for i in range(n):
        ax.add_patch(
            Rectangle(
                (i - 0.5, i - 0.5), 1, 1,
                fill=False, edgecolor="#d1495b", linewidth=1.8,
            )
        )

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"{a}" for a in angle_labels], fontsize=8)
    ax.set_yticklabels([f"{a}" for a in angle_labels], fontsize=8)
    ax.set_xlabel("Probe angle (deg)", fontsize=9.5, labelpad=6)
    ax.set_ylabel("Gallery angle (deg)", fontsize=9.5, labelpad=6)

    cv, sv = crossview_mean(acc), sameview_mean(acc)
    ax.set_title(
        f"{title}\ncross-view {cv:.2f}%   (same-view {sv:.2f}%)",
        fontsize=11, pad=10, weight="bold",
    )

    # Thin grid between cells
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", length=0)

    return im


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot measured CASIA-B cross-view accuracy matrices."
    )
    parser.add_argument(
        "--embeddings", type=str, default="results/embeddings.npz",
        help="Path to the .npz produced by eval.py (default: results/embeddings.npz).",
    )
    parser.add_argument(
        "--output", type=str, default="results/cross_view_accuracy_matrix.png",
        help="Where to save the figure.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.embeddings):
        print(f"[ERROR] Not found: {args.embeddings}")
        print("        Run `python eval.py` first - it produces this file.")
        sys.exit(1)

    data = np.load(args.embeddings, allow_pickle=False)

    angle_labels = [str(int(a)) for a in data["angles"]]
    available = [p for p in PROBE_ORDER if f"matrix_{p}" in data]

    if not available:
        print(f"[ERROR] {args.embeddings} contains no cross-view matrices.")
        sys.exit(1)

    print(f"Loaded {args.embeddings}")
    print(f"  Source weights: {data['weights_path']}")
    print(f"  Created (UTC):  {data['created_utc']}")
    print(f"  Probe sets:     {', '.join(available)}")

    fig, axes = plt.subplots(1, len(available), figsize=(6.2 * len(available), 6.4))
    if len(available) == 1:
        axes = [axes]

    im = None
    for ax, probe_name in zip(axes, available):
        acc = data[f"matrix_{probe_name}"]
        im = draw_panel(ax, acc, angle_labels, PROBE_TITLES[probe_name])
        print(
            f"  {probe_name}: cross-view {crossview_mean(acc):6.2f}%  |  "
            f"same-view {sameview_mean(acc):6.2f}%"
        )

    fig.suptitle(
        "Cross-View Rank-1 Recognition Accuracy - CASIA-B test subjects 075-124",
        fontsize=14, weight="bold", y=0.99,
    )

    # Reserve the bottom strip for the colourbar (placed by fig.colorbar) and
    # the footnote below it, so the two never overlap.
    fig.tight_layout(rect=[0, 0.11, 1, 0.97])

    cbar = fig.colorbar(
        im, ax=axes, orientation="horizontal",
        fraction=0.030, pad=0.10, aspect=60,
    )
    cbar.set_label("Rank-1 Recognition Accuracy (%)", fontsize=10, labelpad=6)

    fig.text(
        0.5, 0.012,
        "Red outline marks the diagonal (gallery angle = probe angle): same-view matching, "
        "excluded from the cross-view mean.",
        ha="center", fontsize=9, color="#444444",
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
