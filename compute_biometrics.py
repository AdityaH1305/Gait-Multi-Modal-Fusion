"""Open-set gait verification metrics: ROC, AUC and Equal Error Rate.

Reads ``results/embeddings.npz`` (produced by ``eval.py``) and scores every
probe against every gallery template, then measures how well a SINGLE global
similarity threshold separates genuine pairs from impostor pairs.

This is a fundamentally harder question than the Rank-1 identification in
``eval.py``.  Rank-1 only asks whether the correct identity scored highest for
one particular probe.  Verification asks whether one fixed threshold can
accept genuine matches and reject impostors across identities the model has
never seen - which requires the embedding space to be geometrically
consistent, not merely locally discriminative.

Protocol:
    Probes    : nm-05/06, bg-01/02, cl-01/02  (subjects 075-124)
    Gallery   : nm-01..04 averaged into one template per (subject, angle)
    Pairs     : every probe x every template, EXCLUDING identical-view pairs,
                to stay consistent with the cross-view Rank-1 headline.

Usage:
    python eval.py                  # must be run first - produces the .npz
    python compute_biometrics.py
    python compute_biometrics.py --include-same-view
"""

import argparse
import os
import sys
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve


PROBE_ORDER = ["NM", "BG", "CL"]
CURVE_COLORS = {
    "NM": "#2563eb",
    "BG": "#16a34a",
    "CL": "#dc2626",
    "ALL": "#7c3aed",
}


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

def calculate_eer(
    fpr: np.ndarray,
    tpr: np.ndarray,
    thresholds: np.ndarray,
) -> Tuple[float, float]:
    """Compute the Equal Error Rate and the threshold at which it occurs.

    The EER is where the False Acceptance Rate equals the False Rejection
    Rate.  In practice the ROC is sampled at discrete points, so FAR and FRR
    rarely land exactly on top of one another; we take the closest crossing
    and report the MIDPOINT of the two.

    (Taking ``min(fpr, fnr)`` there - as an earlier version of this file did -
    systematically reports an EER better than the true operating point.)
    """
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return eer, float(thresholds[idx])


def score_pairs(
    probe_emb: np.ndarray,
    probe_subject: np.ndarray,
    probe_angle: np.ndarray,
    tmpl_emb: np.ndarray,
    tmpl_subject: np.ndarray,
    tmpl_angle: np.ndarray,
    exclude_same_view: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build flat (score, is_genuine) arrays for every probe-template pair.

    Embeddings are L2-normalised, so the dot product IS the cosine similarity.

    Returns:
        ``(scores, y_true)`` - 1-D float32 and uint8 arrays of equal length.
    """
    sims = (probe_emb @ tmpl_emb.T).astype(np.float32)          # (P, T)
    genuine = (probe_subject[:, None] == tmpl_subject[None, :])  # (P, T)

    if exclude_same_view:
        keep = probe_angle[:, None] != tmpl_angle[None, :]
    else:
        keep = np.ones_like(genuine, dtype=bool)

    return sims[keep], genuine[keep].astype(np.uint8)


def compute_metrics(scores: np.ndarray, y_true: np.ndarray) -> Dict[str, object]:
    """Compute ROC, AUC, EER and genuine/impostor score statistics."""
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = float(auc(fpr, tpr))
    eer, eer_threshold = calculate_eer(fpr, tpr, thresholds)

    genuine = scores[y_true == 1]
    impostor = scores[y_true == 0]

    return {
        "fpr": fpr,
        "tpr": tpr,
        "auc": roc_auc,
        "eer": eer,
        "threshold": eer_threshold,
        "n_genuine": int(genuine.size),
        "n_impostor": int(impostor.size),
        "genuine_mean": float(genuine.mean()),
        "impostor_mean": float(impostor.mean()),
        "genuine": genuine,
        "impostor": impostor,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

def plot_roc(results: Dict[str, Dict[str, object]], output_path: str) -> None:
    """Plot one ROC curve per probe condition plus the combined curve."""
    plt.figure(figsize=(9, 8))

    for name, r in results.items():
        style = "-" if name == "ALL" else "--"
        width = 2.6 if name == "ALL" else 1.9
        plt.plot(
            r["fpr"], r["tpr"],
            color=CURVE_COLORS[name], lw=width, linestyle=style,
            label=f"{name}  (AUC = {r['auc']:.4f}, EER = {r['eer'] * 100:.2f}%)",
        )

    plt.plot([0, 1], [0, 1], color="#94a3b8", lw=1.5, linestyle=":",
             label="Random guess (AUC = 0.5000)")

    if "ALL" in results:
        eer = results["ALL"]["eer"]
        plt.plot([eer], [1 - eer], marker="o", markersize=9,
                 color=CURVE_COLORS["ALL"], markeredgecolor="white",
                 markeredgewidth=1.5, zorder=5)

    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel("False Positive Rate (FAR)", fontsize=12, labelpad=10)
    plt.ylabel("True Positive Rate (1 - FRR)", fontsize=12, labelpad=10)
    plt.title(
        "ROC - Open-Set Gait Verification (CASIA-B subjects 075-124)",
        fontsize=14, pad=15, weight="bold",
    )
    plt.legend(loc="lower right", fontsize=10.5)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_score_distributions(
    results: Dict[str, Dict[str, object]],
    output_path: str,
) -> None:
    """Plot genuine vs impostor score histograms per probe condition.

    More diagnostic than the ROC alone: if the two distributions sit on top of
    one another, the embeddings carry no usable verification signal and no
    threshold can help.  A failure of this kind is impossible to miss here,
    and easy to miss in an AUC number.
    """
    names = [n for n in PROBE_ORDER if n in results]
    fig, axes = plt.subplots(1, len(names), figsize=(5.4 * len(names), 4.4), sharey=True)
    if len(names) == 1:
        axes = [axes]

    bins = np.linspace(-1.0, 1.0, 120)

    for ax, name in zip(axes, names):
        r = results[name]
        ax.hist(r["impostor"], bins=bins, density=True, alpha=0.6,
                color="#94a3b8", label=f"Impostor (n={r['n_impostor']:,})")
        ax.hist(r["genuine"], bins=bins, density=True, alpha=0.7,
                color=CURVE_COLORS[name], label=f"Genuine (n={r['n_genuine']:,})")
        ax.axvline(r["threshold"], color="#1a1a1a", linestyle="--", lw=1.4,
                   label=f"EER threshold = {r['threshold']:.3f}")
        ax.set_title(
            f"{name}   AUC {r['auc']:.4f}   EER {r['eer'] * 100:.2f}%",
            fontsize=11.5, weight="bold",
        )
        ax.set_xlabel("Cosine similarity", fontsize=10)
        ax.legend(fontsize=8.5, loc="upper left")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Density", fontsize=10)
    fig.suptitle(
        "Genuine vs Impostor Score Distributions",
        fontsize=13.5, weight="bold", y=1.02,
    )
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Public convenience wrapper
# ═══════════════════════════════════════════════════════════════════════════

def generate_biometric_metrics(
    probe_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    probe_labels: np.ndarray,
    gallery_labels: np.ndarray,
    output_path: str = "results/gait_verification_roc.png",
) -> Dict[str, object]:
    """Score one probe set against one gallery, plot the ROC, return metrics.

    Kept as a single-set entry point; ``main()`` drives the per-condition
    breakdown directly.  Assumes embeddings are already L2-normalised.
    """
    sims = (probe_embeddings @ gallery_embeddings.T).astype(np.float32)
    y_true = (probe_labels[:, None] == gallery_labels[None, :]).astype(np.uint8)

    metrics = compute_metrics(sims.ravel(), y_true.ravel())
    plot_roc({"ALL": metrics}, output_path)
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute open-set gait verification metrics (ROC / AUC / EER)."
    )
    parser.add_argument(
        "--embeddings", type=str, default="results/embeddings.npz",
        help="Path to the .npz produced by eval.py (default: results/embeddings.npz).",
    )
    parser.add_argument(
        "--roc-output", type=str, default="results/gait_verification_roc.png",
        help="Where to save the ROC figure.",
    )
    parser.add_argument(
        "--dist-output", type=str, default="results/score_distributions.png",
        help="Where to save the score-distribution figure.",
    )
    parser.add_argument(
        "--include-same-view", action="store_true",
        help="Include gallery/probe pairs from the identical camera angle "
             "(easier; excluded by default to match the Rank-1 protocol).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.embeddings):
        print(f"[ERROR] Not found: {args.embeddings}")
        print("        Run `python eval.py` first - it produces this file.")
        sys.exit(1)

    data = np.load(args.embeddings, allow_pickle=False)
    exclude_same_view = not args.include_same_view

    print("=" * 74)
    print("  OPEN-SET GAIT VERIFICATION")
    print("=" * 74)
    print(f"  Embeddings:     {args.embeddings}")
    print(f"  Source weights: {data['weights_path']}")
    print(f"  Created (UTC):  {data['created_utc']}")
    print(f"  Same-view pairs: {'INCLUDED' if not exclude_same_view else 'excluded'}")

    tmpl_emb = data["template_emb"]
    tmpl_subject = data["template_subject"]
    tmpl_angle = data["template_angle"]
    print(f"  Gallery templates: {tmpl_emb.shape[0]}")

    results: Dict[str, Dict[str, object]] = {}
    all_scores, all_labels = [], []

    for name in PROBE_ORDER:
        key = f"probe_{name}_emb"
        if key not in data:
            print(f"  [WARN] No {name} probes in the embeddings file - skipping.")
            continue

        scores, y_true = score_pairs(
            data[key], data[f"probe_{name}_subject"], data[f"probe_{name}_angle"],
            tmpl_emb, tmpl_subject, tmpl_angle,
            exclude_same_view=exclude_same_view,
        )

        if y_true.sum() == 0:
            print(f"  [ERROR] {name}: zero genuine pairs - cannot compute a ROC.")
            continue

        results[name] = compute_metrics(scores, y_true)
        all_scores.append(scores)
        all_labels.append(y_true)

    if not results:
        print("[ERROR] No probe sets could be scored.")
        sys.exit(1)

    # Combined curve across all three conditions
    results["ALL"] = compute_metrics(
        np.concatenate(all_scores), np.concatenate(all_labels)
    )

    # ── Report ────────────────────────────────────────────────────────
    print()
    print("=" * 74)
    print(f"  {'Probe':<7}{'AUC':>9}{'EER':>10}{'Threshold':>12}"
          f"{'Genuine':>11}{'Impostor':>11}{'Sep':>8}")
    print("  " + "-" * 70)
    for name, r in results.items():
        sep = r["genuine_mean"] - r["impostor_mean"]
        print(
            f"  {name:<7}{r['auc']:>9.4f}{r['eer'] * 100:>9.2f}%{r['threshold']:>12.4f}"
            f"{r['genuine_mean']:>+11.4f}{r['impostor_mean']:>+11.4f}{sep:>+8.3f}"
        )
    print("=" * 74)

    if results["ALL"]["auc"] < 0.6:
        print()
        print("  [WARNING] Combined AUC is close to chance (0.5). Either the")
        print("            embeddings carry little identity signal, or something")
        print("            upstream is wrong. Check eval.py's sanity report.")

    plot_roc(results, args.roc_output)
    plot_score_distributions(results, args.dist_output)

    print()
    print(f"ROC curve            -> {args.roc_output}")
    print(f"Score distributions  -> {args.dist_output}")
    print()


if __name__ == "__main__":
    main()
