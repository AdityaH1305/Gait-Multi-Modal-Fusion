"""Training script for the Global-Local Multimodal Fusion gait network.

Joint loss:
    Total = CrossEntropy(logits, labels) + lambda * TripletMargin(a, p, n)

CrossEntropy gives a dense gradient (every sample contributes) while the
batch-hard triplet term enforces intra-class compaction and inter-class
separation directly in the 256-D embedding space.

Phase 2 changes
---------------
**The epoch definition was the dominant bug.**  ``PKBatchSampler.__len__``
used to return ``len(unique_labels) // P``, treating one epoch as a single
pass over identities rather than sequences.  With P=4 and 74 identities that
is 18 batches/epoch -- 288 of 8,107 sequences, 3.6% of the data -- and with
2-step gradient accumulation, 9 optimizer steps per epoch, or 1,350 for a
full 150-epoch run.  Cross-entropy started at ln(74)=4.30 and only reached
3.12; the model never converged, it ran out of epochs.

The sampler now defines an epoch by sequence count, giving 126 batches/epoch
at P=8/K=8 -- a full pass over the data and roughly 14x the gradient steps.

**The classifier head was capped.**  The old ``nn.Linear`` head consumed an
L2-normalised embedding, so its largest achievable logit was ``||w||``.  In
the Phase 1 checkpoint those norms averaged 2.16, which floors cross-entropy
at about 2.24 no matter how good the embedding gets.  ``--head cosface``
applies an explicit scale and angular margin, removing the cap.

**Model selection.**  Subjects 001-064 train and 065-074 are held out.  Every
few epochs the held-out identities are scored with the same cross-view
gallery/probe protocol used by eval.py, and the best checkpoint is kept.

Usage:
    python train.py --smoke-test                     # ~1 min, saves nothing
    python train.py --run-name A --epochs 150 --head linear  --lr 1e-3
    python train.py --run-name D --epochs 400 --head cosface --lr 1e-4 --occlusion-aug 0.3
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import eval as eval_mod
from dataset import GaitMultiModalDataset, gait_collate_fn, PKBatchSampler
from model import GlobalLocalFusedNetwork


DEFAULT_DATA_DIR = r"Processed_CASIAB"


# ═══════════════════════════════════════════════════════════════════════════
# Online Batch-Hard Triplet Mining
# ═══════════════════════════════════════════════════════════════════════════

def mine_batch_hard_triplets(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Online batch-hard triplet mining from a PK-structured mini-batch.

    For each sample as anchor, selects the same-identity sample at the
    LARGEST distance (hardest positive) and the different-identity sample at
    the SMALLEST distance (hardest negative).

    Requires PK-sampled batches.  With P identities per batch, the "hardest
    negative" is drawn from only P-1 candidates -- which is why P=4 gave such
    a weak signal, and why P=8 (56 negatives per anchor at K=8) matters.

    Returns:
        ``(anchors, positives, negatives)``, each ``(B, D)``.  Gradient
        connectivity is preserved: these index into the original embeddings.
    """
    B = embeddings.size(0)

    with torch.no_grad():
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)

        labels_col = labels.unsqueeze(1)
        labels_row = labels.unsqueeze(0)
        same_identity = labels_col == labels_row

        not_self = ~torch.eye(B, dtype=torch.bool, device=embeddings.device)
        positive_mask = same_identity & not_self
        negative_mask = ~same_identity

        pos_dists = dist_matrix.clone()
        pos_dists[~positive_mask] = -1.0
        hardest_pos_idx = pos_dists.argmax(dim=1)

        neg_dists = dist_matrix.clone()
        neg_dists[~negative_mask] = float("inf")
        hardest_neg_idx = neg_dists.argmin(dim=1)

    return embeddings, embeddings[hardest_pos_idx], embeddings[hardest_neg_idx]


# ═══════════════════════════════════════════════════════════════════════════
# Validation: cross-view Rank-1 on held-out identities
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_validation(
    model: GlobalLocalFusedNetwork,
    gallery_records: List,
    probe_dict: Dict[str, List],
    device: torch.device,
    frame_budget: int,
) -> Tuple[float, Dict[str, float]]:
    """Score held-out identities with the Phase 1 cross-view protocol.

    Reuses eval.py end to end -- gallery templating, the 11x11 cross-view
    matrix and its summary -- so the validation signal is measured exactly the
    same way as the final reported number, just on different subjects.

    Returns:
        ``(mean_crossview_rank1, per_condition)`` in percent.
    """
    was_training = model.training
    model.eval()

    gallery_emb = eval_mod.extract_embeddings(
        model, gallery_records, device, frame_budget=frame_budget, verbose=False
    )
    tmpl_emb, tmpl_subj, tmpl_ang = eval_mod.build_gallery_templates(
        gallery_records, gallery_emb
    )

    per_condition: Dict[str, float] = {}
    for name in eval_mod.PROBE_ORDER:
        records = probe_dict.get(name, [])
        if not records:
            continue

        probe_emb = eval_mod.extract_embeddings(
            model, records, device, frame_budget=frame_budget, verbose=False
        )
        acc = eval_mod.compute_crossview_matrix(
            tmpl_emb, tmpl_subj, tmpl_ang,
            probe_emb,
            np.array([r.subject for r in records]),
            np.array([r.angle for r in records]),
        )
        per_condition[name] = eval_mod.summarise_matrix(acc)["crossview"]

    if was_training:
        model.train()

    mean = float(np.mean(list(per_condition.values()))) if per_condition else 0.0
    return mean, per_condition


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

CHECKPOINT_NAME = "last_checkpoint.pth"

# Config fields that must match when resuming.  Changing any of these mid-run
# would silently produce a model that is not what either command line asked
# for, so a mismatch is refused rather than warned about.
RESUME_CRITICAL_FIELDS = (
    "head", "neck", "bin_dim", "depth",
    "lr", "p", "k", "frames", "accum", "scale", "margin",
    "data_dir", "train_upper", "val_upper", "occlusion_aug", "flip_prob",
    "warmup_epochs", "epochs", "no_val",
)


def save_checkpoint(
    path: str,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    history: Dict,
    best_val: float,
    best_epoch: int,
    num_classes: int,
    args,
    model_kwargs: Dict,
) -> None:
    """Write a resumable checkpoint atomically.

    Written to a temporary file and then renamed, because ``os.replace`` is
    atomic: an interrupt during the write leaves the previous good checkpoint
    intact rather than a truncated file that would fail to load.

    Note: Python and NumPy RNG states are deliberately not saved.  Restoring
    them would require ``weights_only=False`` on load, and the only cost of
    omitting them is that the augmentation stream differs after a resume,
    which does not affect training.
    """
    payload = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,                 # number of epochs COMPLETED
        "history": history,
        "best_val": best_val,
        "best_epoch": best_epoch,
        "num_classes": num_classes,
        "head": args.head,
        "embed_dim": model_kwargs.get("embed_dim", 256),
        "model_kwargs": model_kwargs,
        "config": vars(args),
    }

    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def save_curves(history: Dict, out_path: str, epochs: int) -> None:
    """Four-panel training summary, including the validation curve.

    The validation panel is the one that matters for Phase 2: it is the only
    signal that distinguishes "still learning" from "overfitting", which the
    project previously had no way to tell apart.
    """
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    x = range(1, len(history["ce"]) + 1)

    axes[0].plot(x, history["ce"], color="#2563eb", linewidth=2)
    axes[0].axhline(np.log(history["num_classes"]), color="#dc2626", linestyle=":",
                    label=f"ln({history['num_classes']}) = random")
    axes[0].set_title("Cross-Entropy Loss", fontsize=14, fontweight="bold")
    axes[0].legend(loc="upper right")

    axes[1].plot(x, history["triplet"], color="#dc2626", linewidth=2)
    axes[1].axhline(0.0, color="gray", linestyle=":", alpha=0.6, label="margin satisfied")
    axes[1].set_title("Triplet Loss", fontsize=14, fontweight="bold")
    axes[1].legend(loc="upper right")

    axes[2].plot(x, history["train_acc"], color="#16a34a", linewidth=2)
    axes[2].set_ylim(0, 100)
    axes[2].set_title("Training Accuracy (CE head)", fontsize=14, fontweight="bold")

    if history["val_epochs"]:
        axes[3].plot(history["val_epochs"], history["val_rank1"],
                     color="#7c3aed", linewidth=2, marker="o", markersize=4)
        best_i = int(np.argmax(history["val_rank1"]))
        axes[3].plot(history["val_epochs"][best_i], history["val_rank1"][best_i],
                     marker="*", markersize=18, color="#f59e0b", zorder=5,
                     label=f"best {history['val_rank1'][best_i]:.2f}% "
                           f"@ epoch {history['val_epochs'][best_i]}")
        axes[3].legend(loc="lower right")
    else:
        axes[3].text(0.5, 0.5, "validation disabled", ha="center", va="center",
                     transform=axes[3].transAxes, color="gray")
    axes[3].set_title("Validation Cross-View Rank-1", fontsize=14, fontweight="bold")
    axes[3].set_ylabel("Accuracy (%)")

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(True, linestyle="--", alpha=0.5)
    axes[0].set_ylabel("Loss")
    axes[1].set_ylabel("Loss")
    axes[2].set_ylabel("Accuracy (%)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the multimodal gait network.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-name", type=str, default="run",
                   help="Outputs go to results/<run-name>/.")
    p.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)

    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--p", type=int, default=8, help="Identities per batch.")
    p.add_argument("--k", type=int, default=8, help="Sequences per identity.")
    p.add_argument("--frames", type=int, default=20, help="Frames sampled per sequence.")
    p.add_argument("--accum", type=int, default=1, help="Gradient accumulation steps.")

    p.add_argument("--head", type=str, default="cosface", choices=["linear", "cosface"])
    p.add_argument("--neck", type=str, default="flatten", choices=["flatten", "hpm"],
                   help="'flatten' is the original head (one Linear holding 97%% of "
                        "the model's parameters); 'hpm' is the part-based "
                        "Horizontal Pyramid Mapping replacement.")
    p.add_argument("--bin-dim", type=int, default=256,
                   help="Output width per HPM bin. 31 bins, so the descriptor is "
                        "31 x bin-dim.")
    p.add_argument("--depth", type=str, default="shallow", choices=["shallow", "deep"],
                   help="'deep' uses GaitSet's 6-layer conv stack instead of 3.")
    p.add_argument("--scale", type=float, default=16.0, help="CosFace logit scale s.")
    p.add_argument("--margin", type=float, default=0.2, help="CosFace angular margin m.")
    p.add_argument("--margin-warmup", type=int, default=20,
                   help="Epochs over which the CosFace margin ramps 0 -> m.")

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--triplet-margin", type=float, default=0.3)
    p.add_argument("--lambda-triplet", type=float, default=1.0)

    p.add_argument("--occlusion-aug", type=float, default=0.0,
                   help="Probability of coat/bag occlusion augmentation.")
    p.add_argument("--flip-prob", type=float, default=0.5,
                   help="Horizontal flip probability (set 0 to ablate).")

    p.add_argument("--train-upper", type=int, default=64,
                   help="Highest subject ID used for training; the rest up to --val-upper validate.")
    p.add_argument("--val-upper", type=int, default=74,
                   help="Highest subject ID used for validation.")
    p.add_argument("--val-every", type=int, default=10, help="Validate every N epochs.")
    p.add_argument("--no-val", action="store_true",
                   help="Train on all 74 identities with no validation or best-checkpoint.")
    p.add_argument("--frame-budget", type=int, default=eval_mod.DEFAULT_FRAME_BUDGET,
                   help="Frame budget for validation embedding extraction.")

    p.add_argument("--resume", action="store_true",
                   help="Continue an interrupted run from results/<run-name>/"
                        "last_checkpoint.pth. All other arguments must match "
                        "the original run.")
    p.add_argument("--allow-partial", action="store_true",
                   help="Train even if the dataset is missing identities or the "
                        "validation split is empty. Off by default, so an "
                        "interrupted preprocessing run fails loudly instead of "
                        "silently training on a fraction of the data.")
    p.add_argument("--smoke-test", action="store_true",
                   help="Run 2 short epochs on a few identities to verify the pipeline. "
                        "Saves nothing. Takes about a minute.")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def train_model() -> None:
    args = parse_args()

    print("=" * 78)
    print("  Global-Local Multimodal Fusion - Training")
    print("  Loss = CrossEntropy + lambda * TripletMargin  |  PK batch sampling")
    print("=" * 78)

    # ── Hardware ───────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"  GPU:  {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Smoke-test overrides ───────────────────────────────────────────
    if args.smoke_test:
        print("\n*** SMOKE TEST: 2 epochs, few identities, nothing is saved ***")
        args.epochs = 2
        args.train_upper = 12
        args.val_upper = 15          # keep the validation cache small too
        args.val_every = 1
        args.p = min(args.p, 4)
        args.k = min(args.k, 4)
        args.margin_warmup = 1
        args.warmup_epochs = 1

    train_range = range(1, args.train_upper + 1)
    val_range = range(args.train_upper + 1, args.val_upper + 1)
    use_val = not args.no_val and len(val_range) > 0

    if args.no_val:
        # Train on every identity up to --val-upper; nothing held back.
        train_range = range(1, args.val_upper + 1)

    micro_batch = args.p * args.k
    out_dir = os.path.join("results", args.run_name)

    print(f"\nConfiguration")
    print(f"  Run name          : {args.run_name}")
    print(f"  Head              : {args.head}"
          + (f" (s={args.scale}, m={args.margin})" if args.head == "cosface" else ""))
    print(f"  PK sampling       : P={args.p}, K={args.k} -> batch={micro_batch}")
    print(f"  Frames/sequence   : {args.frames}")
    print(f"  Grad accumulation : {args.accum} -> effective batch {micro_batch * args.accum}")
    print(f"  Epochs            : {args.epochs}")
    print(f"  Learning rate     : {args.lr} (warmup {args.warmup_epochs} epochs)")
    print(f"  Occlusion aug     : {args.occlusion_aug}")
    print(f"  Flip probability  : {args.flip_prob}")
    print(f"  Train subjects    : {train_range.start:03d}-{train_range.stop - 1:03d}")
    print(f"  Val subjects      : "
          + (f"{val_range.start:03d}-{val_range.stop - 1:03d}" if use_val else "none"))
    print(f"  Output            : {out_dir}/")

    # ── Training data ──────────────────────────────────────────────────
    print()
    train_dataset = GaitMultiModalDataset(
        data_dir=args.data_dir,
        is_train=True,
        subject_range=train_range,
        set_size=args.frames,
        occlusion_aug=args.occlusion_aug,
    )
    train_dataset._flip_prob = args.flip_prob

    num_classes = train_dataset.num_classes

    # ── Dataset completeness guard ─────────────────────────────────────
    # An interrupted preprocessing run leaves a partially populated data
    # directory.  Training silently proceeded on it once, producing a model
    # fitted to a fraction of the identities with no validation split and
    # therefore no best checkpoint.  Fail loudly instead.
    expected_classes = len(train_range)
    if num_classes < expected_classes and not args.smoke_test:
        msg = (f"found only {num_classes} of the {expected_classes} expected "
               f"training identities in '{args.data_dir}'")
        if not args.allow_partial:
            print(f"\n[ERROR] Dataset looks incomplete: {msg}.")
            print("  Preprocessing was probably interrupted. Finish it with:")
            print(f"    python preprocess.py --output-dir {args.data_dir} "
                  f"--npy-only --resume")
            print("  Or pass --allow-partial to train on what is there anyway.")
            sys.exit(1)
        print(f"\n[WARN] Dataset incomplete ({msg}) - continuing on --allow-partial.")

    pk_sampler = PKBatchSampler(train_dataset._labels, P=args.p, K=args.k)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=pk_sampler,
        pin_memory=(device.type == "cuda"),
        num_workers=0,             # safest on Windows
        collate_fn=gait_collate_fn,
    )

    steps_per_epoch = len(train_loader) // max(1, args.accum)
    print(f"\n  Optimizer steps/epoch : {steps_per_epoch}")
    print(f"  Optimizer steps total : {steps_per_epoch * args.epochs:,}")

    # ── Validation data ────────────────────────────────────────────────
    val_gallery, val_probes = None, None
    if use_val:
        print(f"\nCaching validation identities {val_range.start:03d}-{val_range.stop - 1:03d}...")
        val_gallery, val_probes = eval_mod.discover_and_cache_sequences(
            args.data_dir, subject_range=val_range, verbose=False
        )
        n_probe = sum(len(v) for v in val_probes.values())
        print(f"  {len(val_gallery)} gallery + {n_probe} probe sequences")
        if not val_gallery:
            # Without validation there is no best checkpoint, so eval.py would
            # later fail to find best_model.pth. Stop rather than discover that
            # after a 40-minute run.
            if not args.allow_partial:
                print(f"\n[ERROR] No validation sequences for subjects "
                      f"{val_range.start:03d}-{val_range.stop - 1:03d} in "
                      f"'{args.data_dir}'.")
                print("  Without them no best_model.pth is written and model")
                print("  selection is impossible. Finish preprocessing, or pass")
                print("  --no-val to train deliberately without validation.")
                sys.exit(1)
            print("  [WARN] No validation sequences found; disabling validation.")
            use_val = False

    # ── Model, losses, optimizer ───────────────────────────────────────
    # Built once and stored verbatim in every checkpoint, so eval.py can
    # reconstruct any architecture with GlobalLocalFusedNetwork(**model_kwargs)
    # instead of needing a new field for each option added.
    model_kwargs = {
        "num_classes": num_classes,
        "embed_dim": 256,
        "head": args.head,
        "cosface_scale": args.scale,
        "cosface_margin": args.margin,
        "neck": args.neck,
        "bin_dim": args.bin_dim,
        "depth": args.depth,
    }
    model = GlobalLocalFusedNetwork(**model_kwargs).to(device)

    ce_criterion = nn.CrossEntropyLoss()
    triplet_criterion = nn.TripletMarginLoss(margin=args.triplet_margin, p=2)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Linear warmup then cosine decay.  Warmup matters more now that batches
    # are 4x larger and the first steps are correspondingly more disruptive.
    if args.warmup_epochs > 0:
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.1, total_iters=args.warmup_epochs
                ),
                optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(1, args.epochs - args.warmup_epochs), eta_min=1e-6
                ),
            ],
            milestones=[args.warmup_epochs],
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParameters: {trainable:,} trainable | classes: {num_classes}")
    if args.head == "linear":
        print(f"  NOTE: linear head on unit-norm embeddings caps CE at about "
              f"{np.log(num_classes):.2f} -> 2.2. Use --head cosface to remove the cap.")

    if not args.smoke_test:
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dir, "attention_maps"), exist_ok=True)

    history: Dict = {
        "ce": [], "triplet": [], "train_acc": [],
        "val_epochs": [], "val_rank1": [], "val_per_condition": [],
        "num_classes": num_classes, "config": vars(args),
    }
    best_val = -1.0
    best_epoch = -1
    start_epoch = 0

    # ── Resume ─────────────────────────────────────────────────────────
    ckpt_path = os.path.join(out_dir, CHECKPOINT_NAME)
    ckpt_exists = os.path.exists(ckpt_path)

    if args.resume and not ckpt_exists:
        print(f"\n[ERROR] --resume given but no checkpoint at {ckpt_path}.")
        sys.exit(1)

    if ckpt_exists and not args.resume and not args.smoke_test:
        # Refuse to silently overwrite an interrupted run's progress.
        print(f"\n[ERROR] An interrupted run already exists at {ckpt_path}.")
        print("  Continue it with  --resume")
        print(f"  Or start over by deleting {out_dir} or choosing a new --run-name.")
        sys.exit(1)

    if args.resume:
        ck = torch.load(ckpt_path, map_location=device, weights_only=True)

        # A resumed run must be the same experiment, or the result is a model
        # that matches neither command line.
        old_cfg = ck.get("config", {})
        mismatched = [
            f"{k}: {old_cfg.get(k)!r} -> {getattr(args, k)!r}"
            for k in RESUME_CRITICAL_FIELDS
            if k in old_cfg and old_cfg[k] != getattr(args, k, None)
        ]
        if mismatched:
            print("\n[ERROR] Resume config does not match the interrupted run:")
            for m in mismatched:
                print(f"    {m}")
            print("  Use the original arguments, or start a new --run-name.")
            sys.exit(1)

        model.load_state_dict(ck["state_dict"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        history = ck["history"]
        best_val = float(ck["best_val"])
        best_epoch = int(ck["best_epoch"])
        start_epoch = int(ck["epoch"])

        print(f"\nResumed from {ckpt_path}")
        print(f"  Completed epochs : {start_epoch} / {args.epochs}")
        if best_epoch > 0:
            print(f"  Best validation  : {best_val:.2f}% at epoch {best_epoch}")
        if start_epoch >= args.epochs:
            print("  Already complete - nothing to do.")
            return

    # ══════════════════════════════════════════════════════════════════
    # Training loop
    # ══════════════════════════════════════════════════════════════════
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        model.train()

        # CosFace margin ramp: a freshly initialised embedding has no angular
        # structure for a margin to act on, so it is introduced gradually.
        if args.head == "cosface":
            ramp = min(1.0, (epoch + 1) / max(1, args.margin_warmup))
            model.classifier.set_margin_scale(ramp)

        running_ce = 0.0
        running_tri = 0.0
        correct = 0
        total = 0
        grad_norm = torch.tensor(0.0)

        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (frames, gei, labels) in enumerate(train_loader):
            frames = frames.to(device, non_blocking=True)
            gei = gei.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                # Labels are needed by the CosFace head to apply the margin.
                logits, embeddings = model(frames, gei, labels)
                anchors, positives, negatives = mine_batch_hard_triplets(
                    embeddings, labels
                )
                ce_loss = ce_criterion(logits, labels)
                tri_loss = triplet_criterion(anchors, positives, negatives)
                total_loss = ce_loss + args.lambda_triplet * tri_loss
                scaled_loss = total_loss / args.accum

            scaler.scale(scaled_loss).backward()

            running_ce += ce_loss.item()
            running_tri += tri_loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

            is_accum_step = (batch_idx + 1) % args.accum == 0
            is_last = (batch_idx + 1) == len(train_loader)
            if is_accum_step or is_last:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        # ── Attention mask snapshot (every 10 epochs) ──
        # Feeds the attention-evolution figure in the README.  Uses the last
        # batch of the epoch, which is already on the device.
        if not args.smoke_test and (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                attn = torch.softmax(
                    model.fusion.attention(
                        torch.cat([model.branch_a(frames), model.branch_b(gei)], dim=1)
                    ),
                    dim=1,
                )
                mask = attn[0, 0].float().cpu().numpy()   # dynamic-branch weight map
            fig, ax = plt.subplots(figsize=(4, 4))
            im = ax.imshow(mask, cmap="jet", vmin=0.0, vmax=1.0)
            ax.set_title(f"Epoch {epoch + 1} - Dynamic Branch Attention")
            plt.colorbar(im, ax=ax)
            fig.savefig(
                os.path.join(out_dir, "attention_maps", f"attn_e{epoch + 1}.png"),
                dpi=150, bbox_inches="tight",
            )
            plt.close(fig)
            model.train()

        scheduler.step()

        epoch_ce = running_ce / len(train_loader)
        epoch_tri = running_tri / len(train_loader)
        epoch_acc = correct / max(1, total) * 100.0
        history["ce"].append(epoch_ce)
        history["triplet"].append(epoch_tri)
        history["train_acc"].append(epoch_acc)

        lr_now = optimizer.param_groups[0]["lr"]
        line = (f"Epoch {epoch + 1:>4}/{args.epochs} | {time.time() - epoch_start:5.1f}s | "
                f"CE {epoch_ce:6.4f} | Tri {epoch_tri:6.4f} | "
                f"Acc {epoch_acc:5.2f}% | GN {float(grad_norm):5.2f} | LR {lr_now:.2e}")

        # ── Validation + best checkpoint ────────────────────────────────
        due = (epoch + 1) % args.val_every == 0 or (epoch + 1) == args.epochs
        if use_val and due:
            val_mean, val_per = run_validation(
                model, val_gallery, val_probes, device, args.frame_budget
            )
            history["val_epochs"].append(epoch + 1)
            history["val_rank1"].append(val_mean)
            history["val_per_condition"].append(val_per)

            marker = ""
            if val_mean > best_val:
                best_val, best_epoch = val_mean, epoch + 1
                marker = "  <-- best"
                if not args.smoke_test:
                    torch.save(
                        {
                            "state_dict": model.state_dict(),
                            "num_classes": num_classes,
                            "head": args.head,
                            "embed_dim": model.embed_dim,
                            "model_kwargs": model_kwargs,
                            "epoch": epoch + 1,
                            "val_rank1": val_mean,
                        },
                        os.path.join(out_dir, "best_model.pth"),
                    )
            detail = " ".join(f"{k} {v:.1f}" for k, v in val_per.items())
            line += f"\n         VAL cross-view {val_mean:6.2f}%  ({detail}){marker}"

        print(line)

        # ── Resumable checkpoint, written every epoch ──
        # ~34 MB and well under a second, against a ~14 s epoch, so the
        # overhead is a couple of percent and an interrupt costs at most one
        # epoch instead of the whole run.
        if not args.smoke_test:
            save_checkpoint(
                ckpt_path, model, optimizer, scheduler, scaler,
                epoch + 1, history, best_val, best_epoch, num_classes, args,
                model_kwargs,
            )

        if not args.smoke_test and ((epoch + 1) % 25 == 0 or (epoch + 1) == args.epochs):
            save_curves(history, os.path.join(out_dir, "training_curves.png"), args.epochs)
            with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

    # ══════════════════════════════════════════════════════════════════
    # Wrap up
    # ══════════════════════════════════════════════════════════════════
    if args.smoke_test:
        print("\n" + "=" * 78)
        print("  SMOKE TEST PASSED - pipeline runs end to end. Nothing was saved.")
        print("=" * 78)
        return

    final_path = os.path.join(out_dir, "final_model.pth")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "num_classes": num_classes,
            "head": args.head,
            "embed_dim": model.embed_dim,
            "model_kwargs": model_kwargs,
            "epoch": args.epochs,
        },
        final_path,
    )

    save_curves(history, os.path.join(out_dir, "training_curves.png"), args.epochs)
    with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # The run finished, so the resume checkpoint is dead weight - and leaving
    # it would make a later re-run of this name fail the "interrupted run
    # exists" guard.
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    print("\n" + "=" * 78)
    print(f"  Run '{args.run_name}' complete")
    print("=" * 78)
    print(f"  Final CE            : {history['ce'][-1]:.4f}   (started {history['ce'][0]:.4f}, "
          f"random = {np.log(num_classes):.2f})")
    print(f"  Final triplet       : {history['triplet'][-1]:.4f}")
    print(f"  Final train accuracy: {history['train_acc'][-1]:.2f}%")
    if use_val and best_epoch > 0:
        print(f"  Best validation     : {best_val:.2f}% cross-view Rank-1 at epoch {best_epoch}")
        print(f"  Best checkpoint     -> {os.path.join(out_dir, 'best_model.pth')}")
    print(f"  Final checkpoint    -> {final_path}")
    print(f"  Curves              -> {os.path.join(out_dir, 'training_curves.png')}")
    print(f"  History             -> {os.path.join(out_dir, 'history.json')}")
    print()
    print(f"  Next: python eval.py --weights {os.path.join(out_dir, 'best_model.pth')}")
    print()


if __name__ == "__main__":
    train_model()
