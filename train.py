"""Training script for the Global-Local Multimodal Fusion gait network.

Joint Loss Architecture:
    Total_Loss = CrossEntropy(logits, labels) + λ · TripletMargin(anchor, pos, neg)

    CrossEntropy provides dense gradient signal (every sample contributes),
    while TripletMarginLoss enforces explicit intra-class compaction and
    inter-class separation in the 256-D embedding space.

Key components:
    - **PKBatchSampler**: Ensures every batch contains P=4 identities × K=4
      sequences, guaranteeing valid triplets for online mining.
    - **Batch-Hard Mining**: Selects the hardest positive (max distance) and
      hardest negative (min distance) for each anchor, producing the most
      informative gradient signal per step.
    - **Gradient Accumulation**: Accumulates gradients over 2 micro-batches
      (effective batch = 32) for smoother metric learning convergence while
      staying within 6 GB VRAM on RTX 4050.

Bug fixes carried forward:
    - Uses custom collate_fn for variable-length sequences (BUG #3).
    - CosineAnnealingLR instead of StepLR(gamma=0.1) (BUG #5).
    - Fixed deprecated torch.cuda.amp imports.
    - num_workers=0 for Windows stability.

Performance:
    - Dataset pre-caches all images into RAM at init (no per-batch disk I/O).
    - Random Set Sampling (N=30) standardises frame tensor shapes.
"""

import os
import time
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import custom modules
from dataset import GaitMultiModalDataset, gait_collate_fn, PKBatchSampler
from model import GlobalLocalFusedNetwork


# ═══════════════════════════════════════════════════════════════════════════
# Online Batch-Hard Triplet Mining
# ═══════════════════════════════════════════════════════════════════════════


def mine_batch_hard_triplets(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Online batch-hard triplet mining from a PK-structured mini-batch.

    For each sample treated as an **anchor**:
        - **Hardest positive**: the same-identity sample with the LARGEST
          Euclidean distance (the most dissimilar positive — hardest to pull
          closer).
        - **Hardest negative**: the different-identity sample with the SMALLEST
          Euclidean distance (the most similar impostor — hardest to push
          away).

    This strategy produces the most informative triplets and accelerates
    convergence compared to random or semi-hard mining.

    ┌──────────────────────────────────────────────────────────────────────┐
    │  IMPORTANT: Requires PK-sampled batches.  With K=4, every anchor   │
    │  is guaranteed at least 3 positives.  Random batches would often    │
    │  have zero positives for many anchors, causing degenerate mining.   │
    └──────────────────────────────────────────────────────────────────────┘

    Args:
        embeddings: L2-normalised embeddings, shape ``(B, D)`` where B = P×K.
        labels:     Integer identity labels, shape ``(B,)``.

    Returns:
        3-tuple of ``(anchors, positives, negatives)``, each shape ``(B, D)``.
        Gradient connectivity is preserved: the returned tensors are views
        into the original ``embeddings``, so ``TripletMarginLoss.backward()``
        flows gradients back through the embedding head.
    """
    B = embeddings.size(0)

    # ── 1. Compute pairwise Euclidean distance matrix ──
    # Detached: we only need distances for INDEX SELECTION, not for the loss.
    # The actual loss recomputes distances internally with gradient tracking.
    with torch.no_grad():
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)  # (B, B)

        # ── 2. Build identity masks ──
        labels_col = labels.unsqueeze(1)  # (B, 1)
        labels_row = labels.unsqueeze(0)  # (1, B)
        same_identity = (labels_col == labels_row)  # (B, B) — True where same ID

        # Exclude self-pairs from the positive mask (an anchor can't be its
        # own positive)
        not_self = ~torch.eye(B, dtype=torch.bool, device=embeddings.device)
        positive_mask = same_identity & not_self      # (B, B)
        negative_mask = ~same_identity                 # (B, B)

        # ── 3. Hardest positive: argmax distance among same-class ──
        # Mask out non-positives with -1 so they're never selected by argmax
        pos_dists = dist_matrix.clone()
        pos_dists[~positive_mask] = -1.0
        hardest_pos_idx = pos_dists.argmax(dim=1)     # (B,)

        # ── 4. Hardest negative: argmin distance among different-class ──
        # Mask out same-class with +inf so they're never selected by argmin
        neg_dists = dist_matrix.clone()
        neg_dists[~negative_mask] = float("inf")
        hardest_neg_idx = neg_dists.argmin(dim=1)     # (B,)

    # ── 5. Index into ORIGINAL embeddings (gradient-connected) ──
    # The returned tensors are NOT detached — TripletMarginLoss.backward()
    # will propagate gradients through embed_fc → fusion → branches.
    anchors = embeddings                         # (B, D)
    positives = embeddings[hardest_pos_idx]      # (B, D)
    negatives = embeddings[hardest_neg_idx]      # (B, D)

    return anchors, positives, negatives


# ═══════════════════════════════════════════════════════════════════════════
# Main Training Loop
# ═══════════════════════════════════════════════════════════════════════════


def train_model() -> None:
    print("=" * 70)
    print("  Global-Local Multimodal Fusion — Joint Loss Training Pipeline")
    print("  Loss = CrossEntropy + λ · TripletMargin  |  PK Batch Sampling")
    print("=" * 70)

    # ── 1. HARDWARE SETUP ──────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU:  {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── 2. HYPERPARAMETERS ─────────────────────────────────────────────
    # PK Sampling: P identities × K sequences per identity
    P = 4                        # identities per micro-batch
    K = 4                        # sequences per identity
    MICRO_BATCH = P * K          # = 16 samples per forward pass

    # Gradient Accumulation: accumulate over ACCUM_STEPS micro-batches
    # before executing an optimizer step.  This gives an effective batch
    # size of MICRO_BATCH × ACCUM_STEPS = 32, improving gradient quality
    # for metric learning while keeping VRAM usage under 6 GB.
    ACCUM_STEPS = 2              # effective batch = 16 × 2 = 32

    LEARNING_RATE = 1e-3
    EPOCHS = 1
    NUM_CLASSES = 74
    EMBED_DIM = 256

    # Joint loss weights
    TRIPLET_MARGIN = 0.3         # margin for TripletMarginLoss
    LAMBDA_TRIPLET = 1.0         # weight of triplet loss relative to CE

    print(f"\nHyperparameters:")
    print(f"  PK Sampling:       P={P}, K={K} → micro-batch={MICRO_BATCH}")
    print(f"  Gradient Accum:    {ACCUM_STEPS} steps → effective batch={MICRO_BATCH * ACCUM_STEPS}")
    print(f"  Epochs:            {EPOCHS}")
    print(f"  Learning Rate:     {LEARNING_RATE}")
    print(f"  Embedding Dim:     {EMBED_DIM}")
    print(f"  Triplet Margin:    {TRIPLET_MARGIN}")
    print(f"  λ (triplet weight):{LAMBDA_TRIPLET}")

    # ── 3. DATA LOADING ────────────────────────────────────────────────
    train_dataset = GaitMultiModalDataset(
        data_dir=r"C:\Users\adity\Desktop\Gait- MultiModal Fusion\Processed_CASIAB",
        is_train=True,
    )

    # ┌──────────────────────────────────────────────────────────────────┐
    # │  PK BATCH SAMPLER: Replaces random shuffle with identity-aware  │
    # │  batch construction.  Every batch is guaranteed to contain P    │
    # │  identities × K sequences, enabling valid triplet mining.      │
    # │                                                                 │
    # │  When using batch_sampler, DataLoader's batch_size, shuffle,    │
    # │  and drop_last arguments must NOT be specified.                 │
    # └──────────────────────────────────────────────────────────────────┘
    pk_sampler = PKBatchSampler(
        labels=train_dataset._labels,
        P=P,
        K=K,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=pk_sampler,
        pin_memory=(device.type == "cuda"),
        num_workers=0,             # ← safest on Windows
        collate_fn=gait_collate_fn,
    )

    print(f"PK batches per epoch: {len(train_loader)}")

    # ── Quick I/O sanity check: time the first batch ──
    _t0 = time.time()
    _test_batch = next(iter(train_loader))
    _batch_ms = (time.time() - _t0) * 1000
    print(f"First batch load: {_batch_ms:.0f} ms  (should be < 100 ms with RAM cache)")
    print(f"  Frames: {_test_batch[0].shape}  GEI: {_test_batch[1].shape}  Labels: {_test_batch[2].shape}")
    _unique_ids = _test_batch[2].unique()
    print(f"  Unique identities: {len(_unique_ids)} (expected P={P})")
    del _test_batch

    # ── 4. MODEL, LOSSES, OPTIMIZER, SCHEDULER ─────────────────────────
    model = GlobalLocalFusedNetwork(
        num_classes=NUM_CLASSES,
        embed_dim=EMBED_DIM,
    ).to(device)

    # ── Dual loss functions ──
    ce_criterion = nn.CrossEntropyLoss()
    triplet_criterion = nn.TripletMarginLoss(margin=TRIPLET_MARGIN, p=2)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ┌──────────────────────────────────────────────────────────────────┐
    # │  CosineAnnealingLR: smooth LR decay over 150 epochs.           │
    # │  Never reaches near-zero LR (eta_min=1e-6), preserving the     │
    # │  model's ability to fine-tune embeddings in late training.      │
    # └──────────────────────────────────────────────────────────────────┘
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    # ── Mixed Precision (AMP) ──────────────────────────────────────────
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── Tracking ───────────────────────────────────────────────────────
    ce_loss_history: list[float] = []
    tri_loss_history: list[float] = []
    acc_history: list[float] = []

    # Directory for saving results + attention visualisations
    os.makedirs("results", exist_ok=True)
    attn_vis_dir = os.path.join("results", "attention_maps")
    os.makedirs(attn_vis_dir, exist_ok=True)
    ATTN_VIS_EPOCH_INTERVAL = 10   # save attention maps every N epochs

    # ── Parameter count ────────────────────────────────────────────────
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {trainable_params:,} trainable / {total_params:,} total")

    # ══════════════════════════════════════════════════════════════════
    # 5. THE CORE TRAINING LOOP (Joint Loss + Gradient Accumulation)
    # ══════════════════════════════════════════════════════════════════
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        model.train()

        # Per-epoch accumulators
        running_ce_loss = 0.0
        running_tri_loss = 0.0
        correct = 0
        total = 0

        # ── Reset gradients at the start of each epoch ──
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (frames, gei, labels) in enumerate(train_loader):
            frames = frames.to(device, non_blocking=True)
            gei = gei.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # ── Forward pass (mixed precision) ──
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, embeddings = model(frames, gei)

                # ── Online Batch-Hard Triplet Mining ──
                # Mine the hardest positive and hardest negative for each
                # anchor from the PK-structured batch.
                anchors, positives, negatives = mine_batch_hard_triplets(
                    embeddings, labels
                )

                # ── Joint Loss ──
                ce_loss = ce_criterion(logits, labels)
                tri_loss = triplet_criterion(anchors, positives, negatives)
                total_loss = ce_loss + LAMBDA_TRIPLET * tri_loss

                # Scale by accumulation steps so the effective gradient
                # magnitude is independent of ACCUM_STEPS.
                scaled_loss = total_loss / ACCUM_STEPS

            # ── Backward pass (scaled gradients accumulate) ──
            scaler.scale(scaled_loss).backward()

            # ── Track raw (unscaled) metrics ──
            running_ce_loss += ce_loss.item()
            running_tri_loss += tri_loss.item()
            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # ── Optimizer step every ACCUM_STEPS micro-batches ──
            is_accum_step = (batch_idx + 1) % ACCUM_STEPS == 0
            is_last_batch = (batch_idx + 1) == len(train_loader)

            if is_accum_step or is_last_batch:
                # Unscale gradients for clipping
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=5.0
                )

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # ── Per-step logging (every 5 PK batches) ──
            if (batch_idx + 1) % 5 == 0 or is_last_batch:
                step_acc = correct / total * 100 if total > 0 else 0.0
                current_lr = optimizer.param_groups[0]["lr"]

                # Compute diagnostic distances for the current batch
                with torch.no_grad():
                    pos_dist = torch.norm(anchors - positives, p=2, dim=1).mean().item()
                    neg_dist = torch.norm(anchors - negatives, p=2, dim=1).mean().item()

                print(
                    f"  Epoch [{epoch+1}/{EPOCHS}] | "
                    f"Step [{batch_idx+1}/{len(train_loader)}] | "
                    f"CE: {ce_loss.item():.4f} | "
                    f"Tri: {tri_loss.item():.4f} | "
                    f"Acc: {step_acc:.1f}% | "
                    f"d+: {pos_dist:.3f} d-: {neg_dist:.3f} | "
                    f"GN: {grad_norm:.2f} | "
                    f"LR: {current_lr:.2e}"
                )

        # ── Attention mask visualisation (every N epochs) ──
        if (epoch + 1) % ATTN_VIS_EPOCH_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                fa = model.branch_a(frames)
                fb = model.branch_b(gei)
                concat_feat = torch.cat([fa, fb], dim=1)
                attn_raw = model.fusion.attention(concat_feat)
                attn_soft = torch.softmax(attn_raw, dim=1)  # (B, 2, H, W)

                # Dynamic branch weight map, first sample
                mask_a = attn_soft[0, 0].cpu().numpy()  # (16, 16)

            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            im = ax.imshow(mask_a, cmap="jet", vmin=0.0, vmax=1.0)
            ax.set_title(f"Epoch {epoch+1} — Dynamic Branch Attention")
            plt.colorbar(im, ax=ax)
            save_path = os.path.join(
                attn_vis_dir,
                f"attn_e{epoch+1}.png",
            )
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            model.train()

        # ── Step the scheduler at the end of each epoch ──
        scheduler.step()

        # ── Epoch summary ──
        epoch_dur = time.time() - epoch_start
        num_batches = len(train_loader)
        epoch_ce = running_ce_loss / num_batches
        epoch_tri = running_tri_loss / num_batches
        epoch_acc = (correct / total) * 100 if total > 0 else 0.0

        ce_loss_history.append(epoch_ce)
        tri_loss_history.append(epoch_tri)
        acc_history.append(epoch_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"═══► EPOCH {epoch+1}/{EPOCHS} | "
            f"Time: {epoch_dur:.1f}s | "
            f"CE: {epoch_ce:.4f} | "
            f"Tri: {epoch_tri:.4f} | "
            f"Acc: {epoch_acc:.2f}% | "
            f"LR: {current_lr:.2e}\n"
        )

    # ══════════════════════════════════════════════════════════════════
    # 6. EXPORT RESULTS
    # ══════════════════════════════════════════════════════════════════
    torch.save(model.state_dict(), "results/fused_gait_model.pth")
    print("Model weights saved → results/fused_gait_model.pth")

    # ── Loss & Accuracy curves (3-panel plot) ──
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

    epochs_range = range(1, EPOCHS + 1)

    # Panel 1: Cross-Entropy Loss
    ax1.plot(epochs_range, ce_loss_history, color="#2563eb", linewidth=2)
    ax1.set_title("Cross-Entropy Loss", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Panel 2: Triplet Loss
    ax2.plot(epochs_range, tri_loss_history, color="#dc2626", linewidth=2)
    ax2.set_title("Triplet Loss", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.axhline(y=0.0, color="gray", linestyle=":", alpha=0.5, label="Margin satisfied")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Panel 3: Training Accuracy (from CE head)
    ax3.plot(epochs_range, acc_history, color="#16a34a", linewidth=2)
    ax3.set_title("Training Accuracy (CE Head)", fontsize=14, fontweight="bold")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Accuracy (%)")
    ax3.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("results/training_curves.png", dpi=300)
    plt.close()
    print("Training curves saved → results/training_curves.png")


if __name__ == "__main__":
    train_model()