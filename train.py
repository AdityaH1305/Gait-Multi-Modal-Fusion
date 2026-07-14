"""Training script for the Global-Local Multimodal Fusion gait network.

Bug fixes applied:
    - Batch size reduced to 2 for 6GB VRAM safety (was 4).
    - Uses custom collate_fn for variable-length sequences (BUG #3).
    - Replaced StepLR(gamma=0.1) with CosineAnnealingLR (BUG #5).
    - Fixed deprecated torch.cuda.amp imports.
    - Added gradient norm logging for diagnostics.
    - Added accuracy curve to results plot.
    - num_workers=0 for Windows stability.

Performance:
    - Dataset pre-caches all images into RAM at init (no per-batch disk I/O).
    - Random Set Sampling (N=30) standardises frame tensor shapes.
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import your custom modules
from dataset import GaitMultiModalDataset, gait_collate_fn
from model import GlobalLocalFusedNetwork


def train_model() -> None:
    print("=" * 70)
    print("  Global-Local Multimodal Fusion — Training Pipeline")
    print("=" * 70)

    # ── 1. HARDWARE SETUP ──────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU:  {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── 2. HYPERPARAMETERS ─────────────────────────────────────────────
    BATCH_SIZE = 2               # ← FIX: reduced from 4 to 2 for 6GB VRAM safety
    LEARNING_RATE = 1e-3
    EPOCHS = 50
    NUM_CLASSES = 74

    # ── 3. DATA LOADING ────────────────────────────────────────────────
    train_dataset = GaitMultiModalDataset(
        data_dir=r"C:\Users\adity\Desktop\Gait- MultiModal Fusion\Processed_CASIAB",
        is_train=True,
    )

    # ┌──────────────────────────────────────────────────────────────────┐
    # │  FIX #3 (cont.): Use custom collate_fn for variable-length      │
    # │  frame sequences. Without this, DataLoader crashes when          │
    # │  sequences in a batch have different frame counts.               │
    # └──────────────────────────────────────────────────────────────────┘
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
        num_workers=0,             # ← FIX: 0 is safest on Windows
        collate_fn=gait_collate_fn,
    )

    print(f"Batches per epoch: {len(train_loader)}")

    # ── Quick I/O sanity check: time the first batch ──
    _t0 = time.time()
    _test_batch = next(iter(train_loader))
    _batch_ms = (time.time() - _t0) * 1000
    print(f"First batch load: {_batch_ms:.0f} ms  (should be < 100 ms with RAM cache)")
    print(f"  Frames: {_test_batch[0].shape}  GEI: {_test_batch[1].shape}  Labels: {_test_batch[2].shape}")
    del _test_batch

    # ── 4. MODEL, LOSS, OPTIMIZER, SCHEDULER ───────────────────────────
    model = GlobalLocalFusedNetwork(num_classes=NUM_CLASSES).to(device)
    criterion = nn.TripletMarginLoss(margin=0.2, p=2)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ┌──────────────────────────────────────────────────────────────────┐
    # │  FIX #5: CosineAnnealingLR instead of StepLR(gamma=0.1)        │
    # │  Old scheduler killed the LR by 3 orders of magnitude across   │
    # │  50 epochs. Cosine annealing decays smoothly and never reaches  │
    # │  a near-zero LR.                                               │
    # └──────────────────────────────────────────────────────────────────┘
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    # ── Mixed Precision (AMP) ──────────────────────────────────────────
    # FIX: Use the modern, non-deprecated API
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── Tracking ───────────────────────────────────────────────────────
    loss_history: list[float] = []
    acc_history: list[float] = []

    # Directory for saving attention mask visualisations
    os.makedirs("results", exist_ok=True)
    attn_vis_dir = os.path.join("results", "attention_maps")
    os.makedirs(attn_vis_dir, exist_ok=True)
    ATTN_VIS_INTERVAL = 200  # save a visualisation every N batches

    # ── Parameter count ────────────────────────────────────────────────
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {trainable_params:,} trainable / {total_params:,} total")

    # ══════════════════════════════════════════════════════════════════
    # 5. THE CORE TRAINING LOOP
    # ══════════════════════════════════════════════════════════════════
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (frames, gei, labels) in enumerate(train_loader):
            frames = frames.to(device, non_blocking=True)
            gei = gei.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)  # slightly faster than zero_grad()

            # ── Forward pass (mixed precision) ──
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(frames, gei)
                loss = criterion(outputs, labels)

            # ── Backward pass (scaled gradients) ──
            scaler.scale(loss).backward()

            # ── Gradient clipping (prevents explosion) ──
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=5.0
            )

            scaler.step(optimizer)
            scaler.update()

            # ── Metrics ──
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # ── Per-step logging ──
            if (batch_idx + 1) % 25 == 0:
                step_acc = correct / total * 100
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  Epoch [{epoch+1}/{EPOCHS}] | "
                    f"Step [{batch_idx+1}/{len(train_loader)}] | "
                    f"Loss: {loss.item():.4f} | "
                    f"Acc: {step_acc:.1f}% | "
                    f"GradNorm: {grad_norm:.2f} | "
                    f"LR: {current_lr:.2e}"
                )

            # ── Attention mask visualisation ──
            if (batch_idx + 1) % ATTN_VIS_INTERVAL == 0:
                model.eval()
                with torch.no_grad():
                    fa = model.branch_a(frames)
                    fb = model.branch_b(gei)
                    concat_feat = torch.cat([fa, fb], dim=1)
                    attn_raw = model.fusion.attention(concat_feat)
                    attn_soft = torch.softmax(attn_raw, dim=1)  # (B,2,H,W)

                    # Dynamic branch weight map, first sample
                    mask_a = attn_soft[0, 0].cpu().numpy()  # (16, 16)

                fig, ax = plt.subplots(1, 1, figsize=(4, 4))
                im = ax.imshow(mask_a, cmap="jet", vmin=0.0, vmax=1.0)
                ax.set_title(f"Epoch {epoch+1} | Step {batch_idx+1}")
                plt.colorbar(im, ax=ax)
                save_path = os.path.join(
                    attn_vis_dir,
                    f"attn_e{epoch+1}_s{batch_idx+1}.png",
                )
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                model.train()

        # ── Step the scheduler at the end of each epoch ──
        scheduler.step()

        # ── Epoch summary ──
        epoch_dur = time.time() - epoch_start
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = (correct / total) * 100 if total > 0 else 0.0

        loss_history.append(epoch_loss)
        acc_history.append(epoch_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"═══► EPOCH {epoch+1}/{EPOCHS} | "
            f"Time: {epoch_dur:.1f}s | "
            f"Loss: {epoch_loss:.4f} | "
            f"Acc: {epoch_acc:.2f}% | "
            f"LR: {current_lr:.2e}\n"
        )

    # ══════════════════════════════════════════════════════════════════
    # 6. EXPORT RESULTS
    # ══════════════════════════════════════════════════════════════════
    torch.save(model.state_dict(), "results/fused_gait_model.pth")
    print("Model weights saved → results/fused_gait_model.pth")

    # ── Loss & Accuracy curves ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs_range = range(1, EPOCHS + 1)

    ax1.plot(epochs_range, loss_history, color="#2563eb", linewidth=2)
    ax1.set_title("Training Loss", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(epochs_range, acc_history, color="#16a34a", linewidth=2)
    ax2.set_title("Training Accuracy", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("results/training_curves.png", dpi=300)
    plt.close()
    print("Training curves saved → results/training_curves.png")


if __name__ == "__main__":
    train_model()