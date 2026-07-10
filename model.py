"""Dual-branch gait recognition model with attention-based multimodal fusion.

Architecture overview:
    - **DynamicBranch**: Processes variable-length silhouette frame sets via a
      shared CNN followed by Set Pooling (element-wise max across the temporal
      dimension).
    - **StaticBranch**: Processes a single Gait Energy Image (GEI) through an
      identical CNN backbone.
    - **MultimodalFusion**: Learns spatial attention weights to adaptively fuse
      the two branch outputs.
    - **GlobalLocalFusedNetwork**: End-to-end model that wires the branches,
      fusion module, and a linear classifier together.

Bug fixes applied:
    - Added BatchNorm2d after every Conv2d for gradient stability (BUG #4).
    - Added Dropout before the FC head to prevent overfitting on 74 classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Branch A – Dynamic (set of silhouette frames)
# ---------------------------------------------------------------------------

class DynamicBranch(nn.Module):
    """Extracts a fixed-size representation from a *variable-length* set of
    silhouette frames.

    The branch first processes every frame independently through a CNN, then
    collapses the temporal dimension with **Set Pooling** (element-wise max),
    yielding a single feature map that is invariant to frame ordering and count.
    """

    def __init__(self) -> None:
        super().__init__()

        # Layer 1: 1 → 32 channels, 5×5 kernel, same-padding
        # 64×64 → (conv) 64×64 → (pool) 32×32
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(32)     # ← FIX: stabilises activations
        self.pool1 = nn.MaxPool2d(2)

        # Layer 2: 32 → 64 channels, 3×3 kernel, same-padding
        # 32×32 → (conv) 32×32 → (pool) 16×16
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)     # ← FIX
        self.pool2 = nn.MaxPool2d(2)

        # Layer 3: 64 → 128 channels, 3×3 kernel, same-padding
        # 16×16 → (conv) 16×16  (no further pooling)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)    # ← FIX

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with Set Pooling over the frame dimension.

        Args:
            x: Silhouette frames of shape ``(B, N, 64, 64)`` where *N* is the
               number of frames (variable across samples, padded to batch max).

        Returns:
            Pooled feature map of shape ``(B, 128, 16, 16)``.
        """
        B, N, H, W = x.size()

        # 1. INDEPENDENT FRAME PROCESSING
        # Merge Batch and N so Conv2d sees (B*N, 1, H, W).
        x = x.view(B * N, 1, H, W)

        # Conv → BN → ReLU → Pool  (×3 layers)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))

        # 2. SEPARATE THE TIMELINE
        # Reshape back to (B, N, C, H', W') to expose the frame axis.
        _, C, new_H, new_W = x.size()
        x = x.view(B, N, C, new_H, new_W)

        # 3. SET POOLING (element-wise max across the frame dimension)
        # torch.max returns (values, indices); keep values only.
        x, _ = torch.max(x, dim=1)

        return x  # (B, 128, 16, 16)


# ---------------------------------------------------------------------------
# Branch B – Static (Gait Energy Image)
# ---------------------------------------------------------------------------

class StaticBranch(nn.Module):
    """Extracts macro-level appearance features from a single Gait Energy
    Image (GEI) using a standard CNN.

    The architecture mirrors :class:`DynamicBranch` but requires no temporal
    reshaping since the input is a single image.
    """

    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(32)     # ← FIX
        self.pool1 = nn.MaxPool2d(2)      # 64×64 → 32×32

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)     # ← FIX
        self.pool2 = nn.MaxPool2d(2)      # 32×32 → 16×16

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)    # ← FIX

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard CNN forward pass.

        Args:
            x: GEI tensor of shape ``(B, 1, 64, 64)``.

        Returns:
            Feature map of shape ``(B, 128, 16, 16)``.
        """
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))

        return x  # (B, 128, 16, 16)


# ---------------------------------------------------------------------------
# Attention-based Multimodal Fusion
# ---------------------------------------------------------------------------

class MultimodalFusion(nn.Module):
    """Spatially-adaptive attention fusion of two 128-channel feature maps.

    A lightweight 1×1 convolution network predicts per-pixel importance weights
    for each branch, then combines the features via a weighted sum.  Softmax
    normalisation ensures that the two weight maps sum to 1 at every spatial
    location.
    """

    def __init__(self) -> None:
        super().__init__()

        # 256 input channels (128 from each branch) → 2 attention maps
        self.attention = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.15),
            nn.Conv2d(64, 2, kernel_size=1),
        )

    def forward(
        self,
        feat_a: torch.Tensor,
        feat_b: torch.Tensor,
    ) -> torch.Tensor:
        """Compute attention-weighted fusion of two branch outputs.

        Args:
            feat_a: Dynamic branch features, shape ``(B, 128, 16, 16)``.
            feat_b: Static branch features, shape ``(B, 128, 16, 16)``.

        Returns:
            Fused feature map of shape ``(B, 128, 16, 16)``.
        """
        # 1. Concatenate along the channel axis → (B, 256, 16, 16)
        concat_feat = torch.cat([feat_a, feat_b], dim=1)

        # 2. Predict attention weights → (B, 2, 16, 16)
        attn_weights = self.attention(concat_feat)

        # 3. Softmax across dim=1 (the 2-channel branch axis) so weights
        #    for branch A and branch B sum to 1.0 at each spatial location.
        attn_weights = F.softmax(attn_weights, dim=1)

        # Split into per-branch spatial weight maps
        weight_a = attn_weights[:, 0:1, :, :]  # (B, 1, 16, 16)
        weight_b = attn_weights[:, 1:2, :, :]  # (B, 1, 16, 16)

        # 4. Weighted sum fusion
        fused_feat = (feat_a * weight_a) + (feat_b * weight_b)

        return fused_feat  # (B, 128, 16, 16)


# ---------------------------------------------------------------------------
# End-to-end Model
# ---------------------------------------------------------------------------

class GlobalLocalFusedNetwork(nn.Module):
    """Complete dual-branch gait recognition network.

    Combines :class:`DynamicBranch` (frame-set processing),
    :class:`StaticBranch` (GEI processing), and
    :class:`MultimodalFusion` (attention-weighted fusion), followed by a
    fully-connected classifier head.

    Args:
        num_classes: Number of identity classes.  Defaults to ``74``
            (subjects 001–074 in the CASIA-B LST training split).
    """

    # Feature map dimensions after the CNN backbones
    _FEAT_CHANNELS: int = 128
    _FEAT_HEIGHT: int = 16
    _FEAT_WIDTH: int = 16

    def __init__(self, num_classes: int = 74) -> None:
        super().__init__()

        self.branch_a = DynamicBranch()
        self.branch_b = StaticBranch()
        self.fusion = MultimodalFusion()

        flat_dim = self._FEAT_CHANNELS * self._FEAT_HEIGHT * self._FEAT_WIDTH
        # ── FIX: Dropout before FC prevents overfitting on 74 classes ──
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(flat_dim, num_classes)

    def forward(
        self,
        frames: torch.Tensor,
        gei: torch.Tensor,
    ) -> torch.Tensor:
        """Run the full forward pass: branch extraction → fusion → classify.

        Args:
            frames: Silhouette frame set, shape ``(B, N, 64, 64)``.
            gei: Gait Energy Image, shape ``(B, 1, 64, 64)``.

        Returns:
            Class logits of shape ``(B, num_classes)``.
        """
        feat_a = self.branch_a(frames)
        feat_b = self.branch_b(gei)

        fused_feat = self.fusion(feat_a, feat_b)

        # Flatten spatial dims for the linear head
        fused_flat = fused_feat.view(fused_feat.size(0), -1)

        prediction = self.fc(self.dropout(fused_flat))
        return prediction


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing Fully Assembled Fused Network...")

    model = GlobalLocalFusedNetwork(num_classes=74)

    # Simulate DataLoader outputs
    dummy_frames = torch.randn(2, 45, 64, 64)  # 45 silhouette frames, batch=2
    dummy_gei = torch.randn(2, 1, 64, 64)      # 1 GEI image per sample

    final_prediction = model(dummy_frames, dummy_gei)
    print(f"Final Prediction Shape: {final_prediction.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")