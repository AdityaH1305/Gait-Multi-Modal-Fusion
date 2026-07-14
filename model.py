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
      fusion module, and a **split-head** architecture together.

Split-head design (metric learning support):
    The fused feature map is projected into a compact 256-D embedding space
    via a learned linear projection + BatchNorm + L2 normalisation.  A
    separate classifier head maps embeddings → class logits.

    forward() returns ``(logits, embeddings)`` so the training loop can
    compute **joint loss**: CrossEntropy on logits + TripletMargin on
    embeddings simultaneously.

Bug fixes applied:
    - Added BatchNorm2d after every Conv2d for gradient stability (BUG #4).
    - Added Dropout before the embedding head to prevent overfitting on 74 classes.
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
# End-to-end Model (Split-Head for Joint Loss)
# ---------------------------------------------------------------------------

class GlobalLocalFusedNetwork(nn.Module):
    """Complete dual-branch gait recognition network with split-head output.

    Combines :class:`DynamicBranch` (frame-set processing),
    :class:`StaticBranch` (GEI processing), and
    :class:`MultimodalFusion` (attention-weighted fusion), followed by a
    **split head**:

        fused_flat (32768-D)
            │
            ├──► Dropout(0.15)
            │
            ├──► embed_fc (Linear 32768 → 256)
            │       │
            │       ├──► embed_bn (BatchNorm1d)
            │       │
            │       └──► L2 Normalize  ──────────► **embeddings** (256-D, unit norm)
            │                                           │
            │                                           ├──► TripletMarginLoss
            │                                           │
            └──────────────────────────────────────► classifier (Linear 256 → C)
                                                        │
                                                        └──► **logits** (C-D)
                                                                │
                                                                └──► CrossEntropyLoss

    The CE gradient flows *through* the embedding layer, forcing the 256-D
    space to be simultaneously discriminative (for classification) and
    metrically structured (for triplet separation).

    Args:
        num_classes: Number of identity classes.  Defaults to ``74``
            (subjects 001–074 in the CASIA-B LST training split).
        embed_dim:   Dimensionality of the metric embedding space.
            Defaults to ``256`` (standard in GaitSet/GaitPart literature).
    """

    # Feature map dimensions after the CNN backbones
    _FEAT_CHANNELS: int = 128
    _FEAT_HEIGHT: int = 16
    _FEAT_WIDTH: int = 16

    def __init__(self, num_classes: int = 74, embed_dim: int = 256) -> None:
        super().__init__()

        self.embed_dim = embed_dim

        # ── Dual branches + attention fusion ──
        self.branch_a = DynamicBranch()
        self.branch_b = StaticBranch()
        self.fusion = MultimodalFusion()

        flat_dim = self._FEAT_CHANNELS * self._FEAT_HEIGHT * self._FEAT_WIDTH
        # 128 × 16 × 16 = 32,768

        # ── Regularisation ──
        self.dropout = nn.Dropout(p=0.15)

        # ── EMBEDDING HEAD ──
        # Projects the high-dimensional fusion output into a compact metric
        # space.  BatchNorm stabilises the embedding magnitude during early
        # training (prevents collapse), and L2 normalisation maps all
        # embeddings onto the unit hypersphere for cosine-compatible
        # distance computation.
        self.embed_fc = nn.Linear(flat_dim, embed_dim)
        self.embed_bn = nn.BatchNorm1d(embed_dim)

        # ── CLASSIFICATION HEAD ──
        # Operates on the L2-normalised embeddings.  Because the inputs
        # are unit-norm, the weight vectors learn angular decision
        # boundaries (similar to CosFace / ArcFace).
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(
        self,
        frames: torch.Tensor,
        gei: torch.Tensor,
    ) -> tuple:
        """Run the full forward pass: branch extraction → fusion → split head.

        Args:
            frames: Silhouette frame set, shape ``(B, N, 64, 64)``.
            gei: Gait Energy Image, shape ``(B, 1, 64, 64)``.

        Returns:
            A 2-tuple of:
                - **logits** – Class logits of shape ``(B, num_classes)``.
                - **embeddings** – L2-normalised embeddings of shape
                  ``(B, embed_dim)``.
        """
        # ── Branch feature extraction ──
        feat_a = self.branch_a(frames)     # (B, 128, 16, 16)
        feat_b = self.branch_b(gei)        # (B, 128, 16, 16)

        # ── Attention-weighted fusion ──
        fused_feat = self.fusion(feat_a, feat_b)  # (B, 128, 16, 16)

        # ── Flatten spatial dims ──
        fused_flat = fused_feat.view(fused_feat.size(0), -1)  # (B, 32768)
        fused_flat = self.dropout(fused_flat)

        # ── Embedding head: project → normalise → unit hypersphere ──
        embeddings = self.embed_fc(fused_flat)           # (B, 256)
        embeddings = self.embed_bn(embeddings)           # (B, 256)
        embeddings = F.normalize(embeddings, p=2, dim=1) # (B, 256), ||e|| = 1

        # ── Classification head: embeddings → logits ──
        logits = self.classifier(embeddings)             # (B, num_classes)

        return logits, embeddings


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Split-Head Model — Smoke Test")
    print("=" * 60)

    model = GlobalLocalFusedNetwork(num_classes=74, embed_dim=256)

    # Simulate DataLoader outputs
    dummy_frames = torch.randn(2, 45, 64, 64)  # 45 silhouette frames, batch=2
    dummy_gei = torch.randn(2, 1, 64, 64)      # 1 GEI image per sample

    logits, embeddings = model(dummy_frames, dummy_gei)

    print(f"Logits shape:     {logits.shape}")       # (2, 74)
    print(f"Embeddings shape: {embeddings.shape}")   # (2, 256)

    # Verify L2 normalisation
    norms = torch.norm(embeddings, p=2, dim=1)
    print(f"Embedding norms:  {norms.tolist()}")     # Should be [1.0, 1.0]
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
        "Embeddings are not unit-normalised!"
    print("[OK] Embeddings are unit-normalised (L2 norm = 1.0)")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Embedding dimension:  {model.embed_dim}")