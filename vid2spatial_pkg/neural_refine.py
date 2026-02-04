"""
Neural Refinement Module for Vid2Spatial

Refines geometric spatial audio predictions using visual features.
Based on SOTA research (2024-2025):
- Multi-scale visual features (DeepNeRAP 2024)
- Attention fusion (Neural Acoustic Fields 2025)
- Uncertainty estimation (Multi-task Learning)
- Geometric consistency (Cross-modal 2024)

References:
- DeepNeRAP: https://merl.com/publications/docs/TR2024-072.pdf
- Neural Acoustic Fields: https://dl.acm.org/doi/10.1109/TVCG.2025.3549898
- Sep-Stereo: https://link.springer.com/chapter/10.1007/978-3-030-58610-2_4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
from typing import Tuple, Optional


class MultiScaleVisualEncoder(nn.Module):
    """
    Extract multi-scale visual features from RGB frames.

    Inspired by DeepNeRAP (2024) multi-scale approach.
    Uses ResNet-18 intermediate layers for different scales.
    """

    def __init__(self):
        super().__init__()

        # Load pretrained ResNet-18
        resnet = models.resnet18(pretrained=True)

        # Extract intermediate layers for multi-scale features
        self.layer1 = nn.Sequential(*list(resnet.children())[:5])  # 64 channels, /4
        self.layer2 = resnet.layer2  # 128 channels, /8
        self.layer3 = resnet.layer3  # 256 channels, /16
        self.layer4 = resnet.layer4  # 512 channels, /32

        # Adaptive pooling for global features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fusion: combine multi-scale features
        # Total channels: 64 + 128 + 256 + 512 = 960
        self.fusion = nn.Sequential(
            nn.Conv2d(960, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] RGB frames

        Returns:
            features: [B, 512] global visual features
        """
        # Multi-scale feature extraction
        f1 = self.layer1(x)  # [B, 64, H/4, W/4]
        f2 = self.layer2(f1)  # [B, 128, H/8, W/8]
        f3 = self.layer3(f2)  # [B, 256, H/16, W/16]
        f4 = self.layer4(f3)  # [B, 512, H/32, W/32]

        # Resize all to same spatial size (smallest)
        target_size = f4.shape[-2:]
        f1_resized = F.adaptive_avg_pool2d(f1, target_size)
        f2_resized = F.adaptive_avg_pool2d(f2, target_size)
        f3_resized = F.adaptive_avg_pool2d(f3, target_size)

        # Concatenate along channel dimension
        multi_scale = torch.cat([f1_resized, f2_resized, f3_resized, f4], dim=1)

        # Fuse to 512 channels
        fused = self.fusion(multi_scale)  # [B, 512, H/32, W/32]

        # Global pooling
        features = self.pool(fused).squeeze(-1).squeeze(-1)  # [B, 512]

        return features


class AttentionFusion(nn.Module):
    """
    Cross-attention fusion between visual and geometric features.

    Inspired by Neural Acoustic Fields (2025) attention mechanism.
    Visual features attend to geometric predictions to learn what to refine.
    """

    def __init__(self, vis_dim=512, geo_dim=128, hidden=256, num_heads=8):
        super().__init__()

        # Project features to same dimension
        self.vis_proj = nn.Linear(vis_dim, hidden)
        self.geo_proj = nn.Linear(geo_dim, hidden)

        # Multi-head cross-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        # Output projection with residual
        self.out_proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, vis_feat, geo_feat):
        """
        Args:
            vis_feat: [B, vis_dim] visual features
            geo_feat: [B, geo_dim] geometric features

        Returns:
            fused: [B, hidden] fused features
        """
        # Project to same dimension
        V = self.vis_proj(vis_feat).unsqueeze(1)  # [B, 1, hidden]
        G = self.geo_proj(geo_feat).unsqueeze(1)  # [B, 1, hidden]

        # Cross-attention: V attends to G
        # query=V (what we want to refine), key/value=G (geometric context)
        attn_out, attn_weights = self.attention(V, G, G)  # [B, 1, hidden]
        attn_out = attn_out.squeeze(1)  # [B, hidden]

        # Output with residual
        output = self.out_proj(attn_out + V.squeeze(1))

        return output


class UncertaintyRefinement(nn.Module):
    """
    Refinement head with uncertainty estimation.

    Predicts both refinement deltas and uncertainty (log variance).
    Inspired by multi-task learning with uncertainty weighting.
    """

    def __init__(self, input_dim=256):
        super().__init__()

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Refinement head: predict deltas [Δaz, Δel, Δdist]
        self.refine_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Tanh()  # Bounded output
        )

        # Uncertainty head: predict log variance log(σ²)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # Unbounded (log variance)
        )

    def forward(self, x):
        """
        Args:
            x: [B, input_dim] fused features

        Returns:
            delta: [B, 3] refinement deltas (bounded by tanh)
            log_var: [B, 3] log variance for uncertainty
        """
        # Shared features
        feat = self.backbone(x)

        # Refinement prediction
        delta = self.refine_head(feat)  # [-1, 1] from tanh

        # Uncertainty prediction
        log_var = self.uncertainty_head(feat)

        return delta, log_var


class SpatialRefiner(nn.Module):
    """
    Complete neural refinement module for geometric spatial audio.

    Architecture:
        1. Multi-scale visual encoder (ResNet-18 based)
        2. Geometric encoder (MLP)
        3. Attention-based fusion
        4. Uncertainty-aware refinement

    Usage:
        refiner = SpatialRefiner()
        refined = refiner(frame, geo_pred)

    Or with uncertainty:
        refined, log_var = refiner(frame, geo_pred, return_uncertainty=True)
    """

    def __init__(
        self,
        use_multiscale: bool = True,
        use_attention: bool = True,
        max_delta_az: float = 15.0,
        max_delta_el: float = 10.0,
        max_delta_dist: float = 1.0
    ):
        """
        Args:
            use_multiscale: Use multi-scale visual encoder
            use_attention: Use attention fusion
            max_delta_az: Max azimuth correction (degrees)
            max_delta_el: Max elevation correction (degrees)
            max_delta_dist: Max distance correction (meters)
        """
        super().__init__()

        self.use_multiscale = use_multiscale
        self.use_attention = use_attention
        self.max_delta = torch.tensor([max_delta_az, max_delta_el, max_delta_dist])

        # Visual encoder
        if use_multiscale:
            self.visual_encoder = MultiScaleVisualEncoder()
        else:
            # Simple ResNet-18 (last layer only)
            resnet = models.resnet18(pretrained=True)
            self.visual_encoder = nn.Sequential(*list(resnet.children())[:-1])

        self.vis_dim = 512

        # Geometric encoder
        self.geo_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        self.geo_dim = 128

        # Fusion
        if use_attention:
            self.fusion = AttentionFusion(self.vis_dim, self.geo_dim, hidden=256)
            self.fusion_dim = 256
        else:
            # Simple concatenation
            self.fusion_dim = self.vis_dim + self.geo_dim

        # Refinement with uncertainty
        self.refiner = UncertaintyRefinement(input_dim=self.fusion_dim)

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def preprocess_frame(self, frame):
        """
        Preprocess input frame to correct format.

        Args:
            frame: Can be:
                - [B, 3, H, W] tensor (already correct)
                - [B, H, W, 3] tensor (need permute)
                - [H, W, 3] tensor (need unsqueeze + permute)
                - [H, W, 3] numpy (convert to tensor)

        Returns:
            preprocessed: [B, 3, 224, 224] tensor
        """
        # Convert numpy to tensor
        if isinstance(frame, np.ndarray):
            frame = torch.from_numpy(frame).float() / 255.0

        # Add batch dimension if needed
        if frame.ndim == 3:
            frame = frame.unsqueeze(0)

        # Permute if channels last
        if frame.shape[-1] == 3:  # [B, H, W, 3]
            frame = frame.permute(0, 3, 1, 2)  # [B, 3, H, W]

        # Apply normalization and resize
        batch_size = frame.shape[0]
        preprocessed = torch.stack([self.transform(frame[i]) for i in range(batch_size)])

        return preprocessed

    def forward(
        self,
        frame,
        geo_pred,
        return_uncertainty: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Refine geometric prediction using visual features.

        Args:
            frame: Input RGB frame(s)
                - [B, 3, H, W] or [B, H, W, 3] tensor
                - [H, W, 3] numpy array
            geo_pred: [B, 3] or [3] geometric prediction [az, el, dist]
            return_uncertainty: If True, return uncertainty estimates

        Returns:
            refined: [B, 3] refined prediction [az, el, dist]
            log_var: [B, 3] log variance (if return_uncertainty=True)
        """
        # Preprocess frame
        frame = self.preprocess_frame(frame)
        batch_size = frame.shape[0]

        # Ensure geo_pred has batch dimension
        if geo_pred.ndim == 1:
            geo_pred = geo_pred.unsqueeze(0)

        # Move max_delta to same device
        max_delta = self.max_delta.to(frame.device)

        # 1. Extract visual features
        vis_feat = self.visual_encoder(frame)
        if vis_feat.ndim > 2:
            vis_feat = vis_feat.squeeze(-1).squeeze(-1)  # [B, 512]

        # 2. Encode geometric prediction
        geo_feat = self.geo_encoder(geo_pred)  # [B, 128]

        # 3. Fuse features
        if self.use_attention:
            fused = self.fusion(vis_feat, geo_feat)  # [B, 256]
        else:
            fused = torch.cat([vis_feat, geo_feat], dim=-1)  # [B, 640]

        # 4. Predict refinement with uncertainty
        delta, log_var = self.refiner(fused)  # [B, 3], [B, 3]

        # 5. Scale deltas by maximum allowed correction
        # delta is in [-1, 1] from tanh, scale to actual correction range
        delta = delta * max_delta.unsqueeze(0)  # [B, 3]

        # 6. Apply refinement
        refined = geo_pred + delta

        # 7. Clamp to valid ranges
        refined = torch.stack([
            torch.clamp(refined[:, 0], -180, 180),  # azimuth
            torch.clamp(refined[:, 1], -90, 90),    # elevation
            torch.clamp(refined[:, 2], 0.1, 20.0)   # distance
        ], dim=1)

        if return_uncertainty:
            return refined, log_var
        else:
            return refined

    @torch.no_grad()
    def predict(self, frame_np, geo_pred_np):
        """
        Convenience method for inference from numpy arrays.

        Args:
            frame_np: [H, W, 3] numpy array (RGB, 0-255)
            geo_pred_np: [3] numpy array [az, el, dist]

        Returns:
            refined_np: [3] numpy array refined prediction
        """
        self.eval()

        # Convert to tensors
        frame_tensor = torch.from_numpy(frame_np).float() / 255.0
        geo_tensor = torch.from_numpy(geo_pred_np).float()

        # Forward pass
        refined = self.forward(frame_tensor, geo_tensor, return_uncertainty=False)

        # Convert back to numpy
        return refined.squeeze().cpu().numpy()


# Loss functions
def uncertainty_loss(pred, target, log_var):
    """
    Loss with learned uncertainty weighting.

    L = (pred - target)² / (2σ²) + log(σ²) / 2

    This allows the network to learn when predictions are uncertain,
    automatically weighting the loss accordingly.

    Args:
        pred: [B, D] predictions
        target: [B, D] ground truth
        log_var: [B, D] log variance (log σ²)

    Returns:
        loss: scalar loss value
    """
    # Precision = 1 / σ²
    precision = torch.exp(-log_var)

    # Weighted MSE + regularization term
    loss = 0.5 * precision * (pred - target)**2 + 0.5 * log_var

    return loss.mean()


def refinement_loss(
    refined,
    target,
    log_var,
    geo_pred,
    lambda_geo: float = 0.5,
    lambda_reg: float = 0.1
):
    """
    Combined loss for neural refinement training.

    Components:
    1. Uncertainty-weighted MSE (main prediction loss)
    2. Geometric consistency (penalize wild deviations from geometric)
    3. Regularization (prevent over-correction)

    Args:
        refined: [B, 3] refined predictions
        target: [B, 3] ground truth
        log_var: [B, 3] log variance
        geo_pred: [B, 3] geometric predictions (before refinement)
        lambda_geo: Weight for geometric consistency
        lambda_reg: Weight for regularization

    Returns:
        total_loss: Combined loss
        loss_dict: Dictionary of individual loss components
    """
    # 1. Main loss: Uncertainty-weighted MSE
    loss_main = uncertainty_loss(refined, target, log_var)

    # 2. Geometric consistency: don't deviate too far from geometric
    delta = refined - geo_pred
    loss_geo = (delta ** 2).mean()

    # 3. Regularization: penalize large uncertainties
    # Encourage network to be confident when possible
    loss_reg = log_var.mean()

    # Total loss
    total_loss = loss_main + lambda_geo * loss_geo + lambda_reg * loss_reg

    loss_dict = {
        'total': total_loss.item(),
        'main': loss_main.item(),
        'geo': loss_geo.item(),
        'reg': loss_reg.item()
    }

    return total_loss, loss_dict


if __name__ == "__main__":
    # Test the module
    print("Testing SpatialRefiner...")

    # Create model
    model = SpatialRefiner(use_multiscale=True, use_attention=True)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test input
    frame = torch.randn(2, 3, 480, 640)  # 2 frames
    geo_pred = torch.randn(2, 3)  # 2 geometric predictions

    # Forward pass
    refined = model(frame, geo_pred, return_uncertainty=False)
    print(f"Input shape: {frame.shape}")
    print(f"Geometric pred: {geo_pred.shape}")
    print(f"Refined shape: {refined.shape}")

    # With uncertainty
    refined, log_var = model(frame, geo_pred, return_uncertainty=True)
    print(f"Log variance shape: {log_var.shape}")

    # Test loss
    target = torch.randn(2, 3)
    loss, loss_dict = refinement_loss(refined, target, log_var, geo_pred)
    print(f"\nLoss components:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")

    print("\n✅ SpatialRefiner module test passed!")
