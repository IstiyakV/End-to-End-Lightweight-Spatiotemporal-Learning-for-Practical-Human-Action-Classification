"""
Hybrid Transfer Model — binds pre-trained backbone to a custom classification head.
Fully isolated from pre-existing model architectures.
"""

import torch
import torch.nn as nn
import torchvision.models.video as video_models


class HybridTransferModel(nn.Module):
    """
    Binds a heavy spatiotemporal pre-trained backbone (e.g., r3d_18, r2plus1d_18)
    to a custom-built neural network classification head.
    Automatically manages:
      1. Backbone parameter freezing (conv layers requires_grad = False).
      2. Feature dimension projection (nn.Linear adapter if shapes mismatch).
      3. Complete isolation from main UCF-101 pipeline configurations.
    """
    def __init__(
        self,
        backbone: nn.Module,
        custom_head: nn.Module,
        backbone_feature_dim: int = 512,
        freeze_backbone: bool = True
    ):
        super().__init__()
        self.backbone = backbone
        self.custom_head = custom_head
        self.backbone_feature_dim = backbone_feature_dim

        # 1. Strip the original classification head of the backbone (replace with Identity)
        if hasattr(self.backbone, "fc"):
            self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, "classifier"):
            self.backbone.classifier = nn.Identity()

        # 2. Analyze the input dimension of the custom classifier head.
        # We look for the first Linear layer in the custom head to determine its input shape.
        self.custom_in_features = self.backbone_feature_dim
        first_linear = None
        for module in self.custom_head.modules():
            if isinstance(module, nn.Linear):
                first_linear = module
                break

        if first_linear is not None:
            self.custom_in_features = first_linear.in_features
            print(f"[HYBRID BINDER] Detected custom head input feature size: {self.custom_in_features}")
        else:
            print(f"[HYBRID BINDER] [WARNING] No linear layer found in custom head. Defaulting in_features to {self.backbone_feature_dim}")

        # 3. Create a Projection Layer if the backbone feature dimension does not match the custom head input
        self.projection = nn.Identity()
        if self.backbone_feature_dim != self.custom_in_features:
            print(f"[HYBRID BINDER] Dimension mismatch: Backbone outputs {self.backbone_feature_dim} but Custom Head expects {self.custom_in_features}.")
            print(f"[HYBRID BINDER] Dynamically binding nn.Linear({self.backbone_feature_dim}, {self.custom_in_features}) adapter layer.")
            self.projection = nn.Linear(self.backbone_feature_dim, self.custom_in_features)

        # 4. Programmatically freeze the backbone parameters if requested
        if freeze_backbone:
            print("[HYBRID BINDER] Programmatically freezing all spatiotemporal backbone layers (param.requires_grad = False).")
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            print("[HYBRID BINDER] [WARNING] Backbone is unfrozen (all layers active).")

        # Keep custom head and projection active/trainable
        for param in self.projection.parameters():
            param.requires_grad = True
        for param in self.custom_head.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (B, C, T, H, W)
        
        # 1. Spatiotemporal feature extraction through backbone
        features = self.backbone(x) # Output shape: (B, backbone_feature_dim)
        
        # Flatten features if they have spatial/temporal dimensions remaining (e.g. from pool)
        if len(features.shape) > 2:
            features = torch.flatten(features, 1)

        # 2. Shape projection/alignment
        aligned_features = self.projection(features) # Output shape: (B, custom_in_features)

        # 3. Classification output through the custom neural head
        out = self.custom_head(aligned_features)
        return out


def count_parameters(model: nn.Module) -> tuple:
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
