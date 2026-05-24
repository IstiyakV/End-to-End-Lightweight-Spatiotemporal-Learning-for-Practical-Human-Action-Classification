"""
Grad-CAM 3D — spatiotemporal attention visualization.

Hooks into last Conv3D layer -> generates heatmap overlay on video frames.
For paper: "Integrating Grad-CAM visualisations to identify discriminative
spatiotemporal regions" (§V Future Work -> now implemented).
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import imageio
from pathlib import Path
from typing import Optional

from . import config
from .model import Compact3DCNN
from .dataset import VideoDataset


class GradCAM3D:
    """Grad-CAM for 3D CNNs — hooks last conv layer."""

    def __init__(self, model: torch.nn.Module, target_layer: Optional[str] = None):
        self.model = model
        self.model.eval()

        # Default: hook last Conv3D/Conv2D layer dynamically
        if target_layer is None:
            # Dynamically discover the last Conv3d or Conv2d module in the model.
            # This is 100% robust and supports Compact3DCNN, Plain3DCNN, ColabCompact3DCNN,
            # HybridTransferModel, ResNet3D-18, R(2+1)D, and MC3!
            last_conv = None
            for module in model.modules():
                if isinstance(module, (torch.nn.Conv3d, torch.nn.Conv2d)):
                    last_conv = module
            if last_conv is not None:
                self.target_layer = last_conv
            else:
                # Fallback to legacy
                try:
                    self.target_layer = model.conv3[0]
                except Exception:
                    self.target_layer = next(model.children())
        else:
            self.target_layer = dict(model.named_modules())[target_layer]

        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(
        self,
        video_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for a video.

        Args:
            video_tensor: (1, C, T, H, W) tensor.
            target_class: Class index. If None, uses predicted class.

        Returns:
            heatmap: (T, H, W) normalised attention map [0, 1].
        """
        # Need gradients for this
        video_tensor.requires_grad_(True)
        self.model.zero_grad()

        with torch.enable_grad():
            output = self.model(video_tensor)

            if target_class is None:
                target_class = output.argmax(dim=1).item()

            # Backward pass on target class score
            score = output[0, target_class]
            score.backward()

        # Grad-CAM computation
        # gradients: (1, C, T', H', W') — from last conv layer
        # activations: (1, C, T', H', W')
        weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)  # GAP over spatial+temporal
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # Weighted sum
        cam = F.relu(cam)  # ReLU
        cam = cam.squeeze(0).squeeze(0)  # (T', H', W')

        # Upsample to original video dimensions
        cam = cam.unsqueeze(0).unsqueeze(0)  # (1, 1, T', H', W')
        cam = F.interpolate(
            cam,
            size=(config.N_FRAMES, *config.IMG_SIZE),
            mode="trilinear",
            align_corners=False,
        )
        cam = cam.squeeze().cpu().numpy()  # (T, H, W)

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 0:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return cam


def generate_gradcam_overlay(
    model: torch.nn.Module,
    video_path: str,
    save_path: Path,
    target_class: Optional[int] = None,
    device: str = config.DEVICE,
    alpha: float = 0.4,
) -> int:
    """
    Generate Grad-CAM overlay GIF for a video.

    Args:
        model: Trained model.
        video_path: Path to input video.
        save_path: Output GIF path.
        target_class: Class to visualize. None = predicted class.
        device: Device.
        alpha: Heatmap overlay opacity.

    Returns:
        predicted_class: Predicted class index.
    """
    # Extract frames
    ds = VideoDataset([video_path], [0], is_train=False)
    frames_tensor, _ = ds[0]  # (C, T, H, W)
    video_input = frames_tensor.unsqueeze(0).to(device)  # (1, C, T, H, W)

    # Get prediction
    with torch.no_grad():
        pred = model(video_input).argmax(1).item()

    # Generate Grad-CAM
    gradcam = GradCAM3D(model)
    heatmap = gradcam(video_input.clone(), target_class or pred)  # (T, H, W)

    # Overlay on original frames
    frames_np = frames_tensor.permute(1, 2, 3, 0).numpy()  # (T, H, W, C) RGB [0,1]
    overlay_frames = []

    for t in range(config.N_FRAMES):
        frame = (frames_np[t] * 255).astype(np.uint8)
        heat = (heatmap[t] * 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)

        blended = cv2.addWeighted(frame, 1 - alpha, heat_color, alpha, 0)
        overlay_frames.append(blended)

    # Save as GIF
    imageio.mimsave(str(save_path), overlay_frames, duration=100, loop=0)
    print(f"Grad-CAM GIF -> {save_path} (predicted class: {pred})")
    return pred
