"""
Grad-CAM for DenseNet121 and ResNet50.

Generates a heatmap showing which regions of the X-ray influenced the prediction.
Required for clinical validation — radiologists need to see where the model looks.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._gradients: torch.Tensor | None = None
        self._activations: torch.Tensor | None = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def generate(self, img_tensor: torch.Tensor) -> np.ndarray:
        """
        Returns a (H, W) heatmap in [0, 1] for a single image tensor (1, C, H, W).
        """
        self.model.eval()
        img_tensor = img_tensor.requires_grad_(True)

        logit = self.model(img_tensor)
        self.model.zero_grad()
        logit.backward()

        # global average pool the gradients over spatial dims
        weights = self._gradients.mean(dim=(2, 3), keepdim=True) # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1, keepdim=True) # (1, 1, H, W)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam

    def overlay(
        self, original_img: np.ndarray, cam: np.ndarray, alpha: float = 0.4
    ) -> np.ndarray:
        """Overlay the heatmap on the original image as a colored blend."""
        import cv2
        h, w = original_img.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        if original_img.ndim == 2:
            original_img = np.stack([original_img] * 3, axis=-1)
        overlay = (alpha * heatmap + (1 - alpha) * original_img).astype(np.uint8)
        return overlay


def get_target_layer(model: nn.Module, backbone_name: str) -> nn.Module:
    """Returns the last convolutional feature block for Grad-CAM."""
    if backbone_name == "densenet121":
        return model.model.features.denseblock4.denselayer16.conv2
    elif backbone_name == "resnet50":
        return model.model.layer4[-1].conv3
    else:
        raise ValueError(f"No Grad-CAM target defined for backbone: {backbone_name}")
