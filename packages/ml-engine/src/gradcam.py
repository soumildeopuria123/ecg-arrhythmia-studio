"""
Grad-CAM Explainability Module

Generates Grad-CAM heatmaps for the ECG CNN model, overlaying frequency-time
regions that contribute to classification predictions. Uses PyTorch hook
mechanisms to capture activations and gradients from the final convolutional
layer before adaptive average pooling.

Features:
- Forward hook to save activation tensors
- Backward hook to capture gradients
- Channel-weighted CAM generation
- ReLU activation to focus on positive contributions
- Spatial resizing to input dimension
- Jet colormap overlay with transparency
- Returns raw PNG bytes for API response
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, List


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for PyTorch models.

    Generates class-specific heatmaps highlighting the input regions that
    most contribute to the predicted class score.

    Target layer: The final convolutional block before global average pooling.
    For our ECG CNN: block4 output (256 channels, 8x8 spatial).
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        use_logits: bool = True,
    ):
        """
        Initialize Grad-CAM for a specific model and target layer.

        Parameters
        ----------
        model : nn.Module
            The PyTorch model (must be in eval mode).
        target_layer : nn.Module
            The convolutional layer to generate CAM for. Should be the
            last conv layer before pooling.
        use_logits : bool, optional
            If True, backpropagate the target class logit.
            If False, use the output probability. Default True.
        """
        self.model = model
        self.target_layer = target_layer
        self.use_logits = use_logits

        # Storage for activations and gradients
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on the target layer."""
        # Forward hook: save activations
        self.activation_handle = self.target_layer.register_forward_hook(
            self._save_activation
        )

        # Backward hook: save gradients
        self.gradient_handle = self.target_layer.register_backward_hook(
            self._save_gradient
        )

    def _save_activation(self, module: nn.Module, input: Tuple[torch.Tensor],
                         output: torch.Tensor) -> None:
        """
        Save the forward pass activation tensor.

        Parameters
        ----------
        module : nn.Module
            The layer module.
        input : Tuple[torch.Tensor]
            Input to the layer.
        output : torch.Tensor
            Output activation tensor from the layer.
        """
        self.activations = output.detach()

    def _save_gradient(self, module: nn.Module, grad_input: Tuple[torch.Tensor],
                       grad_output: Tuple[torch.Tensor]) -> None:
        """
        Save the backward pass gradient tensor.

        Parameters
        ----------
        module : nn.Module
            The layer module.
        grad_input : Tuple[torch.Tensor]
            Gradient of the loss with respect to the layer input.
        grad_output : Tuple[torch.Tensor]
            Gradient of the loss with respect to the layer output.
        """
        self.gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        if hasattr(self, 'activation_handle'):
            self.activation_handle.remove()
        if hasattr(self, 'gradient_handle'):
            self.gradient_handle.remove()
        self.activations = None
        self.gradients = None

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for the given input and target class.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input batch tensor of shape (1, 1, 128, 128) - single grayscale
            spectrogram with batch dimension.
        target_class : int, optional
            Specific class index to generate CAM for.
            If None, uses the predicted class with highest score.

        Returns
        -------
        np.ndarray
            Grad-CAM heatmap of shape (128, 128) with values in [0, 1].
            Higher values indicate regions most contributing to the class.
        """
        if input_tensor.ndim != 4:
            raise ValueError(
                f"Input tensor must be 4D (batch, channels, height, width), "
                f"got shape {input_tensor.shape}"
            )

        if input_tensor.shape[1] != 1:
            raise ValueError(
                f"Expected 1 input channel (grayscale), got {input_tensor.shape[1]}"
            )

        # Ensure model is in eval mode
        self.model.eval()

        # Zero gradients
        self.model.zero_grad(set_to_none=True)

        # Forward pass
        outputs = self.model(input_tensor)
        logits = outputs["logits"]  # (1, 3)

        # Determine target class
        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        # Zero the gradients
        self.model.zero_grad()

        # Backward pass - compute gradients of the target class score
        if self.use_logits:
            target_score = logits[0, target_class]
        else:
            # Use probability instead
            target_score = outputs["probabilities"][0, target_class]

        target_score.backward()

        # --- Post-processing ---

        # 1. Get gradients: shape (1, 256, 8, 8) - average across spatial dims
        if self.gradients is None:
            raise RuntimeError("Gradients not captured. Hook may not have fired.")

        gradients = self.gradients  # (1, channels, h, w)
        weights = gradients.mean(dim=(2, 3), keepdim=False)  # (1, channels)

        # 2. Weighted combination of activations
        activations = self.activations  # (1, channels, h, w)
        # weights shape: (1, channels) -> broadcast to (1, channels, 1, 1)
        weights_expanded = weights.view(1, -1, 1, 1)

        # CAM: weighted sum of feature maps
        cam = (weights_expanded * activations).sum(dim=1, keepdim=True)
        # cam shape: (1, 1, h, w) - single channel heatmap

        # 3. Apply ReLU to keep only positive contributions
        cam = F.relu(cam)

        # 4. Normalize CAM to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        cam = cam.squeeze()  # Remove dims -> (128, 128) or whatever spatial size

        # 5. Resize to input image size (128x128)
        # Input spectrogram is 128x128, CAM may be smaller due to pooling
        target_size = (128, 128)
        cam_resized = cv2.resize(
            cam.numpy(), target_size, interpolation=cv2.INTER_LINEAR
        )

        # Ensure values are in [0, 1]
        cam_resized = np.clip(cam_resized, 0, 1)

        return cam_resized

    def generate_overlay(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        alpha: float = 0.6,
    ) -> Dict[str, any]:
        """
        Generate Grad-CAM overlay image blending heatmap with original spectrogram.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input batch tensor of shape (1, 1, 128, 128).
        target_class : int, optional
            Class index to explain. If None, uses predicted class.
        alpha : float, optional
            Transparency factor for overlay. 0.0 = only heatmap, 1.0 = only original.
            Default 0.6 (good balance).

        Returns
        -------
        Dict[str, any]
            Dictionary containing:
            - "heatmap": np.ndarray uint8, shape (128, 128, 3), jet colormap
            - "overlay": np.ndarray uint8, shape (128, 128, 3), blended image
            - "original": np.ndarray uint8, shape (128, 128, 3), original spectrogram
            - "cam": np.ndarray float32, shape (128, 128), raw Grad-CAM values [0,1]
            - "target_class": int, the class being explained
        """
        # Generate raw CAM
        cam = self.generate_cam(input_tensor, target_class=target_class)

        # Convert input tensor to numpy (denormalize for visualization)
        # input_tensor shape: (1, 1, 128, 128)
        input_np = input_tensor.squeeze().cpu().numpy()  # (128, 128)

        # Denormalize: assume input was normalized to [0, 1] from preprocessing
        # If already in [0, 1], no further scaling needed
        if input_np.min() < 0:
            # Denormalize assuming z-score or min-max from [-1, 1] or similar
            input_np = (input_np - input_np.min()) / (input_np.max() - input_np.min() + 1e-8)

        # Clip to [0, 1] and convert to uint8
        input_uint8 = (np.clip(input_np, 0, 1) * 255).astype(np.uint8)

        # Apply jet colormap to CAM heatmap
        # cam shape: (128, 128), values in [0, 1]
        heatmap = apply_jet_colormap_cv(cam, alpha=1.0)  # Full intensity heatmap

        # Convert input to 3-channel for blending
        input_3ch = cv2.cvtColor(input_uint8, cv2.COLOR_GRAY2BGR)

        # Create overlay: blend heatmap with original
        # heatmap is already uint8 (BGR), input_3ch is BGR
        overlay = cv2.addWeighted(
            input_3ch, 1.0 - alpha, heatmap, alpha, 0.0
        )

        # --- Build result dict ---
        result = {
            "heatmap": heatmap,      # uint8 (128, 128, 3) jet colormap
            "overlay": overlay,      # uint8 (128, 128, 3) blended
            "original": input_3ch,   # uint8 (128, 128, 3) original gray
            "cam": cam.astype(np.float32),  # float32 (128, 128) [0,1]
            "target_class": target_class
            if target_class is not None
            else int(torch.argmax(self.model(input_tensor)["logits"], dim=1).item()),
        }

        return result


def apply_jet_colormap_cv(
    heatmap_2d: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    """
    Apply OpenCV jet colormap to a 2D heatmap.

    OpenCV's COLORMAP_JET maps values to a color gradient:
    - Low values: blue
    - Medium: cyan/yellow
    - High: red

    Parameters
    ----------
    heatmap_2d : np.ndarray
        2D array of shape (H, W) with values in [0, 1].
    alpha : float, optional
        Not used in CV version but kept for API compatibility.

    Returns
    -------
    np.ndarray
        RGB uint8 array of shape (H, W, 3) with jet colormap applied.
    """
    if not isinstance(heatmap_2d, np.ndarray) or heatmap_2d.ndim != 2:
        raise ValueError("Heatmap must be a 2D numpy array.")

    # OpenCV expects values in [0, 255] for colormap application
    # Our input is already [0, 1], so scale
    heatmap_scaled = (heatmap_2d * 255).astype(np.uint8)

    # Apply jet colormap
    colored = cv2.applyColorMap(heatmap_scaled, cv2.COLORMAP_JET)

    # Convert from BGR (OpenCV default) to RGB
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    # Normalize back to [0, 1] float
    colored_float = colored_rgb.astype(np.float32) / 255.0

    return colored_float


def generate_gradcam_png_bytes(
    model: nn.Module,
    target_layer: nn.Module,
    input_tensor: torch.Tensor,
    target_class: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> bytes:
    """
    Convenience function: generate Grad-CAM as PNG bytes.

    Full pipeline:
    1. Generate CAM overlay
    2. Encode as PNG
    3. Return raw bytes

    Parameters
    ----------
    model : nn.Module
        The trained ECG classification model.
    target_layer : nn.Module
        The target convolutional layer (final conv block before pooling).
    input_tensor : torch.Tensor
        Single spectrogram tensor (1, 1, 128, 128).
    target_class : int, optional
        Class to explain. If None, uses predicted class.
    device : torch.device, optional
        Device for computation.

    Returns
    -------
    bytes
        PNG image bytes of the Grad-CAM overlay.
    """
    if device is None:
        device = next(model.parameters()).device

    model = model.to(device)
    input_tensor = input_tensor.to(device)

    # Create Grad-CAM instance and generate overlay
    grad_cam = GradCAM(model, target_layer)
    result = grad_cam.generate_overlay(input_tensor, target_class=target_class)

    # Encode overlay as PNG bytes
    overlay_image = result["overlay"]  # uint8 (128, 128, 3)

    # Encode to PNG
    success, encoded_image = cv2.imencode(
        ".png", overlay_image[:, :, ::-1]
    )  # RGB->BGR for OpenCV

    if not success:
        raise RuntimeError("Failed to encode Grad-CAM overlay as PNG.")

    png_bytes = encoded_image.tobytes()
    grad_cam.remove_hooks()

    return png_bytes


# ============================================================
# Factory Functions
# ============================================================


def create_grad_cam_hook(
    model: ECGClassifierCNN,
    layer_name: str = "block4",
) -> GradCAM:
    """
    Create a GradCAM instance targeting a specific layer of the ECG CNN.

    For the ECGClassifierCNN, the layer naming corresponds to:
    - "block1": First conv block (32 filters)
    - "block2": Second conv block (64 filters)
    - "block3": Third conv block (128 filters)
    - "block4": Fourth conv block (256 filters) <- Recommended target
    - "block5": Fifth conv block (256 filters, no pooling)

    Parameters
    ----------
    model : ECGClassifierCNN
        The classification model.
    layer_name : str, optional
        Name of the layer to target. Default "block4".

    Returns
    -------
    GradCAM
        Configured GradCAM instance ready for use.
    """
    # Map layer name to module
    layer_map = {
        "block1": model.block1,
        "block2": model.block2,
        "block3": model.block3,
        "block4": model.block4,
        "block5": model.block5,
    }

    target_layer = layer_map.get(layer_name)
    if target_layer is None:
        available = ", ".join(layer_map.keys())
        raise ValueError(
            f"Unknown layer '{layer_name}'. Available: {available}"
        )

    grad_cam = GradCAM(model, target_layer)
    return grad_cam