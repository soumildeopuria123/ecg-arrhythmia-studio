"""
ECG Spectrogram Classification Model (PyTorch)

2D CNN architecture for 3-class arrhythmia classification:
- Normal (N) vs Supraventricular ectopic (S) vs Ventricular ectopic (V)
- Input: (batch, 1, 128, 128) grayscale spectrograms
- Output: (batch, 3) logits or soft probabilities

Architecture:
    Conv2d(32, 3x3) → BatchNorm → ReLU → MaxPool2d(2x2)
    Conv2d(64, 3x3) → BatchNorm → ReLU → MaxPool2d(2x2)
    Conv2d(128, 3x3) → BatchNorm → ReLU → MaxPool2d(2x2)
    Conv2d(256, 3x3) → BatchNorm → ReLU → AvgPool2d(4x4)
    Dropout(0.5)
    Dense(128) → ReLU → Dropout(0.3)
    Dense(3) → Softmax
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any


class ECGClassifierCNN(nn.Module):
    """
    2D Convolutional Neural Network for ECG spectrogram classification.

    Designed for (128, 128) grayscale spectrogram inputs with 3 output classes:
    - Class 0: Normal (N)
    - Class 1: Supraventricular ectopic (S)
    - Class 2: Ventricular ectopic (V)
    """

    def __init__(
        self,
        num_classes: int = 3,
        dropout_rate: float = 0.5,
        dense_units: int = 128,
        input_channels: int = 1,
    ):
        """
        Initialize the CNN model.

        Parameters
        ----------
        num_classes : int, optional
            Number of output classes. Default 3 (N, S, V).
        dropout_rate : float, optional
            Dropout rate for regularization. Default 0.5.
        dense_units : int, optional
            Number of units in the dense hidden layer. Default 128.
        input_channels : int, optional
            Number of input channels. Default 1 (grayscale spectrogram).
        """
        super(ECGClassifierCNN, self).__init__()

        # --- Feature Extraction Blocks ---
        # Each block: Conv2d → BatchNorm2d → ReLU → MaxPool2d

        # Block 1: Input (1, 128, 128) → (32, 31, 31) after Conv → (32, 15, 15) after MaxPool
        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,  # Preserves spatial dimensions
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # halves dimensions: 128 → 64
        )

        # Block 2: (32, 64, 64) → (32, 31, 31) Conv → (32, 15, 15) MaxPool
        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64 → 32
        )

        # Block 3: (64, 32, 32) → (64, 16, 16) after MaxPool
        self.block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32 → 16
        )

        # Block 4: (128, 16, 16) → (128, 8, 8) after MaxPool
        self.block4 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16 → 8
        )

        # Block 5: Conv only (no MaxPool at end), (256, 8, 8) → (256, 8, 8)
        # Using AvgPool at end instead to get fixed size
        self.block5 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # No MaxPool here; will use adaptive avg pool later
        )

        # --- Classification Head ---
        # After all blocks, spatial dimensions depend on input size and pooling.
        # With input 128×128 and 5 maxpools of factor 2:
        # 128 → 64 → 32 → 16 → 8 → 8 (block5 preserves size)
        # So after block5: (256, 8, 8) = 256 * 8 * 8 = 16,384 features

        # Adaptive average pooling to ensure fixed size before classifier
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(256, dense_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(dense_units, dense_units // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate // 2),
            nn.Linear(dense_units // 2, num_classes),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Initialize model weights using He/Kaiming initialization for Conv layers
        and standard initialization for Linear layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He initialization (kaiming_normal)
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the CNN.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 1, 128, 128)
            representing grayscale spectrograms.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:
            - "logits": (batch_size, 3) raw output scores before softmax
            - "probabilities": (batch_size, 3) class probabilities after softmax
            - "features": (batch_size, 256) pooled features before classifier
        """
        # Feature extraction
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        # Adaptive pooling to fixed size (1, 1) regardless of spatial dims
        x = self.adaptive_pool(x)
        # Now shape: (batch_size, 256, 1, 1)

        # Flatten
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        # Shape: (batch_size, 256)

        # Classification
        logits = self.classifier(x)
        probabilities = F.softmax(logits, dim=1)

        return {
            "logits": logits,
            "probabilities": probabilities,
            "features": x,  # Pre-classifier features
        }


def create_model(
    num_classes: int = 3,
    input_channels: int = 1,
    device: Optional[torch.device] = None,
) -> ECGClassifierCNN:
    """
    Factory function to create the ECG classification model.

    Parameters
    ----------
    num_classes : int, optional
        Number of classes. Default 3.
    input_channels : int, optional
        Number of input channels. Default 1.
    device : torch.device, optional
        Device to place model on. If None, uses CPU.

    Returns
    -------
    ECGClassifierCNN
        Initialized model ready for training or inference.
    """
    model = ECGClassifierCNN(
        num_classes=num_classes,
        input_channels=input_channels,
    )

    if device is not None:
        model = model.to(device)

    return model


def count_model_parameters(model: ECGClassifierCNN) -> Dict[str, int]:
    """
    Count total and trainable parameters in the model.

    Parameters
    ----------
    model : ECGClassifierCNN
        The model to analyze.

    Returns
    -------
    Dict[str, int]
        Dictionary with "total" and "trainable" parameter counts.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    return {
        "total": total_params,
        "trainable": trainable_params,
    }


# ============================================================
# Inference / Prediction Helpers
# ============================================================


def predict_signature(
    model: ECGClassifierCNN,
    spectrogram: np.ndarray,
    device: Optional[torch.device] = None,
) -> Dict[str, any]:
    """
    Run inference on a single spectrogram and return prediction results.

    Parameters
    ----------
    model : ECGClassifierCNN
        Model already loaded on the appropriate device.
    spectrogram : np.ndarray
        Single spectrogram of shape (128, 128) or (1, 128, 128).
        If (128, 128), will add batch dimension.
    device : torch.device, optional
        Device for computation. If None, uses model's device.

    Returns
    -------
    Dict[str, any]
        Prediction results including:
        - "class_idx": int, predicted class index (0=N, 1=S, 2=V)
        - "class_name": str, human-readable class name
        - "probabilities": dict mapping class names to probabilities
        - "logits": np.ndarray of shape (3,)
        - "confidence": float, max probability
    """
    model.eval()

    # Ensure correct shape
    if spectrogram.ndim == 2:
        # Add channel and batch dimensions: (128, 128) → (1, 1, 128, 128)
        spectrogram = spectrogram[np.newaxis, np.newaxis, :]
    elif spectrogram.ndim == 3:
        if spectrogram.shape[0] == 1:
            # (1, 128, 128) → add batch dim: (1, 1, 128, 128)
            spectrogram = spectrogram[np.newaxis, :]
        elif spectrogram.shape[-1] == 1:
            # Already (1, 128, 128) or similar
            pass
    else:
        raise ValueError(
            f"Unsupported spectrogram shape: {spectrogram.shape}"
        )

    # Convert to tensor
    if device is None:
        device = next(model.parameters()).device

    input_tensor = torch.from_numpy(spectrogram).float().to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(input_tensor)

    logits = outputs["logits"].cpu().numpy()[0]  # (3,)
    probabilities = outputs["probabilities"].cpu().numpy()[0]  # (3,)

    # Map indices to class names
    class_names = {0: "Normal", 1: "Supraventricular", 2: "Ventricular"}
    class_idx = int(np.argmax(probabilities))
    class_name = class_names.get(class_idx, "Unknown")
    confidence = float(probabilities[class_idx])

    # Build probabilities dict
    prob_dict = {
        class_names[i]: float(probabilities[i]) for i in range(len(class_names))
    }

    return {
        "class_idx": class_idx,
        "class_name": class_name,
        "probabilities": prob_dict,
        "logits": logits,
        "confidence": confidence,
    }