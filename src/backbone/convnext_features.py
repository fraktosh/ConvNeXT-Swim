import torch
import torch.nn as nn
import timm

class ConvNeXtTinyFeatures(nn.Module):
    """
    Extracts multi-scale features from ConvNeXt-Tiny using timm.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        # features_only=True returns the intermediate feature maps instead of the final classification output.
        self.backbone = timm.create_model(
            'convnext_tiny',
            pretrained=pretrained,
            features_only=True
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, C, H, W)
        Returns:
            List of feature tensors at multiple scales.
        """
        return self.backbone(x)
