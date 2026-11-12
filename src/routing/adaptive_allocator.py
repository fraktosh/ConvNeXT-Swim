import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAdaptiveAllocator(nn.Module):
    """
    Learns to route *each spatial region* (pixel) of a feature map 
    to one of 3 processing paths:
    0: CNN-only
    1: CNN + Lightweight Swin
    2: CNN + Deeper Swin
    """
    def __init__(self, in_channels):
        super().__init__()
        # A lightweight convolutional router that preserves spatial dimensions
        # Output is (B, 3, H, W) representing the logits for the 3 choices per region
        self.router = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, 3, kernel_size=3, padding=1)
        )
        
    def forward(self, x, tau=1.0, hard=False):
        """
        x: (B, C, H, W)
        Returns:
            route_weights: (B, 3, H, W) probabilities
        """
        logits = self.router(x) # (B, 3, H, W)
        
        if self.training:
            # Gumbel-Softmax over the 3 choices (dim=1)
            route_weights = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=1)
        else:
            # Hard routing at inference
            idx = logits.argmax(dim=1) # (B, H, W)
            # F.one_hot puts classes at the last dimension, so we permute to (B, 3, H, W)
            route_weights = F.one_hot(idx, num_classes=3).float().permute(0, 3, 1, 2)
            
        return route_weights
