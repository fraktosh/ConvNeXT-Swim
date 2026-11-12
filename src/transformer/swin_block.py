import torch
import torch.nn as nn
from timm.models.swin_transformer import SwinTransformerBlock

class SwinFeatureAdapter(nn.Module):
    """
    Adapts a standard (B, C, H, W) CNN feature map for a Swin Transformer Block,
    and returns it in (B, C, H, W) format.
    """
    def __init__(self, dim, input_resolution, num_heads=4, window_size=7):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        
        # Ensure window size is not larger than input resolution
        actual_window_size = min(window_size, min(input_resolution))
        
        # We initialize a single Swin block. 
        # Shift size is 0 for standard window attention, or window_size // 2 for shifted.
        # We'll use 0 for simplicity in this baseline.
        self.swin_block = SwinTransformerBlock(
            dim=dim,
            input_resolution=input_resolution,
            num_heads=num_heads,
            window_size=actual_window_size,
            shift_size=0
        )
        
    def forward(self, x):
        """
        x: (B, C, H, W)
        Returns: (B, C, H, W)
        """
        B, C, H, W = x.shape
        # CNN to Transformer format: (B, C, H, W) -> (B, H, W, C)
        x_trans = x.permute(0, 2, 3, 1)
        
        # Apply Swin Block
        x_trans = self.swin_block(x_trans)
        
        # Transformer back to CNN format: (B, H, W, C) -> (B, C, H, W)
        x_out = x_trans.permute(0, 3, 1, 2)
        return x_out
