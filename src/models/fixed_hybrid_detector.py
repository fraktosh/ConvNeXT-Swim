import torch
import torch.nn as nn
from src.backbone.convnext_features import ConvNeXtTinyFeatures
from src.models.detection_head import SimpleDetectionHead
from src.transformer.swin_block import SwinFeatureAdapter

class FixedHybridDetector(nn.Module):
    def __init__(self, variant="P4", num_classes=80, pretrained=True, input_size=(640, 640)):
        super().__init__()
        self.variant = variant
        self.backbone = ConvNeXtTinyFeatures(pretrained=pretrained)
        
        # Expected channels for ConvNeXt-Tiny
        channels = [192, 384, 768]
        # Expected spatial resolutions for P3(1/8), P4(1/16), P5(1/32)
        h, w = input_size
        res_p3 = (h // 8, w // 8)
        res_p4 = (h // 16, w // 16)
        res_p5 = (h // 32, w // 32)
        
        self.swin_p3 = None
        self.swin_p4 = None
        
        if variant == "P4" or variant == "P3P4":
            self.swin_p4 = SwinFeatureAdapter(dim=channels[1], input_resolution=res_p4, num_heads=4)
        
        if variant == "P3P4":
            self.swin_p3 = SwinFeatureAdapter(dim=channels[0], input_resolution=res_p3, num_heads=4)
            
        self.head = SimpleDetectionHead(in_channels_list=channels, num_classes=num_classes)

    def forward(self, x):
        features = self.backbone(x)
        p3, p4, p5 = features[1], features[2], features[3]
        
        # Apply Swin processing based on variant
        if self.swin_p3 is not None:
            p3 = self.swin_p3(p3)
            
        if self.swin_p4 is not None:
            p4 = self.swin_p4(p4)
            
        cls_logits, bbox_preds = self.head([p3, p4, p5])
        return cls_logits, bbox_preds
