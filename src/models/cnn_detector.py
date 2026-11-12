import torch
import torch.nn as nn
from src.backbone.convnext_features import ConvNeXtTinyFeatures
from src.models.detection_head import SimpleDetectionHead

class CNNDetector(nn.Module):
    def __init__(self, num_classes=80, pretrained=True):
        super().__init__()
        self.backbone = ConvNeXtTinyFeatures(pretrained=pretrained)
        
        # ConvNeXt-Tiny channels for P3 (1/8), P4 (1/16), P5 (1/32) are typically [192, 384, 768]
        # In timm features_only: feature indices [1, 2, 3] usually correspond to these
        # We will dynamically grab them in forward pass, but hardcode channels here for head projection
        self.head = SimpleDetectionHead(in_channels_list=[192, 384, 768], num_classes=num_classes)

    def forward(self, x):
        features = self.backbone(x)
        # Assuming features from timm convnext_tiny have 4 scales. We need P3, P4, P5 (indices 1, 2, 3)
        p3_p4_p5 = features[1:4] 
        cls_logits, bbox_preds = self.head(p3_p4_p5)
        return cls_logits, bbox_preds
