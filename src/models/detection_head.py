import math
import torch
import torch.nn as nn

class SimpleDetectionHead(nn.Module):
    def __init__(self, in_channels_list, num_classes=80, num_anchors=9, feature_size=256, prior_prob=0.01):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Ensure all incoming feature maps (P3, P4, P5) are projected to the same feature_size
        self.projections = nn.ModuleList([
            nn.Conv2d(in_channels, feature_size, kernel_size=1) 
            for in_channels in in_channels_list
        ])
        
        # Classification branch
        self.cls_tower = nn.Sequential(
            nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_size, num_classes * num_anchors, kernel_size=3, padding=1)
        )
        
        # Regression branch
        self.bbox_tower = nn.Sequential(
            nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_size, 4 * num_anchors, kernel_size=3, padding=1)
        )
        
        self._init_weights(prior_prob)
        
    def _init_weights(self, prior_prob: float):
        """
        RetinaNet weight initialization:
        - Tower convs: normal distribution with std=0.01, bias=0
        - Final bbox conv: normal distribution with std=0.01, bias=0
        - Final cls conv: normal distribution with std=0.01, bias = -log((1 - p) / p)
          This ensures initial sigmoid output is ~p (default 0.01), preventing massive gradient
          spikes from background anchors on the initial steps.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
                    
        # Set prior probability on the final classification conv layer
        # Output shape: (num_classes * num_anchors, feature_size, 3, 3)
        bias_val = -math.log((1.0 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_tower[-1].bias, bias_val)
        
    def forward(self, features):
        """
        features: list of tensors from P3, P4, P5
        """
        cls_logits = []
        bbox_preds = []
        
        for p_feat, proj in zip(features, self.projections):
            feat = proj(p_feat)
            cls_out = self.cls_tower(feat)
            bbox_out = self.bbox_tower(feat)
            
            cls_logits.append(cls_out)
            bbox_preds.append(bbox_out)
            
        return cls_logits, bbox_preds
