import torch
import torch.nn as nn
from src.backbone.convnext_features import ConvNeXtTinyFeatures
from src.models.detection_head import SimpleDetectionHead
from src.transformer.swin_block import SwinFeatureAdapter
from src.routing.adaptive_allocator import SpatialAdaptiveAllocator

class AdaptiveHybridDetector(nn.Module):
    def __init__(self, num_classes=80, pretrained=True, input_size=(640, 640)):
        super().__init__()
        print("WARNING: This AdaptiveHybridDetector implements a NON-CONDITIONAL adaptive baseline.")
        print("It computes all three branches and masks them afterward, rather than skipping execution.")
        print("Any computational savings reported are theoretical/simulated, not measured wall-clock savings.")
        
        self.backbone = ConvNeXtTinyFeatures(pretrained=pretrained)
        
        channels = [192, 384, 768]
        h, w = input_size
        res_p3 = (h // 8, w // 8)
        res_p4 = (h // 16, w // 16)
        res_p5 = (h // 32, w // 32)
        
        # Allocators for P3 and P4 (Spatial region-level routing)
        self.allocator_p3 = SpatialAdaptiveAllocator(channels[0])
        self.allocator_p4 = SpatialAdaptiveAllocator(channels[1])
        
        # Lightweight Swin branches (1 block)
        self.swin_light_p3 = SwinFeatureAdapter(dim=channels[0], input_resolution=res_p3, num_heads=4)
        self.swin_light_p4 = SwinFeatureAdapter(dim=channels[1], input_resolution=res_p4, num_heads=4)
        
        # Deeper Swin branches (simulated by stacking 2 blocks)
        self.swin_deep_p3 = nn.Sequential(
            SwinFeatureAdapter(dim=channels[0], input_resolution=res_p3, num_heads=4),
            SwinFeatureAdapter(dim=channels[0], input_resolution=res_p3, num_heads=4)
        )
        self.swin_deep_p4 = nn.Sequential(
            SwinFeatureAdapter(dim=channels[1], input_resolution=res_p4, num_heads=4),
            SwinFeatureAdapter(dim=channels[1], input_resolution=res_p4, num_heads=4)
        )
        
        # Lightweight projections to give all paths comparable adaptation interfaces
        self.proj_cnn_p3 = nn.Conv2d(channels[0], channels[0], 1)
        self.proj_light_p3 = nn.Conv2d(channels[0], channels[0], 1)
        self.proj_deep_p3 = nn.Conv2d(channels[0], channels[0], 1)
        
        self.proj_cnn_p4 = nn.Conv2d(channels[1], channels[1], 1)
        self.proj_light_p4 = nn.Conv2d(channels[1], channels[1], 1)
        self.proj_deep_p4 = nn.Conv2d(channels[1], channels[1], 1)
        
        self.head = SimpleDetectionHead(in_channels_list=channels, num_classes=num_classes)

    def _apply_routing(self, feat, route_weights, light_branch, deep_branch, proj_cnn, proj_light, proj_deep):
        # feat: (B, C, H, W)
        # route_weights: (B, 3, H, W)
        B, C, H, W = feat.shape
        
        # Apply adaptations
        feat_cnn = proj_cnn(feat)
        feat_light = proj_light(light_branch(feat))
        feat_deep = proj_deep(deep_branch(feat))
        
        # Expand weights across channel dim: (B, 1, H, W)
        w0 = route_weights[:, 0:1, :, :]
        w1 = route_weights[:, 1:2, :, :]
        w2 = route_weights[:, 2:3, :, :]
        
        out = (w0 * feat_cnn) + (w1 * feat_light) + (w2 * feat_deep)
                    
        return out

    def forward(self, x, tau=1.0, hard=False):
        features = self.backbone(x)
        p3, p4, p5 = features[1], features[2], features[3]
        
        # Get routing weights
        weights_p3 = self.allocator_p3(p3, tau=tau, hard=hard)
        weights_p4 = self.allocator_p4(p4, tau=tau, hard=hard)
        
        # Apply routed features
        p3_out = self._apply_routing(p3, weights_p3, self.swin_light_p3, self.swin_deep_p3, self.proj_cnn_p3, self.proj_light_p3, self.proj_deep_p3)
        p4_out = self._apply_routing(p4, weights_p4, self.swin_light_p4, self.swin_deep_p4, self.proj_cnn_p4, self.proj_light_p4, self.proj_deep_p4)
        
        cls_logits, bbox_preds = self.head([p3_out, p4_out, p5])
        return cls_logits, bbox_preds, weights_p3, weights_p4
