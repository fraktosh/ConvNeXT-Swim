import torch
import torch.nn as nn
from torchvision.ops import boxes as box_ops
from torchvision.models.detection._utils import Matcher, BoxCoder
from torchvision.ops import sigmoid_focal_loss
import math

class RetinaNetLoss(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = Matcher(
            high_threshold=0.5,
            low_threshold=0.4,
            allow_low_quality_matches=True,
        )
        self.box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

    def generate_anchors(self, features):
        strides = [8, 16, 32]
        sizes = [32, 64, 128]
        ratios = [0.5, 1.0, 2.0]
        scales = [1, 2**(1/3), 2**(2/3)]
        
        all_anchors = []
        for i, feat in enumerate(features):
            _, _, h, w = feat.shape
            stride = strides[i]
            
            shift_x = (torch.arange(0, w, dtype=torch.float32, device=feat.device) + 0.5) * stride
            shift_y = (torch.arange(0, h, dtype=torch.float32, device=feat.device) + 0.5) * stride
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=2)
            
            base_anchors = []
            for r in ratios:
                for s in scales:
                    w_a = sizes[i] * s * math.sqrt(r)
                    h_a = sizes[i] * s / math.sqrt(r)
                    base_anchors.append([-w_a/2, -h_a/2, w_a/2, h_a/2])
            base_anchors = torch.tensor(base_anchors, dtype=torch.float32, device=feat.device)
            
            anchors = shifts.view(h, w, 1, 4) + base_anchors.view(1, 1, -1, 4)
            all_anchors.append(anchors.reshape(-1, 4))
            
        return torch.cat(all_anchors, dim=0)

    def forward(self, cls_logits, bbox_preds, targets, image_size=(640, 640)):
        """
        cls_logits: list of (B, 9*80, H, W)
        bbox_preds: list of (B, 9*4, H, W)
        targets: list of dicts with 'boxes' (N, 4) in xyxy and 'labels' (N,)
        """
        anchors = self.generate_anchors(cls_logits) # (N_anchors, 4)
        
        # Flatten predictions
        # For each level: (B, 9*C, H, W) -> (B, H*W*9, C)
        B = cls_logits[0].shape[0]
        cls_flat, bbox_flat = [], []
        for c, b in zip(cls_logits, bbox_preds):
            c = c.view(B, -1, self.num_classes, c.shape[2], c.shape[3]).permute(0, 3, 4, 1, 2).reshape(B, -1, self.num_classes)
            b = b.view(B, -1, 4, b.shape[2], b.shape[3]).permute(0, 3, 4, 1, 2).reshape(B, -1, 4)
            cls_flat.append(c)
            bbox_flat.append(b)
            
        cls_flat = torch.cat(cls_flat, dim=1) # (B, N_anchors, C)
        bbox_flat = torch.cat(bbox_flat, dim=1) # (B, N_anchors, 4)
        
        cls_loss = torch.tensor(0.0, device=cls_flat.device)
        box_loss = torch.tensor(0.0, device=cls_flat.device)
        
        num_foreground = 0
        
        for b_idx in range(B):
            gt_boxes = targets[b_idx]['boxes']
            gt_labels = targets[b_idx]['labels']
            
            if len(gt_boxes) == 0:
                cls_loss += sigmoid_focal_loss(
                    cls_flat[b_idx], 
                    torch.zeros_like(cls_flat[b_idx]),
                    alpha=0.25, gamma=2.0, reduction='sum'
                )
                continue
                
            match_quality_matrix = box_ops.box_iou(gt_boxes, anchors)
            matches = self.matcher(match_quality_matrix)
            
            matched_idxs = matches.clamp(min=0)
            
            # Classification target
            cls_target = torch.zeros_like(cls_flat[b_idx])
            foreground_idxs = matches >= 0
            
            # matched labels (0-indexed in our dataset)
            matched_labels = gt_labels[matched_idxs[foreground_idxs]]
            cls_target[foreground_idxs, matched_labels] = 1.0
            
            # Loss computation
            # Ignored anchors (matches == -1) don't contribute to classification loss
            valid_idxs = matches != -1
            cls_loss += sigmoid_focal_loss(
                cls_flat[b_idx][valid_idxs],
                cls_target[valid_idxs],
                alpha=0.25, gamma=2.0, reduction='sum'
            )
            
            # Regression target
            if foreground_idxs.sum() > 0:
                matched_gt_boxes = gt_boxes[matched_idxs[foreground_idxs]]
                # Encode boxes
                box_target = self.box_coder.encode_single(matched_gt_boxes, anchors[foreground_idxs])
                box_pred = bbox_flat[b_idx][foreground_idxs]
                
                box_loss += torch.nn.functional.smooth_l1_loss(
                    box_pred, box_target, beta=1.0, reduction='sum'
                )
                num_foreground += foreground_idxs.sum().item()
                
        # Normalize
        num_foreground = max(1, num_foreground)
        cls_loss = cls_loss / num_foreground
        box_loss = box_loss / num_foreground
        
        return cls_loss, box_loss
