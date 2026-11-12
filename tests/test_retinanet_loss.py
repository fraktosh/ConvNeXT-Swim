import os
import sys
import unittest
import torch
from torchvision.ops import boxes as box_ops

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.losses.retinanet_loss import RetinaNetLoss
from src.models.detection_head import SimpleDetectionHead

class TestRetinaNetLoss(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss_fn = RetinaNetLoss(num_classes=80).to(self.device)
        self.head = SimpleDetectionHead(in_channels_list=[192, 384, 768], num_classes=80).to(self.device)

    def test_anchor_generation_shapes_and_counts(self):
        # 640x640 inputs give:
        # P3: (B, 192, 80, 80)
        # P4: (B, 384, 40, 40)
        # P5: (B, 768, 20, 20)
        features = [
            torch.zeros(1, 192, 80, 80, device=self.device),
            torch.zeros(1, 384, 40, 40, device=self.device),
            torch.zeros(1, 768, 20, 20, device=self.device)
        ]
        cls_logits, bbox_preds = self.head(features)
        
        # Verify logits shape
        self.assertEqual(cls_logits[0].shape, (1, 9 * 80, 80, 80))
        self.assertEqual(cls_logits[1].shape, (1, 9 * 80, 40, 40))
        self.assertEqual(cls_logits[2].shape, (1, 9 * 80, 20, 20))
        
        # Verify bbox preds shape
        self.assertEqual(bbox_preds[0].shape, (1, 9 * 4, 80, 80))
        self.assertEqual(bbox_preds[1].shape, (1, 9 * 4, 40, 40))
        self.assertEqual(bbox_preds[2].shape, (1, 9 * 4, 20, 20))
        
        anchors = self.loss_fn.generate_anchors(cls_logits)
        # Total anchors = 80*80*9 + 40*40*9 + 20*20*9 = 57600 + 14400 + 3600 = 75600
        self.assertEqual(anchors.shape, (75600, 4))
        # Anchors must be valid xyxy (x2 > x1, y2 > y1)
        self.assertTrue(torch.all(anchors[:, 2] > anchors[:, 0]))
        self.assertTrue(torch.all(anchors[:, 3] > anchors[:, 1]))

    def test_box_encoding_decoding_roundtrip(self):
        anchors = torch.tensor([[100.0, 100.0, 200.0, 200.0]], device=self.device)
        gt_boxes = torch.tensor([[110.0, 105.0, 215.0, 195.0]], device=self.device)
        encoded = self.loss_fn.box_coder.encode_single(gt_boxes, anchors)
        decoded = self.loss_fn.box_coder.decode_single(encoded, anchors)
        max_err = torch.max(torch.abs(decoded - gt_boxes)).item()
        self.assertLess(max_err, 1e-4)

    def test_loss_with_positive_targets(self):
        features = [
            torch.randn(2, 192, 80, 80, device=self.device),
            torch.randn(2, 384, 40, 40, device=self.device),
            torch.randn(2, 768, 20, 20, device=self.device)
        ]
        cls_logits, bbox_preds = self.head(features)
        
        targets = [
            {
                'boxes': torch.tensor([[100.0, 100.0, 250.0, 250.0], [50.0, 50.0, 120.0, 120.0]], device=self.device),
                'labels': torch.tensor([1, 79], dtype=torch.int64, device=self.device)
            },
            {
                'boxes': torch.tensor([[300.0, 300.0, 450.0, 450.0]], device=self.device),
                'labels': torch.tensor([50], dtype=torch.int64, device=self.device)
            }
        ]
        
        cls_loss, box_loss = self.loss_fn(cls_logits, bbox_preds, targets)
        
        self.assertTrue(torch.isfinite(cls_loss).item())
        self.assertTrue(torch.isfinite(box_loss).item())
        self.assertGreater(cls_loss.item(), 0.0)
        self.assertGreater(box_loss.item(), 0.0)
        
        # Test backward pass
        total_loss = cls_loss + box_loss
        total_loss.backward()
        
        # Verify gradients exist and are finite
        for p in self.head.parameters():
            if p.grad is not None:
                self.assertTrue(torch.all(torch.isfinite(p.grad)))

    def test_empty_targets(self):
        features = [
            torch.zeros(1, 192, 80, 80, device=self.device),
            torch.zeros(1, 384, 40, 40, device=self.device),
            torch.zeros(1, 768, 20, 20, device=self.device)
        ]
        cls_logits, bbox_preds = self.head(features)
        
        empty_targets = [{
            'boxes': torch.empty((0, 4), device=self.device),
            'labels': torch.empty((0,), dtype=torch.int64, device=self.device)
        }]
        
        cls_loss, box_loss = self.loss_fn(cls_logits, bbox_preds, empty_targets)
        self.assertTrue(torch.isfinite(cls_loss).item())
        self.assertEqual(box_loss.item(), 0.0)

if __name__ == '__main__':
    unittest.main()
