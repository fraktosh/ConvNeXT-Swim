import os
import sys
import torch
import json
import yaml
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.losses.retinanet_loss import RetinaNetLoss
from src.data.coco import RealCOCODataset
from scripts.train_cnn_baseline import simple_transform, collate_fn
from src.models.cnn_detector import CNNDetector

def run_audit():
    print("="*50)
    print("Phase 0.5: RetinaNet Loss Audit")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = RetinaNetLoss(num_classes=80).to(device)
    
    # 1. Anchor Encoding / Decoding Inverse Test
    print("\n[3] Box Encoding/Decoding Inverse Test")
    anchors = torch.tensor([[100.0, 100.0, 200.0, 200.0]], device=device) # xyxy
    gt_boxes = torch.tensor([[110.0, 105.0, 215.0, 195.0]], device=device) # xyxy
    
    encoded = loss_fn.box_coder.encode_single(gt_boxes, anchors)
    decoded = loss_fn.box_coder.decode_single(encoded, anchors)
    
    max_err = torch.max(torch.abs(decoded - gt_boxes)).item()
    print(f"  Max Absolute Reconstruction Error: {max_err:.8f}")
    if max_err < 1e-4:
        print("  Status: PASS")
    else:
        print("  Status: FAIL")
        
    # 2. IoU Matching & Empty targets
    print("\n[4 & 7] IoU Matching and Empty Target Handling")
    with open('configs/cnn_baseline.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    dataset = RealCOCODataset(
        root=config['data']['train_dir'],
        annFile=config['data']['train_ann'],
        transform=simple_transform
    )
    
    dataloader = DataLoader(Subset(dataset, range(8)), batch_size=2, shuffle=False, collate_fn=collate_fn)
    
    images, targets = next(iter(dataloader))
    
    # Mock features to generate anchors
    mock_features = [
        torch.zeros(2, 256, 80, 80, device=device), # P3 (640/8)
        torch.zeros(2, 256, 40, 40, device=device), # P4 (640/16)
        torch.zeros(2, 256, 20, 20, device=device)  # P5 (640/32)
    ]
    
    anchors = loss_fn.generate_anchors(mock_features)
    print(f"  Total anchors generated: {len(anchors)}")
    
    for i in range(2):
        gt = targets[i]['boxes'].to(device)
        print(f"  Image {i} GT boxes: {len(gt)}")
        if len(gt) > 0:
            from torchvision.ops import boxes as box_ops
            match_quality = box_ops.box_iou(gt, anchors)
            matches = loss_fn.matcher(match_quality)
            pos = (matches >= 0).sum().item()
            neg = (matches == loss_fn.matcher.BELOW_LOW_THRESHOLD).sum().item()
            ign = (matches == loss_fn.matcher.BETWEEN_THRESHOLDS).sum().item()
            print(f"  Positives (>=0.5 or max): {pos}")
            print(f"  Negatives (<0.4): {neg}")
            print(f"  Ignored [0.4, 0.5): {ign}")
            print(f"  Total (pos+neg+ign): {pos+neg+ign} == {len(anchors)}")
            
    # Test empty target
    empty_target = [{'boxes': torch.empty((0, 4), device=device), 'labels': torch.empty((0,), dtype=torch.int64, device=device)}]
    mock_cls = [torch.zeros(1, 9*80, f.shape[2], f.shape[3], device=device) for f in mock_features]
    mock_box = [torch.zeros(1, 9*4, f.shape[2], f.shape[3], device=device) for f in mock_features]
    
    cls_loss, box_loss = loss_fn(mock_cls, mock_box, empty_target)
    print("\n  Empty Target Test:")
    print(f"  Cls Loss (finite): {torch.isfinite(cls_loss).item()} ({cls_loss.item():.4f})")
    print(f"  Box Loss (zero): {box_loss.item() == 0.0} ({box_loss.item():.4f})")
    
    print("\n[8] COCO Category ID Mapping")
    print("  WARNING: Currently simple_transform maps 'category_id' directly to [0, 79] using a naive clamp for the smoke test.")
    print("  PyCOCOtools uses original category IDs (1-90 with gaps). A proper contiguous mapping dictionary is missing and must be added to RealCOCODataset.")

    print("\n[12] Checkpoint Compatibility")
    model = CNNDetector(num_classes=80)
    print("  WARNING: train_cnn_baseline.py currently saves ONLY model.state_dict(). It does NOT save optimizer state, epoch, config, or random seed.")
    
if __name__ == '__main__':
    run_audit()
