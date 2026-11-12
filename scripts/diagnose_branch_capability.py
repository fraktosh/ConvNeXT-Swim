import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.adaptive_hybrid_detector import AdaptiveHybridDetector
from src.data.coco import RealCOCODataset
from scripts.train_cnn_baseline import simple_transform, collate_fn

def main():
    with open('configs/adaptive_hybrid.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*50}\nBranch Capability Diagnostic\n{'='*50}")
    
    dataset = RealCOCODataset(
        root=config['data']['train_dir'],
        annFile=config['data']['train_ann'],
        transform=simple_transform
    )
    
    # Use batch size 4 for stable statistics
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=0)
    
    model = AdaptiveHybridDetector(num_classes=config['model']['num_classes'], pretrained=config['model']['pretrained_backbone']).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['lr'])
    model.train()
    
    images, targets = next(iter(dataloader))
    images = torch.stack(images).to(device)
    
    branches = ['CNN', 'Light-Swin', 'Deep-Swin']
    w_masks = [
        torch.tensor([1.0, 0.0, 0.0], device=device).view(1, 3, 1, 1),
        torch.tensor([0.0, 1.0, 0.0], device=device).view(1, 3, 1, 1),
        torch.tensor([0.0, 0.0, 1.0], device=device).view(1, 3, 1, 1),
    ]
    
    print("\n--- P3 Diagnostics ---")
    for b_idx, b_name in enumerate(branches):
        optimizer.zero_grad()
        
        # We manually intercept the routing.
        # To do this safely, we will mock the _apply_routing function temporarily.
        original_apply_routing = model._apply_routing
        
        def mock_apply_p3(feat, route_weights, light_branch, deep_branch):
            # Force the mask to strictly one branch across all regions
            forced_weights = w_masks[b_idx].expand_as(route_weights)
            
            feat_light = light_branch(feat)
            feat_deep = deep_branch(feat)
            
            # Retain grad on outputs to see gradient magnitude flowing through
            feat.retain_grad()
            feat_light.retain_grad()
            feat_deep.retain_grad()
            
            w0 = forced_weights[:, 0:1, :, :]
            w1 = forced_weights[:, 1:2, :, :]
            w2 = forced_weights[:, 2:3, :, :]
            
            out = (w0 * feat) + (w1 * feat_light) + (w2 * feat_deep)
            
            # Record stats
            global branch_stats
            branch_stats = {
                'feat_mean': feat.mean().item(), 'feat_std': feat.std().item(),
                'light_mean': feat_light.mean().item(), 'light_std': feat_light.std().item(),
                'deep_mean': feat_deep.mean().item(), 'deep_std': feat_deep.std().item(),
                'feat_tensor': feat, 'light_tensor': feat_light, 'deep_tensor': feat_deep
            }
            return out
            
        model._apply_routing = mock_apply_p3
        
        # In this test, we only override P3. We let P4 be whatever so it's a fair comparison just on P3.
        cls_logits, bbox_preds, _, _ = model(images)
        det_loss = sum([c.mean() + b.mean() for c, b in zip(cls_logits, bbox_preds)])
        
        det_loss.backward()
        
        grad_norm = 0.0
        if b_idx == 0:
            grad_norm = branch_stats['feat_tensor'].grad.norm().item()
        elif b_idx == 1:
            grad_norm = branch_stats['light_tensor'].grad.norm().item()
        elif b_idx == 2:
            grad_norm = branch_stats['deep_tensor'].grad.norm().item()
            
        print(f"\n{b_name} Forced at P3:")
        print(f"  Detection Loss: {det_loss.item():.4f}")
        print(f"  Gradient Norm (on branch output): {grad_norm:.4f}")
        if b_idx == 0:
            print(f"  Output Stats: Mean={branch_stats['feat_mean']:.4f}, Std={branch_stats['feat_std']:.4f}")
        elif b_idx == 1:
            print(f"  Output Stats: Mean={branch_stats['light_mean']:.4f}, Std={branch_stats['light_std']:.4f}")
        else:
            print(f"  Output Stats: Mean={branch_stats['deep_mean']:.4f}, Std={branch_stats['deep_std']:.4f}")
            
        # Restore
        model._apply_routing = original_apply_routing

if __name__ == '__main__':
    main()
