import os
import sys
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.adaptive_hybrid_detector import AdaptiveHybridDetector
from src.losses.compute_budget_loss import ComputeBudgetLoss
from src.data.coco import RealCOCODataset
from scripts.train_cnn_baseline import simple_transform, collate_fn

def main():
    with open('configs/adaptive_hybrid.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    torch.manual_seed(config['training']['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Smoke Test on: {device}")
    
    dataset = RealCOCODataset(
        root=config['data']['train_dir'],
        annFile=config['data']['train_ann'],
        transform=simple_transform
    )
    
    dataloader = DataLoader(dataset, batch_size=config['data']['batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=0)
    
    model = AdaptiveHybridDetector(num_classes=config['model']['num_classes'], pretrained=config['model']['pretrained_backbone']).to(device)
    budget_loss_fn = ComputeBudgetLoss(target_budget=config['training']['budget_target']).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['lr'])
    scaler = torch.amp.GradScaler('cuda', enabled=config['training']['mixed_precision'])
    
    model.train()
    
    for i, (images, targets) in enumerate(dataloader):
        if i >= 10:
            break
            
        images = torch.stack(images).to(device)
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda', enabled=config['training']['mixed_precision']):
            cls_logits, bbox_preds, w_p3, w_p4 = model(images, tau=config['training']['tau'])
            det_loss = sum([c.mean() + b.mean() for c, b in zip(cls_logits, bbox_preds)])
            b_loss = budget_loss_fn(w_p3) + budget_loss_fn(w_p4)
            total_loss = det_loss + config['training']['budget_weight'] * b_loss
            
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Calculate distribution over the batch
        dist_p3 = w_p3.mean(dim=(0, 2, 3)).detach().cpu().numpy()
        dist_p4 = w_p4.mean(dim=(0, 2, 3)).detach().cpu().numpy()
        
        print(f"Step {i} | Total Loss: {total_loss.item():.4f} | Budget Loss: {b_loss.item():.4f}")
        print(f"  P3 Routing Dist (CNN, Light, Deep): {dist_p3}")
        print(f"  P4 Routing Dist (CNN, Light, Deep): {dist_p4}")
        
    print("Smoke test complete! Check if distributions are non-degenerate.")

if __name__ == '__main__':
    main()
