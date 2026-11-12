import os
import sys
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.adaptive_hybrid_detector import AdaptiveHybridDetector
from src.losses.compute_budget_loss import ComputeBudgetLoss
from src.data.coco import RealCOCODataset
from scripts.train_cnn_baseline import simple_transform, collate_fn

def run_experiment(exp_name, tau_start, tau_end, num_batches=10):
    with open('configs/adaptive_hybrid.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    torch.manual_seed(config['training']['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*50}\nExperiment {exp_name}: Tau Start={tau_start}, Tau End={tau_end}\n{'='*50}")
    
    dataset = RealCOCODataset(
        root=config['data']['train_dir'],
        annFile=config['data']['train_ann'],
        transform=simple_transform
    )
    
    dataloader = DataLoader(dataset, batch_size=config['data']['batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=0)
    
    model = AdaptiveHybridDetector(num_classes=config['model']['num_classes'], pretrained=config['model']['pretrained_backbone']).to(device)
    budget_loss_fn = ComputeBudgetLoss(target_budget=config['training']['budget_target']).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['lr'])
    
    model.train()
    cost_weights = torch.tensor([0.0, 0.5, 1.0], device=device)
    
    for i, (images, targets) in enumerate(dataloader):
        if i >= num_batches:
            break
            
        images = torch.stack(images).to(device)
        optimizer.zero_grad()
        
        # Calculate tau
        progress = i / (num_batches - 1) if num_batches > 1 else 1.0
        current_tau = tau_start + progress * (tau_end - tau_start)
        
        # Forward pass
        cls_logits, bbox_preds, w_p3, w_p4 = model(images, tau=current_tau)
        
        det_loss = sum([c.mean() + b.mean() for c, b in zip(cls_logits, bbox_preds)])
        
        b_loss_p3 = budget_loss_fn(w_p3)
        b_loss_p4 = budget_loss_fn(w_p4)
        b_loss = b_loss_p3 + b_loss_p4
        
        total_loss = det_loss + config['training']['budget_weight'] * b_loss
        
        w_p3.retain_grad()
        w_p4.retain_grad()
        
        total_loss.backward()
        
        dist_p3 = w_p3.mean(dim=(0, 2, 3)).detach().cpu().numpy()
        dist_p4 = w_p4.mean(dim=(0, 2, 3)).detach().cpu().numpy()
        
        eps = 1e-8
        ent_p3 = -np.sum(dist_p3 * np.log(dist_p3 + eps))
        ent_p4 = -np.sum(dist_p4 * np.log(dist_p4 + eps))
        
        exp_cost_p3 = (dist_p3 * cost_weights.cpu().numpy()).sum()
        exp_cost_p4 = (dist_p4 * cost_weights.cpu().numpy()).sum()
        
        grad_p3_norm = w_p3.grad.norm().item() if w_p3.grad is not None else 0.0
        grad_p4_norm = w_p4.grad.norm().item() if w_p4.grad is not None else 0.0
        
        if i == num_batches - 1:
            print(f"--- Final Step {i+1} [Tau = {current_tau:.4f}] ---")
            print(f"Detection Loss: {det_loss.item():.4f}")
            print(f"Budget Loss: {b_loss.item():.4f} (Weighted: {(config['training']['budget_weight'] * b_loss).item():.4f})")
            print(f"P3: Dist [CNN, Light, Deep] = {dist_p3}")
            print(f"    Expected Cost = {exp_cost_p3:.4f}, Entropy = {ent_p3:.4f}, Grad Norm = {grad_p3_norm:.6f}")
            print(f"P4: Dist [CNN, Light, Deep] = {dist_p4}")
            print(f"    Expected Cost = {exp_cost_p4:.4f}, Entropy = {ent_p4:.4f}, Grad Norm = {grad_p4_norm:.6f}")
            
        optimizer.step()

def main():
    run_experiment("A (Fixed Tau=1.0)", tau_start=1.0, tau_end=1.0)
    run_experiment("B (Fixed Tau=2.0)", tau_start=2.0, tau_end=2.0)
    run_experiment("C (Annealed Tau 2.0 -> 0.5)", tau_start=2.0, tau_end=0.5)

if __name__ == '__main__':
    main()
