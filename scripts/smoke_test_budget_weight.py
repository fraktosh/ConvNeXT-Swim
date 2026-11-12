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

def main():
    if len(sys.argv) < 2:
        print("Usage: python smoke_test_budget_weight.py <weight>")
        sys.exit(1)
        
    budget_weight = float(sys.argv[1])
    
    with open('configs/adaptive_hybrid.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    torch.manual_seed(config['training']['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*50}\nBudget Weight Ablation: Weight = {budget_weight}\n{'='*50}")
    
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
    
    num_batches = 10
    
    for i, (images, targets) in enumerate(dataloader):
        if i >= num_batches:
            break
            
        images = torch.stack(images).to(device)
        
        # Forward pass
        cls_logits, bbox_preds, w_p3, w_p4 = model(images, tau=1.0)
        
        det_loss = sum([c.mean() + b.mean() for c, b in zip(cls_logits, bbox_preds)])
        
        b_loss_p3 = budget_loss_fn(w_p3)
        b_loss_p4 = budget_loss_fn(w_p4)
        b_loss = b_loss_p3 + b_loss_p4
        
        weighted_b_loss = budget_weight * b_loss
        
        # Calculate isolated gradients
        optimizer.zero_grad()
        det_loss.backward(retain_graph=True)
        det_grad_p3 = model.allocator_p3.router[-1].weight.grad.norm().item() if model.allocator_p3.router[-1].weight.grad is not None else 0.0
        det_grad_p4 = model.allocator_p4.router[-1].weight.grad.norm().item() if model.allocator_p4.router[-1].weight.grad is not None else 0.0
        
        optimizer.zero_grad()
        weighted_b_loss.backward(retain_graph=True)
        b_grad_p3 = model.allocator_p3.router[-1].weight.grad.norm().item() if model.allocator_p3.router[-1].weight.grad is not None else 0.0
        b_grad_p4 = model.allocator_p4.router[-1].weight.grad.norm().item() if model.allocator_p4.router[-1].weight.grad is not None else 0.0
        
        # Total update
        optimizer.zero_grad()
        total_loss = det_loss + weighted_b_loss
        total_loss.backward()
        tot_grad_p3 = model.allocator_p3.router[-1].weight.grad.norm().item() if model.allocator_p3.router[-1].weight.grad is not None else 0.0
        tot_grad_p4 = model.allocator_p4.router[-1].weight.grad.norm().item() if model.allocator_p4.router[-1].weight.grad is not None else 0.0
        
        dist_p3 = w_p3.mean(dim=(0, 2, 3)).detach().cpu().numpy()
        dist_p4 = w_p4.mean(dim=(0, 2, 3)).detach().cpu().numpy()
        
        eps = 1e-8
        ent_p3 = -np.sum(dist_p3 * np.log(dist_p3 + eps))
        ent_p4 = -np.sum(dist_p4 * np.log(dist_p4 + eps))
        
        exp_cost_p3 = (dist_p3 * cost_weights.cpu().numpy()).sum()
        exp_cost_p4 = (dist_p4 * cost_weights.cpu().numpy()).sum()
        
        if i == num_batches - 1:
            print(f"--- Final Step {i+1} ---")
            print(f"Detection Loss: {det_loss.item():.4f}")
            print(f"Budget Loss (Unweighted): {b_loss.item():.4f}")
            print(f"Weighted Budget Loss: {weighted_b_loss.item():.4f}")
            print(f"P3:")
            print(f"  Distribution [CNN, Light, Deep]: [{dist_p3[0]:.4f}, {dist_p3[1]:.4f}, {dist_p3[2]:.4f}]")
            print(f"  Entropy: {ent_p3:.4f}")
            print(f"  Expected Cost: {exp_cost_p3:.4f}")
            print(f"  Det Grad Norm: {det_grad_p3:.6f}")
            print(f"  Budget Grad Norm: {b_grad_p3:.6f}")
            print(f"  Total Grad Norm: {tot_grad_p3:.6f}")
            print(f"  Grad Ratio (Budget/Det): {b_grad_p3 / (det_grad_p3 + 1e-8):.6f}")
            print(f"P4:")
            print(f"  Distribution [CNN, Light, Deep]: [{dist_p4[0]:.4f}, {dist_p4[1]:.4f}, {dist_p4[2]:.4f}]")
            print(f"  Entropy: {ent_p4:.4f}")
            print(f"  Expected Cost: {exp_cost_p4:.4f}")
            print(f"  Det Grad Norm: {det_grad_p4:.6f}")
            print(f"  Budget Grad Norm: {b_grad_p4:.6f}")
            print(f"  Total Grad Norm: {tot_grad_p4:.6f}")
            print(f"  Grad Ratio (Budget/Det): {b_grad_p4 / (det_grad_p4 + 1e-8):.6f}")
            
        optimizer.step()

if __name__ == '__main__':
    main()
