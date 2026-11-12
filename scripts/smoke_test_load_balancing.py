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
from src.losses.compute_budget_loss import ComputeBudgetLoss
from src.data.coco import RealCOCODataset
from scripts.train_cnn_baseline import simple_transform, collate_fn

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_params_module(module):
    if module is None: return 0
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def main():
    if len(sys.argv) < 2:
        print("Usage: python smoke_test_load_balancing.py <lambda_load>")
        sys.exit(1)
    lambda_load = float(sys.argv[1])
    
    with open('configs/adaptive_hybrid.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    torch.manual_seed(config['training']['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*50}\nPhase 4B: Aggregate Load Balancing Diagnostic\nLambda Load = {lambda_load}\n{'='*50}")
    
    model = AdaptiveHybridDetector(num_classes=config['model']['num_classes'], pretrained=config['model']['pretrained_backbone']).to(device)
    
    total_params = count_parameters(model)
    cnn_path_params = count_params_module(model.proj_cnn_p3)
    light_path_params = count_params_module(model.swin_light_p3) + count_params_module(model.proj_light_p3)
    deep_path_params = count_params_module(model.swin_deep_p3) + count_params_module(model.proj_deep_p3)
    
    dataset = RealCOCODataset(
        root=config['data']['train_dir'],
        annFile=config['data']['train_ann'],
        transform=simple_transform
    )
    
    dataloader = DataLoader(dataset, batch_size=config['data']['batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=0)
    budget_loss_fn = ComputeBudgetLoss(target_budget=0.5).to(device)
    
    budget_weight = 1.0

    
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['lr'])
    model.train()
    
    cost_weights = torch.tensor([0.0, 0.5, 1.0], device=device)
    num_batches = 10
    
    for i, (images, targets) in enumerate(dataloader):
        if i >= num_batches: break
            
        images = torch.stack(images).to(device)
        
        cls_logits, bbox_preds, w_p3, w_p4 = model(images, tau=1.0)
        
        det_loss = sum([c.mean() + b.mean() for c, b in zip(cls_logits, bbox_preds)])
        
        b_loss_p3 = budget_loss_fn(w_p3)
        b_loss_p4 = budget_loss_fn(w_p4)
        b_loss = b_loss_p3 + b_loss_p4
        weighted_b_loss = budget_weight * b_loss
        
        # Load balancing penalty (sum p log p -> negative entropy -> minimized at uniform distribution)
        p_bar_p3 = w_p3.mean(dim=(0, 2, 3))
        p_bar_p4 = w_p4.mean(dim=(0, 2, 3))
        
        load_loss_p3 = (p_bar_p3 * torch.log(p_bar_p3 + 1e-8)).sum()
        load_loss_p4 = (p_bar_p4 * torch.log(p_bar_p4 + 1e-8)).sum()
        load_loss = load_loss_p3 + load_loss_p4
        weighted_load_loss = lambda_load * load_loss
        
        # Compute grad norms
        optimizer.zero_grad()
        det_loss.backward(retain_graph=True)
        det_grad_p3 = model.allocator_p3.router[-1].weight.grad.norm().item() if model.allocator_p3.router[-1].weight.grad is not None else 0.0
        det_grad_p4 = model.allocator_p4.router[-1].weight.grad.norm().item() if model.allocator_p4.router[-1].weight.grad is not None else 0.0
        
        optimizer.zero_grad()
        weighted_b_loss.backward(retain_graph=True)
        b_grad_p3 = model.allocator_p3.router[-1].weight.grad.norm().item() if model.allocator_p3.router[-1].weight.grad is not None else 0.0
        b_grad_p4 = model.allocator_p4.router[-1].weight.grad.norm().item() if model.allocator_p4.router[-1].weight.grad is not None else 0.0
        
        optimizer.zero_grad()
        weighted_load_loss.backward(retain_graph=True)
        l_grad_p3 = model.allocator_p3.router[-1].weight.grad.norm().item() if model.allocator_p3.router[-1].weight.grad is not None else 0.0
        l_grad_p4 = model.allocator_p4.router[-1].weight.grad.norm().item() if model.allocator_p4.router[-1].weight.grad is not None else 0.0
        
        optimizer.zero_grad()
        total_loss = det_loss + weighted_b_loss + weighted_load_loss
        total_loss.backward()
        optimizer.step()
        
        if i == num_batches - 1:
            dist_p3 = p_bar_p3.detach().cpu().numpy()
            dist_p4 = p_bar_p4.detach().cpu().numpy()
            
            ent_p3 = -np.sum(dist_p3 * np.log(dist_p3 + 1e-8))
            ent_p4 = -np.sum(dist_p4 * np.log(dist_p4 + 1e-8))
            
            exp_cost_p3 = (dist_p3 * cost_weights.cpu().numpy()).sum()
            exp_cost_p4 = (dist_p4 * cost_weights.cpu().numpy()).sum()
            
            eps = 1e-8
            
            print(f"\n--- Final Step {i+1} Diagnostics ---")
            print(f"Detection Loss: {det_loss.item():.4f}")
            print(f"Unweighted Budget MSE: {b_loss.item():.6f}")
            print(f"Weighted Budget Loss: {weighted_b_loss.item():.6f}")
            print(f"Load Balancing Loss (Sum p log p): {load_loss.item():.6f}")
            print(f"Weighted Load Balancing Loss: {weighted_load_loss.item():.6f}")
            
            print("\n[P3]")
            print(f"  Dist [CNN, Light, Deep]: [{dist_p3[0]:.4f}, {dist_p3[1]:.4f}, {dist_p3[2]:.4f}]")
            print(f"  Aggregate Entropy: {ent_p3:.4f}, Expected Cost: {exp_cost_p3:.4f}")
            print(f"  Det Grad: {det_grad_p3:.6f}, Budget Grad: {b_grad_p3:.6f}, Load Grad: {l_grad_p3:.6f}")
            print(f"  Grad Ratio (Budget/Det): {b_grad_p3 / (det_grad_p3 + eps):.6f}")
            print(f"  Grad Ratio (Load/Det): {l_grad_p3 / (det_grad_p3 + eps):.6f}")
            print(f"  Starvation Check: CNN={dist_p3[0]<0.01}, Light={dist_p3[1]<0.01}, Deep={dist_p3[2]<0.01}")
            
            print("\n[P4]")
            print(f"  Dist [CNN, Light, Deep]: [{dist_p4[0]:.4f}, {dist_p4[1]:.4f}, {dist_p4[2]:.4f}]")
            print(f"  Aggregate Entropy: {ent_p4:.4f}, Expected Cost: {exp_cost_p4:.4f}")
            print(f"  Det Grad: {det_grad_p4:.6f}, Budget Grad: {b_grad_p4:.6f}, Load Grad: {l_grad_p4:.6f}")
            print(f"  Grad Ratio (Budget/Det): {b_grad_p4 / (det_grad_p4 + eps):.6f}")
            print(f"  Grad Ratio (Load/Det): {l_grad_p4 / (det_grad_p4 + eps):.6f}")
            print(f"  Starvation Check: CNN={dist_p4[0]<0.01}, Light={dist_p4[1]<0.01}, Deep={dist_p4[2]<0.01}")

if __name__ == '__main__':
    main()
