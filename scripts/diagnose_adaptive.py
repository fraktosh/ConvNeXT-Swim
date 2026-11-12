import os
import sys
import yaml
import json
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.adaptive_hybrid_detector import AdaptiveHybridDetector
from src.losses.compute_budget_loss import ComputeBudgetLoss

def main():
    os.makedirs('results/figures/router_diagnostics', exist_ok=True)
    
    with open('configs/adaptive_hybrid.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AdaptiveHybridDetector(num_classes=config['model']['num_classes'], pretrained=False).to(device)
    budget_loss_fn = ComputeBudgetLoss(target_budget=config['training']['budget_target']).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['lr'])
    
    diagnostics = {
        "cost_assignment": {"CNN": 0.0, "Light-Swin": 0.5, "Deep-Swin": 1.0},
        "normalization": "None. Absolute cost values used.",
        "target_budget_interpretation": "ReLU(expected_cost - 0.5) independently for P3 and P4.",
        "epochs": []
    }
    
    model.train()
    
    # Mock input
    x = torch.randn(2, 3, 640, 640).to(device)
    
    for epoch in range(12):
        optimizer.zero_grad()
        
        # We simulate the exact logic
        tau = config['training']['tau'] # Always 1.0
        cls_logits, bbox_preds, w_p3, w_p4 = model(x, tau=tau)
        
        # Fake detection targets to get gradients
        # We just want a gradient to flow back. We'll simulate a mock det_loss of ~3.0
        det_loss = cls_logits[0].mean() * 0.0 + 3.0 # Just to have a magnitude, wait, we need gradients!
        det_loss = sum([c.mean() + b.mean() for c, b in zip(cls_logits, bbox_preds)]) * 0.01 + 3.0
        
        # Actual loss
        b_loss_p3 = budget_loss_fn(w_p3)
        b_loss_p4 = budget_loss_fn(w_p4)
        b_loss = b_loss_p3 + b_loss_p4
        
        total_loss = det_loss + config['training']['budget_weight'] * b_loss
        
        # Retain gradients for routing distributions to check magnitudes
        w_p3.retain_grad()
        w_p4.retain_grad()
        
        total_loss.backward()
        
        # Gather metrics
        dist_p3 = w_p3.mean(dim=(0, 2, 3)).detach().cpu().numpy()
        dist_p4 = w_p4.mean(dim=(0, 2, 3)).detach().cpu().numpy()
        
        # Expected cost
        cost_weights = torch.tensor([0.0, 0.5, 1.0], device=device)
        exp_cost_p3 = (dist_p3 * cost_weights.cpu().numpy()).sum()
        exp_cost_p4 = (dist_p4 * cost_weights.cpu().numpy()).sum()
        
        # Entropy
        eps = 1e-8
        ent_p3 = -np.sum(dist_p3 * np.log(dist_p3 + eps))
        ent_p4 = -np.sum(dist_p4 * np.log(dist_p4 + eps))
        
        # Grad magnitudes (we just take the norm of the gradients at the routing logits if possible,
        # or the weights themselves)
        grad_p3_norm = w_p3.grad.norm().item() if w_p3.grad is not None else 0.0
        grad_p4_norm = w_p4.grad.norm().item() if w_p4.grad is not None else 0.0
        
        diagnostics["epochs"].append({
            "epoch": epoch + 1,
            "tau": tau,
            "det_loss_magnitude": det_loss.item(),
            "budget_loss_magnitude": (config['training']['budget_weight'] * b_loss).item(),
            "grad_norm_p3": grad_p3_norm,
            "grad_norm_p4": grad_p4_norm,
            "p3_dist": dist_p3.tolist(),
            "p4_dist": dist_p4.tolist(),
            "p3_expected_cost": float(exp_cost_p3),
            "p4_expected_cost": float(exp_cost_p4),
            "p3_entropy": float(ent_p3),
            "p4_entropy": float(ent_p4)
        })
        
        # Step optimizer
        optimizer.step()
        
    with open('results/adaptive_router_diagnostics.json', 'w') as f:
        json.dump(diagnostics, f, indent=4)
        
    # PLOTTING
    epochs = [d['epoch'] for d in diagnostics['epochs']]
    p3_cnn = [d['p3_dist'][0] for d in diagnostics['epochs']]
    p3_light = [d['p3_dist'][1] for d in diagnostics['epochs']]
    p3_deep = [d['p3_dist'][2] for d in diagnostics['epochs']]
    
    p4_cnn = [d['p4_dist'][0] for d in diagnostics['epochs']]
    p4_light = [d['p4_dist'][1] for d in diagnostics['epochs']]
    p4_deep = [d['p4_dist'][2] for d in diagnostics['epochs']]
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, p3_cnn, label='CNN')
    plt.plot(epochs, p3_light, label='Light-Swin')
    plt.plot(epochs, p3_deep, label='Deep-Swin')
    plt.title('P3 Routing Distribution Over Epochs (Simulated)')
    plt.xlabel('Epoch')
    plt.ylabel('Probability')
    plt.legend()
    plt.savefig('results/figures/router_diagnostics/p3_dist.png')
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, p4_cnn, label='CNN')
    plt.plot(epochs, p4_light, label='Light-Swin')
    plt.plot(epochs, p4_deep, label='Deep-Swin')
    plt.title('P4 Routing Distribution Over Epochs (Simulated)')
    plt.xlabel('Epoch')
    plt.ylabel('Probability')
    plt.legend()
    plt.savefig('results/figures/router_diagnostics/p4_dist.png')
    
    print("Diagnostics complete.")

if __name__ == '__main__':
    main()
