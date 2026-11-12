import os
import sys
import yaml
import json
import time
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.adaptive_hybrid_detector import AdaptiveHybridDetector
from src.losses.compute_budget_loss import ComputeBudgetLoss
from src.losses.retinanet_loss import RetinaNetLoss
from src.data.coco import RealCOCODataset, coco_train_transform, collate_fn

def main():
    parser = argparse.ArgumentParser(description="Train Adaptive CNN-Swin Detector")
    parser.add_argument('--config', type=str, default='configs/adaptive_hybrid.yaml', help='Path to config file')
    parser.add_argument('--max_batches', type=int, default=None, help='Max batches per epoch for diagnostic/debug runs')
    parser.add_argument('--epochs', type=int, default=None, help='Override epochs from config')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    epochs = args.epochs if args.epochs is not None else config['training']['epochs']
    torch.manual_seed(config['training']['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Training: {config['model']['name']}")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Epochs: {epochs}, Max batches per epoch: {args.max_batches if args.max_batches else 'Full dataset'}")
    print(f"Budget target: {config['training']['budget_target']}, Budget weight: {config['training']['budget_weight']}")
    print(f"{'='*60}\n")
    
    dataset = RealCOCODataset(
        root=config['data']['train_dir'],
        annFile=config['data']['train_ann'],
        transform=coco_train_transform
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=config['data']['batch_size'], 
        shuffle=True, 
        collate_fn=collate_fn, 
        num_workers=2, 
        pin_memory=True
    )
    
    model = AdaptiveHybridDetector(
        num_classes=config['model']['num_classes'], 
        pretrained=config['model']['pretrained_backbone']
    ).to(device)
    
    det_loss_fn = RetinaNetLoss(num_classes=config['model']['num_classes']).to(device)
    budget_loss_fn = ComputeBudgetLoss(target_budget=config['training']['budget_target']).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=config['training']['mixed_precision'])
    
    os.makedirs(config['training']['output_dir'], exist_ok=True)
    history = []
    start_train_time = time.time()
    
    final_p3_dist = None
    final_p4_dist = None
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_det = 0.0
        epoch_budget = 0.0
        steps_run = 0
        
        for batch_idx, (images, targets) in enumerate(dataloader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
                
            images = torch.stack(images).to(device)
            targets = [{k: (v.to(device) if hasattr(v, 'to') else v) for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=config['training']['mixed_precision']):
                cls_logits, bbox_preds, w_p3, w_p4 = model(images, tau=config['training']['tau'])
                cls_loss, box_loss = det_loss_fn(cls_logits, bbox_preds, targets)
                det_loss = cls_loss + box_loss
                
                b_loss = budget_loss_fn(w_p3) + budget_loss_fn(w_p4)
                total_loss = det_loss + config['training']['budget_weight'] * b_loss
                
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0).item()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += total_loss.item()
            epoch_det += det_loss.item()
            epoch_budget += b_loss.item()
            steps_run += 1
            
            final_p3_dist = w_p3.mean(dim=(0, 2, 3)).detach().cpu().numpy().tolist()
            final_p4_dist = w_p4.mean(dim=(0, 2, 3)).detach().cpu().numpy().tolist()
            
            if (batch_idx + 1) % 10 == 0 or (args.max_batches is not None and (batch_idx + 1) == args.max_batches):
                lr = optimizer.param_groups[0]['lr']
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(dataloader) if args.max_batches is None else args.max_batches}] | Total: {total_loss.item():.4f} (Det: {det_loss.item():.4f}, Budget: {b_loss.item():.4f}) | GradNorm: {grad_norm:.2f} | LR: {lr:.2e}", flush=True)
                
        scheduler.step()
        avg_loss = epoch_loss / max(1, steps_run)
        avg_det = epoch_det / max(1, steps_run)
        avg_budget = epoch_budget / max(1, steps_run)
        
        history.append({
            'epoch': epoch + 1,
            'total_loss': avg_loss,
            'det_loss': avg_det,
            'budget_loss': avg_budget,
            'p3_dist': final_p3_dist,
            'p4_dist': final_p4_dist,
            'steps': steps_run
        })
        print(f"--> Epoch {epoch+1} Summary: Avg Total = {avg_loss:.4f} (Det: {avg_det:.4f}, Budget: {avg_budget:.4f}) | P3 Dist: {[round(x, 3) for x in final_p3_dist]} | P4 Dist: {[round(x, 3) for x in final_p4_dist]}\n", flush=True)
        
    total_time = time.time() - start_train_time
    max_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if torch.cuda.is_available() else 0.0
    params = sum(p.numel() for p in model.parameters())
    
    checkpoint = {
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': config,
        'seed': config['training']['seed'],
        'history': history,
        'total_time_s': round(total_time, 2),
        'max_memory_mb': round(max_memory, 2),
        'params': params,
        'final_p3_dist': final_p3_dist,
        'final_p4_dist': final_p4_dist
    }
    ckpt_path = os.path.join(config['training']['output_dir'], 'final_model.pth')
    torch.save(checkpoint, ckpt_path)
    print(f"\nTraining Complete! Saved validated checkpoint to {ckpt_path}")
    
    metrics = {
        "Model": config['model']['name'],
        "Params": params,
        "GPU_memory_mb": round(max_memory, 2),
        "total_time_s": round(total_time, 2),
        "P3_dist": final_p3_dist,
        "P4_dist": final_p4_dist,
        "history": history
    }
    with open('results/adaptive_training_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == '__main__':
    main()
