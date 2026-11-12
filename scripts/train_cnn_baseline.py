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
from src.models.cnn_detector import CNNDetector
from src.data.coco import RealCOCODataset, coco_train_transform, collate_fn
from src.losses.retinanet_loss import RetinaNetLoss

def main():
    parser = argparse.ArgumentParser(description="Train CNN-only Baseline Detector")
    parser.add_argument('--config', type=str, default='configs/cnn_baseline.yaml', help='Path to config file')
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
    
    model = CNNDetector(num_classes=config['model']['num_classes'], pretrained=config['model']['pretrained_backbone']).to(device)
    loss_fn = RetinaNetLoss(num_classes=config['model']['num_classes']).to(device)
    
    if config['training']['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=1e-4)
    else:
        optimizer = optim.SGD(model.parameters(), lr=config['training']['lr'], momentum=0.9, weight_decay=1e-4)
        
    if config['training']['scheduler'] == 'CosineAnnealing':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        
    scaler = torch.amp.GradScaler('cuda', enabled=config['training']['mixed_precision'])
    
    os.makedirs(config['training']['output_dir'], exist_ok=True)
    history = []
    start_train_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_cls = 0.0
        epoch_box = 0.0
        steps_run = 0
        
        for batch_idx, (images, targets) in enumerate(dataloader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
                
            images = torch.stack(images).to(device)
            targets = [{k: (v.to(device) if hasattr(v, 'to') else v) for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=config['training']['mixed_precision']):
                cls_logits, bbox_preds = model(images)
                cls_loss, box_loss = loss_fn(cls_logits, bbox_preds, targets)
                loss = cls_loss + box_loss
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            
            # Gradient norm tracking and clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0).item()
            
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            epoch_cls += cls_loss.item()
            epoch_box += box_loss.item()
            steps_run += 1
            
            if (batch_idx + 1) % 10 == 0 or (args.max_batches is not None and (batch_idx + 1) == args.max_batches):
                lr = optimizer.param_groups[0]['lr']
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(dataloader) if args.max_batches is None else args.max_batches}] | Loss: {loss.item():.4f} (Cls: {cls_loss.item():.4f}, Box: {box_loss.item():.4f}) | GradNorm: {grad_norm:.2f} | LR: {lr:.2e}", flush=True)
                
        scheduler.step()
        avg_loss = epoch_loss / max(1, steps_run)
        avg_cls = epoch_cls / max(1, steps_run)
        avg_box = epoch_box / max(1, steps_run)
        
        history.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'cls_loss': avg_cls,
            'box_loss': avg_box,
            'steps': steps_run
        })
        print(f"--> Epoch {epoch+1} Summary: Avg Loss = {avg_loss:.4f} (Cls: {avg_cls:.4f}, Box: {avg_box:.4f})\n", flush=True)
        
    total_time = time.time() - start_train_time
    max_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if torch.cuda.is_available() else 0.0
    params = sum(p.numel() for p in model.parameters())
    
    # Save checkpoint with complete metadata
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
        'params': params
    }
    ckpt_path = os.path.join(config['training']['output_dir'], 'final_model.pth')
    torch.save(checkpoint, ckpt_path)
    print(f"\nTraining Complete! Saved validated checkpoint to {ckpt_path}")
    
    # Save training metrics log
    metrics = {
        "Model": config['model']['name'],
        "Params": params,
        "GPU_memory_mb": round(max_memory, 2),
        "total_time_s": round(total_time, 2),
        "history": history
    }
    with open('results/cnn_baseline_training_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == '__main__':
    main()
