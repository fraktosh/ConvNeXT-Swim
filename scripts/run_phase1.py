import sys
import os
import time
import json
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.backbone.convnext_features import ConvNeXtTinyFeatures

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_phase1():
    config = load_config('configs/phase1.yaml')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = ConvNeXtTinyFeatures(pretrained=config['model']['pretrained'])
    model.to(device)
    model.eval()
    
    # 1. Feature Shapes
    dummy_input = torch.randn(
        config['input']['batch_size'],
        config['input']['channels'],
        config['input']['height'],
        config['input']['width']
    ).to(device)
    
    with torch.no_grad():
        features = model(dummy_input)
        
    shapes_info = []
    p3, p4, p5 = None, None, None
    for i, f in enumerate(features):
        b, c, h, w = f.shape
        ratio = config['input']['height'] // h
        
        level_info = {
            "level_index": i,
            "tensor_shape": [b, c, h, w],
            "channels": c,
            "spatial_resolution": [h, w],
            "downsampling_ratio": ratio
        }
        shapes_info.append(level_info)
        
        if ratio == 8: p3 = level_info
        if ratio == 16: p4 = level_info
        if ratio == 32: p5 = level_info

    shapes_output = {
        "all_levels": shapes_info,
        "identified_levels": {
            "P3 (1/8)": p3,
            "P4 (1/16)": p4,
            "P5 (1/32)": p5
        }
    }
    
    with open('results/phase1_feature_shapes.json', 'w') as f:
        json.dump(shapes_output, f, indent=4)
        
    # 2. Profiling
    num_params = sum(p.numel() for p in model.parameters())
    
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    
    # warmup
    for _ in range(config['profiling']['warmup_iters']):
        _ = model(dummy_input)
    torch.cuda.synchronize()
        
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(config['profiling']['measure_iters']):
        _ = model(dummy_input)
    end_event.record()
    torch.cuda.synchronize()
    
    total_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = total_time_ms / config['profiling']['measure_iters']
    throughput = 1000.0 / avg_time_ms * config['input']['batch_size']
    max_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    
    report = f"""# Phase 1 Profiling Report
## Model: ConvNeXt-Tiny (Features Only)

- Parameter Count: {num_params:,}
- GPU Memory Allocated: {max_memory_mb:.2f} MB
- Average Inference Time (ms/batch): {avg_time_ms:.2f} ms
- Throughput (images/sec): {throughput:.2f}
"""
    with open('results/phase1_report.md', 'w') as f:
        f.write(report)
        
    # 3. Visualization
    # take a dummy image, make it look somewhat like an image for vis
    img_np = dummy_input[0].cpu().numpy().transpose(1, 2, 0)
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img_np)
    axes[0].set_title('Input Image (Dummy)')
    axes[0].axis('off')
    
    target_ratios = [8, 16, 32]
    
    plot_idx = 1
    for i, f in enumerate(features):
        ratio = config['input']['height'] // f.shape[2]
        if ratio in target_ratios:
            # summarize activation by taking mean over channels
            act = f[0].mean(dim=0).cpu().numpy()
            axes[plot_idx].imshow(act, cmap='viridis')
            axes[plot_idx].set_title(f'Level (1/{ratio}) Activation')
            axes[plot_idx].axis('off')
            plot_idx += 1
            if plot_idx > 3: break
            
    plt.tight_layout()
    plt.savefig('results/phase1_feature_visualization.png')
    print("Phase 1 run complete. Check results/ folder.")

if __name__ == '__main__':
    run_phase1()
