import os
import sys
import yaml
import json
import torch
import time
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.cnn_detector import CNNDetector
from src.data.coco import RealCOCODataset
from scripts.train_cnn_baseline import simple_transform, collate_fn

def profile():
    with open('configs/cnn_baseline.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Profiling Inference on: {device}")
    
    # Validation dataset
    dataset = RealCOCODataset(
        root=config['data']['val_dir'],
        annFile=config['data']['val_ann'],
        transform=simple_transform
    )
    
    # Inference batch size is typically 1
    batch_size = 1
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Load Model
    model = CNNDetector(num_classes=config['model']['num_classes'], pretrained=False).to(device)
    checkpoint_path = os.path.join(config['training']['output_dir'], 'final_model.pth')
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    
    warmup_iters = 50
    measure_iters = 200
    
    print(f"Starting warmup ({warmup_iters} iterations)...")
    # Reset peak memory stats before measurement
    torch.cuda.reset_peak_memory_stats(device)
    
    iterator = iter(dataloader)
    
    # Warmup
    for _ in range(warmup_iters):
        images, _ = next(iterator)
        images = torch.stack(images).to(device)
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=config['training']['mixed_precision']):
                _ = model(images)
                
    torch.cuda.synchronize()
    
    print(f"Measuring model forward-pass latency ({measure_iters} iterations)...")
    forward_times = []
    
    for _ in range(measure_iters):
        try:
            images, _ = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            images, _ = next(iterator)
            
        images = torch.stack(images).to(device)
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=config['training']['mixed_precision']):
                _ = model(images)
                
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        forward_times.append((end - start) * 1000) # in ms
        
    avg_forward_ms = sum(forward_times) / len(forward_times)
    
    print(f"Measuring end-to-end inference latency (including data transfer to GPU)...")
    e2e_times = []
    
    for _ in range(measure_iters):
        try:
            images, _ = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            images, _ = next(iterator)
            
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        images = torch.stack(images).to(device)
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=config['training']['mixed_precision']):
                _ = model(images)
                
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        e2e_times.append((end - start) * 1000)
        
    avg_e2e_ms = sum(e2e_times) / len(e2e_times)
    fps = 1000.0 / avg_e2e_ms
    
    max_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    
    results = {
        "model": "CNN-ConvNeXt-Tiny-Baseline",
        "batch_size": batch_size,
        "warmup_iters": warmup_iters,
        "measure_iters": measure_iters,
        "forward_latency_ms": round(avg_forward_ms, 2),
        "e2e_latency_ms": round(avg_e2e_ms, 2),
        "inference_fps": round(fps, 2),
        "peak_inference_memory_mb": round(max_memory, 2)
    }
    
    with open('results/cnn_inference_profile.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Inference profiling complete! Saved to results/cnn_inference_profile.json")
    print(f"Forward Latency: {results['forward_latency_ms']} ms")
    print(f"E2E Latency: {results['e2e_latency_ms']} ms")
    print(f"FPS: {results['inference_fps']}")

if __name__ == '__main__':
    profile()
