import os
import sys
import json
import time
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.cnn_detector import CNNDetector
from src.models.fixed_hybrid_detector import FixedHybridDetector
from src.models.adaptive_hybrid_detector import AdaptiveHybridDetector

def profile_all():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Profiling on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    models = {
        'CNN-only': (CNNDetector(num_classes=80, pretrained=False), 'checkpoints/cnn_baseline/final_model.pth'),
        'Fixed Swin P4': (FixedHybridDetector(variant='P4', num_classes=80, pretrained=False), 'checkpoints/fixed_p4/final_model.pth'),
        'Fixed Swin P3/P4': (FixedHybridDetector(variant='P3P4', num_classes=80, pretrained=False), 'checkpoints/fixed_p3p4/final_model.pth'),
        'Adaptive routing': (AdaptiveHybridDetector(num_classes=80, pretrained=False), 'checkpoints/adaptive/final_model.pth')
    }
    
    dummy = torch.randn(1, 3, 640, 640, device=device)
    results = {}
    
    warmup_iters = 50
    measure_iters = 200
    
    for name, (model, ckpt_path) in models.items():
        print(f"\nBenchmarking {name}...")
        model = model.to(device)
        if os.path.exists(ckpt_path):
            sd = torch.load(ckpt_path, map_location=device, weights_only=False)
            if isinstance(sd, dict) and 'model_state_dict' in sd:
                sd = sd['model_state_dict']
            model.load_state_dict(sd, strict=False)
            print(f"  Loaded checkpoint {ckpt_path}")
        model.eval()
        
        total_params = sum(p.numel() for p in model.parameters())
        
        # Warmup
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize()
            
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                for _ in range(warmup_iters):
                    _ = model(dummy)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        # Measurement
        times = []
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                for _ in range(measure_iters):
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    _ = model(dummy)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    times.append((t1 - t0) * 1000) # ms
                    
        avg_latency = sum(times) / len(times)
        fps = 1000.0 / avg_latency
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if torch.cuda.is_available() else 0.0
        
        results[name] = {
            "Model": name,
            "Params": total_params,
            "Params_M": round(total_params / 1e6, 2),
            "Latency_ms": round(avg_latency, 2),
            "FPS": round(fps, 2),
            "Peak_Mem_MB": round(peak_mem, 2),
            "Protocol": {
                "Batch_size": 1,
                "Resolution": "640x640",
                "Device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                "Warmup_iters": warmup_iters,
                "Measure_iters": measure_iters,
                "Mixed_precision": True
            }
        }
        print(f"  Result: {total_params / 1e6:.2f}M params | {avg_latency:.2f} ms ({fps:.1f} FPS) | Peak Mem: {peak_mem:.1f} MB")
        
    os.makedirs('results', exist_ok=True)
    with open('results/benchmark_profile_consistent.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nSaved benchmark results to results/benchmark_profile_consistent.json")
    return results

if __name__ == '__main__':
    profile_all()
