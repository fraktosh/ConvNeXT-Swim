import sys
import torch
import platform
import os

try:
    import timm
    timm_version = timm.__version__
except ImportError:
    timm_version = "Not installed"

def check_environment():
    env_info = []
    env_info.append("=== Environment Information ===")
    env_info.append(f"OS: {platform.system()} {platform.release()}")
    env_info.append(f"Python version: {sys.version.split(' ')[0]}")
    env_info.append(f"PyTorch version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    env_info.append(f"CUDA availability: {cuda_available}")
    
    if cuda_available:
        env_info.append(f"CUDA version: {torch.version.cuda}")
        env_info.append(f"GPU Name: {torch.cuda.get_device_name(0)}")
        env_info.append(f"GPU Count: {torch.cuda.device_count()}")
    
    env_info.append(f"timm version: {timm_version}")
    
    output = "\n".join(env_info)
    print(output)
    
    with open("results/environment.txt", "w") as f:
        f.write(output)

if __name__ == "__main__":
    check_environment()
