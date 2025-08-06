# 🔧 TensorRT CUDA Error 35 Solutions

## Problem Diagnosis
You're encountering **CUDA Error 35** during TensorRT initialization. This is caused by a **version mismatch** between your CUDA components:

- **NVIDIA Driver**: 575.64.03 (supports CUDA 12.9)
- **CUDA Toolkit**: 12.9
- **PyTorch CUDA**: 12.6 ⚠️ MISMATCH
- **TensorRT**: 10.13.2.6

## 🎯 Solution 1: Update PyTorch to Match CUDA 12.9 (RECOMMENDED)

```bash
# Uninstall current PyTorch
pip uninstall torch torchvision torchaudio

# Install PyTorch with CUDA 12.6 (matching your current environment)
# OR upgrade to CUDA 12.9 compatible version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Alternative: Install latest PyTorch with CUDA 12.4+ (most compatible)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## 🎯 Solution 2: Downgrade CUDA Toolkit to 12.6

```bash
# Remove current CUDA
sudo apt remove --purge '*cuda*' '*cublas*' '*cufft*' '*cufile*' '*curand*' '*cusolver*' '*cusparse*' '*npp*' '*nvjpeg*' 'nsight*'

# Install CUDA 12.6
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-6

# Update PATH
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
```

## 🎯 Solution 3: Use CPU-Only Mode (Temporary Workaround)

```python
# Modified test script for CPU-only
import torch
from diffusers import StableDiffusionPipeline

# Force CPU usage
device = "cpu"
torch_dtype = torch.float32  # Use float32 for CPU

pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch_dtype,
    use_safetensors=True
).to(device)

# Generate image (slower but works)
image = pipe("A beautiful landscape").images[0]
image.save("cpu_output.png")
print("✅ CPU generation successful!")
```

## 🎯 Solution 4: Container/Environment Reset

```bash
# Option A: Restart the container completely
docker restart <container_name>

# Option B: Reset CUDA context
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia
sudo modprobe nvidia nvidia_modeset nvidia_drm nvidia_uvm

# Option C: Clear all CUDA processes
sudo fuser -v /dev/nvidia*
sudo kill -9 <process_ids_if_any>
```

## 🎯 Solution 5: Alternative TensorRT Installation

```bash
# Remove current TensorRT
pip uninstall tensorrt tensorrt-cu13 tensorrt_cu13_bindings tensorrt_cu13_libs

# Install TensorRT that matches your CUDA version
pip install tensorrt==10.5.0  # or another compatible version

# Or try the NVIDIA official installation
pip install nvidia-tensorrt
```

## 🧪 Testing Script

Save this as `test_solutions.py`:

```python
#!/usr/bin/env python3
import torch
import sys

def test_cuda_compatibility():
    print("🔍 CUDA Compatibility Check")
    print("=" * 40)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA version: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    
    # Test basic CUDA operations
    try:
        if torch.cuda.is_available():
            x = torch.randn(100, 100).cuda()
            y = torch.mm(x, x.t())
            print("✅ Basic CUDA operations work")
        else:
            print("⚠️ CUDA not available")
    except Exception as e:
        print(f"❌ CUDA operations failed: {e}")
    
    # Test TensorRT import
    try:
        import tensorrt
        print(f"✅ TensorRT available: {tensorrt.__version__}")
    except ImportError:
        print("❌ TensorRT not available")
    
    # Test TensorRT basic functionality
    try:
        import tensorrt as trt
        logger = trt.Logger(trt.Logger.ERROR)  # Suppress warnings
        builder = trt.Builder(logger)
        print("✅ TensorRT basic functionality works")
        return True
    except Exception as e:
        print(f"❌ TensorRT failed: {e}")
        return False

if __name__ == "__main__":
    success = test_cuda_compatibility()
    if success:
        print("\n🎉 Environment is ready for TensorRT!")
    else:
        print("\n❌ Please apply one of the solutions above.")
```

## 🚀 Quick Fix Commands

```bash
# Quick test to verify the fix
cd /code/diffusers
python3 test_solutions.py

# If successful, try the original script
python3 test.py
```

## 📋 Next Steps

1. **Choose Solution 1 (PyTorch update)** - This is usually the easiest
2. **Test with the verification script**
3. **Run your original diffusers script**
4. **If still failing, try Solution 2 or 4**

## 💡 Prevention

- Always match PyTorch CUDA version with your CUDA toolkit
- Use virtual environments to avoid version conflicts
- Check compatibility matrices before installing packages

---
**Error 35 specifically means**: CUDA driver/runtime version mismatch or context creation failure.
The solutions above address both potential causes.