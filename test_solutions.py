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