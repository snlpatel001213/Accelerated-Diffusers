#!/usr/bin/env python3
"""
Test script with multiple CUDA/TensorRT fixes for common issues.
This script tries different approaches to solve CUDA initialization errors.
"""

import os
import sys
import torch
import subprocess

def check_environment():
    """Check CUDA and TensorRT environment."""
    print("🔍 Checking environment...")
    
    # Check CUDA
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    
    # Check TensorRT
    try:
        import tensorrt
        print(f"TensorRT version: {tensorrt.__version__}")
    except ImportError:
        print("❌ TensorRT not available")
        return False
    
    return True

def fix_cuda_context():
    """Apply various CUDA context fixes."""
    print("🔧 Applying CUDA context fixes...")
    
    # Fix 1: Set CUDA device and initialize context
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.cuda.init()
        torch.cuda.empty_cache()
        print("✅ CUDA context initialized")
    
    # Fix 2: Set environment variables
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # Fix 3: Ensure proper CUDA memory allocation
    if torch.cuda.is_available():
        # Allocate a small tensor to ensure CUDA context is active
        dummy = torch.ones(1).cuda()
        del dummy
        torch.cuda.synchronize()
        print("✅ CUDA memory allocation verified")

def test_tensorrt_basic():
    """Test basic TensorRT functionality without diffusers."""
    print("🧪 Testing basic TensorRT functionality...")
    
    try:
        import tensorrt as trt
        
        # Create a simple TensorRT logger
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        
        # Test basic TensorRT operations
        network = builder.create_network()
        config = builder.create_builder_config()
        
        print("✅ TensorRT basic operations successful")
        return True
    except Exception as e:
        print(f"❌ TensorRT basic test failed: {e}")
        return False

def test_diffusers_pipeline():
    """Test the diffusers TensorRT pipeline."""
    print("🚀 Testing diffusers TensorRT pipeline...")
    
    try:
        from diffusers.pipelines.stable_diffusion import StableDiffusionTensorRTPipeline
        
        # Load pipeline with minimal settings
        print("Loading pipeline...")
        pipe = StableDiffusionTensorRTPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to("cuda")
        
        print("✅ Pipeline loaded successfully!")
        
        # Enable TensorRT with minimal settings
        print("Enabling TensorRT optimization...")
        pipe.enable_tensorrt_optimization(
            cache_dir="./tensorrt_cache",
            fp16=True,
            verbose=False,  # Reduce verbosity to avoid cluttering
            max_workspace_size=512 * 1024 * 1024  # 512MB workspace (smaller)
        )
        
        print("✅ TensorRT optimization enabled!")
        
        # Generate a test image
        print("Generating test image...")
        image = pipe("A simple test image").images[0]
        image.save("test_success.png")
        print("✅ Image generated successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False

def main():
    """Main test function with multiple fallback strategies."""
    print("🔧 TensorRT CUDA Initialization Fix Script")
    print("=" * 50)
    
    # Step 1: Check environment
    if not check_environment():
        print("❌ Environment check failed. Install TensorRT first.")
        return
    
    # Step 2: Apply CUDA fixes
    fix_cuda_context()
    
    # Step 3: Test basic TensorRT
    if not test_tensorrt_basic():
        print("❌ Basic TensorRT test failed. Check TensorRT installation.")
        return
    
    # Step 4: Test diffusers pipeline
    if test_diffusers_pipeline():
        print("\n🎉 SUCCESS! TensorRT with diffusers is working!")
    else:
        print("\n❌ Pipeline test failed. Trying alternative solutions...")
        
        # Alternative solution: Reset CUDA context
        print("🔄 Resetting CUDA context...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        # Try again with even smaller workspace
        print("🔄 Retrying with minimal TensorRT settings...")
        try:
            from diffusers.pipelines.stable_diffusion import StableDiffusionTensorRTPipeline
            pipe = StableDiffusionTensorRTPipeline.from_pretrained(
                "stable-diffusion-v1-5/stable-diffusion-v1-5",
                torch_dtype=torch.float16
            ).to("cuda")
            
            pipe.enable_tensorrt_optimization(
                cache_dir="./tensorrt_cache_minimal",
                fp16=True,
                verbose=False,
                max_workspace_size=256 * 1024 * 1024  # 256MB
            )
            
            image = pipe("test").images[0]
            print("✅ SUCCESS with minimal settings!")
            
        except Exception as e:
            print(f"❌ Still failing: {e}")
            print("\n💡 Additional troubleshooting suggestions:")
            print("1. Restart the container/session")
            print("2. Check if other processes are using CUDA")
            print("3. Try with CPU-only mode first")
            print("4. Update TensorRT to latest version")

if __name__ == "__main__":
    main()