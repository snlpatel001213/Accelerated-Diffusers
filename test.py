"""
🚀 Diffusers Pipeline Test Script

EASY SWITCHING GUIDE:
=====================
To switch between accelerated and unaccelerated versions, simply change the value below:

✅ For ACCELERATED (TensorRT optimized):  USE_ACCELERATION = True
✅ For STANDARD (unaccelerated):         USE_ACCELERATION = False

No other changes needed! The script will automatically use the right imports and settings.
"""

import os
import torch

# 🔧 CONFIGURATION: Switch between accelerated and unaccelerated versions
USE_ACCELERATION = True  # 👈 CHANGE THIS LINE: True = accelerated, False = standard

# Conditional imports based on acceleration setting
if USE_ACCELERATION:
    # Accelerated version with TensorRT
    from diffusers.pipelines.stable_diffusion import StableDiffusionTensorRTPipeline as StableDiffusionPipeline
    print("✅ Using ACCELERATED version with TensorRT optimization")
else:
    # Standard unaccelerated version
    from diffusers import StableDiffusionPipeline
    print("✅ Using STANDARD unaccelerated version")

print("✅ Import successful!")
if USE_ACCELERATION:
    print("✅ TensorRT optimization available:", hasattr(StableDiffusionPipeline, 'enable_tensorrt_optimization'))

# Create a simple test without loading models
print("✅ Pipeline class loaded successfully!")
print("✅ Base classes:", [cls.__name__ for cls in StableDiffusionPipeline.__mro__])

print("\n🎉 Development setup working correctly!")
print("You can now make changes to the local repo and they will be reflected immediately.")

# Set CUDA device properly and initialize CUDA context
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA devices: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    # Initialize CUDA context properly
    torch.cuda.set_device(0)  # Use first GPU
    torch.cuda.init()  # Initialize CUDA context
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Clear any existing CUDA memory
    torch.cuda.empty_cache()
    
    try:
        print("Loading pipeline...")
        # Load pipeline with proper error handling
        pipe = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to("cuda")
        
        print("Pipeline loaded successfully!")
        
        # Conditionally enable TensorRT optimization only for accelerated version
        if USE_ACCELERATION:
            print("Enabling TensorRT optimization...")
            
            # Smart caching demonstration
            cache_dir = "./tensorrt_cache"
            onnx_cache_dir = "./onnx_cache"
            
            # Check existing files with dimension-specific naming
            engine_files = ["text_encoder_b1.trt", "unet_b1_h768_w512.trt", "vae_decoder_b1_h768_w512.trt"]
            onnx_files = ["text_encoder_b1.onnx", "unet_b1_h768_w512.onnx", "vae_decoder_b1_h768_w512.onnx"]
            
            print(f"\n📁 Checking cache directories:")
            print(f"   TensorRT engines: {cache_dir}/")
            print(f"   ONNX models: {onnx_cache_dir}/")
            
            print(f"\n🔍 Current cache status:")
            for i, (engine_file, onnx_file) in enumerate(zip(engine_files, onnx_files)):
                engine_path = os.path.join(cache_dir, engine_file)
                onnx_path = os.path.join(onnx_cache_dir, onnx_file)
                
                engine_exists = os.path.exists(engine_path)
                onnx_exists = os.path.exists(onnx_path)
                
                model_name = engine_file.replace('.trt', '').split('_')[0]  # Extract base model name
                print(f"   {model_name} ({engine_file.replace('.trt', '')}):")
                
                if engine_exists and onnx_exists:
                    print(f"     ✅ TensorRT engine exists → Will load directly (fastest)")
                    print(f"     ✅ ONNX model exists → Preserved for future use")
                elif onnx_exists and not engine_exists:
                    print(f"     🔄 ONNX exists, TensorRT missing → Will create TensorRT from ONNX")
                    print(f"     ⚡ No PyTorch conversion needed!")
                elif engine_exists and not onnx_exists:
                    print(f"     ✅ TensorRT engine exists → Will load directly")
                    print(f"     ⚠️  ONNX missing → Engine was created without ONNX preservation")
                else:
                    print(f"     🔄 Neither exists → Full conversion: PyTorch → ONNX → TensorRT")
            
            # Enable optimization with separate cache directories and custom dimensions
            print(f"\n🚀 Enabling TensorRT optimization with smart caching...")
            pipe.enable_tensorrt_optimization(
                cache_dir=cache_dir,
                onnx_cache_dir=onnx_cache_dir,
                fp16=True,
                verbose=True,
                max_workspace_size=1 << 30,  # 1GB workspace
                batch_size=1,
                height=768,  # Custom height for testing variable dimensions
                width=512    # Custom width for testing variable dimensions
            )
            print("TensorRT optimization enabled!")
            
            # Verify optimization results
            print("\n✅ Optimization complete! Verifying results...")
            
            print(f"\n📊 Final cache status:")
            for engine_file, onnx_file in zip(engine_files, onnx_files):
                engine_path = os.path.join(cache_dir, engine_file)
                onnx_path = os.path.join(onnx_cache_dir, onnx_file)
                
                model_name = engine_file.replace('.trt', '').split('_')[0]  # Extract base model name
                print(f"   {model_name} ({engine_file.replace('.trt', '')}):")
                
                if os.path.exists(engine_path):
                    size_mb = os.path.getsize(engine_path) / (1024 * 1024)
                    print(f"     ✅ TensorRT engine: {engine_file} ({size_mb:.1f} MB)")
                else:
                    print(f"     ❌ TensorRT engine: {engine_file} (MISSING)")
                
                if os.path.exists(onnx_path):
                    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
                    print(f"     ✅ ONNX model: {onnx_file} ({size_mb:.1f} MB)")
                else:
                    print(f"     ⚠️  ONNX model: {onnx_file} (not preserved)")
            
            print(f"\n📁 Smart cache organization:")
            print(f"   📂 {cache_dir}/ - TensorRT engine files (.trt) for fast loading")
            print(f"   📂 {onnx_cache_dir}/ - ONNX files (.onnx) for reuse without PyTorch conversion")
            print(f"\n💡 Next run will be faster:")
            print(f"   - TensorRT engines will load directly (no conversion)")
            print(f"   - ONNX files are preserved for rebuilding engines if needed")
        else:
            print("Running with standard (unaccelerated) pipeline - no TensorRT optimization")
        
        # Generate images with custom dimensions
        print("Generating image with custom dimensions (768x512)...")
        image = pipe(
            "A beautiful landscape", 
            height=768, 
            width=512,
            num_inference_steps=20,  # Faster generation for testing
            guidance_scale=7.5
        ).images[0]
        image.save("test_output.png")
        print("✅ Image generated successfully and saved as test_output.png!")
        print(f"✅ Generated image dimensions: {image.size} (width x height)")
        
        if USE_ACCELERATION:
            print("✅ Image was generated using TensorRT-optimized models with variable dimensions!")
        else:
            print("✅ Image was generated using standard pipeline")
        
    except Exception as e:
        if USE_ACCELERATION:
            print(f"❌ Error during accelerated execution: {e}")
            print("This might be due to TensorRT or CUDA context issues.")
            print("Try setting USE_ACCELERATION = False to use the standard pipeline.")
        else:
            print(f"❌ Error during standard execution: {e}")
            print("This might be due to CUDA context initialization issues.")
        
else:
    print("❌ CUDA not available!")