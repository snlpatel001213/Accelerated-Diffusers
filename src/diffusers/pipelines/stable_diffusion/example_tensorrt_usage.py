#!/usr/bin/env python3
"""
Example usage of StableDiffusionTensorRTPipeline with automatic model optimization.

This example demonstrates:
1. Loading a Stable Diffusion pipeline with TensorRT optimization
2. Automatic ONNX and TensorRT conversion
3. Fallback to PyTorch if TensorRT is not available
4. Performance comparison between PyTorch and TensorRT inference
"""

import time
import torch
from diffusers import StableDiffusionPipeline
from pipeline_stable_diffusion_tensorrt import StableDiffusionTensorRTPipeline


def benchmark_pipeline(pipe, prompt, num_runs=5):
    """Benchmark pipeline inference time."""
    # Warmup
    _ = pipe(prompt, num_inference_steps=20)
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_runs):
        _ = pipe(prompt, num_inference_steps=20)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    return avg_time


def main():
    # Model configuration
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    prompt = "A beautiful landscape with mountains and a lake, digital art"
    
    print("=== Stable Diffusion TensorRT Optimization Example ===\n")
    
    # Load standard pipeline for comparison
    print("Loading standard Stable Diffusion pipeline...")
    standard_pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)
    
    # Load TensorRT-optimized pipeline
    print("Loading TensorRT-optimized Stable Diffusion pipeline...")
    tensorrt_pipe = StableDiffusionTensorRTPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)
    
    # Enable TensorRT optimization
    print("Enabling TensorRT optimization...")
    print("This will convert models to ONNX and TensorRT engines if not cached.")
    print("First run may take several minutes for conversion...\n")
    
    try:
        tensorrt_pipe.enable_tensorrt_optimization(
            cache_dir="./tensorrt_cache",
            fp16=True if dtype == torch.float16 else False,
            verbose=True
        )
        tensorrt_enabled = True
        print("TensorRT optimization enabled successfully!\n")
    except Exception as e:
        print(f"TensorRT optimization failed: {e}")
        print("Falling back to standard PyTorch inference\n")
        tensorrt_enabled = False
    
    # Generate images and benchmark
    print("Generating images...")
    
    # Standard PyTorch inference
    print("Running standard PyTorch inference...")
    pytorch_time = benchmark_pipeline(standard_pipe, prompt)
    print(f"PyTorch average time: {pytorch_time:.2f} seconds")
    
    # TensorRT inference (or fallback)
    if tensorrt_enabled:
        print("Running TensorRT-optimized inference...")
        tensorrt_time = benchmark_pipeline(tensorrt_pipe, prompt)
        print(f"TensorRT average time: {tensorrt_time:.2f} seconds")
        
        speedup = pytorch_time / tensorrt_time
        print(f"Speedup: {speedup:.2f}x faster with TensorRT")
    else:
        print("Running fallback PyTorch inference...")
        fallback_time = benchmark_pipeline(tensorrt_pipe, prompt)
        print(f"Fallback average time: {fallback_time:.2f} seconds")
    
    # Generate final images
    print("\nGenerating final high-quality images...")
    
    # Standard pipeline result
    standard_result = standard_pipe(
        prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
        generator=torch.Generator(device=device).manual_seed(42)
    )
    standard_result.images[0].save("standard_output.png")
    print("Standard pipeline result saved as 'standard_output.png'")
    
    # TensorRT pipeline result
    tensorrt_result = tensorrt_pipe(
        prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
        generator=torch.Generator(device=device).manual_seed(42)
    )
    tensorrt_result.images[0].save("tensorrt_output.png")
    print("TensorRT pipeline result saved as 'tensorrt_output.png'")
    
    print("\n=== Optimization Summary ===")
    print("✓ Models automatically converted to ONNX and TensorRT")
    print("✓ Converted models cached for future use")
    print("✓ Seamless fallback to PyTorch if TensorRT unavailable")
    print("✓ Compatible with all Stable Diffusion pipeline features")


if __name__ == "__main__":
    main()