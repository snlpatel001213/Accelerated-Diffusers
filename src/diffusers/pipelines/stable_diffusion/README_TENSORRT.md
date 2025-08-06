# TensorRT Optimization for Stable Diffusion Pipelines

This directory contains TensorRT optimization capabilities for Stable Diffusion pipelines, providing significant performance improvements for inference on NVIDIA GPUs.

## Overview

The TensorRT optimization system automatically:

1. **Checks for existing optimized models** - Looks for cached ONNX and TensorRT engine files
2. **Converts PyTorch models** - Automatically converts to ONNX format using Polygraphy optimization
3. **Builds TensorRT engines** - Creates optimized TensorRT engines for maximum performance
4. **Replaces forward calls** - Seamlessly replaces PyTorch forward calls with TensorRT inference
5. **Provides fallback** - Falls back to PyTorch if TensorRT is unavailable

## Files

- `tensorrt_optimization.py` - Core TensorRT optimization manager and mixin class
- `pipeline_stable_diffusion_tensorrt.py` - TensorRT-optimized Stable Diffusion pipeline
- `example_tensorrt_usage.py` - Complete usage example with benchmarking
- `README_TENSORRT.md` - This documentation file

## Requirements

### Required Dependencies
```bash
pip install torch torchvision transformers diffusers
```

### TensorRT Dependencies (for optimization)
```bash
# Install TensorRT (NVIDIA official installation)
pip install tensorrt

# Install CUDA-related packages
pip install pycuda

# Install ONNX tools
pip install onnx onnx-graphsurgeon

# Install Polygraphy (NVIDIA's toolkit)
pip install polygraphy
```

### Hardware Requirements
- NVIDIA GPU with CUDA support
- CUDA 11.0+ or 12.0+
- TensorRT 8.0+

## Quick Start

### Basic Usage

```python
import torch
from diffusers.pipelines.stable_diffusion import StableDiffusionTensorRTPipeline

# Load the TensorRT-optimized pipeline
pipe = StableDiffusionTensorRTPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# Enable TensorRT optimization (converts models automatically)
pipe.enable_tensorrt_optimization(
    cache_dir="./tensorrt_cache",  # Where to store optimized models
    fp16=True,                     # Use FP16 for better performance
    verbose=True                   # Show conversion progress
)

# Generate images (uses TensorRT for inference)
prompt = "A beautiful landscape with mountains and a lake"
image = pipe(prompt).images[0]
image.save("output.png")
```

### Advanced Configuration

```python
# Configure TensorRT optimization settings
pipe.enable_tensorrt_optimization(
    cache_dir="./my_tensorrt_cache",
    fp16=True,                    # Enable FP16 precision
    max_workspace_size=2 << 30,   # 2GB workspace (default: 1GB)
    verbose=True                  # Show detailed logs
)

# Check optimization status
if pipe.tensorrt_enabled:
    print("TensorRT optimization active")
    print("Optimized models:", list(pipe.tensorrt_optimizer.engines.keys()))
else:
    print("Running with PyTorch fallback")
```

## How It Works

### 1. File Checking
The system first checks for existing optimized models:
```
tensorrt_cache/
├── text_encoder.onnx
├── text_encoder.trt
├── unet.onnx
├── unet.trt
├── vae_decoder.onnx
└── vae_decoder.trt
```

### 2. Model Conversion Pipeline
If optimized models don't exist:

**PyTorch → ONNX → TensorRT**

1. **ONNX Export**: Converts PyTorch models to ONNX format
2. **Polygraphy Optimization**: Applies graph optimizations using NVIDIA Polygraphy
3. **TensorRT Compilation**: Builds optimized TensorRT engines
4. **Caching**: Saves optimized models for future use

### 3. Runtime Inference
During generation:
- **Text Encoder**: TensorRT inference for prompt encoding
- **UNet**: TensorRT inference for denoising (most performance-critical)
- **VAE Decoder**: TensorRT inference for latent-to-image conversion

### 4. Automatic Fallback
If TensorRT is unavailable or optimization fails:
- Seamlessly falls back to standard PyTorch inference
- No changes needed in user code
- Full compatibility maintained

## Performance Benefits

Typical performance improvements on NVIDIA GPUs:

| Model Component | Speedup | Notes |
|----------------|---------|-------|
| Text Encoder   | 1.5-2x  | Moderate improvement |
| UNet          | 2-4x    | Largest improvement |
| VAE Decoder   | 1.5-2x  | Good improvement |
| **Overall**   | **2-3x** | **Total pipeline speedup** |

### Benchmark Example
```python
# See example_tensorrt_usage.py for complete benchmarking code
pytorch_time = 3.2  # seconds
tensorrt_time = 1.1  # seconds
speedup = 2.9x      # improvement
```

## Troubleshooting

### Common Issues

1. **TensorRT Not Found**
   ```
   ImportError: TensorRT is not available
   ```
   - Install TensorRT following NVIDIA's official guide
   - Ensure CUDA is properly installed

2. **CUDA Memory Issues**
   ```
   RuntimeError: CUDA out of memory
   ```
   - Reduce `max_workspace_size` parameter
   - Use FP16 instead of FP32
   - Close other GPU applications

3. **Model Conversion Fails**
   ```
   RuntimeError: Failed to build TensorRT engine
   ```
   - Check CUDA/TensorRT version compatibility
   - Verify model architecture is supported
   - Enable verbose logging for detailed error info

### Debug Mode
```python
pipe.enable_tensorrt_optimization(verbose=True)
```

This will show detailed conversion progress and any errors.

### Cache Management
```python
# Clear cache to force re-conversion
import shutil
shutil.rmtree("./tensorrt_cache")

# Check cache size
import os
cache_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, dirnames, filenames in os.walk("./tensorrt_cache")
                for filename in filenames)
print(f"Cache size: {cache_size / (1024**3):.1f} GB")
```

## Integration with Existing Code

The TensorRT pipeline is a drop-in replacement:

```python
# Change this:
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained(...)

# To this:
from diffusers.pipelines.stable_diffusion import StableDiffusionTensorRTPipeline
pipe = StableDiffusionTensorRTPipeline.from_pretrained(...)
pipe.enable_tensorrt_optimization()

# All other code remains the same!
```

## Technical Details

### Model Modifications
- **Text Encoder**: Converts CLIP text encoder to TensorRT
- **UNet**: Optimizes the core denoising model with dynamic shapes
- **VAE Decoder**: Creates specialized decoder-only TensorRT engine

### Memory Management
- Uses CUDA streams for async execution
- Efficient GPU memory allocation
- Automatic cleanup on destruction

### Optimization Features
- **Graph Fusion**: Combines operations for efficiency
- **Kernel Auto-tuning**: Optimizes kernels for specific GPU
- **Mixed Precision**: FP16/FP32 optimizations
- **Dynamic Batching**: Supports variable batch sizes

## Contributing

To extend TensorRT support to other pipelines:

1. Inherit from `TensorRTStableDiffusionMixin`
2. Implement model-specific optimization methods
3. Override forward calls to use TensorRT inference
4. Add proper error handling and fallbacks

Example for a new pipeline:
```python
class MyCustomTensorRTPipeline(TensorRTStableDiffusionMixin, MyCustomPipeline):
    def _optimize_custom_model(self):
        # Add custom model optimization
        pass
    
    def _tensorrt_custom_call(self, *args, **kwargs):
        # Add custom TensorRT inference
        pass
```

## License

This code follows the same Apache 2.0 license as the main diffusers library.