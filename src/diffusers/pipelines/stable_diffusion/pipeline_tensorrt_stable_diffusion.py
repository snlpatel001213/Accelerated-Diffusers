# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import numpy as np
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...image_processor import PipelineImageInput
from ...loaders import TextualInversionLoaderMixin
from ...models import AutoencoderKL, UNet2DConditionModel
from ...utils import replace_example_docstring
from .pipeline_stable_diffusion import StableDiffusionPipeline
from .pipeline_output import StableDiffusionPipelineOutput
from .safety_checker import StableDiffusionSafetyChecker

try:
    import onnx
    from polygraphy.backend.onnx.loader import fold_constants
    import onnx_graphsurgeon as gs
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    warnings.warn(
        "ONNX or Polygraphy not available. Install onnx, onnx-graphsurgeon, and polygraphy for model conversion."
    )

try:
    import tensorrt as trt
    import pycuda.autoinit
    import pycuda.driver as cuda
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    warnings.warn(
        "TensorRT not available. Install tensorrt and pycuda to use TensorRT optimization."
    )


class TensorRTOptimizer:
    """
    TensorRT optimization manager for Stable Diffusion pipelines.
    
    This class handles:
    1. Checking for existing ONNX/TensorRT engine files
    2. Converting PyTorch models to ONNX and then to TensorRT engines using Polygraphy CLI
    3. Replacing PyTorch forward calls with TensorRT inference
    """
    
    def __init__(
        self,
        cache_dir: str = "./tensorrt_cache",
        onnx_cache_dir: Optional[str] = None,
        fp16: bool = True,
        max_workspace_size: int = 1 << 30,  # 1GB
        verbose: bool = False
    ):
        """
        Initialize TensorRT optimizer.
        
        Args:
            cache_dir: Directory to store TensorRT engine files
            onnx_cache_dir: Directory to store ONNX files (defaults to cache_dir if None)
            fp16: Whether to use FP16 precision
            max_workspace_size: Maximum workspace size for TensorRT
            verbose: Whether to print verbose logs
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX tools are not available. Please install onnx, onnx-graphsurgeon, and polygraphy.")
        
        # Check if Polygraphy CLI is available
        try:
            subprocess.run(["polygraphy", "--help"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise ImportError("Polygraphy CLI not found. Please install polygraphy with: pip install polygraphy[tensorflow]")
        
        if not TRT_AVAILABLE:
            warnings.warn("TensorRT Python API not available, but Polygraphy CLI will be used for conversion. Install tensorrt and pycuda for runtime inference.")
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.onnx_cache_dir = Path(onnx_cache_dir) if onnx_cache_dir else self.cache_dir
        self.onnx_cache_dir.mkdir(parents=True, exist_ok=True)
        self.fp16 = fp16
        self.max_workspace_size = max_workspace_size
        self.verbose = verbose
        
        # Initialize TensorRT runtime components only if TensorRT is available
        if TRT_AVAILABLE:
            # TensorRT logger
            self.trt_logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
            
            # Store engines and contexts
            self.engines = {}
            self.contexts = {}
            self.cuda_inputs = {}
            self.cuda_outputs = {}
            self.cuda_bindings = {}
            self.stream = None
        else:
            self.trt_logger = None
            self.engines = {}
            self.contexts = {}
            self.cuda_inputs = {}
            self.cuda_outputs = {}
            self.cuda_bindings = {}
            self.stream = None
        
    def _get_model_paths(self, model_name: str, batch_size: int = 1, height: int = 512, width: int = 512) -> Tuple[Path, Path]:
        """Get ONNX and TensorRT engine file paths for a model with specific dimensions."""
        # Create meaningful names that include dimensions
        if model_name == "text_encoder":
            # Text encoder doesn't depend on height/width, only batch size
            model_suffix = f"_b{batch_size}"
        else:
            # UNet and VAE depend on all dimensions
            model_suffix = f"_b{batch_size}_h{height}_w{width}"
        
        onnx_path = self.onnx_cache_dir / f"{model_name}{model_suffix}.onnx"
        engine_path = self.cache_dir / f"{model_name}{model_suffix}.trt"
        return onnx_path, engine_path
    
    def _validate_dimensions(self, height: int, width: int) -> bool:
        """
        Validate that height and width meet diffusion model requirements.
        
        Args:
            height: Image height
            width: Image width
            
        Returns:
            bool: True if dimensions are valid
            
        Raises:
            ValueError: If dimensions are invalid
        """
        # Check range (128-1024)
        if not (128 <= height <= 1024):
            raise ValueError(f"Height must be between 128 and 1024, got {height}")
        if not (128 <= width <= 1024):
            raise ValueError(f"Width must be between 128 and 1024, got {width}")
        
        # Check multiple of 16
        if height % 16 != 0:
            raise ValueError(f"Height must be multiple of 16, got {height}")
        if width % 16 != 0:
            raise ValueError(f"Width must be multiple of 16, got {width}")
        
        return True
    
    def _check_files_exist(self, model_name: str, batch_size: int = 1, height: int = 512, width: int = 512) -> Tuple[bool, bool]:
        """Check if ONNX and TensorRT engine files exist for specific dimensions."""
        onnx_path, engine_path = self._get_model_paths(model_name, batch_size, height, width)
        return onnx_path.exists(), engine_path.exists()
    
    def _convert_to_onnx(
        self,
        model: torch.nn.Module,
        model_args: Tuple,
        model_name: str,
        input_names: List[str],
        output_names: List[str],
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        opset_version: int = 14
    ) -> Path:
        """Convert PyTorch model to ONNX format."""
        onnx_path, _ = self._get_model_paths(model_name)
        
        if self.verbose:
            print(f"Converting {model_name} to ONNX...")
        
        # Set model to evaluation mode
        model.eval()
        
        # Export to ONNX
        with torch.inference_mode():
            if self.fp16:
                with torch.autocast("cuda"):
                    torch.onnx.export(
                        model,
                        model_args,
                        onnx_path.as_posix(),
                        input_names=input_names,
                        output_names=output_names,
                        dynamic_axes=dynamic_axes,
                        do_constant_folding=True,
                        opset_version=opset_version,
                        export_params=True,
                        keep_initializers_as_inputs=False
                    )
            else:
                torch.onnx.export(
                    model,
                    model_args,
                    onnx_path.as_posix(),
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    do_constant_folding=True,
                    opset_version=opset_version,
                    export_params=True,
                    keep_initializers_as_inputs=False
                )
        
        # Optimize ONNX with Polygraphy
        if self.verbose:
            print(f"Optimizing ONNX model with Polygraphy...")
        
        onnx_model = onnx.load(onnx_path)
        optimized_model = fold_constants(onnx_model, allow_onnxruntime_shape_inference=True)
        onnx.save(optimized_model, onnx_path)
        
        if self.verbose:
            print(f"ONNX model saved to {onnx_path}")
        
        return onnx_path
    
    def _convert_to_tensorrt(self, onnx_path: Path, model_name: str, input_shapes: Dict[str, Tuple], batch_size: int = 1, height: int = 512, width: int = 512) -> Path:
        """Convert ONNX model to TensorRT engine using Polygraphy CLI."""
        _, engine_path = self._get_model_paths(model_name, batch_size, height, width)
        
        if self.verbose:
            print(f"Converting {model_name} to TensorRT engine using Polygraphy CLI...")
        
        # Build Polygraphy convert command
        cmd = [
            "polygraphy", "convert",
            str(onnx_path),
            "--output", str(engine_path)
        ]
        
        # Add precision settings
        if self.fp16:
            cmd.extend(["--fp16"])
        
        # Add workspace size using the correct option
        workspace_mb = self.max_workspace_size // (1024 * 1024)
        cmd.extend(["--pool-limit", f"workspace:{workspace_mb}M"])
        
        # Add optimization profiles for dynamic shapes
        if input_shapes:
            for input_name, shape in input_shapes.items():
                shape_str = "[" + ",".join(map(str, shape)) + "]"
                cmd.extend([
                    "--trt-min-shapes", f"{input_name}:{shape_str}",
                    "--trt-opt-shapes", f"{input_name}:{shape_str}",
                    "--trt-max-shapes", f"{input_name}:{shape_str}"
                ])
        
        # Add verbose flag if needed
        if self.verbose:
            cmd.extend(["--verbose"])
        
        # Add convert-to option at the end
        cmd.extend(["--convert-to", "trt"])
        
        try:
            if self.verbose:
                print(f"Running command: {' '.join(cmd)}")
            
            # Run Polygraphy convert command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            if self.verbose and result.stdout:
                print("Polygraphy output:", result.stdout)
                
        except subprocess.CalledProcessError as e:
            error_msg = f"Polygraphy conversion failed for {model_name}: {e.stderr}"
            print(error_msg)
            raise RuntimeError(error_msg)
        except FileNotFoundError:
            raise RuntimeError("Polygraphy CLI not found. Please install polygraphy with: pip install polygraphy[tensorflow]")
        
        if not engine_path.exists():
            raise RuntimeError(f"TensorRT engine was not created at {engine_path}")
        
        if self.verbose:
            print(f"TensorRT engine saved to {engine_path}")
        
        return engine_path
    
    def _load_engine(self, model_name: str, batch_size: int = 1, height: int = 512, width: int = 512) -> None:
        """Load TensorRT engine and create execution context."""
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT Python API not available. Install tensorrt and pycuda for runtime inference.")
        
        _, engine_path = self._get_model_paths(model_name, batch_size, height, width)
        
        if not engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")
        
        # Load engine
        runtime = trt.Runtime(self.trt_logger)
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        engine = runtime.deserialize_cuda_engine(engine_data)
        context = engine.create_execution_context()
        
        # Store engine and context with dimension-specific key
        engine_key = f"{model_name}_b{batch_size}_h{height}_w{width}"
        self.engines[engine_key] = engine
        self.contexts[engine_key] = context
        
        # Create CUDA stream if not exists
        if self.stream is None:
            self.stream = cuda.Stream()
        
        # Allocate GPU memory for inputs and outputs
        inputs = []
        outputs = []
        bindings = []
        
        # Use newer TensorRT API
        num_io_tensors = engine.num_io_tensors if hasattr(engine, 'num_io_tensors') else engine.num_bindings
        
        for i in range(num_io_tensors):
            # Use newer API if available, fallback to older API
            if hasattr(engine, 'get_tensor_name'):
                binding_name = engine.get_tensor_name(i)
                tensor_shape = engine.get_tensor_shape(binding_name)
                tensor_dtype = engine.get_tensor_dtype(binding_name)
                is_input = engine.get_tensor_mode(binding_name) == trt.TensorIOMode.INPUT
            else:
                binding_name = engine.get_binding_name(i)
                tensor_shape = engine.get_binding_shape(i)
                tensor_dtype = engine.get_binding_dtype(i)
                is_input = engine.binding_is_input(i)
            
            # Handle dynamic shapes by using maximum possible size
            if -1 in tensor_shape:
                # For dynamic shapes, use a reasonable maximum size
                # This is a conservative approach for dynamic batch/sequence lengths
                max_shape = []
                for dim in tensor_shape:
                    if dim == -1:
                        max_shape.append(16)  # Conservative max batch size
                    else:
                        max_shape.append(dim)
                size = np.prod(max_shape)
            else:
                size = trt.volume(tensor_shape)
            
            dtype = trt.nptype(tensor_dtype)
            
            # Allocate GPU memory with padding for safety
            memory_size = size * np.dtype(dtype).itemsize
            if memory_size <= 0:
                raise RuntimeError(f"Invalid memory size {memory_size} for tensor {binding_name}")
            
            device_mem = cuda.mem_alloc(memory_size)
            bindings.append(int(device_mem))
            
            if self.verbose:
                print(f"  Allocated {memory_size} bytes for {binding_name} (shape: {tensor_shape}, dtype: {dtype})")
            
            if is_input:
                inputs.append(device_mem)
            else:
                outputs.append(device_mem)
        
        self.cuda_inputs[engine_key] = inputs
        self.cuda_outputs[engine_key] = outputs
        self.cuda_bindings[engine_key] = bindings
        
        if self.verbose:
            print(f"Loaded TensorRT engine for {model_name}")
    
    def optimize_model(
        self,
        model: torch.nn.Module,
        model_args: Tuple,
        model_name: str,
        input_names: List[str],
        output_names: List[str],
        batch_size: int = 1,
        height: int = 512,
        width: int = 512,
        input_shapes: Optional[Dict[str, Tuple]] = None,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        opset_version: int = 14
    ) -> bool:
        """
        Optimize a PyTorch model for TensorRT inference with specific dimensions.
        
        Smart caching logic:
        - If TensorRT engine exists for these dimensions: Load directly
        - If ONNX exists but TensorRT doesn't: Create TensorRT from ONNX
        - If neither exists: Convert PyTorch → ONNX → TensorRT
        
        Args:
            model: PyTorch model to optimize
            model_args: Arguments for model forward pass
            model_name: Name of the model
            input_names: Names of input tensors
            output_names: Names of output tensors
            batch_size: Batch size for optimization
            height: Image height (ignored for text_encoder)
            width: Image width (ignored for text_encoder)
            input_shapes: Input tensor shapes
            dynamic_axes: Dynamic axes for ONNX export
            opset_version: ONNX opset version
            
        Returns:
            bool: True if optimization successful, False otherwise
        """
        try:
            # Validate dimensions for models that use height/width
            if model_name != "text_encoder":
                self._validate_dimensions(height, width)
            
            # Test forward pass to ensure model works with current shapes
            if self.verbose:
                print(f"Testing forward pass for {model_name} with current shapes...")
            
            try:
                model.eval()
                with torch.no_grad():
                    test_output = model(*model_args)
                if self.verbose:
                    print(f"✅ Forward pass successful for {model_name}")
            except Exception as e:
                error_msg = f"Forward pass failed for {model_name} with current shapes: {e}"
                print(f"❌ {error_msg}")
                return False
            
            onnx_exists, engine_exists = self._check_files_exist(model_name, batch_size, height, width)
            onnx_path, engine_path = self._get_model_paths(model_name, batch_size, height, width)
            
            if self.verbose:
                print(f"Optimizing {model_name} (batch={batch_size}, h={height}, w={width}):")
                print(f"  ONNX file exists: {onnx_exists} ({onnx_path})")
                print(f"  TensorRT engine exists: {engine_exists} ({engine_path})")
            
            # Case 1: TensorRT engine exists - load directly (fastest path)
            if engine_exists:
                if self.verbose:
                    print(f"✅ Loading existing TensorRT engine: {engine_path}")
                self._load_engine(model_name, batch_size, height, width)
                return True
            
            # Case 2: ONNX exists but TensorRT doesn't - create TensorRT from ONNX
            elif onnx_exists:
                if self.verbose:
                    print(f"🔄 Converting existing ONNX to TensorRT: {onnx_path} → {engine_path}")
                self._convert_to_tensorrt(onnx_path, model_name, input_shapes or {}, batch_size, height, width)
                self._load_engine(model_name, batch_size, height, width)
                return True
            
            # Case 3: Neither exists - full conversion pipeline
            else:
                if self.verbose:
                    print(f"🔄 Full conversion pipeline: PyTorch → ONNX → TensorRT")
                # Convert PyTorch to ONNX
                onnx_path = self._convert_to_onnx(
                    model, model_args, model_name, input_names, output_names, dynamic_axes, opset_version
                )
                # Convert ONNX to TensorRT
                self._convert_to_tensorrt(onnx_path, model_name, input_shapes or {}, batch_size, height, width)
                # Load engine
                self._load_engine(model_name, batch_size, height, width)
                return True
            
        except Exception as e:
            print(f"❌ Failed to optimize {model_name}: {e}")
            return False
    
    def infer_tensorrt(self, model_name: str, inputs: List[torch.Tensor], batch_size: int = 1, height: int = 512, width: int = 512) -> List[torch.Tensor]:
        """
        Perform TensorRT inference instead of PyTorch forward call.
        
        Args:
            model_name: Name of the optimized model
            inputs: List of input tensors
            batch_size: Batch size for inference
            height: Image height (ignored for text_encoder)
            width: Image width (ignored for text_encoder)
            
        Returns:
            List of output tensors
        """
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT Python API not available. Install tensorrt and pycuda for runtime inference.")
        
        # Create engine key with dimensions
        engine_key = f"{model_name}_b{batch_size}_h{height}_w{width}"
        
        if engine_key not in self.engines:
            raise RuntimeError(f"Model {model_name} not optimized for dimensions (batch={batch_size}, h={height}, w={width}). Call optimize_model first.")
        
        engine = self.engines[engine_key]
        context = self.contexts[engine_key]
        
        # Validate and copy inputs to GPU
        for i, inp in enumerate(inputs):
            if i >= len(self.cuda_inputs[engine_key]):
                raise RuntimeError(f"Too many inputs provided for model {model_name}. Expected {len(self.cuda_inputs[engine_key])}, got {len(inputs)}")
            
            # Convert input to numpy with proper dtype matching
            input_tensor = inp.detach().cpu()
            
            # Get expected dtype from engine
            if hasattr(engine, 'get_tensor_name'):
                binding_name = engine.get_tensor_name(i)
                expected_dtype_trt = engine.get_tensor_dtype(binding_name)
            else:
                expected_dtype_trt = engine.get_binding_dtype(i)
            
            expected_dtype_np = trt.nptype(expected_dtype_trt)
            
            # Convert tensor to expected dtype if needed
            if input_tensor.dtype != torch.from_numpy(np.array([], dtype=expected_dtype_np)).dtype:
                if self.verbose:
                    print(f"Converting input {i} from {input_tensor.dtype} to {expected_dtype_np}")
                if expected_dtype_np == np.float32:
                    input_tensor = input_tensor.float()
                elif expected_dtype_np == np.float16:
                    input_tensor = input_tensor.half()
                elif expected_dtype_np == np.int32:
                    input_tensor = input_tensor.int()
            
            # Convert to numpy and ensure contiguous memory layout
            input_np = np.ascontiguousarray(input_tensor.numpy())
            
            # Calculate actual size
            input_size = input_np.nbytes
            
            # Get allocated memory info for debugging
            if hasattr(engine, 'get_tensor_name'):
                binding_name = engine.get_tensor_name(i)
                expected_shape = engine.get_tensor_shape(binding_name)
                expected_dtype = trt.nptype(engine.get_tensor_dtype(binding_name))
            else:
                expected_shape = engine.get_binding_shape(i)
                expected_dtype = trt.nptype(engine.get_binding_dtype(i))
            
            expected_size = np.prod(expected_shape) * np.dtype(expected_dtype).itemsize
            
            # Check if input size exceeds allocated memory
            if -1 in expected_shape:
                # For dynamic shapes, we need to be more careful
                max_allocated_size = 16 * np.prod([d for d in expected_shape if d != -1]) * np.dtype(expected_dtype).itemsize
                if input_size > max_allocated_size:
                    raise RuntimeError(f"Input size {input_size} exceeds allocated memory {max_allocated_size} for {model_name} input {i}")
            elif input_size != expected_size:
                if self.verbose:
                    print(f"Warning: Input size mismatch for {model_name} input {i}")
                    print(f"  Expected shape: {expected_shape}, dtype: {expected_dtype}")
                    print(f"  Actual shape: {input_np.shape}, dtype: {input_np.dtype}")
                    print(f"  Expected size: {expected_size} bytes, Actual size: {input_size} bytes")
                
                # If size mismatch is significant, it's an error
                if abs(input_size - expected_size) > expected_size * 0.1:  # 10% tolerance
                    raise RuntimeError(f"Input size mismatch too large for {model_name} input {i}")
            
            # Set input shape for dynamic shapes (newer TensorRT API)
            if hasattr(context, 'set_input_shape') and -1 in expected_shape:
                try:
                    context.set_input_shape(binding_name, input_np.shape)
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: Could not set input shape for {binding_name}: {e}")
            elif hasattr(context, 'set_binding_shape') and -1 in expected_shape:
                try:
                    context.set_binding_shape(i, input_np.shape)
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: Could not set binding shape for binding {i}: {e}")
            
            # Copy to GPU with error handling
            try:
                cuda.memcpy_htod_async(
                    self.cuda_inputs[engine_key][i],
                    input_np,
                    self.stream
                )
            except Exception as e:
                raise RuntimeError(f"Failed to copy input {i} to GPU for model {model_name}: {e}. Input shape: {input_np.shape}, size: {input_size} bytes")
            
        
        # Execute TensorRT inference with proper API detection
        try:
            # Try the newest TensorRT 10.x API first
            if hasattr(context, 'execute_async_v3'):
                # Set tensor addresses for TensorRT 10.x
                if hasattr(context, 'set_tensor_address') and hasattr(engine, 'get_tensor_name'):
                    # Set ALL tensor addresses (inputs and outputs) using set_tensor_address
                    for i, inp in enumerate(inputs):
                        binding_name = engine.get_tensor_name(i)
                        context.set_tensor_address(binding_name, int(self.cuda_inputs[engine_key][i]))
                    
                    # Set output tensor addresses
                    for i, cuda_output in enumerate(self.cuda_outputs[engine_key]):
                        output_idx = len(self.cuda_inputs[engine_key]) + i
                        if output_idx < engine.num_io_tensors:
                            output_name = engine.get_tensor_name(output_idx)
                            context.set_tensor_address(output_name, int(cuda_output))
                elif hasattr(context, 'set_input_tensor_address') and hasattr(engine, 'get_tensor_name'):
                    # Fallback to separate input/output address setting
                    for i, inp in enumerate(inputs):
                        binding_name = engine.get_tensor_name(i)
                        context.set_input_tensor_address(binding_name, int(self.cuda_inputs[engine_key][i]))
                    
                    # Set output tensor addresses
                    for i, cuda_output in enumerate(self.cuda_outputs[engine_key]):
                        output_idx = len(self.cuda_inputs[engine_key]) + i
                        if output_idx < engine.num_io_tensors:
                            output_name = engine.get_tensor_name(output_idx)
                            context.set_output_tensor_address(output_name, int(cuda_output))
                
                # Execute with TensorRT 10.x API
                success = context.execute_async_v3(stream_handle=self.stream.handle)
                
            elif hasattr(context, 'execute_async_v2'):
                # Use TensorRT 8.x/9.x API
                success = context.execute_async_v2(
                    bindings=self.cuda_bindings[engine_key],
                    stream_handle=self.stream.handle
                )
            elif hasattr(context, 'execute_async'):
                # Fallback to older TensorRT API
                success = context.execute_async(
                    bindings=self.cuda_bindings[engine_key],
                    stream_handle=self.stream.handle
                )
            else:
                # Use synchronous execution as last resort
                if hasattr(context, 'execute_v2'):
                    success = context.execute_v2(bindings=self.cuda_bindings[engine_key])
                else:
                    success = context.execute(bindings=self.cuda_bindings[engine_key])
                # Synchronize manually since we used sync execution
                if self.stream:
                    self.stream.synchronize()
            
            if not success:
                raise RuntimeError(f"TensorRT inference execution failed for {model_name}")
                
        except Exception as e:
            raise RuntimeError(f"TensorRT inference failed for {model_name}: {e}")
        
        # Copy outputs back to CPU
        outputs = []
        for i, cuda_output in enumerate(self.cuda_outputs[engine_key]):
            # Get output shape from engine using newer API
            if hasattr(engine, 'get_tensor_name'):
                # Find the output tensor name
                output_idx = len(self.cuda_inputs[engine_key]) + i
                if output_idx < engine.num_io_tensors:
                    output_name = engine.get_tensor_name(output_idx)
                    output_shape = engine.get_tensor_shape(output_name)
                    output_dtype = trt.nptype(engine.get_tensor_dtype(output_name))
                else:
                    # Fallback for edge cases
                    output_shape = (1,)
                    output_dtype = np.float32
            else:
                # Use older API
                output_shape = engine.get_binding_shape(len(self.cuda_inputs[engine_key]) + i)
                output_dtype = trt.nptype(engine.get_binding_dtype(len(self.cuda_inputs[engine_key]) + i))
            
            # Allocate host memory
            host_output = np.empty(output_shape, dtype=output_dtype)
            
            # Copy from GPU to CPU
            cuda.memcpy_dtoh_async(host_output, cuda_output, self.stream)
            
            # Convert to torch tensor
            output_tensor = torch.from_numpy(host_output)
            if inputs[0].device.type == 'cuda':
                output_tensor = output_tensor.cuda()
            
            outputs.append(output_tensor)
        
        # Synchronize stream
        self.stream.synchronize()
        
        return outputs
    
    def __del__(self):
        """Cleanup GPU memory."""
        if hasattr(self, 'stream') and self.stream:
            self.stream.synchronize()


class TensorRTStableDiffusionMixin:
    """
    Mixin class to add TensorRT optimization capabilities to Stable Diffusion pipelines.
    
    Uses Polygraphy CLI for model conversion, providing better flexibility and reduced complexity
    compared to direct TensorRT Python API usage.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tensorrt_optimizer = None
        self.tensorrt_enabled = False
    
    def _extract_shapes_and_dtypes(self, tensors):
        """
        Extract shapes and dtypes from tensors (single tensor or tuple/list of tensors).
        
        Args:
            tensors: Single tensor or tuple/list of tensors
            
        Returns:
            Tuple of (shapes, dtypes) where each is a list
        """
        if isinstance(tensors, torch.Tensor):
            return [tuple(tensors.shape)], [tensors.dtype]
        elif isinstance(tensors, (list, tuple)):
            shapes = []
            dtypes = []
            for tensor in tensors:
                if hasattr(tensor, 'shape') and hasattr(tensor, 'dtype'):
                    shapes.append(tuple(tensor.shape))
                    dtypes.append(tensor.dtype)
                elif hasattr(tensor, 'last_hidden_state'):  # Handle text encoder output
                    shapes.append(tuple(tensor.last_hidden_state.shape))
                    dtypes.append(tensor.last_hidden_state.dtype)
                    if hasattr(tensor, 'pooler_output') and tensor.pooler_output is not None:
                        shapes.append(tuple(tensor.pooler_output.shape))
                        dtypes.append(tensor.pooler_output.dtype)
                elif hasattr(tensor, 'sample'):  # Handle UNet output
                    shapes.append(tuple(tensor.sample.shape))
                    dtypes.append(tensor.sample.dtype)
            return shapes, dtypes
        else:
            # Handle individual model outputs that have special attributes
            if hasattr(tensors, 'last_hidden_state'):  # Text encoder output
                shapes = [tuple(tensors.last_hidden_state.shape)]
                dtypes = [tensors.last_hidden_state.dtype]
                if hasattr(tensors, 'pooler_output') and tensors.pooler_output is not None:
                    shapes.append(tuple(tensors.pooler_output.shape))
                    dtypes.append(tensors.pooler_output.dtype)
                return shapes, dtypes
            elif hasattr(tensors, 'sample'):  # UNet output
                return [tuple(tensors.sample.shape)], [tensors.sample.dtype]
            elif hasattr(tensors, 'shape') and hasattr(tensors, 'dtype'):  # Regular tensor
                return [tuple(tensors.shape)], [tensors.dtype]
            else:
                return [], []

    def enable_tensorrt_optimization(
        self,
        cache_dir: str = "./tensorrt_cache",
        onnx_cache_dir: Optional[str] = None,
        fp16: bool = True,
        max_workspace_size: int = 1 << 30,
        verbose: bool = False,
        batch_size: int = 1,
        height: int = 512,
        width: int = 512
    ):
        """
        Enable TensorRT optimization for the pipeline using Polygraphy CLI.
        
        Args:
            cache_dir: Directory to store TensorRT engine files
            onnx_cache_dir: Directory to store ONNX files (defaults to cache_dir if None)
            fp16: Whether to use FP16 precision
            max_workspace_size: Maximum workspace size for TensorRT
            verbose: Whether to print verbose logs
            batch_size: Batch size for optimization (default: 1)
            height: Image height for optimization (default: 512, must be 128-1024 and multiple of 16)
            width: Image width for optimization (default: 512, must be 128-1024 and multiple of 16)
        """
        try:
            self.tensorrt_optimizer = TensorRTOptimizer(
                cache_dir=cache_dir,
                onnx_cache_dir=onnx_cache_dir,
                fp16=fp16,
                max_workspace_size=max_workspace_size,
                verbose=verbose
            )
            # Validate dimensions early
            self.tensorrt_optimizer._validate_dimensions(height, width)
        except ImportError as e:
            print(f"Failed to initialize TensorRT optimizer: {e}")
            return
        except ValueError as e:
            print(f"Invalid dimensions for TensorRT optimization: {e}")
            return
        
        # Store original model methods for fallback
        self._original_text_encoder_forward = getattr(self.text_encoder, 'forward', None) if hasattr(self, 'text_encoder') else None
        self._original_unet_forward = getattr(self.unet, 'forward', None) if hasattr(self, 'unet') else None
        self._original_vae_decode = getattr(self.vae, 'decode', None) if hasattr(self, 'vae') else None
        
        # Store optimization dimensions
        self.tensorrt_batch_size = batch_size
        self.tensorrt_height = height
        self.tensorrt_width = width
        
        # Optimize models with specified dimensions
        self._optimize_text_encoder(batch_size, height, width)
        self._optimize_unet(batch_size, height, width)
        self._optimize_vae(batch_size, height, width)
        
        self.tensorrt_enabled = True
    
    def _optimize_text_encoder(self, batch_size: int = 1, height: int = 512, width: int = 512):
        """Optimize text encoder model."""
        if not hasattr(self, 'text_encoder') or self.text_encoder is None:
            return
        
        # Perform forward pass to get actual input/output shapes
        if self.tensorrt_optimizer.verbose:
            print("Performing forward pass to determine text encoder shapes...")
        
        # Use a representative input (similar to what would be used in practice)
        # Note: Input IDs for text encoder are always int32, regardless of model dtype
        sample_input = torch.randint(0, 1000, (batch_size, 77), dtype=torch.int32)
        if torch.cuda.is_available():
            sample_input = sample_input.cuda()
        
        # Perform forward pass with original model
        self.text_encoder.eval()
        with torch.no_grad():
            sample_output = self.text_encoder(sample_input)
        
        # Extract actual shapes from forward pass
        input_shapes, _ = self._extract_shapes_and_dtypes(sample_input)
        output_shapes, _ = self._extract_shapes_and_dtypes(sample_output)
        
        if self.tensorrt_optimizer.verbose:
            print(f"Text encoder input shapes: {input_shapes}")
            print(f"Text encoder output shapes: {output_shapes}")
        
        # Build input_shapes dict from actual shapes
        input_shapes_dict = {"input_ids": input_shapes[0]}
        
        success = self.tensorrt_optimizer.optimize_model(
            model=self.text_encoder,
            model_args=(sample_input,),
            model_name="text_encoder",
            input_names=["input_ids"],
            output_names=["last_hidden_state", "pooler_output"],
            batch_size=batch_size,
            height=height,
            width=width,
            input_shapes=input_shapes_dict,
            dynamic_axes={"input_ids": {0: "batch"}},
        )
        
        if success and self.tensorrt_optimizer.verbose:
            print("Text encoder optimized for TensorRT")
    
    def _optimize_unet(self, batch_size: int = 1, height: int = 512, width: int = 512):
        """Optimize UNet model."""
        if not hasattr(self, 'unet') or self.unet is None:
            return
        
        # Perform forward pass to get actual input/output shapes
        if self.tensorrt_optimizer.verbose:
            print("Performing forward pass to determine UNet shapes...")
        
        # Use representative inputs (similar to what would be used in practice)
        # Match the model's dtype for proper compatibility
        # Calculate latent dimensions (height and width are divided by 8 for latent space)
        model_dtype = next(self.unet.parameters()).dtype
        latent_height = height // 8
        latent_width = width // 8
        sample_latents = torch.randn(batch_size, 4, latent_height, latent_width, dtype=model_dtype)
        sample_timestep = torch.tensor([1.0], dtype=model_dtype)
        sample_encoder_hidden_states = torch.randn(batch_size, 77, 768, dtype=model_dtype)
        
        if torch.cuda.is_available():
            sample_latents = sample_latents.cuda()
            sample_timestep = sample_timestep.cuda()
            sample_encoder_hidden_states = sample_encoder_hidden_states.cuda()
        
        # Perform forward pass with original model
        self.unet.eval()
        with torch.no_grad():
            sample_output = self.unet(sample_latents, sample_timestep, sample_encoder_hidden_states)
        
        # Extract actual shapes from forward pass
        input_shapes, _ = self._extract_shapes_and_dtypes([sample_latents, sample_timestep, sample_encoder_hidden_states])
        output_shapes, _ = self._extract_shapes_and_dtypes(sample_output)
        
        if self.tensorrt_optimizer.verbose:
            print(f"UNet input shapes: {input_shapes}")
            print(f"UNet output shapes: {output_shapes}")
        
        # Build input_shapes dict from actual shapes
        input_shapes_dict = {
            "sample": input_shapes[0],
            "timestep": input_shapes[1],
            "encoder_hidden_states": input_shapes[2]
        }
        
        success = self.tensorrt_optimizer.optimize_model(
            model=self.unet,
            model_args=(sample_latents, sample_timestep, sample_encoder_hidden_states),
            model_name="unet",
            input_names=["sample", "timestep", "encoder_hidden_states"],
            output_names=["noise_pred"],
            batch_size=batch_size,
            height=height,
            width=width,
            input_shapes=input_shapes_dict,
            dynamic_axes={
                "sample": {0: "batch", 2: "height", 3: "width"},
                "encoder_hidden_states": {0: "batch"}
            },
        )
        
        if success and self.tensorrt_optimizer.verbose:
            print("UNet optimized for TensorRT")
    
    def _optimize_vae(self, batch_size: int = 1, height: int = 512, width: int = 512):
        """Optimize VAE decoder."""
        if not hasattr(self, 'vae') or self.vae is None:
            return
        
        # Perform forward pass to get actual input/output shapes
        if self.tensorrt_optimizer.verbose:
            print("Performing forward pass to determine VAE shapes...")
        
        # Use representative inputs for VAE decoder
        # Match the model's dtype for proper compatibility
        # Calculate latent dimensions (height and width are divided by 8 for latent space)
        model_dtype = next(self.vae.parameters()).dtype
        latent_height = height // 8
        latent_width = width // 8
        sample_latents = torch.randn(batch_size, 4, latent_height, latent_width, dtype=model_dtype)
        if torch.cuda.is_available():
            sample_latents = sample_latents.cuda()
        
        # Create a wrapper for VAE decoder only
        class VAEDecoder(torch.nn.Module):
            def __init__(self, vae):
                super().__init__()
                self.vae = vae
            
            def forward(self, latents):
                return self.vae.decode(latents, return_dict=False)[0]
        
        vae_decoder = VAEDecoder(self.vae)
        
        # Perform forward pass with original model
        vae_decoder.eval()
        with torch.no_grad():
            sample_output = vae_decoder(sample_latents)
        
        # Extract actual shapes from forward pass
        input_shapes, _ = self._extract_shapes_and_dtypes(sample_latents)
        output_shapes, _ = self._extract_shapes_and_dtypes(sample_output)
        
        if self.tensorrt_optimizer.verbose:
            print(f"VAE input shapes: {input_shapes}")
            print(f"VAE output shapes: {output_shapes}")
        
        # Build input_shapes dict from actual shapes
        input_shapes_dict = {"latents": input_shapes[0]}
        
        success = self.tensorrt_optimizer.optimize_model(
            model=vae_decoder,
            model_args=(sample_latents,),
            model_name="vae_decoder",
            input_names=["latents"],
            output_names=["image"],
            batch_size=batch_size,
            height=height,
            width=width,
            input_shapes=input_shapes_dict,
            dynamic_axes={
                "latents": {0: "batch", 2: "height", 3: "width"}
            },
        )
        
        if success and self.tensorrt_optimizer.verbose:
            print("VAE decoder optimized for TensorRT")
    
    def _tensorrt_text_encoder_call(self, input_ids, **kwargs):
        """TensorRT-optimized text encoder call with fallback."""
        # Get dimensions for engine lookup
        batch_size = input_ids.shape[0] if input_ids.dim() > 0 else self.tensorrt_batch_size
        height = getattr(self, 'tensorrt_height', 512)
        width = getattr(self, 'tensorrt_width', 512)
        engine_key = f"text_encoder_b{batch_size}_h{height}_w{width}"
        
        if self.tensorrt_enabled and engine_key in self.tensorrt_optimizer.engines and not getattr(self, '_in_fallback', False):
            try:
                outputs = self.tensorrt_optimizer.infer_tensorrt("text_encoder", [input_ids], batch_size, height, width)
                return type('TextEncoderOutput', (), {
                    'last_hidden_state': outputs[0],
                    'pooler_output': outputs[1] if len(outputs) > 1 else None
                })()
            except Exception as e:
                if self.tensorrt_optimizer.verbose:
                    print(f"TensorRT text encoder failed, falling back to PyTorch: {e}")
                # Set fallback flag to prevent recursion
                self._in_fallback = True
                try:
                    # Use original forward method to avoid recursion
                    if self._original_text_encoder_forward:
                        result = self._original_text_encoder_forward(input_ids, **kwargs)
                    else:
                        result = self.text_encoder(input_ids, **kwargs)
                finally:
                    self._in_fallback = False
                return result
        else:
            if self._original_text_encoder_forward:
                return self._original_text_encoder_forward(input_ids, **kwargs)
            else:
                return self.text_encoder(input_ids, **kwargs)
    
    def _tensorrt_unet_call(self, sample, timestep, encoder_hidden_states, **kwargs):
        """TensorRT-optimized UNet call with fallback."""
        # Get dimensions for engine lookup
        batch_size = sample.shape[0] if sample.dim() > 0 else self.tensorrt_batch_size
        # Calculate actual image dimensions from latent dimensions (multiply by 8)
        height = sample.shape[2] * 8 if sample.dim() > 2 else getattr(self, 'tensorrt_height', 512)
        width = sample.shape[3] * 8 if sample.dim() > 3 else getattr(self, 'tensorrt_width', 512)
        engine_key = f"unet_b{batch_size}_h{height}_w{width}"
        
        if self.tensorrt_enabled and engine_key in self.tensorrt_optimizer.engines and not getattr(self, '_in_fallback', False):
            try:
                outputs = self.tensorrt_optimizer.infer_tensorrt(
                    "unet", [sample, timestep, encoder_hidden_states], batch_size, height, width
                )
                return (outputs[0],)
            except Exception as e:
                if self.tensorrt_optimizer.verbose:
                    print(f"TensorRT UNet failed, falling back to PyTorch: {e}")
                # Set fallback flag to prevent recursion
                self._in_fallback = True
                try:
                    # Use original forward method to avoid recursion
                    if self._original_unet_forward:
                        result = self._original_unet_forward(sample, timestep, encoder_hidden_states=encoder_hidden_states, **kwargs)
                    else:
                        result = self.unet(sample, timestep, encoder_hidden_states=encoder_hidden_states, **kwargs)
                finally:
                    self._in_fallback = False
                return result
        else:
            if self._original_unet_forward:
                return self._original_unet_forward(sample, timestep, encoder_hidden_states=encoder_hidden_states, **kwargs)
            else:
                return self.unet(sample, timestep, encoder_hidden_states=encoder_hidden_states, **kwargs)
    
    def _tensorrt_vae_decode_call(self, latents, **kwargs):
        """TensorRT-optimized VAE decode call with fallback."""
        # Get dimensions for engine lookup
        batch_size = latents.shape[0] if latents.dim() > 0 else self.tensorrt_batch_size
        # Calculate actual image dimensions from latent dimensions (multiply by 8)
        height = latents.shape[2] * 8 if latents.dim() > 2 else getattr(self, 'tensorrt_height', 512)
        width = latents.shape[3] * 8 if latents.dim() > 3 else getattr(self, 'tensorrt_width', 512)
        engine_key = f"vae_decoder_b{batch_size}_h{height}_w{width}"
        
        if self.tensorrt_enabled and engine_key in self.tensorrt_optimizer.engines and not getattr(self, '_in_fallback', False):
            try:
                outputs = self.tensorrt_optimizer.infer_tensorrt("vae_decoder", [latents], batch_size, height, width)
                return (outputs[0],)
            except Exception as e:
                if self.tensorrt_optimizer.verbose:
                    print(f"TensorRT VAE decoder failed, falling back to PyTorch: {e}")
                # Set fallback flag to prevent recursion
                self._in_fallback = True
                try:
                    # Use original method to avoid recursion
                    if self._original_vae_decode:
                        result = self._original_vae_decode(latents, **kwargs)
                    else:
                        result = self.vae.decode(latents, **kwargs)
                finally:
                    self._in_fallback = False
                return result
        else:
            if self._original_vae_decode:
                return self._original_vae_decode(latents, **kwargs)
            else:
                return self.vae.decode(latents, **kwargs)


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusionTensorRTPipeline

        >>> pipe = StableDiffusionTensorRTPipeline.from_pretrained(
        ...     "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")
        >>> pipe.enable_tensorrt_optimization()

        >>> prompt = "a photo of an astronaut riding a horse on mars"
        >>> image = pipe(prompt).images[0]
        ```
"""


class StableDiffusionTensorRTPipeline(TensorRTStableDiffusionMixin, StableDiffusionPipeline):
    """
    Pipeline for text-to-image generation using Stable Diffusion with TensorRT optimization.

    This pipeline extends the standard StableDiffusionPipeline with TensorRT acceleration capabilities.
    It automatically converts PyTorch models to ONNX and then to TensorRT engines for faster inference.

    The pipeline inherits all the functionality from StableDiffusionPipeline and adds:
        - Automatic model conversion to TensorRT
        - Intelligent caching of converted models
        - Seamless fallback to PyTorch if TensorRT is not available
        - Optimized inference paths for text encoder, UNet, and VAE
        - Variable batch, height, and width support with dimension-specific engines
        - Forward pass validation to ensure model compatibility
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            requires_safety_checker=requires_safety_checker,
        )

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        """
        Override _encode_prompt to use TensorRT-optimized text encoder calls.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            if clip_skip is None:
                # Use TensorRT-optimized text encoder call
                if self.tensorrt_enabled:
                    prompt_embeds = self._tensorrt_text_encoder_call(text_input_ids.to(device), attention_mask=attention_mask)
                else:
                    prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
                prompt_embeds = prompt_embeds.last_hidden_state
            else:
                # Fallback to original implementation for clip_skip
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
                )
                prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            # Use TensorRT-optimized text encoder call for negative prompt
            if self.tensorrt_enabled:
                negative_prompt_embeds = self._tensorrt_text_encoder_call(
                    uncond_input.input_ids.to(device),
                    attention_mask=attention_mask,
                )
            else:
                negative_prompt_embeds = self.text_encoder(
                    uncond_input.input_ids.to(device),
                    attention_mask=attention_mask,
                )
            negative_prompt_embeds = negative_prompt_embeds.last_hidden_state

        if do_classifier_free_guidance:
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds

    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation with TensorRT optimization.

        This method extends the base StableDiffusionPipeline.__call__ with TensorRT acceleration.
        All arguments are the same as the base pipeline.

        Examples:

        """
        # Store original forward methods to restore them later if needed
        original_unet_forward = None
        original_vae_decode = None

        try:
            # Replace UNet forward with TensorRT-optimized version if available
            if self.tensorrt_enabled and hasattr(self.tensorrt_optimizer, 'engines'):
                # Check if we have a UNet engine for current dimensions
                if height is not None and width is not None:
                    batch_size = num_images_per_prompt
                    if prompt is not None:
                        if isinstance(prompt, list):
                            batch_size *= len(prompt)
                        else:
                            batch_size *= 1
                    
                    engine_key = f"unet_b{batch_size}_h{height}_w{width}"
                    if engine_key in self.tensorrt_optimizer.engines:
                        original_unet_forward = self.unet.forward
                        
                        def tensorrt_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
                            return self._tensorrt_unet_call(sample, timestep, encoder_hidden_states, **kwargs)
                        
                        self.unet.forward = tensorrt_unet_forward

            # Replace VAE decode with TensorRT-optimized version if available
            if self.tensorrt_enabled and hasattr(self.tensorrt_optimizer, 'engines'):
                # Check if we have a VAE engine for current dimensions
                if height is not None and width is not None:
                    batch_size = num_images_per_prompt
                    if prompt is not None:
                        if isinstance(prompt, list):
                            batch_size *= len(prompt)
                        else:
                            batch_size *= 1
                    
                    engine_key = f"vae_decoder_b{batch_size}_h{height}_w{width}"
                    if engine_key in self.tensorrt_optimizer.engines:
                        original_vae_decode = self.vae.decode
                        
                        def tensorrt_vae_decode(latents, **kwargs):
                            return self._tensorrt_vae_decode_call(latents, **kwargs)
                        
                        self.vae.decode = tensorrt_vae_decode

            # Call the parent's __call__ method with optimized models
            return super().__call__(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                timesteps=timesteps,
                sigmas=sigmas,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
                eta=eta,
                generator=generator,
                latents=latents,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                ip_adapter_image=ip_adapter_image,
                ip_adapter_image_embeds=ip_adapter_image_embeds,
                output_type=output_type,
                return_dict=return_dict,
                cross_attention_kwargs=cross_attention_kwargs,
                guidance_rescale=guidance_rescale,
                clip_skip=clip_skip,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                **kwargs,
            )

        finally:
            # Restore original forward methods
            if original_unet_forward is not None:
                self.unet.forward = original_unet_forward
            if original_vae_decode is not None:
                self.vae.decode = original_vae_decode