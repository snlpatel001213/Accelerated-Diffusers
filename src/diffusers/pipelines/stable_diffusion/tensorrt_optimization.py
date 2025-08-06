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
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import numpy as np

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
        
    def _get_model_paths(self, model_name: str) -> Tuple[Path, Path]:
        """Get ONNX and TensorRT engine file paths for a model."""
        onnx_path = self.onnx_cache_dir / f"{model_name}.onnx"
        engine_path = self.cache_dir / f"{model_name}.trt"
        return onnx_path, engine_path
    
    def _check_files_exist(self, model_name: str) -> Tuple[bool, bool]:
        """Check if ONNX and TensorRT engine files exist."""
        onnx_path, engine_path = self._get_model_paths(model_name)
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
    
    def _convert_to_tensorrt(self, onnx_path: Path, model_name: str, input_shapes: Dict[str, Tuple]) -> Path:
        """Convert ONNX model to TensorRT engine using Polygraphy CLI."""
        _, engine_path = self._get_model_paths(model_name)
        
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
    
    def _load_engine(self, model_name: str) -> None:
        """Load TensorRT engine and create execution context."""
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT Python API not available. Install tensorrt and pycuda for runtime inference.")
        
        _, engine_path = self._get_model_paths(model_name)
        
        if not engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")
        
        # Load engine
        runtime = trt.Runtime(self.trt_logger)
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        engine = runtime.deserialize_cuda_engine(engine_data)
        context = engine.create_execution_context()
        
        # Store engine and context
        self.engines[model_name] = engine
        self.contexts[model_name] = context
        
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
        
        self.cuda_inputs[model_name] = inputs
        self.cuda_outputs[model_name] = outputs
        self.cuda_bindings[model_name] = bindings
        
        if self.verbose:
            print(f"Loaded TensorRT engine for {model_name}")
    
    def optimize_model(
        self,
        model: torch.nn.Module,
        model_args: Tuple,
        model_name: str,
        input_names: List[str],
        output_names: List[str],
        input_shapes: Optional[Dict[str, Tuple]] = None,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        opset_version: int = 14
    ) -> bool:
        """
        Optimize a PyTorch model for TensorRT inference.
        
        Smart caching logic:
        - If TensorRT engine exists: Load directly
        - If ONNX exists but TensorRT doesn't: Create TensorRT from ONNX
        - If neither exists: Convert PyTorch → ONNX → TensorRT
        
        Returns:
            bool: True if optimization successful, False otherwise
        """
        try:
            onnx_exists, engine_exists = self._check_files_exist(model_name)
            onnx_path, engine_path = self._get_model_paths(model_name)
            
            if self.verbose:
                print(f"Optimizing {model_name}:")
                print(f"  ONNX file exists: {onnx_exists} ({onnx_path})")
                print(f"  TensorRT engine exists: {engine_exists} ({engine_path})")
            
            # Case 1: TensorRT engine exists - load directly (fastest path)
            if engine_exists:
                if self.verbose:
                    print(f"✅ Loading existing TensorRT engine: {engine_path}")
                self._load_engine(model_name)
                return True
            
            # Case 2: ONNX exists but TensorRT doesn't - create TensorRT from ONNX
            elif onnx_exists:
                if self.verbose:
                    print(f"🔄 Converting existing ONNX to TensorRT: {onnx_path} → {engine_path}")
                self._convert_to_tensorrt(onnx_path, model_name, input_shapes or {})
                self._load_engine(model_name)
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
                self._convert_to_tensorrt(onnx_path, model_name, input_shapes or {})
                # Load engine
                self._load_engine(model_name)
                return True
            
        except Exception as e:
            print(f"❌ Failed to optimize {model_name}: {e}")
            return False
    
    def infer_tensorrt(self, model_name: str, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Perform TensorRT inference instead of PyTorch forward call.
        
        Args:
            model_name: Name of the optimized model
            inputs: List of input tensors
            
        Returns:
            List of output tensors
        """
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT Python API not available. Install tensorrt and pycuda for runtime inference.")
        
        if model_name not in self.engines:
            raise RuntimeError(f"Model {model_name} not optimized. Call optimize_model first.")
        
        engine = self.engines[model_name]
        context = self.contexts[model_name]
        
        # Validate and copy inputs to GPU
        for i, inp in enumerate(inputs):
            if i >= len(self.cuda_inputs[model_name]):
                raise RuntimeError(f"Too many inputs provided for model {model_name}. Expected {len(self.cuda_inputs[model_name])}, got {len(inputs)}")
            
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
                    self.cuda_inputs[model_name][i],
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
                        context.set_tensor_address(binding_name, int(self.cuda_inputs[model_name][i]))
                    
                    # Set output tensor addresses
                    for i, cuda_output in enumerate(self.cuda_outputs[model_name]):
                        output_idx = len(self.cuda_inputs[model_name]) + i
                        if output_idx < engine.num_io_tensors:
                            output_name = engine.get_tensor_name(output_idx)
                            context.set_tensor_address(output_name, int(cuda_output))
                elif hasattr(context, 'set_input_tensor_address') and hasattr(engine, 'get_tensor_name'):
                    # Fallback to separate input/output address setting
                    for i, inp in enumerate(inputs):
                        binding_name = engine.get_tensor_name(i)
                        context.set_input_tensor_address(binding_name, int(self.cuda_inputs[model_name][i]))
                    
                    # Set output tensor addresses
                    for i, cuda_output in enumerate(self.cuda_outputs[model_name]):
                        output_idx = len(self.cuda_inputs[model_name]) + i
                        if output_idx < engine.num_io_tensors:
                            output_name = engine.get_tensor_name(output_idx)
                            context.set_output_tensor_address(output_name, int(cuda_output))
                
                # Execute with TensorRT 10.x API
                success = context.execute_async_v3(stream_handle=self.stream.handle)
                
            elif hasattr(context, 'execute_async_v2'):
                # Use TensorRT 8.x/9.x API
                success = context.execute_async_v2(
                    bindings=self.cuda_bindings[model_name],
                    stream_handle=self.stream.handle
                )
            elif hasattr(context, 'execute_async'):
                # Fallback to older TensorRT API
                success = context.execute_async(
                    bindings=self.cuda_bindings[model_name],
                    stream_handle=self.stream.handle
                )
            else:
                # Use synchronous execution as last resort
                if hasattr(context, 'execute_v2'):
                    success = context.execute_v2(bindings=self.cuda_bindings[model_name])
                else:
                    success = context.execute(bindings=self.cuda_bindings[model_name])
                # Synchronize manually since we used sync execution
                if self.stream:
                    self.stream.synchronize()
            
            if not success:
                raise RuntimeError(f"TensorRT inference execution failed for {model_name}")
                
        except Exception as e:
            raise RuntimeError(f"TensorRT inference failed for {model_name}: {e}")
        
        # Copy outputs back to CPU
        outputs = []
        for i, cuda_output in enumerate(self.cuda_outputs[model_name]):
            # Get output shape from engine using newer API
            if hasattr(engine, 'get_tensor_name'):
                # Find the output tensor name
                output_idx = len(self.cuda_inputs[model_name]) + i
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
                output_shape = engine.get_binding_shape(len(self.cuda_inputs[model_name]) + i)
                output_dtype = trt.nptype(engine.get_binding_dtype(len(self.cuda_inputs[model_name]) + i))
            
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
    
    def enable_tensorrt_optimization(
        self,
        cache_dir: str = "./tensorrt_cache",
        onnx_cache_dir: Optional[str] = None,
        fp16: bool = True,
        max_workspace_size: int = 1 << 30,
        verbose: bool = False
    ):
        """
        Enable TensorRT optimization for the pipeline using Polygraphy CLI.
        
        Args:
            cache_dir: Directory to store TensorRT engine files
            onnx_cache_dir: Directory to store ONNX files (defaults to cache_dir if None)
            fp16: Whether to use FP16 precision
            max_workspace_size: Maximum workspace size for TensorRT
            verbose: Whether to print verbose logs
        """
        try:
            self.tensorrt_optimizer = TensorRTOptimizer(
                cache_dir=cache_dir,
                onnx_cache_dir=onnx_cache_dir,
                fp16=fp16,
                max_workspace_size=max_workspace_size,
                verbose=verbose
            )
        except ImportError as e:
            print(f"Failed to initialize TensorRT optimizer: {e}")
            return
        
        # Store original model methods for fallback
        self._original_text_encoder_forward = getattr(self.text_encoder, 'forward', None) if hasattr(self, 'text_encoder') else None
        self._original_unet_forward = getattr(self.unet, 'forward', None) if hasattr(self, 'unet') else None
        self._original_vae_decode = getattr(self.vae, 'decode', None) if hasattr(self, 'vae') else None
        
        # Optimize models
        self._optimize_text_encoder()
        self._optimize_unet()
        self._optimize_vae()
        
        self.tensorrt_enabled = True
    
    def _optimize_text_encoder(self):
        """Optimize text encoder model."""
        if not hasattr(self, 'text_encoder') or self.text_encoder is None:
            return
        
        # Prepare sample inputs
        sample_input = torch.randint(0, 1000, (1, 77), dtype=torch.int32)
        if torch.cuda.is_available():
            sample_input = sample_input.cuda()
        
        success = self.tensorrt_optimizer.optimize_model(
            model=self.text_encoder,
            model_args=(sample_input,),
            model_name="text_encoder",
            input_names=["input_ids"],
            output_names=["last_hidden_state", "pooler_output"],
            input_shapes={"input_ids": (1, 77)},
            dynamic_axes={"input_ids": {0: "batch"}},
        )
        
        if success and self.tensorrt_optimizer.verbose:
            print("Text encoder optimized for TensorRT")
    
    def _optimize_unet(self):
        """Optimize UNet model."""
        if not hasattr(self, 'unet') or self.unet is None:
            return
        
        # Prepare sample inputs
        sample_latents = torch.randn(2, 4, 64, 64)
        sample_timestep = torch.tensor([1.0])
        sample_encoder_hidden_states = torch.randn(2, 77, 768)
        
        if torch.cuda.is_available():
            sample_latents = sample_latents.cuda()
            sample_timestep = sample_timestep.cuda()
            sample_encoder_hidden_states = sample_encoder_hidden_states.cuda()
        
        success = self.tensorrt_optimizer.optimize_model(
            model=self.unet,
            model_args=(sample_latents, sample_timestep, sample_encoder_hidden_states),
            model_name="unet",
            input_names=["sample", "timestep", "encoder_hidden_states"],
            output_names=["noise_pred"],
            input_shapes={
                "sample": (2, 4, 64, 64),
                "timestep": (1,),
                "encoder_hidden_states": (2, 77, 768)
            },
            dynamic_axes={
                "sample": {0: "batch", 2: "height", 3: "width"},
                "encoder_hidden_states": {0: "batch"}
            },
        )
        
        if success and self.tensorrt_optimizer.verbose:
            print("UNet optimized for TensorRT")
    
    def _optimize_vae(self):
        """Optimize VAE decoder."""
        if not hasattr(self, 'vae') or self.vae is None:
            return
        
        # Prepare sample inputs for VAE decoder
        sample_latents = torch.randn(1, 4, 64, 64)
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
        
        success = self.tensorrt_optimizer.optimize_model(
            model=vae_decoder,
            model_args=(sample_latents,),
            model_name="vae_decoder",
            input_names=["latents"],
            output_names=["image"],
            input_shapes={"latents": (1, 4, 64, 64)},
            dynamic_axes={
                "latents": {0: "batch", 2: "height", 3: "width"}
            },
        )
        
        if success and self.tensorrt_optimizer.verbose:
            print("VAE decoder optimized for TensorRT")
    
    def _tensorrt_text_encoder_call(self, input_ids, **kwargs):
        """TensorRT-optimized text encoder call with fallback."""
        if self.tensorrt_enabled and "text_encoder" in self.tensorrt_optimizer.engines and not getattr(self, '_in_fallback', False):
            try:
                outputs = self.tensorrt_optimizer.infer_tensorrt("text_encoder", [input_ids])
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
            return self.text_encoder(input_ids, **kwargs)
    
    def _tensorrt_unet_call(self, sample, timestep, encoder_hidden_states, **kwargs):
        """TensorRT-optimized UNet call with fallback."""
        if self.tensorrt_enabled and "unet" in self.tensorrt_optimizer.engines and not getattr(self, '_in_fallback', False):
            try:
                outputs = self.tensorrt_optimizer.infer_tensorrt(
                    "unet", [sample, timestep, encoder_hidden_states]
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
            return self.unet(sample, timestep, encoder_hidden_states=encoder_hidden_states, **kwargs)
    
    def _tensorrt_vae_decode_call(self, latents, **kwargs):
        """TensorRT-optimized VAE decode call with fallback."""
        if self.tensorrt_enabled and "vae_decoder" in self.tensorrt_optimizer.engines and not getattr(self, '_in_fallback', False):
            try:
                outputs = self.tensorrt_optimizer.infer_tensorrt("vae_decoder", [latents])
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
            return self.vae.decode(latents, **kwargs)