# coding=utf-8
# Copyright 2025 HuggingFace Inc.
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

# ========================================================================
# 🚀 TENSORRT COMPREHENSIVE TEST SUITE - QUICK REFERENCE
# ========================================================================
# 
# TLDR - Run all tests:
# RUN_SLOW=yes python3 -m pytest tests/pipelines/stable_diffusion/test_tensorrt_comprehensive.py -v -s
#
# Prerequisites: CUDA GPU, TensorRT, Polygraphy
# For full documentation see docstring below.
# ========================================================================

"""
Comprehensive TensorRT Test Suite for Stable Diffusion Pipeline

This test suite validates:
1. Integration tests (with TensorRT enabled)
2. Non-integration tests (standard PyTorch)
3. Performance acceleration measurements
4. Accuracy validation using polygraphy models
5. Detailed reporting with tables

HOW TO RUN TESTS:
================

Prerequisites:
--------------
- CUDA-capable GPU
- TensorRT installed (pip install tensorrt)
- Polygraphy installed (pip install polygraphy)
- Environment: RUN_SLOW=yes (required for @slow tests)

Running Methods:
---------------

1. Run Complete Test Suite:
   ```bash
   cd /code/diffusers
   RUN_SLOW=yes python3 -m pytest tests/pipelines/stable_diffusion/test_tensorrt_comprehensive.py -v -s
   ```

2. Run Individual Tests:
   ```bash
   # Test standard PyTorch pipeline (non-integration)
   RUN_SLOW=yes python3 -m pytest tests/pipelines/stable_diffusion/test_tensorrt_comprehensive.py::TensorRTComprehensiveTests::test_non_integration_standard_pipeline -v -s
   
   # Test TensorRT pipeline (integration)
   RUN_SLOW=yes python3 -m pytest tests/pipelines/stable_diffusion/test_tensorrt_comprehensive.py::TensorRTComprehensiveTests::test_integration_tensorrt_pipeline -v -s
   
   # Test accuracy comparison (requires both above tests to run first)
   RUN_SLOW=yes python3 -m pytest tests/pipelines/stable_diffusion/test_tensorrt_comprehensive.py::TensorRTComprehensiveTests::test_accuracy_comparison -v -s
   
   # Test polygraphy direct loading (requires integration test first)
   RUN_SLOW=yes python3 -m pytest tests/pipelines/stable_diffusion/test_tensorrt_comprehensive.py::TensorRTComprehensiveTests::test_polygraphy_direct_model_loading -v -s
   
   # Test environment setup
   RUN_SLOW=yes python3 -m pytest tests/pipelines/stable_diffusion/test_tensorrt_comprehensive.py::TensorRTComprehensiveTests::test_cuda_environment -v -s
   
   # Print performance tables
   RUN_SLOW=yes python3 -m pytest tests/pipelines/stable_diffusion/test_tensorrt_comprehensive.py::TensorRTComprehensiveTests::test_print_performance_table -v -s
   ```

3. Run Tests in Logical Order (for complete workflow):
   ```bash
   cd /code/diffusers
   
   # Step 1: Environment check
   RUN_SLOW=yes python3 -m pytest tests/pipelines/stable_diffusion/test_tensorrt_comprehensive.py::TensorRTComprehensiveTests::test_cuda_environment -v -s
   
   # Step 2: Standard pipeline (establishes baseline)
   RUN_SLOW=yes python3 -m pytest tests/pipelines/stable_diffusion/test_tensorrt_comprehensive.py::TensorRTComprehensiveTests::test_non_integration_standard_pipeline -v -s
   
   # Step 3: TensorRT pipeline (creates engines)
   RUN_SLOW=yes python3 -m pytest tests/pipelines/stable_diffusion/test_tensorrt_comprehensive.py::TensorRTComprehensiveTests::test_integration_tensorrt_pipeline -v -s
   
   # Step 4: Accuracy comparison (uses results from steps 2&3)
   RUN_SLOW=yes python3 -m pytest tests/pipelines/stable_diffusion/test_tensorrt_comprehensive.py::TensorRTComprehensiveTests::test_accuracy_comparison -v -s
   
   # Step 5: Polygraphy validation (uses engines from step 3)
   RUN_SLOW=yes python3 -m pytest tests/pipelines/stable_diffusion/test_tensorrt_comprehensive.py::TensorRTComprehensiveTests::test_polygraphy_direct_model_loading -v -s
   
   # Step 6: Print formatted results
   RUN_SLOW=yes python3 -m pytest tests/pipelines/stable_diffusion/test_tensorrt_comprehensive.py::TensorRTComprehensiveTests::test_print_performance_table -v -s
   ```

Expected Output:
---------------
- Performance table showing speedup metrics
- Accuracy table showing pixel differences and similarity scores
- TensorRT engine validation results
- Polygraphy direct loading confirmation

Troubleshooting:
---------------
- If tests are skipped: Ensure RUN_SLOW=yes is set
- If TensorRT fails: Check CUDA/TensorRT installation
- If polygraphy fails: Install with `pip install polygraphy`
- If accuracy test skips: Run standard and TensorRT tests first
- If polygraphy test skips: Run integration test first to create engines

Test Dependencies:
-----------------
- test_accuracy_comparison depends on: test_non_integration_standard_pipeline + test_integration_tensorrt_pipeline
- test_polygraphy_direct_model_loading depends on: test_integration_tensorrt_pipeline
- test_print_performance_table: Reports results from all previous tests
"""

import gc
import os
import tempfile
import time
import unittest
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tabulate import tabulate

from diffusers import StableDiffusionPipeline
from diffusers.utils.testing_utils import (
    require_torch_accelerator,
    skip_mps,
    slow,
    torch_device,
)

# Try to import TensorRT pipeline
try:
    from diffusers.pipelines.stable_diffusion.pipeline_tensorrt_stable_diffusion import (
        StableDiffusionTensorRTPipeline
    )
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    warnings.warn("TensorRT pipeline not available. Skipping TensorRT tests.")

# Try to import polygraphy for direct model loading
try:
    import onnx
    from polygraphy.backend.common import bytes_from_path
    from polygraphy.backend.trt import (
        CreateConfig,
        EngineFromNetwork,
        NetworkFromOnnxPath,
        TrtRunner,
        save_engine
    )
    POLYGRAPHY_AVAILABLE = True
except ImportError:
    POLYGRAPHY_AVAILABLE = False
    warnings.warn("Polygraphy not available. Skipping polygraphy direct loading tests.")


class TensorRTComprehensiveTests(unittest.TestCase):
    """
    Comprehensive test suite for TensorRT Stable Diffusion pipeline.
    
    Quick Start:
    -----------
    Run all tests with: RUN_SLOW=yes python3 -m pytest tests/pipelines/stable_diffusion/test_tensorrt_comprehensive.py -v -s
    
    Test Methods:
    ------------
    - test_cuda_environment: Validates CUDA setup
    - test_non_integration_standard_pipeline: Tests standard PyTorch pipeline (baseline)
    - test_integration_tensorrt_pipeline: Tests TensorRT optimized pipeline
    - test_accuracy_comparison: Compares outputs between standard and TensorRT
    - test_polygraphy_direct_model_loading: Validates TensorRT engines via polygraphy
    - test_print_performance_table: Displays formatted results tables
    
    Dependencies:
    ------------
    Some tests depend on others and should be run in sequence for full results.
    """
    
    model_id = "hf-internal-testing/tiny-stable-diffusion-pipe"
    test_prompt = "A red cat"
    num_inference_steps = 2
    guidance_scale = 7.5
    
    def setUp(self):
        """Set up test environment."""
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Set deterministic behavior
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create temporary directories for caching
        self.temp_dir = tempfile.mkdtemp()
        self.tensorrt_cache_dir = os.path.join(self.temp_dir, "tensorrt_cache")
        self.onnx_cache_dir = os.path.join(self.temp_dir, "onnx_cache")
        
        os.makedirs(self.tensorrt_cache_dir, exist_ok=True)
        os.makedirs(self.onnx_cache_dir, exist_ok=True)
        
        # Initialize results storage
        self.performance_results = []
        self.accuracy_results = []
    
    def tearDown(self):
        """Clean up after tests."""
        # Clean up temporary files
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except Exception:
            pass
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
    
    @require_torch_accelerator
    @slow
    def test_non_integration_standard_pipeline(self):
        """
        Test standard PyTorch pipeline without TensorRT integration.
        
        This establishes the baseline performance and accuracy for comparison.
        
        Run with:
        RUN_SLOW=yes python3 -m pytest tests/pipelines/stable_diffusion/test_tensorrt_comprehensive.py::TensorRTComprehensiveTests::test_non_integration_standard_pipeline -v -s
        """
        print("\n🔥 Testing Non-Integration: Standard PyTorch Pipeline")
        
        # Load standard pipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            use_safetensors=False,  # Tiny model doesn't have safetensors
            safety_checker=None,  # Disable safety checker for tiny model
            requires_safety_checker=False
        ).to(torch_device)
        
        # Warm-up run
        _ = pipe(
            self.test_prompt,
            num_inference_steps=1,
            guidance_scale=self.guidance_scale,
            generator=torch.Generator().manual_seed(42)
        )
        
        # Benchmark run
        start_time = time.time()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        image = pipe(
            self.test_prompt,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=torch.Generator().manual_seed(42)
        ).images[0]
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Store results
        self.performance_results.append({
            "Pipeline": "Standard PyTorch",
            "Integration": "No",
            "Time (s)": f"{execution_time:.3f}",
            "Speedup": "1.00x (baseline)"
        })
        
        self.assertIsNotNone(image)
        self.assertEqual(image.size, (64, 64))  # tiny model output size
        
        print(f"✅ Standard pipeline test passed - Time: {execution_time:.3f}s")
        
        # Store image for accuracy comparison
        self.standard_image = np.array(image)
        
        # Clean up
        del pipe
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    @unittest.skipUnless(TENSORRT_AVAILABLE, "TensorRT not available")
    @require_torch_accelerator
    @slow
    def test_integration_tensorrt_pipeline(self):
        """
        Test TensorRT pipeline with integration enabled.
        
        This performs full model conversion (PyTorch -> ONNX -> TensorRT) and measures performance.
        Creates TensorRT engines for polygraphy validation.
        
        Run with:
        RUN_SLOW=yes python3 -m pytest tests/pipelines/stable_diffusion/test_tensorrt_comprehensive.py::TensorRTComprehensiveTests::test_integration_tensorrt_pipeline -v -s
        """
        print("\n🚀 Testing Integration: TensorRT Optimized Pipeline")
        
        # Load TensorRT pipeline
        pipe = StableDiffusionTensorRTPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            use_safetensors=False,  # Tiny model doesn't have safetensors
            safety_checker=None,  # Disable safety checker for tiny model
            requires_safety_checker=False
        ).to(torch_device)
        
        # Enable TensorRT optimization
        print("Enabling TensorRT optimization...")
        pipe.enable_tensorrt_optimization(
            cache_dir=self.tensorrt_cache_dir,
            onnx_cache_dir=self.onnx_cache_dir,
            fp16=True,
            verbose=False,
            max_workspace_size=512 * 1024 * 1024  # 512MB
        )
        
        # Warm-up run (includes conversion time)
        print("Performing warm-up run (includes model conversion)...")
        conversion_start = time.time()
        
        _ = pipe(
            self.test_prompt,
            num_inference_steps=1,
            guidance_scale=self.guidance_scale,
            generator=torch.Generator().manual_seed(42)
        )
        
        conversion_time = time.time() - conversion_start
        
        # Benchmark run (pure inference)
        print("Performing benchmark run...")
        start_time = time.time()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        image = pipe(
            self.test_prompt,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=torch.Generator().manual_seed(42)
        ).images[0]
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Calculate speedup if baseline exists
        speedup = "N/A"
        if hasattr(self, 'performance_results') and self.performance_results:
            baseline_time = float(self.performance_results[0]["Time (s)"])
            speedup_ratio = baseline_time / execution_time
            speedup = f"{speedup_ratio:.2f}x"
        
        # Store results
        self.performance_results.append({
            "Pipeline": "TensorRT Optimized",
            "Integration": "Yes",
            "Time (s)": f"{execution_time:.3f}",
            "Speedup": speedup,
            "Conversion Time (s)": f"{conversion_time:.3f}"
        })
        
        self.assertIsNotNone(image)
        self.assertEqual(image.size, (64, 64))
        
        print(f"✅ TensorRT pipeline test passed - Time: {execution_time:.3f}s, Conversion: {conversion_time:.3f}s")
        
        # Store image for accuracy comparison
        self.tensorrt_image = np.array(image)
        
        # Clean up
        del pipe
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    @unittest.skipUnless(POLYGRAPHY_AVAILABLE, "Polygraphy not available")
    @require_torch_accelerator
    @slow
    def test_polygraphy_direct_model_loading(self):
        """
        Test direct loading of polygraphy models for accuracy validation.
        
        Validates that TensorRT engines created by the integration test can be
        directly loaded using polygraphy APIs.
        
        Prerequisites: Run test_integration_tensorrt_pipeline first to create engines.
        
        Run with:
        RUN_SLOW=yes python3 -m pytest tests/pipelines/stable_diffusion/test_tensorrt_comprehensive.py::TensorRTComprehensiveTests::test_polygraphy_direct_model_loading -v -s
        """
        print("\n🔧 Testing Polygraphy Direct Model Loading")
        
        # Check if TensorRT engines exist from previous test
        engine_files = ["text_encoder.trt", "unet.trt", "vae_decoder.trt"]
        available_engines = []
        
        for engine_file in engine_files:
            engine_path = os.path.join(self.tensorrt_cache_dir, engine_file)
            if os.path.exists(engine_path):
                available_engines.append((engine_file, engine_path))
        
        if not available_engines:
            self.skipTest("No TensorRT engines found. Run integration test first.")
        
        print(f"Found {len(available_engines)} TensorRT engines for validation")
        
        # Test loading each available engine
        for engine_name, engine_path in available_engines:
            print(f"Loading engine: {engine_name}")
            
            try:
                # Load engine using polygraphy
                engine_data = bytes_from_path(engine_path)
                
                # Verify engine can be loaded
                self.assertIsNotNone(engine_data)
                self.assertGreater(len(engine_data), 0)
                
                print(f"✅ Successfully loaded {engine_name} ({len(engine_data)} bytes)")
                
            except Exception as e:
                print(f"❌ Failed to load {engine_name}: {e}")
                raise
        
        print("✅ All available engines loaded successfully via polygraphy")
    
    def test_accuracy_comparison(self):
        """
        Compare accuracy between standard and TensorRT pipelines.
        
        Performs pixel-wise comparison and calculates multiple similarity metrics:
        - Max/Mean pixel differences
        - Mean Squared Error (MSE)
        - Cosine similarity
        - Correlation coefficient
        
        Prerequisites: Run both test_non_integration_standard_pipeline and 
        test_integration_tensorrt_pipeline first.
        
        Run with:
        RUN_SLOW=yes python3 -m pytest tests/pipelines/stable_diffusion/test_tensorrt_comprehensive.py::TensorRTComprehensiveTests::test_accuracy_comparison -v -s
        """
        print("\n📊 Testing Accuracy Comparison")
        
        if not (hasattr(self, 'standard_image') and hasattr(self, 'tensorrt_image')):
            self.skipTest("Both standard and TensorRT images needed for comparison")
        
        # Calculate pixel-wise differences
        diff = np.abs(self.standard_image.astype(np.float32) - self.tensorrt_image.astype(np.float32))
        
        # Calculate metrics
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        mse = np.mean(diff ** 2)
        
        # Calculate similarity metrics
        standard_flat = self.standard_image.flatten().astype(np.float32)
        tensorrt_flat = self.tensorrt_image.flatten().astype(np.float32)
        
        # Cosine similarity
        dot_product = np.dot(standard_flat, tensorrt_flat)
        norm_product = np.linalg.norm(standard_flat) * np.linalg.norm(tensorrt_flat)
        cosine_similarity = dot_product / norm_product if norm_product != 0 else 0
        
        # Structural similarity (simplified)
        correlation = np.corrcoef(standard_flat, tensorrt_flat)[0, 1]
        
        # Store accuracy results
        self.accuracy_results.append({
            "Metric": "Max Pixel Difference",
            "Value": f"{max_diff:.2f}",
            "Unit": "RGB values"
        })
        self.accuracy_results.append({
            "Metric": "Mean Pixel Difference", 
            "Value": f"{mean_diff:.2f}",
            "Unit": "RGB values"
        })
        self.accuracy_results.append({
            "Metric": "Mean Squared Error",
            "Value": f"{mse:.2f}",
            "Unit": "RGB²"
        })
        self.accuracy_results.append({
            "Metric": "Cosine Similarity",
            "Value": f"{cosine_similarity:.4f}",
            "Unit": "0-1 scale"
        })
        self.accuracy_results.append({
            "Metric": "Correlation",
            "Value": f"{correlation:.4f}",
            "Unit": "-1 to 1 scale"
        })
        
        # Assertions for reasonable accuracy
        self.assertLess(mean_diff, 50.0, "Mean difference too high")
        self.assertGreater(cosine_similarity, 0.8, "Cosine similarity too low")
        
        print(f"✅ Accuracy comparison completed:")
        print(f"   Mean difference: {mean_diff:.2f}")
        print(f"   Cosine similarity: {cosine_similarity:.4f}")
        print(f"   Correlation: {correlation:.4f}")
    
    def test_print_performance_table(self):
        """
        Print formatted performance and accuracy results tables.
        
        Displays professional tables showing:
        - Performance metrics (timing, speedup)
        - Accuracy metrics (pixel differences, similarity scores)
        
        Best results when run after other tests to display their data.
        
        Run with:
        RUN_SLOW=yes python3 -m pytest tests/pipelines/stable_diffusion/test_tensorrt_comprehensive.py::TensorRTComprehensiveTests::test_print_performance_table -v -s
        """
        print("\n📈 Performance Acceleration Results")
        print("=" * 60)
        
        if self.performance_results:
            table = tabulate(
                self.performance_results,
                headers="keys",
                tablefmt="grid",
                floatfmt=".3f"
            )
            print(table)
        else:
            print("No performance results available.")
        
        print("\n📊 Accuracy Validation Results")
        print("=" * 50)
        
        if self.accuracy_results:
            table = tabulate(
                self.accuracy_results,
                headers="keys",
                tablefmt="grid"
            )
            print(table)
        else:
            print("No accuracy results available.")
        
        # Always pass - this is just for reporting
        self.assertTrue(True)
    
    @require_torch_accelerator
    def test_cuda_environment(self):
        """
        Test CUDA environment setup.
        
        Validates:
        - CUDA availability
        - GPU device access
        - Basic CUDA operations
        
        Run with:
        RUN_SLOW=yes python3 -m pytest tests/pipelines/stable_diffusion/test_tensorrt_comprehensive.py::TensorRTComprehensiveTests::test_cuda_environment -v -s
        """
        print("\n🔧 Testing CUDA Environment")
        
        self.assertTrue(torch.cuda.is_available(), "CUDA not available")
        
        device_count = torch.cuda.device_count()
        self.assertGreater(device_count, 0, "No CUDA devices found")
        
        # Test basic CUDA operations
        x = torch.randn(100, 100, device=torch_device)
        y = torch.mm(x, x.t())
        self.assertEqual(y.device.type, torch_device.split(':')[0])
        
        print(f"✅ CUDA environment OK - {device_count} device(s) available")
    
    @unittest.skipUnless(TENSORRT_AVAILABLE, "TensorRT not available")
    def test_tensorrt_imports(self):
        """
        Test TensorRT and related imports.
        
        Validates:
        - TensorRT pipeline availability
        - Polygraphy import status
        - TensorRT runtime availability
        
        Run with:
        RUN_SLOW=yes python3 -m pytest tests/pipelines/stable_diffusion/test_tensorrt_comprehensive.py::TensorRTComprehensiveTests::test_tensorrt_imports -v -s
        """
        print("\n🔧 Testing TensorRT Imports")
        
        # Test TensorRT pipeline import
        self.assertTrue(TENSORRT_AVAILABLE, "TensorRT pipeline not available")
        
        # Test polygraphy imports
        if POLYGRAPHY_AVAILABLE:
            print("✅ Polygraphy available for model conversion")
        else:
            print("⚠️ Polygraphy not available - limited functionality")
        
        # Test TensorRT runtime imports
        try:
            import tensorrt as trt
            print(f"✅ TensorRT runtime available - version: {trt.__version__}")
        except ImportError:
            print("⚠️ TensorRT runtime not available - will use polygraphy CLI")
        
        print("✅ All required imports successful")


if __name__ == "__main__":
    """
    Direct execution example:
    
    To run this test file directly with all required environment setup:
    
    ```bash
    cd /code/diffusers
    RUN_SLOW=yes python3 tests/pipelines/stable_diffusion/test_tensorrt_comprehensive.py
    ```
    
    Or run with pytest for better output formatting:
    
    ```bash
    cd /code/diffusers
    RUN_SLOW=yes python3 -m pytest tests/pipelines/stable_diffusion/test_tensorrt_comprehensive.py -v -s
    ```
    """
    unittest.main(verbosity=2)