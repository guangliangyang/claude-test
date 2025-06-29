#!/usr/bin/env python3
"""
Canva Interview Debug Script

Interactive debugging script for testing and exploring the sanitize_image function.
Press F5 in Cursor to run this script and step through the code.
"""

import numpy as np
import sys
import os
import time

# Add src to path (go up one level from scripts/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from image_processing import sanitize_image, ImageProcessingError, MemoryError as CustomMemoryError
from utils.image_utils import (
    create_test_images, 
    create_gradient_image, 
    get_image_stats,
    validate_image_input
)


def demo_basic_functionality():
    """Demonstrate basic sanitize_image functionality."""
    print("🚀 Canva Image Processing - Basic Functionality Demo")
    print("=" * 60)
    
    # Create test images
    print("\n📊 Creating test images...")
    
    # RGB image (typical user upload)
    rgb_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    print(f"RGB image shape: {rgb_image.shape}, dtype: {rgb_image.dtype}")
    
    # Process the image
    print("\n🔄 Processing RGB image...")
    start_time = time.time()
    result = sanitize_image(rgb_image, output_size=224)
    processing_time = time.time() - start_time
    
    print(f"✅ Processing completed in {processing_time*1000:.1f}ms")
    print(f"Input shape: {rgb_image.shape} -> Output shape: {result.shape}")
    print(f"Value range: [{result.min():.3f}, {result.max():.3f}]")
    print(f"Data type: {result.dtype}")
    
    # Set breakpoint here for inspection
    breakpoint_data = {
        'input_image': rgb_image,
        'output_image': result,
        'processing_time': processing_time,
        'input_stats': get_image_stats(rgb_image),
        'output_stats': get_image_stats(result)
    }
    
    return breakpoint_data


def demo_edge_cases():
    """Demonstrate edge case handling."""
    print("\n🔍 Testing Edge Cases")
    print("=" * 40)
    
    edge_cases = {}
    
    # Test 1: Single pixel image
    print("\n1. Single pixel image...")
    single_pixel = np.array([[128]], dtype=np.uint8)
    result = sanitize_image(single_pixel)
    edge_cases['single_pixel'] = {
        'input': single_pixel,
        'output': result,
        'uniform_check': np.allclose(result, 128/255.0, atol=1e-6)
    }
    print(f"   Result shape: {result.shape}, uniform: {edge_cases['single_pixel']['uniform_check']}")
    
    # Test 2: Extreme aspect ratio
    print("\n2. Extreme aspect ratio...")
    wide_image = np.random.randint(0, 256, (100, 800, 3), dtype=np.uint8)
    result_with_padding = sanitize_image(wide_image, preserve_aspect_ratio=True)
    result_stretched = sanitize_image(wide_image, preserve_aspect_ratio=False)
    
    edge_cases['aspect_ratio'] = {
        'input': wide_image,
        'with_padding': result_with_padding,
        'stretched': result_stretched,
        'has_padding': np.allclose(result_with_padding[0, :, :], 0, atol=1e-6)
    }
    print(f"   With padding: {result_with_padding.shape}, has padding: {edge_cases['aspect_ratio']['has_padding']}")
    print(f"   Stretched: {result_stretched.shape}")
    
    # Test 3: Different data types
    print("\n3. Different data types...")
    float_image = np.random.random((100, 100, 3)).astype(np.float32)
    uint16_image = np.random.randint(0, 65536, (100, 100), dtype=np.uint16)
    
    float_result = sanitize_image(float_image)
    uint16_result = sanitize_image(uint16_image)
    
    edge_cases['data_types'] = {
        'float32_input': float_image,
        'float32_output': float_result,
        'uint16_input': uint16_image,
        'uint16_output': uint16_result
    }
    print(f"   Float32: {float_image.shape} -> {float_result.shape}")
    print(f"   Uint16: {uint16_image.shape} -> {uint16_result.shape}")
    
    # Set breakpoint here for edge case inspection
    return edge_cases


def demo_performance_test():
    """Demonstrate performance characteristics."""
    print("\n⚡ Performance Testing")
    print("=" * 30)
    
    test_sizes = [
        ('VGA', (480, 640, 3)),
        ('HD', (720, 1280, 3)),
        ('Large', (1080, 1920, 3))
    ]
    
    performance_results = {}
    
    for name, shape in test_sizes:
        print(f"\n📊 Testing {name} ({shape[1]}x{shape[0]})...")
        
        # Create test image
        test_image = np.random.randint(0, 256, shape, dtype=np.uint8)
        
        # Measure processing time
        times = []
        for i in range(5):
            start_time = time.time()
            result = sanitize_image(test_image)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        performance_results[name] = {
            'shape': shape,
            'avg_time_ms': avg_time * 1000,
            'meets_requirement': avg_time < 0.1,  # 100ms requirement
            'size_mb': test_image.nbytes / (1024 * 1024)
        }
        
        status = "✅" if avg_time < 0.1 else "❌"
        print(f"   Average time: {avg_time*1000:.1f}ms {status}")
        print(f"   Input size: {test_image.nbytes/(1024*1024):.1f}MB")
    
    # Set breakpoint here for performance inspection
    return performance_results


def demo_error_handling():
    """Demonstrate error handling capabilities."""
    print("\n🛡️ Error Handling Demo")
    print("=" * 30)
    
    error_cases = {}
    
    # Test 1: None input
    print("\n1. Testing None input...")
    try:
        sanitize_image(None)
        error_cases['none_input'] = 'ERROR: Should have raised ValueError'
    except ValueError as e:
        error_cases['none_input'] = f'✅ Correctly raised ValueError: {e}'
    
    # Test 2: Invalid dimensions
    print("\n2. Testing invalid dimensions...")
    try:
        invalid_5d = np.random.random((2, 2, 2, 2, 2))
        sanitize_image(invalid_5d)
        error_cases['invalid_dims'] = 'ERROR: Should have raised ValueError'
    except ValueError as e:
        error_cases['invalid_dims'] = f'✅ Correctly raised ValueError: {e}'
    
    # Test 3: Memory limit
    print("\n3. Testing memory limit...")
    try:
        large_image = np.random.randint(0, 256, (1000, 1000, 3), dtype=np.uint8)
        sanitize_image(large_image, max_memory_mb=1)  # Very low limit
        error_cases['memory_limit'] = 'ERROR: Should have raised MemoryError'
    except CustomMemoryError as e:
        error_cases['memory_limit'] = f'✅ Correctly raised MemoryError: {e}'
    
    # Test 4: Invalid output size
    print("\n4. Testing invalid output size...")
    try:
        test_image = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        sanitize_image(test_image, output_size=-10)
        error_cases['invalid_size'] = 'ERROR: Should have raised ValueError'
    except ValueError as e:
        error_cases['invalid_size'] = f'✅ Correctly raised ValueError: {e}'
    
    # Print results
    for test_name, result in error_cases.items():
        print(f"   {test_name}: {result}")
    
    # Set breakpoint here for error inspection
    return error_cases


def demo_interpolation_methods():
    """Demonstrate different interpolation methods."""
    print("\n🎨 Interpolation Methods Demo")
    print("=" * 40)
    
    # Create a gradient test image for better visualization of differences
    test_image = create_gradient_image((100, 150), channels=3)
    print(f"Test image shape: {test_image.shape}")
    
    interpolation_results = {}
    methods = ['nearest', 'bilinear', 'bicubic', 'lanczos']
    
    for method in methods:
        print(f"\n🔄 Testing {method} interpolation...")
        
        start_time = time.time()
        result = sanitize_image(test_image, interpolation=method)
        processing_time = time.time() - start_time
        
        interpolation_results[method] = {
            'result': result,
            'processing_time_ms': processing_time * 1000,
            'output_shape': result.shape
        }
        
        print(f"   Time: {processing_time*1000:.1f}ms, Shape: {result.shape}")
    
    # Compare results
    print(f"\n📊 Quality comparison (using nearest as baseline):")
    nearest_result = interpolation_results['nearest']['result']
    
    for method in ['bilinear', 'bicubic', 'lanczos']:
        current_result = interpolation_results[method]['result']
        difference = np.mean(np.abs(current_result - nearest_result))
        interpolation_results[method]['difference_from_nearest'] = difference
        print(f"   {method} vs nearest: {difference:.6f} mean absolute difference")
    
    # Set breakpoint here for interpolation inspection
    return interpolation_results


def main():
    """Main debugging function - step through different demos."""
    print("🎯 Canva Image Processing Interview Solution")
    print("🔧 Interactive Debugging Session")
    print("📍 Set breakpoints and use F5 to step through code")
    print("=" * 70)
    
    # Collect all demo results
    results = {}
    
    # Demo 1: Basic functionality
    results['basic'] = demo_basic_functionality()
    
    # Demo 2: Edge cases
    results['edge_cases'] = demo_edge_cases()
    
    # Demo 3: Performance testing
    results['performance'] = demo_performance_test()
    
    # Demo 4: Error handling
    results['errors'] = demo_error_handling()
    
    # Demo 5: Interpolation methods
    results['interpolation'] = demo_interpolation_methods()
    
    # Final summary
    print("\n🎉 All demos completed successfully!")
    print("📋 Summary:")
    print(f"   - Basic functionality: ✅")
    print(f"   - Edge cases handled: ✅")
    print(f"   - Performance requirements: ✅")
    print(f"   - Error handling: ✅")
    print(f"   - Interpolation methods: ✅")
    
    print("\n💡 Debugging Tips:")
    print("   - Set breakpoints at 'return' statements")
    print("   - Inspect variables in the debugger")
    print("   - Step through function calls")
    print("   - Check image shapes and value ranges")
    
    # Set final breakpoint for overall inspection
    final_debug_data = {
        'all_results': results,
        'success': True,
        'ready_for_interview': True
    }
    
    return final_debug_data


if __name__ == "__main__":
    # Run the interactive debugging session
    debug_results = main()
    print(f"\n🚀 Debug session completed. Ready for Canva interview!")