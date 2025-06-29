"""
Comprehensive test suite for Canva image processing functionality.

This module tests the sanitize_image function and related utilities
with various edge cases, performance requirements, and error conditions.
"""

import pytest
import numpy as np
import time
from typing import Tuple, List
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from image_processing import (
    sanitize_image, 
    ImageProcessingError, 
    MemoryError as CustomMemoryError,
    benchmark_sanitize_image
)
from utils.image_utils import (
    validate_image_input,
    get_image_stats,
    create_test_images,
    create_gradient_image
)


class TestSanitizeImageBasicFunctionality:
    """Test basic functionality of sanitize_image function."""
    
    def test_grayscale_image_processing(self):
        """Test processing of grayscale images."""
        # Create test grayscale image
        input_image = np.random.randint(0, 256, (100, 150), dtype=np.uint8)
        
        result = sanitize_image(input_image, output_size=224)
        
        # Check output properties
        assert result.shape == (224, 224), f"Expected (224, 224), got {result.shape}"
        assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"
        assert 0 <= result.min() <= result.max() <= 1, f"Values not in [0,1]: min={result.min()}, max={result.max()}"
    
    def test_rgb_image_processing(self):
        """Test processing of RGB images."""
        input_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        
        result = sanitize_image(input_image, output_size=224)
        
        assert result.shape == (224, 224, 3), f"Expected (224, 224, 3), got {result.shape}"
        assert result.dtype == np.float32
        assert 0 <= result.min() <= result.max() <= 1
    
    def test_rgba_image_processing(self):
        """Test processing of RGBA images."""
        input_image = np.random.randint(0, 256, (300, 400, 4), dtype=np.uint8)
        
        result = sanitize_image(input_image, output_size=224)
        
        assert result.shape == (224, 224, 4), f"Expected (224, 224, 4), got {result.shape}"
        assert result.dtype == np.float32
        assert 0 <= result.min() <= result.max() <= 1
    
    def test_different_output_sizes(self):
        """Test different output sizes."""
        input_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        for size in [64, 128, 256, 512]:
            result = sanitize_image(input_image, output_size=size)
            assert result.shape == (size, size, 3), f"Failed for size {size}"
    
    def test_aspect_ratio_preservation(self):
        """Test aspect ratio preservation with padding."""
        # Wide image
        wide_image = np.random.randint(0, 256, (100, 300, 3), dtype=np.uint8)
        result = sanitize_image(wide_image, preserve_aspect_ratio=True)
        
        assert result.shape == (224, 224, 3)
        # Check that padding was applied (should have black bars on top/bottom)
        assert np.allclose(result[0, :, :], 0, atol=1e-6)  # Top row should be padding
        
        # Tall image
        tall_image = np.random.randint(0, 256, (300, 100, 3), dtype=np.uint8)
        result = sanitize_image(tall_image, preserve_aspect_ratio=True)
        
        assert result.shape == (224, 224, 3)
        # Check that padding was applied (should have black bars on left/right)
        assert np.allclose(result[:, 0, :], 0, atol=1e-6)  # Left column should be padding
    
    def test_no_aspect_ratio_preservation(self):
        """Test direct resize without aspect ratio preservation."""
        input_image = np.random.randint(0, 256, (100, 300, 3), dtype=np.uint8)
        
        result = sanitize_image(input_image, preserve_aspect_ratio=False)
        
        assert result.shape == (224, 224, 3)
        # Should not have uniform padding rows/columns
        assert not np.allclose(result[0, :, :], 0, atol=1e-6)


class TestSanitizeImageEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_pixel_image(self):
        """Test processing of single pixel image."""
        input_image = np.array([[128]], dtype=np.uint8)
        
        result = sanitize_image(input_image, output_size=224)
        
        assert result.shape == (224, 224)
        # Should be uniform with the normalized pixel value
        expected_value = 128 / 255.0
        assert np.allclose(result, expected_value, atol=1e-6)
    
    def test_uniform_images(self):
        """Test processing of uniform (all same value) images."""
        # All black
        black_image = np.zeros((50, 50, 3), dtype=np.uint8)
        result = sanitize_image(black_image)
        assert np.allclose(result, 0.0, atol=1e-6)
        
        # All white
        white_image = np.full((50, 50, 3), 255, dtype=np.uint8)
        result = sanitize_image(white_image)
        assert np.allclose(result, 1.0, atol=1e-6)
        
        # Uniform gray
        gray_image = np.full((50, 50), 128, dtype=np.uint8)
        result = sanitize_image(gray_image)
        expected = 128 / 255.0
        assert np.allclose(result, expected, atol=1e-6)
    
    def test_different_data_types(self):
        """Test processing of different numpy data types."""
        base_shape = (64, 64, 3)
        
        # uint8 (0-255)
        uint8_img = np.random.randint(0, 256, base_shape, dtype=np.uint8)
        result = sanitize_image(uint8_img)
        assert result.dtype == np.float32
        assert 0 <= result.min() <= result.max() <= 1
        
        # uint16 (0-65535)
        uint16_img = np.random.randint(0, 65536, base_shape, dtype=np.uint16)
        result = sanitize_image(uint16_img)
        assert result.dtype == np.float32
        assert 0 <= result.min() <= result.max() <= 1
        
        # float32 (0-1)
        float32_img = np.random.random(base_shape).astype(np.float32)
        result = sanitize_image(float32_img)
        assert result.dtype == np.float32
        assert 0 <= result.min() <= result.max() <= 1
        
        # float64 (0-1)
        float64_img = np.random.random(base_shape).astype(np.float64)
        result = sanitize_image(float64_img)
        assert result.dtype == np.float32
        assert 0 <= result.min() <= result.max() <= 1
    
    def test_1d_array_processing(self):
        """Test processing of 1D arrays (flattened square images)."""
        # Perfect square (64x64 = 4096)
        input_1d = np.random.randint(0, 256, (4096,), dtype=np.uint8)
        
        result = sanitize_image(input_1d, output_size=64)
        
        assert result.shape == (64, 64)
        assert result.dtype == np.float32
        assert 0 <= result.min() <= result.max() <= 1
    
    def test_channel_dimension_handling(self):
        """Test handling of different channel dimensions."""
        # (H, W, 1) -> should squeeze to (H, W)
        input_image = np.random.randint(0, 256, (50, 50, 1), dtype=np.uint8)
        result = sanitize_image(input_image)
        assert result.shape == (224, 224)
        
        # Already correct size
        correct_size = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        result = sanitize_image(correct_size, output_size=224)
        assert result.shape == (224, 224, 3)


class TestSanitizeImageErrorHandling:
    """Test error handling and validation."""
    
    def test_none_input(self):
        """Test handling of None input."""
        with pytest.raises(ValueError, match="Input image cannot be None"):
            sanitize_image(None)
    
    def test_empty_array(self):
        """Test handling of empty arrays."""
        empty_array = np.array([])
        with pytest.raises(ValueError, match="Input image cannot be empty"):
            sanitize_image(empty_array)
    
    def test_invalid_data_type(self):
        """Test handling of invalid input types."""
        with pytest.raises(ValueError, match="Input must be numpy array"):
            sanitize_image([1, 2, 3, 4])
    
    def test_invalid_dimensions(self):
        """Test handling of invalid array dimensions."""
        # 5D array
        invalid_5d = np.random.random((2, 2, 2, 2, 2))
        with pytest.raises(ValueError, match="Unsupported image dimensions"):
            sanitize_image(invalid_5d)
    
    def test_invalid_channel_count(self):
        """Test handling of invalid channel counts."""
        # 5 channels (unsupported)
        invalid_channels = np.random.randint(0, 256, (50, 50, 5), dtype=np.uint8)
        with pytest.raises(ValueError, match="Unsupported number of channels"):
            sanitize_image(invalid_channels)
    
    def test_invalid_output_size(self):
        """Test handling of invalid output sizes."""
        input_image = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        
        with pytest.raises(ValueError, match="Output size must be positive"):
            sanitize_image(input_image, output_size=0)
        
        with pytest.raises(ValueError, match="Output size must be positive"):
            sanitize_image(input_image, output_size=-10)
    
    def test_memory_limit_exceeded(self):
        """Test memory limit enforcement."""
        # Create large image that would exceed memory limit
        large_image = np.random.randint(0, 256, (1000, 1000, 3), dtype=np.uint8)
        
        with pytest.raises(CustomMemoryError, match="Estimated memory usage.*exceeds limit"):
            sanitize_image(large_image, max_memory_mb=1)  # Very low limit
    
    def test_non_square_1d_array(self):
        """Test handling of 1D arrays that can't form squares."""
        # 99 elements can't form a perfect square (sqrt(99) ≈ 9.95)
        non_square_1d = np.random.randint(0, 256, (99,), dtype=np.uint8)
        
        with pytest.raises(ValueError, match="Cannot reshape 1D array"):
            sanitize_image(non_square_1d)


class TestSanitizeImageInterpolation:
    """Test different interpolation methods."""
    
    @pytest.mark.parametrize("interpolation", ['nearest', 'bilinear', 'bicubic', 'lanczos'])
    def test_interpolation_methods(self, interpolation):
        """Test all interpolation methods."""
        input_image = create_gradient_image((100, 150), channels=3)
        
        result = sanitize_image(input_image, interpolation=interpolation)
        
        assert result.shape == (224, 224, 3)
        assert result.dtype == np.float32
        assert 0 <= result.min() <= result.max() <= 1
    
    def test_interpolation_quality_differences(self):
        """Test that different interpolation methods produce different results."""
        input_image = create_gradient_image((50, 100), channels=1)
        
        nearest = sanitize_image(input_image, interpolation='nearest')
        bilinear = sanitize_image(input_image, interpolation='bilinear')
        
        # Results should be different due to different interpolation
        assert not np.allclose(nearest, bilinear, atol=1e-3)


class TestSanitizeImagePerformance:
    """Test performance requirements."""
    
    def test_processing_time_vga(self):
        """Test processing time for VGA-sized images (480x640)."""
        input_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        
        start_time = time.time()
        result = sanitize_image(input_image)
        processing_time = time.time() - start_time
        
        # Should process VGA in less than 100ms
        assert processing_time < 0.1, f"Processing took {processing_time:.3f}s, expected < 0.1s"
        assert result.shape == (224, 224, 3)
    
    def test_processing_time_hd(self):
        """Test processing time for HD-sized images (720x1280)."""
        input_image = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
        
        start_time = time.time()
        result = sanitize_image(input_image)
        processing_time = time.time() - start_time
        
        # HD should still be reasonably fast
        assert processing_time < 0.2, f"Processing took {processing_time:.3f}s, expected < 0.2s"
        assert result.shape == (224, 224, 3)
    
    @pytest.mark.slow
    def test_processing_time_4k(self):
        """Test processing time for 4K images (2160x3840)."""
        input_image = np.random.randint(0, 256, (2160, 3840, 3), dtype=np.uint8)
        
        start_time = time.time()
        result = sanitize_image(input_image, max_memory_mb=1000)  # Increase memory limit
        processing_time = time.time() - start_time
        
        # 4K should process in reasonable time
        assert processing_time < 1.0, f"Processing took {processing_time:.3f}s, expected < 1.0s"
        assert result.shape == (224, 224, 3)
    
    def test_memory_usage_within_limits(self):
        """Test that memory usage stays within reasonable limits."""
        # This is a basic test - in production would use memory profiling
        input_image = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
        
        # Should not raise memory error with default limits
        result = sanitize_image(input_image)
        assert result.shape == (224, 224, 3)


class TestSanitizeImageConsistency:
    """Test consistency and reproducibility."""
    
    def test_deterministic_output(self):
        """Test that same input produces same output."""
        input_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        result1 = sanitize_image(input_image.copy())
        result2 = sanitize_image(input_image.copy())
        
        assert np.array_equal(result1, result2), "Same input should produce identical output"
    
    def test_value_range_consistency(self):
        """Test that output values are always in [0, 1] range."""
        test_images = create_test_images()
        
        for name, image in test_images.items():
            if validate_image_input(image):
                try:
                    result = sanitize_image(image)
                    assert 0 <= result.min(), f"Min value below 0 for {name}: {result.min()}"
                    assert result.max() <= 1, f"Max value above 1 for {name}: {result.max()}"
                except (ValueError, CustomMemoryError):
                    # Expected errors for invalid inputs
                    pass


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_validate_image_input(self):
        """Test image input validation."""
        # Valid inputs
        assert validate_image_input(np.random.randint(0, 256, (50, 50), dtype=np.uint8))
        assert validate_image_input(np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8))
        
        # Invalid inputs
        assert not validate_image_input(None)
        assert not validate_image_input([1, 2, 3])
        assert not validate_image_input(np.array([]))
        assert not validate_image_input(np.random.random((50, 50, 5)))  # 5 channels
    
    def test_get_image_stats(self):
        """Test image statistics calculation."""
        test_image = np.random.randint(0, 256, (100, 150, 3), dtype=np.uint8)
        stats = get_image_stats(test_image)
        
        assert stats['shape'] == (100, 150, 3)
        assert stats['height'] == 100
        assert stats['width'] == 150
        assert stats['channels'] == 3
        assert stats['is_rgb'] == True
        assert stats['aspect_ratio'] == 1.5
        assert 0 <= stats['min_value'] <= stats['max_value'] <= 255
    
    def test_create_gradient_image(self):
        """Test gradient image creation."""
        gradient = create_gradient_image((64, 128), channels=3)
        
        assert gradient.shape == (64, 128, 3)
        assert gradient.dtype == np.uint8
        
        # Check that it's actually a gradient
        assert gradient[0, 0, 0] != gradient[0, -1, 0]  # Horizontal gradient in red channel
        assert gradient[0, 0, 1] != gradient[-1, 0, 1]  # Vertical gradient in green channel


class TestBenchmarkFunction:
    """Test benchmarking functionality."""
    
    def test_benchmark_basic(self):
        """Test basic benchmarking functionality."""
        # Run with small test shapes for speed
        test_shapes = [(100, 100, 3), (200, 300, 3)]
        
        results = benchmark_sanitize_image(
            test_shapes=test_shapes,
            output_size=64,  # Smaller for faster testing
            num_runs=3
        )
        
        assert len(results) == 2
        
        for shape_key, metrics in results.items():
            assert 'mean_time' in metrics
            assert 'std_time' in metrics
            assert 'min_time' in metrics
            assert 'max_time' in metrics
            assert 'input_size_mb' in metrics
            
            assert metrics['mean_time'] > 0
            assert metrics['min_time'] <= metrics['mean_time']
            assert metrics['mean_time'] <= metrics['max_time']


# Pytest markers for different test categories
pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
]


if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--cov=src",
        "--cov-report=term-missing",
        "--cov-report=html"
    ])