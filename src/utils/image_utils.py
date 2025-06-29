"""
Image utilities for Canva image processing pipeline.

This module provides helper functions for image validation, conversion,
and manipulation tasks that support the main sanitize_image function.
"""

import numpy as np
from typing import Tuple, List, Union, Optional
import logging

logger = logging.getLogger(__name__)


def validate_image_input(image: np.ndarray) -> bool:
    """
    Validate if input array is a valid image format.
    
    Args:
        image: Input numpy array
        
    Returns:
        True if valid image format, False otherwise
    """
    if not isinstance(image, np.ndarray):
        return False
    
    if image.size == 0:
        return False
    
    if image.ndim < 1 or image.ndim > 4:
        return False
    
    # Check for reasonable dimensions
    if image.ndim >= 2:
        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            return False
        
        # Check for extremely large images that might cause memory issues
        if h * w > 50_000_000:  # 50 megapixels
            logger.warning(f"Very large image detected: {h}x{w}")
    
    # Check channel count for color images
    if image.ndim == 3:
        channels = image.shape[2]
        if channels not in [1, 3, 4]:
            return False
    
    return True


def get_image_stats(image: np.ndarray) -> dict:
    """
    Get comprehensive statistics about an image array.
    
    Args:
        image: Input image array
        
    Returns:
        Dictionary containing image statistics
    """
    stats = {
        'shape': image.shape,
        'dtype': str(image.dtype),
        'size_bytes': image.nbytes,
        'size_mb': image.nbytes / (1024 * 1024),
        'min_value': float(image.min()),
        'max_value': float(image.max()),
        'mean_value': float(image.mean()),
        'std_value': float(image.std()),
    }
    
    # Add dimension-specific info
    if image.ndim >= 2:
        stats['height'] = image.shape[0]
        stats['width'] = image.shape[1]
        stats['aspect_ratio'] = image.shape[1] / image.shape[0]
    
    if image.ndim == 3:
        stats['channels'] = image.shape[2]
        stats['is_grayscale'] = image.shape[2] == 1
        stats['is_rgb'] = image.shape[2] == 3
        stats['is_rgba'] = image.shape[2] == 4
    else:
        stats['channels'] = 1
        stats['is_grayscale'] = True
        stats['is_rgb'] = False
        stats['is_rgba'] = False
    
    return stats


def create_test_images() -> dict:
    """
    Create a variety of test images for validation and testing.
    
    Returns:
        Dictionary of test images with different characteristics
    """
    test_images = {}
    
    # Basic grayscale images
    test_images['small_gray'] = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
    test_images['medium_gray'] = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
    test_images['large_gray'] = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
    
    # RGB images
    test_images['small_rgb'] = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    test_images['medium_rgb'] = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    test_images['large_rgb'] = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    
    # RGBA images
    test_images['rgba'] = np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8)
    
    # Different aspect ratios
    test_images['wide'] = np.random.randint(0, 256, (100, 300, 3), dtype=np.uint8)
    test_images['tall'] = np.random.randint(0, 256, (300, 100, 3), dtype=np.uint8)
    
    # Different data types
    test_images['float32'] = np.random.random((64, 64, 3)).astype(np.float32)
    test_images['float64'] = np.random.random((64, 64)).astype(np.float64)
    test_images['uint16'] = np.random.randint(0, 65536, (64, 64), dtype=np.uint16)
    
    # Edge cases
    test_images['single_pixel'] = np.array([[128]], dtype=np.uint8)
    test_images['all_black'] = np.zeros((50, 50, 3), dtype=np.uint8)
    test_images['all_white'] = np.full((50, 50, 3), 255, dtype=np.uint8)
    test_images['uniform_gray'] = np.full((50, 50), 128, dtype=np.uint8)
    
    # Unusual formats
    test_images['1d_array'] = np.random.randint(0, 256, (64*64,), dtype=np.uint8)
    test_images['channel_last'] = np.random.randint(0, 256, (64, 64, 1), dtype=np.uint8)
    
    return test_images


def create_gradient_image(size: Tuple[int, int], channels: int = 3) -> np.ndarray:
    """
    Create a gradient test image for consistent testing.
    
    Args:
        size: (height, width) of the image
        channels: Number of channels (1, 3, or 4)
        
    Returns:
        Gradient image array
    """
    h, w = size
    
    # Create horizontal gradient
    gradient = np.linspace(0, 255, w, dtype=np.uint8)
    gradient = np.tile(gradient, (h, 1))
    
    if channels == 1:
        return gradient
    elif channels == 3:
        # RGB gradient (R=horizontal, G=vertical, B=diagonal)
        rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
        rgb_image[:, :, 0] = gradient  # Red: horizontal gradient
        rgb_image[:, :, 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, np.newaxis]  # Green: vertical
        rgb_image[:, :, 2] = (gradient + np.linspace(0, 255, h, dtype=np.uint8)[:, np.newaxis]) // 2  # Blue: diagonal
        return rgb_image
    elif channels == 4:
        # RGBA with full alpha
        rgba_image = np.zeros((h, w, 4), dtype=np.uint8)
        rgba_image[:, :, :3] = create_gradient_image(size, 3)
        rgba_image[:, :, 3] = 255  # Full alpha
        return rgba_image
    else:
        raise ValueError(f"Unsupported channel count: {channels}")


def check_memory_usage(image_shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> dict:
    """
    Calculate memory usage for an image of given shape and dtype.
    
    Args:
        image_shape: Shape of the image
        dtype: Data type of the image
        
    Returns:
        Dictionary with memory usage information
    """
    # Calculate bytes per element
    bytes_per_element = np.dtype(dtype).itemsize
    
    # Total elements
    total_elements = np.prod(image_shape)
    
    # Memory usage
    bytes_total = total_elements * bytes_per_element
    mb_total = bytes_total / (1024 * 1024)
    gb_total = mb_total / 1024
    
    return {
        'total_elements': int(total_elements),
        'bytes_per_element': bytes_per_element,
        'total_bytes': int(bytes_total),
        'total_mb': mb_total,
        'total_gb': gb_total,
        'dtype': str(dtype)
    }


def safe_cast_dtype(image: np.ndarray, target_dtype: np.dtype) -> np.ndarray:
    """
    Safely cast image to target dtype with proper value scaling.
    
    Args:
        image: Input image array
        target_dtype: Target data type
        
    Returns:
        Image cast to target dtype
    """
    if image.dtype == target_dtype:
        return image.copy()
    
    # Get value ranges for source and target dtypes
    source_info = np.iinfo(image.dtype) if np.issubdtype(image.dtype, np.integer) else None
    target_info = np.iinfo(target_dtype) if np.issubdtype(target_dtype, np.integer) else None
    
    # Handle different casting scenarios
    if source_info and target_info:
        # Integer to integer
        source_range = source_info.max - source_info.min
        target_range = target_info.max - target_info.min
        
        if source_range == target_range:
            # Same range, direct cast
            return image.astype(target_dtype)
        else:
            # Scale values
            normalized = (image.astype(np.float64) - source_info.min) / source_range
            scaled = normalized * target_range + target_info.min
            return np.clip(scaled, target_info.min, target_info.max).astype(target_dtype)
    
    elif source_info and not target_info:
        # Integer to float
        if target_dtype in [np.float32, np.float64]:
            return (image.astype(target_dtype) - source_info.min) / (source_info.max - source_info.min)
        else:
            return image.astype(target_dtype)
    
    elif not source_info and target_info:
        # Float to integer
        if image.min() >= 0 and image.max() <= 1:
            # Assume [0, 1] range
            scaled = image * target_info.max
            return np.clip(scaled, target_info.min, target_info.max).astype(target_dtype)
        else:
            # Unknown range, normalize first
            img_min, img_max = image.min(), image.max()
            if img_min == img_max:
                return np.full_like(image, target_info.max // 2, dtype=target_dtype)
            normalized = (image - img_min) / (img_max - img_min)
            scaled = normalized * target_info.max
            return np.clip(scaled, target_info.min, target_info.max).astype(target_dtype)
    
    else:
        # Float to float or other cases
        return image.astype(target_dtype)


def is_power_of_two(n: int) -> bool:
    """
    Check if a number is a power of two.
    
    Args:
        n: Number to check
        
    Returns:
        True if n is a power of two, False otherwise
    """
    return n > 0 and (n & (n - 1)) == 0


def next_power_of_two(n: int) -> int:
    """
    Find the next power of two greater than or equal to n.
    
    Args:
        n: Input number
        
    Returns:
        Next power of two
    """
    if n <= 0:
        return 1
    
    # Handle the case where n is already a power of two
    if is_power_of_two(n):
        return n
    
    # Find the next power of two
    power = 1
    while power < n:
        power <<= 1
    
    return power


def calculate_optimal_tile_size(image_shape: Tuple[int, int], max_memory_mb: int = 100) -> Tuple[int, int]:
    """
    Calculate optimal tile size for processing large images in chunks.
    
    Args:
        image_shape: (height, width) of the image
        max_memory_mb: Maximum memory per tile in MB
        
    Returns:
        (tile_height, tile_width) optimal for processing
    """
    h, w = image_shape
    
    # Assume float32 processing (4 bytes per pixel)
    max_pixels = (max_memory_mb * 1024 * 1024) // 4
    
    # If image fits in memory, return full size
    if h * w <= max_pixels:
        return h, w
    
    # Calculate tile dimensions
    aspect_ratio = w / h
    
    # Start with square tiles and adjust for aspect ratio
    tile_pixels = int(np.sqrt(max_pixels))
    
    if aspect_ratio >= 1:
        # Wide image
        tile_w = min(tile_pixels, w)
        tile_h = min(max_pixels // tile_w, h)
    else:
        # Tall image
        tile_h = min(tile_pixels, h)
        tile_w = min(max_pixels // tile_h, w)
    
    # Ensure minimum tile size
    tile_h = max(16, tile_h)
    tile_w = max(16, tile_w)
    
    # Prefer powers of two for better performance
    tile_h = next_power_of_two(tile_h // 2)
    tile_w = next_power_of_two(tile_w // 2)
    
    return tile_h, tile_w