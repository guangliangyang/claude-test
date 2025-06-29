"""
Canva Image Processing Module

This module provides image sanitization functionality for Canva's AI pipeline,
converting raw image data into standardized formats suitable for machine learning models.
"""

import numpy as np
import logging
import time
from typing import Union, Tuple, Optional, Literal
from scipy.ndimage import zoom

logger = logging.getLogger(__name__)

InterpolationMethod = Literal['nearest', 'bilinear', 'bicubic', 'lanczos']


class ImageProcessingError(Exception):
    """Custom exception for image processing errors."""
    pass


class MemoryError(ImageProcessingError):
    """Exception raised when memory constraints are exceeded."""
    pass


def sanitize_image(
    image: np.ndarray,
    output_size: int = 224,
    preserve_aspect_ratio: bool = True,
    interpolation: InterpolationMethod = 'bilinear',
    max_memory_mb: int = 500
) -> np.ndarray:
    """
    Sanitize image array to normalized square matrix for Canva AI pipeline.
    
    Converts any input image format to a standardized square matrix with pixel
    values normalized to [0, 1] range, suitable for ML model preprocessing.
    
    Args:
        image: Input image array of shape (H,W), (H,W,1), (H,W,3), or (H,W,4)
        output_size: Target square size (default: 224 for ImageNet compatibility)
        preserve_aspect_ratio: Whether to preserve aspect ratio with padding
        interpolation: Resampling method ('nearest', 'bilinear', 'bicubic', 'lanczos')
        max_memory_mb: Maximum memory usage limit in MB
    
    Returns:
        Normalized square image array in [0, 1] range
        - Grayscale: (output_size, output_size)
        - RGB/RGBA: (output_size, output_size, channels)
        
    Raises:
        ValueError: Invalid input format or parameters
        MemoryError: Insufficient memory for processing
        ImageProcessingError: Other processing errors
        
    Examples:
        >>> # Process RGB image
        >>> rgb_img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        >>> result = sanitize_image(rgb_img, output_size=224)
        >>> assert result.shape == (224, 224, 3)
        >>> assert 0 <= result.min() and result.max() <= 1
        
        >>> # Process grayscale with aspect ratio preservation
        >>> gray_img = np.random.randint(0, 256, (300, 600), dtype=np.uint8)
        >>> result = sanitize_image(gray_img, preserve_aspect_ratio=True)
        >>> assert result.shape == (224, 224)
    """
    start_time = time.time()
    
    # Input validation
    if image is None:
        raise ValueError("Input image cannot be None. Please provide a valid numpy array.")
    
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Input must be numpy array, got {type(image).__name__}. "
                        f"Convert your data to numpy array first.")
    
    if image.size == 0:
        raise ValueError("Input image cannot be empty. Please provide an image with at least one pixel.")
    
    if output_size <= 0:
        raise ValueError(f"Output size must be positive integer, got {output_size}. "
                        f"Common sizes are 64, 128, 224, 256, 512.")
    
    # Memory check
    estimated_memory = _estimate_memory_usage(image, output_size)
    if estimated_memory > max_memory_mb:
        raise MemoryError(
            f"Estimated memory usage ({estimated_memory:.1f}MB) exceeds limit ({max_memory_mb}MB). "
            f"Input shape: {image.shape}, target size: {output_size}x{output_size}. "
            f"Consider reducing image size or increasing memory limit."
        )
    
    try:
        # Normalize image dimensions and data type
        normalized_image = _normalize_image_format(image)
        
        # Resize to target dimensions
        resized_image = _resize_image(
            normalized_image, 
            output_size, 
            preserve_aspect_ratio, 
            interpolation
        )
        
        # Normalize pixel values to [0, 1]
        final_image = _normalize_pixel_values(resized_image)
        
        # Log processing time for monitoring
        processing_time = time.time() - start_time
        logger.debug(f"Image processed in {processing_time:.3f}s, shape: {image.shape} -> {final_image.shape}")
        
        return final_image
        
    except Exception as e:
        logger.error(f"Image processing failed: {str(e)}")
        if isinstance(e, (ValueError, MemoryError, ImageProcessingError)):
            raise
        else:
            raise ImageProcessingError(f"Unexpected error during processing: {str(e)}")


def _normalize_image_format(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to standard format (H, W) or (H, W, C).
    
    Args:
        image: Input image array
        
    Returns:
        Normalized image array
        
    Raises:
        ValueError: Unsupported image format
    """
    # Handle different input dimensions
    if image.ndim == 1:
        # Try to infer if it's a flattened square image
        size = int(np.sqrt(image.size))
        if size * size == image.size:
            image = image.reshape(size, size)
            logger.debug(f"Reshaped 1D array to {size}x{size} grayscale image")
        else:
            raise ValueError(f"Cannot reshape 1D array of size {image.size} to square image. "
                           f"Array size must be a perfect square (e.g., 64, 100, 144, 256, etc.). "
                           f"Closest perfect squares: {int(np.sqrt(image.size))**2}, {(int(np.sqrt(image.size))+1)**2}")
    
    elif image.ndim == 2:
        # Already in (H, W) format - grayscale
        pass
    
    elif image.ndim == 3:
        h, w, c = image.shape
        if c == 1:
            # Convert (H, W, 1) to (H, W)
            image = image.squeeze(axis=2)
        elif c in [3, 4]:
            # RGB or RGBA - keep as is
            pass
        else:
            raise ValueError(f"Unsupported number of channels: {c}, expected 1, 3, or 4. "
                           f"Got shape {image.shape}. Supported formats: "
                           f"grayscale (H,W,1), RGB (H,W,3), RGBA (H,W,4)")
    
    elif image.ndim == 4:
        # Might be batch dimension or unusual format
        if image.shape[0] == 1:
            # Remove batch dimension
            image = image.squeeze(axis=0)
            return _normalize_image_format(image)  # Recursive call
        else:
            raise ValueError(f"4D arrays not supported, got shape {image.shape}")
    
    else:
        raise ValueError(f"Unsupported image dimensions: {image.ndim}D, expected 1-3D")
    
    # Ensure minimum size
    if image.ndim >= 2 and (image.shape[0] < 1 or image.shape[1] < 1):
        raise ValueError(f"Image dimensions too small: {image.shape}. "
                       f"Each dimension must be at least 1 pixel.")
    
    return image


def _resize_image(
    image: np.ndarray, 
    output_size: int, 
    preserve_aspect_ratio: bool,
    interpolation: InterpolationMethod
) -> np.ndarray:
    """
    Resize image to target square dimensions.
    
    Args:
        image: Normalized image array
        output_size: Target square size
        preserve_aspect_ratio: Whether to preserve aspect ratio
        interpolation: Interpolation method
        
    Returns:
        Resized image array
    """
    if image.ndim == 2:
        h, w = image.shape
        target_shape = (output_size, output_size)
    else:
        h, w, c = image.shape
        target_shape = (output_size, output_size, c)
    
    # If already correct size, return copy
    if (h, w) == (output_size, output_size):
        return image.copy()
    
    if preserve_aspect_ratio:
        return _resize_with_padding(image, output_size, interpolation)
    else:
        return _resize_direct(image, target_shape, interpolation)


def _resize_with_padding(
    image: np.ndarray, 
    output_size: int, 
    interpolation: InterpolationMethod
) -> np.ndarray:
    """
    Resize image while preserving aspect ratio using padding.
    
    Args:
        image: Input image
        output_size: Target size
        interpolation: Interpolation method
        
    Returns:
        Resized and padded image
    """
    if image.ndim == 2:
        h, w = image.shape
        is_grayscale = True
    else:
        h, w, c = image.shape
        is_grayscale = False
    
    # Calculate scaling factor to fit within output_size
    scale = min(output_size / h, output_size / w)
    
    # New dimensions after scaling
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    # Resize to new dimensions
    if is_grayscale:
        zoom_factors = (new_h / h, new_w / w)
        resized = zoom(image, zoom_factors, order=_get_scipy_order(interpolation))
    else:
        zoom_factors = (new_h / h, new_w / w, 1)
        resized = zoom(image, zoom_factors, order=_get_scipy_order(interpolation))
    
    # Create output array with padding
    if is_grayscale:
        result = np.zeros((output_size, output_size), dtype=image.dtype)
    else:
        result = np.zeros((output_size, output_size, c), dtype=image.dtype)
    
    # Calculate padding offsets
    pad_h = (output_size - new_h) // 2
    pad_w = (output_size - new_w) // 2
    
    # Place resized image in center
    if is_grayscale:
        result[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
    else:
        result[pad_h:pad_h + new_h, pad_w:pad_w + new_w, :] = resized
    
    return result


def _resize_direct(
    image: np.ndarray, 
    target_shape: Tuple[int, ...], 
    interpolation: InterpolationMethod
) -> np.ndarray:
    """
    Direct resize without preserving aspect ratio.
    
    Args:
        image: Input image
        target_shape: Target dimensions
        interpolation: Interpolation method
        
    Returns:
        Resized image
    """
    if image.ndim == 2:
        h, w = image.shape
        zoom_factors = (target_shape[0] / h, target_shape[1] / w)
    else:
        h, w, c = image.shape
        zoom_factors = (target_shape[0] / h, target_shape[1] / w, 1)
    
    return zoom(image, zoom_factors, order=_get_scipy_order(interpolation))


def _normalize_pixel_values(image: np.ndarray) -> np.ndarray:
    """
    Normalize pixel values to [0, 1] range.
    
    Args:
        image: Input image with any value range
        
    Returns:
        Image with values normalized to [0, 1]
    """
    # Handle different data types
    if image.dtype == np.uint8:
        # Standard 8-bit image
        return image.astype(np.float32) / 255.0
    
    elif image.dtype in [np.uint16, np.uint32, np.uint64]:
        # Higher bit depth integers
        max_val = np.iinfo(image.dtype).max
        return image.astype(np.float32) / max_val
    
    elif image.dtype in [np.int8, np.int16, np.int32, np.int64]:
        # Signed integers - shift to positive range then normalize
        min_val = np.iinfo(image.dtype).min
        max_val = np.iinfo(image.dtype).max
        shifted = image.astype(np.float32) - min_val
        return shifted / (max_val - min_val)
    
    elif image.dtype in [np.float16, np.float32, np.float64]:
        # Floating point - normalize based on actual range
        img_min = image.min()
        img_max = image.max()
        
        # Handle edge cases
        if img_min == img_max:
            # Uniform image
            if img_min == 0:
                return image.astype(np.float32)
            else:
                return np.full_like(image, 0.5, dtype=np.float32)
        
        # Check if already in [0, 1] range
        if 0 <= img_min and img_max <= 1:
            return image.astype(np.float32)
        
        # Normalize to [0, 1]
        normalized = (image.astype(np.float32) - img_min) / (img_max - img_min)
        return normalized
    
    else:
        # Unknown dtype - try to convert and normalize
        logger.warning(f"Unknown dtype {image.dtype}, attempting conversion")
        float_image = image.astype(np.float32)
        img_min = float_image.min()
        img_max = float_image.max()
        
        if img_min == img_max:
            return np.full_like(float_image, 0.5)
        
        return (float_image - img_min) / (img_max - img_min)


def _get_scipy_order(interpolation: InterpolationMethod) -> int:
    """
    Convert interpolation method name to scipy order parameter.
    
    Args:
        interpolation: Interpolation method name
        
    Returns:
        Scipy interpolation order
    """
    mapping = {
        'nearest': 0,
        'bilinear': 1,
        'bicubic': 3,
        'lanczos': 3  # Approximate with cubic
    }
    
    return mapping.get(interpolation, 1)  # Default to bilinear


def _estimate_memory_usage(image: np.ndarray, output_size: int) -> float:
    """
    Estimate memory usage for image processing in MB with improved accuracy.
    
    Args:
        image: Input image
        output_size: Target output size
        
    Returns:
        Estimated memory usage in MB
    """
    # Input image size (always kept in memory during processing)
    input_bytes = image.nbytes
    
    # Determine number of channels for output
    if image.ndim == 1:
        channels = 1  # 1D arrays become grayscale
    elif image.ndim == 2:
        channels = 1  # Grayscale
    elif image.ndim == 3:
        channels = image.shape[2]  # RGB/RGBA
    else:
        channels = 1  # Fallback
    
    # Output image size (always float32)
    output_bytes = output_size * output_size * channels * 4
    
    # Intermediate arrays during processing:
    # 1. Normalized format array (same size as input, converted to float32)
    normalized_bytes = image.size * 4
    
    # 2. Resizing might create temporary arrays (scipy.zoom)
    # Worst case: 2x the larger of input/output for interpolation buffers
    resize_temp_bytes = max(normalized_bytes, output_bytes) * 2
    
    # 3. Additional overhead for function call stack, variables, etc.
    overhead_bytes = min(input_bytes * 0.1, 50 * 1024 * 1024)  # Max 50MB overhead
    
    # Total memory estimation
    total_bytes = (input_bytes +           # Original input
                  normalized_bytes +       # Normalized copy
                  resize_temp_bytes +      # Resizing temporaries
                  output_bytes +           # Final output
                  overhead_bytes)          # System overhead
    
    return total_bytes / (1024 * 1024)  # Convert to MB


# Performance monitoring utilities
def benchmark_sanitize_image(
    test_shapes: list = None,
    output_size: int = 224,
    num_runs: int = 10
) -> dict:
    """
    Benchmark sanitize_image function performance.
    
    Args:
        test_shapes: List of (H, W, C) shapes to test
        output_size: Target output size
        num_runs: Number of runs per test
        
    Returns:
        Dictionary with benchmark results
    """
    if test_shapes is None:
        test_shapes = [
            (480, 640, 3),    # VGA
            (720, 1280, 3),   # HD
            (1080, 1920, 3),  # Full HD
            (2160, 3840, 3),  # 4K
        ]
    
    results = {}
    
    for shape in test_shapes:
        # Generate random test image
        test_image = np.random.randint(0, 256, shape, dtype=np.uint8)
        
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            _ = sanitize_image(test_image, output_size=output_size)
            times.append(time.time() - start_time)
        
        results[f"{shape[1]}x{shape[0]}"] = {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'input_size_mb': test_image.nbytes / (1024 * 1024)
        }
    
    return results