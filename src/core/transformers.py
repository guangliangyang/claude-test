"""
Image transformation components with single responsibilities.

Each transformer handles one specific aspect of image processing.
"""

import numpy as np
import time
from scipy.ndimage import zoom
from typing import Tuple

from .interfaces import ImageTransformer
from ..types import ProcessingConfig, TransformationResult, InterpolationMethod
from ..exceptions import TransformationError


class ImageFormatTransformer(ImageTransformer):
    """Transforms image format and dimensions."""
    
    def transform(self, image: np.ndarray, config: ProcessingConfig) -> TransformationResult:
        """Normalize image to standard format (H, W) or (H, W, C)."""
        start_time = time.time()
        
        try:
            normalized_image = self._normalize_image_format(image)
            processing_time = time.time() - start_time
            
            return TransformationResult(
                image=normalized_image,
                metadata={
                    'operation': 'format_normalization',
                    'input_shape': image.shape,
                    'output_shape': normalized_image.shape,
                    'input_dtype': str(image.dtype),
                    'output_dtype': str(normalized_image.dtype)
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            raise TransformationError(f"Format transformation failed: {str(e)}")
    
    def _normalize_image_format(self, image: np.ndarray) -> np.ndarray:
        """Convert image to standard format."""
        if image.ndim == 1:
            # Try to reshape 1D to 2D if perfect square
            size = int(np.sqrt(image.size))
            if size * size == image.size:
                image = image.reshape(size, size)
            else:
                raise ValueError(
                    f"Cannot reshape 1D array of size {image.size} to square image. "
                    f"Array size must be a perfect square (e.g., 64, 100, 144, 256, etc.). "
                    f"Closest perfect squares: {int(np.sqrt(image.size))**2}, {(int(np.sqrt(image.size))+1)**2}"
                )
        
        elif image.ndim == 2:
            # Already in (H, W) format
            pass
        
        elif image.ndim == 3:
            h, w, c = image.shape
            if c == 1:
                # Convert (H, W, 1) to (H, W)
                image = image.squeeze(axis=2)
            # RGB/RGBA stay as is
        
        elif image.ndim == 4:
            if image.shape[0] == 1:
                # Remove batch dimension
                image = image.squeeze(axis=0)
                return self._normalize_image_format(image)
            else:
                raise ValueError(f"4D arrays not supported, got shape {image.shape}")
        
        # Ensure minimum size
        if image.ndim >= 2 and (image.shape[0] < 1 or image.shape[1] < 1):
            raise ValueError(
                f"Image dimensions too small: {image.shape}. "
                f"Each dimension must be at least 1 pixel."
            )
        
        return image


class ImageResizer(ImageTransformer):
    """Handles image resizing operations."""
    
    def transform(self, image: np.ndarray, config: ProcessingConfig) -> TransformationResult:
        """Resize image to target dimensions."""
        start_time = time.time()
        
        try:
            if config.preserve_aspect_ratio:
                resized_image = self._resize_with_padding(
                    image, config.output_size, config.interpolation
                )
            else:
                resized_image = self._resize_direct(
                    image, config.output_size, config.interpolation
                )
            
            processing_time = time.time() - start_time
            
            return TransformationResult(
                image=resized_image,
                metadata={
                    'operation': 'resize',
                    'input_shape': image.shape,
                    'output_shape': resized_image.shape,
                    'preserve_aspect_ratio': config.preserve_aspect_ratio,
                    'interpolation': config.interpolation
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            raise TransformationError(f"Resize transformation failed: {str(e)}")
    
    def _resize_with_padding(
        self, 
        image: np.ndarray, 
        output_size: int, 
        interpolation: InterpolationMethod
    ) -> np.ndarray:
        """Resize with aspect ratio preservation using padding."""
        if image.ndim == 2:
            h, w = image.shape
            is_grayscale = True
        else:
            h, w, c = image.shape
            is_grayscale = False
        
        # Calculate scaling factor
        scale = min(output_size / h, output_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize to new dimensions
        if is_grayscale:
            zoom_factors = (new_h / h, new_w / w)
        else:
            zoom_factors = (new_h / h, new_w / w, 1)
        
        resized = zoom(image, zoom_factors, order=self._get_scipy_order(interpolation))
        
        # Create output with padding
        if is_grayscale:
            result = np.zeros((output_size, output_size), dtype=image.dtype)
        else:
            result = np.zeros((output_size, output_size, c), dtype=image.dtype)
        
        # Center the resized image
        pad_h = (output_size - new_h) // 2
        pad_w = (output_size - new_w) // 2
        
        if is_grayscale:
            result[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        else:
            result[pad_h:pad_h + new_h, pad_w:pad_w + new_w, :] = resized
        
        return result
    
    def _resize_direct(
        self, 
        image: np.ndarray, 
        output_size: int, 
        interpolation: InterpolationMethod
    ) -> np.ndarray:
        """Direct resize without preserving aspect ratio."""
        if image.ndim == 2:
            h, w = image.shape
            zoom_factors = (output_size / h, output_size / w)
        else:
            h, w, c = image.shape
            zoom_factors = (output_size / h, output_size / w, 1)
        
        return zoom(image, zoom_factors, order=self._get_scipy_order(interpolation))
    
    def _get_scipy_order(self, interpolation: InterpolationMethod) -> int:
        """Convert interpolation method to scipy order."""
        mapping = {
            'nearest': 0,
            'bilinear': 1,
            'bicubic': 3,
            'lanczos': 3
        }
        return mapping.get(interpolation, 1)


class ImageNormalizer(ImageTransformer):
    """Normalizes pixel values to target range."""
    
    def transform(self, image: np.ndarray, config: ProcessingConfig) -> TransformationResult:
        """Normalize pixel values to [0, 1] range."""
        start_time = time.time()
        
        try:
            normalized_image = self._normalize_pixel_values(image)
            processing_time = time.time() - start_time
            
            return TransformationResult(
                image=normalized_image,
                metadata={
                    'operation': 'normalization',
                    'input_dtype': str(image.dtype),
                    'output_dtype': str(normalized_image.dtype),
                    'input_range': [float(image.min()), float(image.max())],
                    'output_range': [float(normalized_image.min()), float(normalized_image.max())]
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            raise TransformationError(f"Normalization failed: {str(e)}")
    
    def _normalize_pixel_values(self, image: np.ndarray) -> np.ndarray:
        """Normalize pixel values based on data type."""
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        
        elif image.dtype in [np.uint16, np.uint32, np.uint64]:
            max_val = np.iinfo(image.dtype).max
            return image.astype(np.float32) / max_val
        
        elif image.dtype in [np.int8, np.int16, np.int32, np.int64]:
            min_val = np.iinfo(image.dtype).min
            max_val = np.iinfo(image.dtype).max
            shifted = image.astype(np.float32) - min_val
            return shifted / (max_val - min_val)
        
        elif image.dtype in [np.float16, np.float32, np.float64]:
            img_min, img_max = image.min(), image.max()
            
            if img_min == img_max:
                return np.full_like(image, 0.5, dtype=np.float32) if img_min != 0 else image.astype(np.float32)
            
            if 0 <= img_min and img_max <= 1:
                return image.astype(np.float32)
            
            return (image.astype(np.float32) - img_min) / (img_max - img_min)
        
        else:
            # Unknown dtype - normalize to [0, 1]
            float_image = image.astype(np.float32)
            img_min, img_max = float_image.min(), float_image.max()
            
            if img_min == img_max:
                return np.full_like(float_image, 0.5)
            
            return (float_image - img_min) / (img_max - img_min)