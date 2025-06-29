"""
Input validation components with high cohesion.

Each validator has a single, well-defined responsibility.
"""

import numpy as np
from typing import Any

from .interfaces import ImageValidator
from ..types import ValidationResult, ImageMetadata
from ..exceptions import ValidationError


class ImageInputValidator(ImageValidator):
    """Validates basic image input requirements."""
    
    def validate(self, image: Any) -> ValidationResult:
        """Validate image input with detailed error reporting."""
        try:
            # Check for None
            if image is None:
                return ValidationResult(
                    is_valid=False,
                    error_message="Input image cannot be None. Please provide a valid numpy array."
                )
            
            # Check type
            if not isinstance(image, np.ndarray):
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Input must be numpy array, got {type(image).__name__}. "
                                f"Convert your data to numpy array first."
                )
            
            # Check emptiness
            if image.size == 0:
                return ValidationResult(
                    is_valid=False,
                    error_message="Input image cannot be empty. Please provide an image with at least one pixel."
                )
            
            # Check dimensions
            if image.ndim < 1 or image.ndim > 4:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Unsupported image dimensions: {image.ndim}D, expected 1-3D"
                )
            
            # Validate channels for 3D images
            if image.ndim == 3:
                channels = image.shape[2]
                if channels not in [1, 3, 4]:
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"Unsupported number of channels: {channels}, expected 1, 3, or 4. "
                                    f"Got shape {image.shape}. Supported formats: "
                                    f"grayscale (H,W,1), RGB (H,W,3), RGBA (H,W,4)"
                    )
            
            # Generate metadata for valid images
            metadata = self._extract_metadata(image)
            
            return ValidationResult(
                is_valid=True,
                metadata=metadata._asdict()
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Validation failed: {str(e)}"
            )
    
    def _extract_metadata(self, image: np.ndarray) -> ImageMetadata:
        """Extract comprehensive metadata from image."""
        shape = image.shape
        dtype = image.dtype
        
        # Determine channel information
        if image.ndim == 1:
            channels = 1
            is_grayscale, is_rgb, is_rgba = True, False, False
        elif image.ndim == 2:
            channels = 1
            is_grayscale, is_rgb, is_rgba = True, False, False
        elif image.ndim == 3:
            channels = shape[2]
            is_grayscale = channels == 1
            is_rgb = channels == 3
            is_rgba = channels == 4
        else:
            channels = 1
            is_grayscale, is_rgb, is_rgba = True, False, False
        
        size_mb = image.nbytes / (1024 * 1024)
        
        return ImageMetadata(
            shape=shape,
            dtype=dtype,
            channels=channels,
            is_grayscale=is_grayscale,
            is_rgb=is_rgb,
            is_rgba=is_rgba,
            size_mb=size_mb
        )


class ConfigurationValidator:
    """Validates processing configuration parameters."""
    
    @staticmethod
    def validate_output_size(output_size: int) -> ValidationResult:
        """Validate output size parameter."""
        if output_size <= 0:
            return ValidationResult(
                is_valid=False,
                error_message=f"Output size must be positive integer, got {output_size}. "
                            f"Common sizes are 64, 128, 224, 256, 512."
            )
        return ValidationResult(is_valid=True)
    
    @staticmethod
    def validate_interpolation(interpolation: str) -> ValidationResult:
        """Validate interpolation method."""
        valid_methods = ['nearest', 'bilinear', 'bicubic', 'lanczos']
        if interpolation not in valid_methods:
            return ValidationResult(
                is_valid=False,
                error_message=f"Invalid interpolation method: {interpolation}. "
                            f"Valid methods: {valid_methods}"
            )
        return ValidationResult(is_valid=True)


class MemoryValidator:
    """Validates memory usage constraints."""
    
    def __init__(self, max_memory_mb: int = 500):
        self.max_memory_mb = max_memory_mb
    
    def validate_memory_requirements(
        self, 
        image: np.ndarray, 
        output_size: int
    ) -> ValidationResult:
        """Validate memory requirements for processing."""
        estimated_memory = self._estimate_memory_usage(image, output_size)
        
        if estimated_memory > self.max_memory_mb:
            return ValidationResult(
                is_valid=False,
                error_message=f"Estimated memory usage ({estimated_memory:.1f}MB) exceeds limit ({self.max_memory_mb}MB). "
                            f"Input shape: {image.shape}, target size: {output_size}x{output_size}. "
                            f"Consider reducing image size or increasing memory limit.",
                metadata={'estimated_memory_mb': estimated_memory}
            )
        
        return ValidationResult(
            is_valid=True,
            metadata={'estimated_memory_mb': estimated_memory}
        )
    
    def _estimate_memory_usage(self, image: np.ndarray, output_size: int) -> float:
        """Estimate memory usage with improved accuracy."""
        input_bytes = image.nbytes
        
        # Determine output channels
        if image.ndim == 1:
            channels = 1
        elif image.ndim == 2:
            channels = 1
        elif image.ndim == 3:
            channels = image.shape[2]
        else:
            channels = 1
        
        # Output size (float32)
        output_bytes = output_size * output_size * channels * 4
        
        # Intermediate arrays
        normalized_bytes = image.size * 4
        resize_temp_bytes = max(normalized_bytes, output_bytes) * 2
        overhead_bytes = min(input_bytes * 0.1, 50 * 1024 * 1024)
        
        total_bytes = (input_bytes + normalized_bytes + 
                      resize_temp_bytes + output_bytes + overhead_bytes)
        
        return total_bytes / (1024 * 1024)