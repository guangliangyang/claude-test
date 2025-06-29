"""
Canva Image Processing Module V2 - Refactored Architecture

This module provides the new modular image processing system while maintaining
backward compatibility with the original sanitize_image function.
"""

import numpy as np
from typing import Optional

# Import new modular components
from .core.processors import CanvaImageProcessor, ProcessorFactory
from .types import ProcessingConfig, InterpolationMethod
from .exceptions import ImageProcessingError, MemoryLimitError, ValidationError

# Re-export for backward compatibility
from .exceptions import ImageProcessingError
MemoryError = MemoryLimitError  # Alias for backward compatibility


def sanitize_image(
    image: np.ndarray,
    output_size: int = 224,
    preserve_aspect_ratio: bool = True,
    interpolation: InterpolationMethod = 'bilinear',
    max_memory_mb: int = 500
) -> np.ndarray:
    """
    Sanitize image array to normalized square matrix for Canva AI pipeline.
    
    This is the backward-compatible interface that uses the new modular architecture.
    
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
        MemoryLimitError: Insufficient memory for processing
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
    # Create configuration object
    config = ProcessingConfig(
        output_size=output_size,
        preserve_aspect_ratio=preserve_aspect_ratio,
        interpolation=interpolation,
        max_memory_mb=max_memory_mb
    )
    
    # Use the new processor architecture
    processor = ProcessorFactory.create_standard_processor()
    
    try:
        return processor.process(image, config)
    except ValidationError as e:
        # Convert to ValueError for backward compatibility
        raise ValueError(str(e))
    except MemoryLimitError as e:
        # Re-raise with original name for compatibility
        raise MemoryError(str(e))


class ImageProcessorV2:
    """
    Advanced image processor with full access to new architecture.
    
    Provides the complete API for the new modular system while maintaining
    ease of use for common operations.
    """
    
    def __init__(self, processor: Optional[CanvaImageProcessor] = None):
        """
        Initialize with optional custom processor.
        
        Args:
            processor: Custom processor instance, defaults to standard processor
        """
        self.processor = processor or ProcessorFactory.create_standard_processor()
    
    def process(
        self,
        image: np.ndarray,
        config: Optional[ProcessingConfig] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Process image with full configuration control.
        
        Args:
            image: Input image array
            config: Processing configuration object
            **kwargs: Configuration parameters (alternative to config object)
            
        Returns:
            Processed image array
        """
        if config is None and kwargs:
            config = ProcessingConfig(**kwargs)
        elif config is None:
            config = ProcessingConfig()
        
        return self.processor.process(image, config)
    
    def estimate_memory(
        self,
        image: np.ndarray,
        config: Optional[ProcessingConfig] = None,
        **kwargs
    ) -> float:
        """
        Estimate memory usage for processing.
        
        Args:
            image: Input image array
            config: Processing configuration
            **kwargs: Configuration parameters
            
        Returns:
            Estimated memory usage in MB
        """
        if config is None and kwargs:
            config = ProcessingConfig(**kwargs)
        elif config is None:
            config = ProcessingConfig()
        
        return self.processor.estimate_memory_usage(image, config)
    
    def process_batch(
        self,
        images: list[np.ndarray],
        config: Optional[ProcessingConfig] = None,
        **kwargs
    ) -> list[np.ndarray]:
        """
        Process multiple images with the same configuration.
        
        Args:
            images: List of input images
            config: Processing configuration
            **kwargs: Configuration parameters
            
        Returns:
            List of processed images
        """
        if config is None and kwargs:
            config = ProcessingConfig(**kwargs)
        elif config is None:
            config = ProcessingConfig()
        
        return [self.processor.process(img, config) for img in images]


# Factory functions for common use cases
def create_quality_processor() -> ImageProcessorV2:
    """Create processor optimized for image quality."""
    return ImageProcessorV2(ProcessorFactory.create_high_quality_processor())


def create_speed_processor() -> ImageProcessorV2:
    """Create processor optimized for processing speed."""
    return ImageProcessorV2(ProcessorFactory.create_fast_processor())


def create_custom_processor(**components) -> ImageProcessorV2:
    """Create processor with custom components."""
    return ImageProcessorV2(ProcessorFactory.create_custom_processor(**components))