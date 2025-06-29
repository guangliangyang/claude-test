"""
Main image processing orchestrator using dependency injection.

Implements the Strategy pattern and Template Method pattern for loose coupling.
"""

import numpy as np
import time
import logging
from typing import Optional, Dict, Any

from .interfaces import ImageProcessor, ImageValidator, ImageTransformer
from .validators import ImageInputValidator, ConfigurationValidator, MemoryValidator
from .transformers import ImageFormatTransformer, ImageResizer, ImageNormalizer
from ..types import ProcessingConfig, ValidationResult
from ..exceptions import ValidationError, MemoryLimitError, ConfigurationError

logger = logging.getLogger(__name__)


class CanvaImageProcessor(ImageProcessor):
    """
    Main image processor implementing Strategy and Template Method patterns.
    
    Uses dependency injection for loose coupling and high testability.
    """
    
    def __init__(
        self,
        input_validator: Optional[ImageValidator] = None,
        format_transformer: Optional[ImageTransformer] = None,
        resizer: Optional[ImageTransformer] = None,
        normalizer: Optional[ImageTransformer] = None
    ):
        """
        Initialize processor with injectable dependencies.
        
        Args:
            input_validator: Validator for input images
            format_transformer: Transformer for format normalization
            resizer: Transformer for image resizing
            normalizer: Transformer for pixel normalization
        """
        # Use dependency injection with defaults
        self.input_validator = input_validator or ImageInputValidator()
        self.format_transformer = format_transformer or ImageFormatTransformer()
        self.resizer = resizer or ImageResizer()
        self.normalizer = normalizer or ImageNormalizer()
        
        # Configuration validator (stateless, no injection needed)
        self.config_validator = ConfigurationValidator()
    
    def process(
        self, 
        image: Any, 
        config: Optional[ProcessingConfig] = None
    ) -> np.ndarray:
        """
        Process image through complete pipeline using Template Method pattern.
        
        Args:
            image: Input image
            config: Processing configuration
            
        Returns:
            Processed image array
            
        Raises:
            ValidationError: Invalid input or configuration
            MemoryLimitError: Memory limit exceeded
            ConfigurationError: Invalid configuration parameters
        """
        start_time = time.time()
        
        # Use default config if not provided
        if config is None:
            config = ProcessingConfig()
        
        try:
            # Template method steps
            self._validate_configuration(config)
            validated_image = self._validate_input(image)
            self._check_memory_constraints(validated_image, config)
            
            # Process through transformation pipeline
            formatted_image = self._transform_format(validated_image, config)
            resized_image = self._transform_resize(formatted_image, config)
            normalized_image = self._transform_normalize(resized_image, config)
            
            # Log successful processing
            processing_time = time.time() - start_time
            logger.debug(
                f"Image processed successfully in {processing_time:.3f}s: "
                f"{image.shape if hasattr(image, 'shape') else 'unknown'} -> {normalized_image.shape}"
            )
            
            return normalized_image
            
        except (ValidationError, MemoryLimitError, ConfigurationError):
            # Re-raise expected errors
            raise
        except Exception as e:
            # Wrap unexpected errors
            logger.error(f"Unexpected error during image processing: {str(e)}")
            raise ValidationError(f"Processing failed: {str(e)}")
    
    def estimate_memory_usage(self, image: Any, config: ProcessingConfig) -> float:
        """
        Estimate memory usage for processing.
        
        Args:
            image: Input image
            config: Processing configuration
            
        Returns:
            Estimated memory usage in MB
        """
        if not isinstance(image, np.ndarray):
            return 0.0
        
        memory_validator = MemoryValidator(config.max_memory_mb)
        result = memory_validator.validate_memory_requirements(image, config.output_size)
        
        return result.metadata.get('estimated_memory_mb', 0.0) if result.metadata else 0.0
    
    # Template method steps (protected methods)
    
    def _validate_configuration(self, config: ProcessingConfig) -> None:
        """Validate processing configuration."""
        # Validate output size
        size_result = self.config_validator.validate_output_size(config.output_size)
        if not size_result.is_valid:
            raise ConfigurationError(size_result.error_message)
        
        # Validate interpolation method
        interp_result = self.config_validator.validate_interpolation(config.interpolation)
        if not interp_result.is_valid:
            raise ConfigurationError(interp_result.error_message)
    
    def _validate_input(self, image: Any) -> np.ndarray:
        """Validate input image."""
        result = self.input_validator.validate(image)
        
        if not result.is_valid:
            raise ValidationError(result.error_message)
        
        return image  # Already validated as np.ndarray
    
    def _check_memory_constraints(self, image: np.ndarray, config: ProcessingConfig) -> None:
        """Check memory usage constraints."""
        memory_validator = MemoryValidator(config.max_memory_mb)
        result = memory_validator.validate_memory_requirements(image, config.output_size)
        
        if not result.is_valid:
            raise MemoryLimitError(result.error_message)
    
    def _transform_format(self, image: np.ndarray, config: ProcessingConfig) -> np.ndarray:
        """Transform image format."""
        result = self.format_transformer.transform(image, config)
        return result.image
    
    def _transform_resize(self, image: np.ndarray, config: ProcessingConfig) -> np.ndarray:
        """Transform image size."""
        # Skip resize if already correct size
        if image.ndim == 2:
            h, w = image.shape
        else:
            h, w = image.shape[:2]
        
        if h == config.output_size and w == config.output_size:
            return image.copy()
        
        result = self.resizer.transform(image, config)
        return result.image
    
    def _transform_normalize(self, image: np.ndarray, config: ProcessingConfig) -> np.ndarray:
        """Transform pixel values."""
        result = self.normalizer.transform(image, config)
        return result.image


class ProcessingPipelineBuilder:
    """
    Builder pattern for creating custom processing pipelines.
    
    Allows flexible configuration of processing components.
    """
    
    def __init__(self):
        self._input_validator = None
        self._format_transformer = None
        self._resizer = None
        self._normalizer = None
    
    def with_input_validator(self, validator: ImageValidator) -> 'ProcessingPipelineBuilder':
        """Set custom input validator."""
        self._input_validator = validator
        return self
    
    def with_format_transformer(self, transformer: ImageTransformer) -> 'ProcessingPipelineBuilder':
        """Set custom format transformer."""
        self._format_transformer = transformer
        return self
    
    def with_resizer(self, resizer: ImageTransformer) -> 'ProcessingPipelineBuilder':
        """Set custom resizer."""
        self._resizer = resizer
        return self
    
    def with_normalizer(self, normalizer: ImageTransformer) -> 'ProcessingPipelineBuilder':
        """Set custom normalizer."""
        self._normalizer = normalizer
        return self
    
    def build(self) -> CanvaImageProcessor:
        """Build the configured processor."""
        return CanvaImageProcessor(
            input_validator=self._input_validator,
            format_transformer=self._format_transformer,
            resizer=self._resizer,
            normalizer=self._normalizer
        )


class ProcessorFactory:
    """
    Factory pattern for creating pre-configured processors.
    
    Provides common processor configurations.
    """
    
    @staticmethod
    def create_standard_processor() -> CanvaImageProcessor:
        """Create processor with standard configuration."""
        return CanvaImageProcessor()
    
    @staticmethod
    def create_high_quality_processor() -> CanvaImageProcessor:
        """Create processor optimized for quality."""
        # Could inject custom transformers optimized for quality
        return CanvaImageProcessor()
    
    @staticmethod
    def create_fast_processor() -> CanvaImageProcessor:
        """Create processor optimized for speed."""
        # Could inject custom transformers optimized for speed
        return CanvaImageProcessor()
    
    @staticmethod
    def create_custom_processor(
        **components
    ) -> CanvaImageProcessor:
        """Create processor with custom components."""
        return CanvaImageProcessor(**components)