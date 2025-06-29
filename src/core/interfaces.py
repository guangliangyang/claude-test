"""
Abstract interfaces for image processing components.

Defines the contracts for different image processing responsibilities,
promoting loose coupling and testability.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np
from ..types import ProcessingConfig, ValidationResult, TransformationResult


class ImageValidator(ABC):
    """Abstract interface for image input validation."""
    
    @abstractmethod
    def validate(self, image: Any) -> ValidationResult:
        """
        Validate image input.
        
        Args:
            image: Input to validate
            
        Returns:
            ValidationResult with validation status and details
        """
        pass


class ImageTransformer(ABC):
    """Abstract interface for image transformations."""
    
    @abstractmethod
    def transform(self, image: np.ndarray, config: ProcessingConfig) -> TransformationResult:
        """
        Transform image according to configuration.
        
        Args:
            image: Input image array
            config: Transformation configuration
            
        Returns:
            TransformationResult with transformed image and metadata
        """
        pass


class ImageProcessor(ABC):
    """Main interface for complete image processing pipeline."""
    
    @abstractmethod
    def process(
        self, 
        image: Any, 
        config: Optional[ProcessingConfig] = None
    ) -> np.ndarray:
        """
        Process image through complete pipeline.
        
        Args:
            image: Input image
            config: Processing configuration
            
        Returns:
            Processed image array
        """
        pass
    
    @abstractmethod
    def estimate_memory_usage(self, image: Any, config: ProcessingConfig) -> float:
        """
        Estimate memory usage for processing.
        
        Args:
            image: Input image
            config: Processing configuration
            
        Returns:
            Estimated memory usage in MB
        """
        pass