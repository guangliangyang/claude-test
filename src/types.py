"""
Type definitions for image processing system.

Centralizes all type definitions to improve code clarity and maintainability.
"""

from typing import Literal, NamedTuple, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np

# Type aliases
InterpolationMethod = Literal['nearest', 'bilinear', 'bicubic', 'lanczos']
ImageShape = tuple[int, ...]
ImageDtype = type[np.number]


@dataclass
class ProcessingConfig:
    """Configuration for image processing operations."""
    output_size: int = 224
    preserve_aspect_ratio: bool = True
    interpolation: InterpolationMethod = 'bilinear'
    max_memory_mb: int = 500
    target_dtype: ImageDtype = np.float32


class ValidationResult(NamedTuple):
    """Result of image validation."""
    is_valid: bool
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class TransformationResult(NamedTuple):
    """Result of image transformation."""
    image: np.ndarray
    metadata: Dict[str, Any]
    processing_time: float


class ImageMetadata(NamedTuple):
    """Metadata about an image."""
    shape: ImageShape
    dtype: ImageDtype
    channels: int
    is_grayscale: bool
    is_rgb: bool
    is_rgba: bool
    size_mb: float