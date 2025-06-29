"""
Custom exceptions for image processing system.

Provides specific exception types for better error handling and debugging.
"""


class ImageProcessingError(Exception):
    """Base exception for image processing errors."""
    pass


class ValidationError(ImageProcessingError):
    """Exception raised for input validation failures."""
    pass


class TransformationError(ImageProcessingError):
    """Exception raised during image transformation."""
    pass


class MemoryLimitError(ImageProcessingError):
    """Exception raised when memory constraints are exceeded."""
    pass


class ConfigurationError(ImageProcessingError):
    """Exception raised for invalid configuration parameters."""
    pass