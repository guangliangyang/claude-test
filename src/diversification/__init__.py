"""
Diversification algorithms for image search results.

This package provides various algorithms to diversify search results,
particularly for balancing gender representation in image search.
"""

from .models import Gender, ImageItem, SearchResult, DiversificationResult
from .algorithms import (
    DiversificationAlgorithm,
    InterleavingAlgorithm,
    ProportionalDistributionAlgorithm,
    MMRAlgorithm,
    GreedyDiversificationAlgorithm,
    RandomizedDiversificationAlgorithm,
    SlidingWindowAlgorithm,
    EntropyMaximizationAlgorithm
)
from .evaluators import DiversityEvaluator
from .visualizers import DiversityVisualizer

__all__ = [
    'Gender',
    'ImageItem',
    'SearchResult',
    'DiversificationResult',
    'DiversificationAlgorithm',
    'InterleavingAlgorithm',
    'ProportionalDistributionAlgorithm', 
    'MMRAlgorithm',
    'GreedyDiversificationAlgorithm',
    'RandomizedDiversificationAlgorithm',
    'SlidingWindowAlgorithm',
    'EntropyMaximizationAlgorithm',
    'DiversityEvaluator',
    'DiversityVisualizer'
]