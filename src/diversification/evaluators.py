"""
Evaluation metrics for diversification algorithms.

This module provides comprehensive metrics to evaluate the quality
of diversification algorithms across multiple dimensions.
"""

import math
from typing import List, Dict, Any, Tuple
from collections import Counter, defaultdict
import numpy as np

from .models import ImageItem, SearchResult, DiversificationResult, Gender


class DiversityEvaluator:
    """Comprehensive evaluator for diversification algorithm performance."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.metrics = [
            'alternation_score',
            'gini_coefficient', 
            'entropy_score',
            'gender_balance_score',
            'prefix_diversity',
            'consecutive_penalty',
            'relevance_preservation',
            'overall_score'
        ]
    
    def evaluate_all(self, result: DiversificationResult) -> Dict[str, float]:
        """
        Evaluate all metrics for a diversification result.
        
        Args:
            result: DiversificationResult to evaluate
            
        Returns:
            Dictionary with all metric scores
        """
        scores = {}
        
        # Extract gender sequence
        gender_sequence = [item.gender for item in result.diversified_items]
        
        # Calculate individual metrics
        scores['alternation_score'] = self.alternation_score(gender_sequence)
        scores['gini_coefficient'] = self.gini_coefficient(gender_sequence)
        scores['entropy_score'] = self.entropy_score(gender_sequence)
        scores['gender_balance_score'] = self.gender_balance_score(gender_sequence)
        scores['prefix_diversity'] = self.prefix_diversity(gender_sequence)
        scores['consecutive_penalty'] = self.consecutive_penalty(gender_sequence)
        scores['relevance_preservation'] = self.relevance_preservation(
            result.original_result.items, 
            result.diversified_items
        )
        
        # Calculate overall score (weighted combination)
        scores['overall_score'] = self.overall_score(scores)
        
        return scores
    
    def alternation_score(self, gender_sequence: List[Gender]) -> float:
        """
        Measure how well genders alternate in the sequence.
        
        Score is 1.0 for perfect alternation, decreases with clustering.
        
        Args:
            gender_sequence: List of genders in sequence
            
        Returns:
            Alternation score between 0 and 1
        """
        if len(gender_sequence) <= 1:
            return 1.0
        
        alternations = 0
        for i in range(1, len(gender_sequence)):
            if gender_sequence[i] != gender_sequence[i-1]:
                alternations += 1
        
        # Maximum possible alternations
        max_alternations = len(gender_sequence) - 1
        
        return alternations / max_alternations if max_alternations > 0 else 1.0
    
    def gini_coefficient(self, gender_sequence: List[Gender]) -> float:
        """
        Calculate Gini coefficient for gender distribution inequality.
        
        0 = perfect equality, 1 = maximum inequality
        We return 1 - gini so higher scores are better.
        
        Args:
            gender_sequence: List of genders in sequence
            
        Returns:
            Inverted Gini coefficient (higher is more equal)
        """
        if not gender_sequence:
            return 1.0
        
        # Count each gender
        counts = Counter(gender_sequence)
        values = list(counts.values())
        
        if len(values) <= 1:
            return 1.0  # Perfect equality if only one gender
        
        # Sort values
        values.sort()
        n = len(values)
        
        # Calculate Gini coefficient
        numerator = sum((2 * i - n - 1) * val for i, val in enumerate(values, 1))
        denominator = n * sum(values)
        
        gini = numerator / denominator if denominator > 0 else 0
        
        return 1 - gini  # Invert so higher is better
    
    def entropy_score(self, gender_sequence: List[Gender]) -> float:
        """
        Calculate entropy of gender distribution.
        
        Higher entropy indicates more diversity.
        Normalized to [0, 1] where 1 is maximum entropy.
        
        Args:
            gender_sequence: List of genders in sequence
            
        Returns:
            Normalized entropy score
        """
        if not gender_sequence:
            return 0.0
        
        # Count each gender
        counts = Counter(gender_sequence)
        total = len(gender_sequence)
        
        # Calculate entropy
        entropy = 0
        for count in counts.values():
            if count > 0:
                prob = count / total
                entropy -= prob * math.log2(prob)
        
        # Normalize by maximum possible entropy
        num_genders = len(counts)
        max_entropy = math.log2(num_genders) if num_genders > 1 else 1
        
        return entropy / max_entropy if max_entropy > 0 else 1.0
    
    def gender_balance_score(self, gender_sequence: List[Gender]) -> float:
        """
        Evaluate how balanced the gender distribution is.
        
        Perfect balance gets score 1.0, maximum imbalance gets 0.0.
        
        Args:
            gender_sequence: List of genders in sequence
            
        Returns:
            Balance score between 0 and 1
        """
        if not gender_sequence:
            return 1.0
        
        counts = Counter(gender_sequence)
        total = len(gender_sequence)
        
        if len(counts) <= 1:
            return 0.0  # Only one gender = no balance
        
        # Calculate deviation from perfect balance
        expected_count = total / len(counts)
        total_deviation = sum(abs(count - expected_count) for count in counts.values())
        max_deviation = total - expected_count  # Maximum possible deviation
        
        balance_score = 1 - (total_deviation / (2 * max_deviation)) if max_deviation > 0 else 1.0
        
        return max(0.0, balance_score)
    
    def prefix_diversity(self, gender_sequence: List[Gender], window_sizes: List[int] = None) -> float:
        """
        Evaluate diversity in prefixes of the sequence.
        
        Measures how diverse the first k items are for various values of k.
        
        Args:
            gender_sequence: List of genders in sequence
            window_sizes: Sizes of prefixes to evaluate
            
        Returns:
            Average diversity score across all prefixes
        """
        if window_sizes is None:
            window_sizes = [2, 3, 5, 10]
        
        if not gender_sequence:
            return 1.0
        
        diversity_scores = []
        
        for k in window_sizes:
            if k > len(gender_sequence):
                continue
            
            prefix = gender_sequence[:k]
            unique_genders = len(set(prefix))
            max_possible = min(k, 2)  # Assuming only male/female
            
            if max_possible > 0:
                diversity_scores.append(unique_genders / max_possible)
        
        return np.mean(diversity_scores) if diversity_scores else 1.0
    
    def consecutive_penalty(self, gender_sequence: List[Gender], max_consecutive: int = 3) -> float:
        """
        Penalize long runs of the same gender.
        
        Returns 1.0 if no runs exceed max_consecutive, decreases with longer runs.
        
        Args:
            gender_sequence: List of genders in sequence
            max_consecutive: Maximum allowed consecutive items of same gender
            
        Returns:
            Penalty score between 0 and 1 (higher is better)
        """
        if len(gender_sequence) <= 1:
            return 1.0
        
        max_run_length = 1
        current_run = 1
        
        for i in range(1, len(gender_sequence)):
            if gender_sequence[i] == gender_sequence[i-1]:
                current_run += 1
                max_run_length = max(max_run_length, current_run)
            else:
                current_run = 1
        
        if max_run_length <= max_consecutive:
            return 1.0
        else:
            # Exponential penalty for longer runs
            penalty = math.exp(-(max_run_length - max_consecutive))
            return max(0.0, penalty)
    
    def relevance_preservation(self, original_items: List[ImageItem], diversified_items: List[ImageItem]) -> float:
        """
        Measure how well the diversification preserves relevance ordering.
        
        Uses rank correlation to compare original vs diversified ordering.
        
        Args:
            original_items: Original items in relevance order
            diversified_items: Diversified items
            
        Returns:
            Relevance preservation score between 0 and 1
        """
        if not original_items or not diversified_items:
            return 1.0
        
        # Create relevance rankings
        original_ranking = {item.id: i for i, item in enumerate(original_items)}
        diversified_ranking = {item.id: i for i, item in enumerate(diversified_items)}
        
        # Calculate Spearman's rank correlation
        common_ids = set(original_ranking.keys()) & set(diversified_ranking.keys())
        
        if len(common_ids) <= 1:
            return 1.0
        
        original_ranks = [original_ranking[item_id] for item_id in common_ids]
        diversified_ranks = [diversified_ranking[item_id] for item_id in common_ids]
        
        # Spearman correlation
        correlation = np.corrcoef(original_ranks, diversified_ranks)[0, 1]
        
        # Handle NaN case (constant rankings)
        if np.isnan(correlation):
            correlation = 1.0
        
        # Convert to 0-1 scale (correlation is in [-1, 1])
        return (correlation + 1) / 2
    
    def overall_score(self, scores: Dict[str, float]) -> float:
        """
        Calculate weighted overall score from individual metrics.
        
        Args:
            scores: Dictionary of individual metric scores
            
        Returns:
            Weighted overall score
        """
        weights = {
            'alternation_score': 0.2,
            'gini_coefficient': 0.15,
            'entropy_score': 0.15,
            'gender_balance_score': 0.2,
            'prefix_diversity': 0.15,
            'consecutive_penalty': 0.1,
            'relevance_preservation': 0.05
        }
        
        weighted_sum = 0
        total_weight = 0
        
        for metric, weight in weights.items():
            if metric in scores:
                weighted_sum += scores[metric] * weight
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def compare_algorithms(self, results: List[DiversificationResult]) -> Dict[str, Any]:
        """
        Compare multiple algorithm results.
        
        Args:
            results: List of DiversificationResult objects
            
        Returns:
            Dictionary with comparison statistics
        """
        comparison = {
            'algorithms': [],
            'metrics': defaultdict(list),
            'rankings': {},
            'best_algorithm': None,
            'summary': {}
        }
        
        # Evaluate each result
        all_scores = {}
        for result in results:
            algorithm_name = result.algorithm_name
            scores = self.evaluate_all(result)
            
            comparison['algorithms'].append(algorithm_name)
            all_scores[algorithm_name] = scores
            
            for metric, score in scores.items():
                comparison['metrics'][metric].append(score)
        
        # Calculate rankings for each metric
        for metric in self.metrics:
            if metric in comparison['metrics']:
                scores_with_names = list(zip(comparison['metrics'][metric], comparison['algorithms']))
                scores_with_names.sort(reverse=True)  # Higher scores are better
                
                comparison['rankings'][metric] = [name for _, name in scores_with_names]
        
        # Find best overall algorithm
        if 'overall_score' in comparison['rankings']:
            comparison['best_algorithm'] = comparison['rankings']['overall_score'][0]
        
        # Calculate summary statistics
        for metric in self.metrics:
            if metric in comparison['metrics']:
                values = comparison['metrics'][metric]
                comparison['summary'][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'best_algorithm': comparison['rankings'][metric][0]
                }
        
        return comparison