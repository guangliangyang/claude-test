"""
Diversification algorithms for search result optimization.

This module implements various algorithms to diversify search results,
with a focus on gender balance in image search.
"""

import random
import time
import math
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter

from .models import ImageItem, SearchResult, DiversificationResult, Gender


class DiversificationAlgorithm(ABC):
    """Abstract base class for all diversification algorithms."""
    
    def __init__(self, **parameters):
        """Initialize algorithm with parameters."""
        self.parameters = parameters
        self.name = self.__class__.__name__
    
    @abstractmethod
    def _diversify_impl(self, items: List[ImageItem]) -> List[ImageItem]:
        """
        Core diversification logic to be implemented by subclasses.
        
        Args:
            items: Original list of ImageItems
            
        Returns:
            Diversified list of ImageItems
        """
        pass
    
    def diversify(self, search_result: SearchResult) -> DiversificationResult:
        """
        Apply diversification algorithm to search results.
        
        Args:
            search_result: Original search results
            
        Returns:
            DiversificationResult with diversified items and metadata
        """
        start_time = time.time()
        
        # Apply the diversification algorithm
        diversified_items = self._diversify_impl(search_result.items.copy())
        
        execution_time = time.time() - start_time
        
        return DiversificationResult(
            algorithm_name=self.name,
            original_result=search_result,
            diversified_items=diversified_items,
            execution_time=execution_time,
            parameters=self.parameters.copy()
        )
    
    def __str__(self) -> str:
        return f"{self.name}({self.parameters})"


class InterleavingAlgorithm(DiversificationAlgorithm):
    """
    Simple interleaving algorithm that alternates between genders.
    
    Example: mmmmmffffff -> mfmfmfmfmfm
    """
    
    def __init__(self, prioritize_minority: bool = True):
        """
        Initialize interleaving algorithm.
        
        Args:
            prioritize_minority: Whether to prioritize the minority gender first
        """
        super().__init__(prioritize_minority=prioritize_minority)
    
    def _diversify_impl(self, items: List[ImageItem]) -> List[ImageItem]:
        """Implement interleaving diversification."""
        # Separate by gender
        male_items = [item for item in items if item.gender == Gender.MALE]
        female_items = [item for item in items if item.gender == Gender.FEMALE]
        
        # Sort by relevance score (descending)
        male_items.sort(key=lambda x: x.relevance_score, reverse=True)
        female_items.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Determine starting gender
        if self.parameters['prioritize_minority']:
            # Start with the minority gender
            if len(male_items) < len(female_items):
                first_list, second_list = male_items, female_items
            else:
                first_list, second_list = female_items, male_items
        else:
            # Start with the majority gender
            if len(male_items) >= len(female_items):
                first_list, second_list = male_items, female_items
            else:
                first_list, second_list = female_items, male_items
        
        # Interleave the lists
        result = []
        max_len = max(len(first_list), len(second_list))
        
        for i in range(max_len):
            if i < len(first_list):
                result.append(first_list[i])
            if i < len(second_list):
                result.append(second_list[i])
        
        return result


class ProportionalDistributionAlgorithm(DiversificationAlgorithm):
    """
    Distributes items proportionally to maintain gender balance throughout the list.
    
    Ensures that at any prefix of the result, the gender ratio is close to the overall ratio.
    """
    
    def __init__(self, window_size: int = 5):
        """
        Initialize proportional distribution algorithm.
        
        Args:
            window_size: Size of sliding window to maintain proportions
        """
        super().__init__(window_size=window_size)
    
    def _diversify_impl(self, items: List[ImageItem]) -> List[ImageItem]:
        """Implement proportional distribution."""
        # Count genders
        gender_counts = Counter(item.gender for item in items)
        total_items = len(items)
        
        # Calculate target proportions
        target_proportions = {
            gender: count / total_items 
            for gender, count in gender_counts.items()
        }
        
        # Separate and sort by relevance
        gender_items = defaultdict(list)
        for item in items:
            gender_items[item.gender].append(item)
        
        for gender in gender_items:
            gender_items[gender].sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Proportional selection
        result = []
        gender_indices = defaultdict(int)
        
        for position in range(total_items):
            # Calculate how many of each gender we should have had by now
            target_counts = {
                gender: int((position + 1) * proportion)
                for gender, proportion in target_proportions.items()
            }
            
            # Calculate current counts
            current_counts = Counter(item.gender for item in result)
            
            # Find which gender is most "behind" its target
            best_gender = None
            max_deficit = -1
            
            for gender in gender_counts:
                if gender_indices[gender] < len(gender_items[gender]):
                    deficit = target_counts[gender] - current_counts[gender]
                    if deficit > max_deficit:
                        max_deficit = deficit
                        best_gender = gender
            
            # If no gender is behind, pick the one with highest remaining relevance
            if best_gender is None or max_deficit <= 0:
                best_score = -1
                for gender in gender_counts:
                    if gender_indices[gender] < len(gender_items[gender]):
                        score = gender_items[gender][gender_indices[gender]].relevance_score
                        if score > best_score:
                            best_score = score
                            best_gender = gender
            
            # Add the selected item
            if best_gender is not None:
                result.append(gender_items[best_gender][gender_indices[best_gender]])
                gender_indices[best_gender] += 1
        
        return result


class MMRAlgorithm(DiversificationAlgorithm):
    """
    Maximal Marginal Relevance algorithm for balancing relevance and diversity.
    
    Score = λ × Relevance - (1-λ) × max_similarity_to_selected_items
    """
    
    def __init__(self, lambda_param: float = 0.6):
        """
        Initialize MMR algorithm.
        
        Args:
            lambda_param: Balance parameter (0=pure diversity, 1=pure relevance)
        """
        super().__init__(lambda_param=lambda_param)
    
    def _diversify_impl(self, items: List[ImageItem]) -> List[ImageItem]:
        """Implement MMR diversification."""
        if not items:
            return []
        
        lambda_param = self.parameters['lambda_param']
        
        # Normalize relevance scores
        max_relevance = max(item.relevance_score for item in items)
        min_relevance = min(item.relevance_score for item in items)
        relevance_range = max_relevance - min_relevance or 1
        
        result = []
        remaining_items = items.copy()
        
        # Select the first item (highest relevance)
        first_item = max(remaining_items, key=lambda x: x.relevance_score)
        result.append(first_item)
        remaining_items.remove(first_item)
        
        # Select remaining items using MMR
        while remaining_items:
            best_item = None
            best_score = -float('inf')
            
            for candidate in remaining_items:
                # Normalize relevance
                relevance = (candidate.relevance_score - min_relevance) / relevance_range
                
                # Calculate maximum similarity to already selected items
                max_similarity = 0
                if result:
                    max_similarity = max(
                        candidate.similarity(selected_item) 
                        for selected_item in result
                    )
                
                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_item = candidate
            
            if best_item:
                result.append(best_item)
                remaining_items.remove(best_item)
        
        return result


class GreedyDiversificationAlgorithm(DiversificationAlgorithm):
    """
    Greedy algorithm that always selects the item most different from already selected items.
    """
    
    def __init__(self, diversity_weight: float = 1.0):
        """
        Initialize greedy diversification.
        
        Args:
            diversity_weight: Weight for diversity vs relevance
        """
        super().__init__(diversity_weight=diversity_weight)
    
    def _diversify_impl(self, items: List[ImageItem]) -> List[ImageItem]:
        """Implement greedy diversification."""
        if not items:
            return []
        
        diversity_weight = self.parameters['diversity_weight']
        result = []
        remaining_items = items.copy()
        
        # Select first item (highest relevance)
        first_item = max(remaining_items, key=lambda x: x.relevance_score)
        result.append(first_item)
        remaining_items.remove(first_item)
        
        # Greedily select most diverse items
        while remaining_items:
            best_item = None
            best_score = -float('inf')
            
            for candidate in remaining_items:
                # Calculate average similarity to selected items
                if result:
                    avg_similarity = sum(
                        candidate.similarity(selected_item) 
                        for selected_item in result
                    ) / len(result)
                else:
                    avg_similarity = 0
                
                # Score = relevance - diversity_weight * similarity
                score = candidate.relevance_score - diversity_weight * avg_similarity
                
                if score > best_score:
                    best_score = score
                    best_item = candidate
            
            if best_item:
                result.append(best_item)
                remaining_items.remove(best_item)
        
        return result


class RandomizedDiversificationAlgorithm(DiversificationAlgorithm):
    """
    Randomized algorithm that maintains diversity constraints while introducing randomness.
    """
    
    def __init__(self, randomness: float = 0.3, seed: Optional[int] = None):
        """
        Initialize randomized diversification.
        
        Args:
            randomness: Level of randomness (0=deterministic, 1=fully random)
            seed: Random seed for reproducibility
        """
        super().__init__(randomness=randomness, seed=seed)
        if seed is not None:
            random.seed(seed)
    
    def _diversify_impl(self, items: List[ImageItem]) -> List[ImageItem]:
        """Implement randomized diversification."""
        randomness = self.parameters['randomness']
        
        # Separate by gender and sort by relevance
        male_items = [item for item in items if item.gender == Gender.MALE]
        female_items = [item for item in items if item.gender == Gender.FEMALE]
        
        male_items.sort(key=lambda x: x.relevance_score, reverse=True)
        female_items.sort(key=lambda x: x.relevance_score, reverse=True)
        
        result = []
        male_idx = female_idx = 0
        
        while male_idx < len(male_items) or female_idx < len(female_items):
            # Determine which gender to pick next
            male_available = male_idx < len(male_items)
            female_available = female_idx < len(female_items)
            
            if male_available and female_available:
                # Both available - use randomness and current balance
                current_males = sum(1 for item in result if item.gender == Gender.MALE)
                current_females = len(result) - current_males
                
                # Bias towards balancing
                if current_males > current_females:
                    female_prob = 0.7 + randomness * 0.3
                elif current_females > current_males:
                    female_prob = 0.3 - randomness * 0.3
                else:
                    female_prob = 0.5
                
                if random.random() < female_prob:
                    result.append(female_items[female_idx])
                    female_idx += 1
                else:
                    result.append(male_items[male_idx])
                    male_idx += 1
            elif male_available:
                result.append(male_items[male_idx])
                male_idx += 1
            elif female_available:
                result.append(female_items[female_idx])
                female_idx += 1
        
        return result


class SlidingWindowAlgorithm(DiversificationAlgorithm):
    """
    Ensures diversity within sliding windows of the result list.
    """
    
    def __init__(self, window_size: int = 4, max_same_gender: int = 2):
        """
        Initialize sliding window algorithm.
        
        Args:
            window_size: Size of the sliding window
            max_same_gender: Maximum consecutive items of same gender
        """
        super().__init__(window_size=window_size, max_same_gender=max_same_gender)
    
    def _diversify_impl(self, items: List[ImageItem]) -> List[ImageItem]:
        """Implement sliding window diversification."""
        window_size = self.parameters['window_size']
        max_same_gender = self.parameters['max_same_gender']
        
        # Separate by gender and sort by relevance
        gender_items = defaultdict(list)
        for item in items:
            gender_items[item.gender].append(item)
        
        for gender in gender_items:
            gender_items[gender].sort(key=lambda x: x.relevance_score, reverse=True)
        
        result = []
        gender_indices = defaultdict(int)
        
        for position in range(len(items)):
            # Find valid candidates that don't violate constraints
            valid_candidates = []
            
            for gender in gender_items:
                if gender_indices[gender] < len(gender_items[gender]):
                    # Check consecutive constraint
                    consecutive_count = 0
                    for i in range(len(result) - 1, max(len(result) - max_same_gender, -1), -1):
                        if result[i].gender == gender:
                            consecutive_count += 1
                        else:
                            break
                    
                    if consecutive_count < max_same_gender:
                        # Check window constraint
                        window_start = max(0, position - window_size + 1)
                        window_items = result[window_start:]
                        window_gender_count = sum(1 for item in window_items if item.gender == gender)
                        
                        # Allow if window wouldn't be dominated by this gender
                        if window_gender_count < window_size // 2 + 1:
                            valid_candidates.append((gender, gender_items[gender][gender_indices[gender]]))
            
            # If no valid candidates, relax constraints
            if not valid_candidates:
                for gender in gender_items:
                    if gender_indices[gender] < len(gender_items[gender]):
                        valid_candidates.append((gender, gender_items[gender][gender_indices[gender]]))
            
            # Select best candidate by relevance
            if valid_candidates:
                best_gender, best_item = max(valid_candidates, key=lambda x: x[1].relevance_score)
                result.append(best_item)
                gender_indices[best_gender] += 1
        
        return result


class EntropyMaximizationAlgorithm(DiversificationAlgorithm):
    """
    Maximizes the entropy of gender distribution in the result sequence.
    """
    
    def __init__(self, look_ahead: int = 3):
        """
        Initialize entropy maximization algorithm.
        
        Args:
            look_ahead: Number of positions to look ahead when optimizing
        """
        super().__init__(look_ahead=look_ahead)
    
    def _calculate_entropy(self, sequence: List[Gender]) -> float:
        """Calculate entropy of a gender sequence."""
        if not sequence:
            return 0
        
        counts = Counter(sequence)
        total = len(sequence)
        entropy = 0
        
        for count in counts.values():
            if count > 0:
                prob = count / total
                entropy -= prob * math.log2(prob)
        
        return entropy
    
    def _diversify_impl(self, items: List[ImageItem]) -> List[ImageItem]:
        """Implement entropy maximization diversification."""
        look_ahead = self.parameters['look_ahead']
        
        # Separate by gender and sort by relevance
        gender_items = defaultdict(list)
        for item in items:
            gender_items[item.gender].append(item)
        
        for gender in gender_items:
            gender_items[gender].sort(key=lambda x: x.relevance_score, reverse=True)
        
        result = []
        gender_indices = defaultdict(int)
        
        while len(result) < len(items):
            best_choice = None
            best_score = -float('inf')
            
            # Try each available item
            for gender in gender_items:
                if gender_indices[gender] < len(gender_items[gender]):
                    candidate_item = gender_items[gender][gender_indices[gender]]
                    
                    # Create temporary sequence with this choice
                    temp_sequence = [item.gender for item in result] + [gender]
                    
                    # Calculate entropy of this choice
                    entropy = self._calculate_entropy(temp_sequence)
                    
                    # Add relevance component
                    relevance = candidate_item.relevance_score
                    
                    # Combined score (entropy weighted higher for diversity)
                    score = 0.7 * entropy + 0.3 * relevance
                    
                    if score > best_score:
                        best_score = score
                        best_choice = (gender, candidate_item)
            
            # Add the best choice
            if best_choice:
                gender, item = best_choice
                result.append(item)
                gender_indices[gender] += 1
        
        return result