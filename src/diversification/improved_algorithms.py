"""
Human-Aligned Diversification Algorithms
========================================

Improved algorithms that incorporate human feedback and preferences
for gender diversification in search results.
"""

from typing import List
import random
from .models import ImageItem, SearchResult, DiversificationResult, Gender
from .algorithms import DiversificationAlgorithm


class HumanAlignedInterleavingAlgorithm(DiversificationAlgorithm):
    """
    Interleaving algorithm optimized based on human preferences.
    Ensures early minority appearance while maintaining alternation.
    """
    
    def __init__(self, preferred_first_minority_pos: int = 2):
        """
        Args:
            preferred_first_minority_pos: Preferred position for first minority (1-indexed)
        """
        super().__init__()
        self.preferred_first_pos = preferred_first_minority_pos - 1  # Convert to 0-indexed
    
    def _diversify_impl(self, items: List[ImageItem]) -> List[ImageItem]:
        if not items:
            return []
        
        # Separate by gender
        males = [item for item in items if item.gender == Gender.MALE]
        females = [item for item in items if item.gender == Gender.FEMALE]
        
        if not males or not females:
            return items
        
        # Determine minority and majority
        if len(females) <= len(males):
            minority, majority = females, males
        else:
            minority, majority = males, females
        
        result = []
        
        # Place items before first minority position
        for i in range(min(self.preferred_first_pos, len(majority))):
            result.append(majority[i])
        
        # Now interleave starting with minority at preferred position
        maj_idx = self.preferred_first_pos
        min_idx = 0
        
        # Start with minority at preferred position
        if min_idx < len(minority):
            result.append(minority[min_idx])
            min_idx += 1
        
        # Continue alternating
        while maj_idx < len(majority) and min_idx < len(minority):
            result.append(majority[maj_idx])
            maj_idx += 1
            if min_idx < len(minority):
                result.append(minority[min_idx])
                min_idx += 1
        
        # Add remaining items
        while maj_idx < len(majority):
            result.append(majority[maj_idx])
            maj_idx += 1
        
        while min_idx < len(minority):
            result.append(minority[min_idx])
            min_idx += 1
        
        return result


class EarlyDiversityOptimizedAlgorithm(DiversificationAlgorithm):
    """
    Algorithm optimized for early diversity based on human feedback analysis.
    Prioritizes early minority appearance while maintaining reasonable balance.
    """
    
    def __init__(self, target_first_minority_pos: int = 3):
        """
        Args:
            target_first_minority_pos: Target position for first minority (1-indexed)
        """
        super().__init__()
        self.target_pos = target_first_minority_pos - 1  # Convert to 0-indexed
    
    def _diversify_impl(self, items: List[ImageItem]) -> List[ImageItem]:
        if not items:
            return []
        
        # Separate by gender
        males = [item for item in items if item.gender == Gender.MALE]
        females = [item for item in items if item.gender == Gender.FEMALE]
        
        if not males or not females:
            return items
        
        # Determine minority and majority
        if len(females) <= len(males):
            minority, majority = females, males
        else:
            minority, majority = males, females
        
        result = []
        
        # Place majority items before target position
        maj_used = 0
        for i in range(min(self.target_pos, len(majority))):
            result.append(majority[maj_used])
            maj_used += 1
        
        # Place first minority at target position
        min_used = 0
        if min_used < len(minority):
            result.append(minority[min_used])
            min_used += 1
        
        # Continue with optimized distribution
        remaining_positions = len(items) - len(result)
        remaining_majority = len(majority) - maj_used
        remaining_minority = len(minority) - min_used
        
        # Calculate ideal spacing for remaining minorities
        if remaining_minority > 0 and remaining_positions > 0:
            spacing = max(1, remaining_positions // (remaining_minority + 1))
        else:
            spacing = remaining_positions
        
        next_minority_in = spacing
        position = 0
        
        while maj_used < len(majority) or min_used < len(minority):
            if position == next_minority_in and min_used < len(minority):
                result.append(minority[min_used])
                min_used += 1
                remaining_minority -= 1
                if remaining_minority > 0:
                    remaining_positions = len(items) - len(result)
                    next_minority_in = position + max(1, remaining_positions // (remaining_minority + 1))
                else:
                    next_minority_in = float('inf')
            elif maj_used < len(majority):
                result.append(majority[maj_used])
                maj_used += 1
            
            position += 1
        
        return result


class ConsensusBasedAlgorithm(DiversificationAlgorithm):
    """
    Algorithm that tries to match human consensus patterns from survey data.
    Uses weighted probability based on human expert preferences.
    """
    
    def __init__(self, human_preference_weights: dict = None):
        """
        Args:
            human_preference_weights: Dict mapping first minority position to weight
        """
        super().__init__()
        # Default weights based on survey data: pos 2: 4 votes, pos 3: 8 votes, pos 4: 2 votes
        self.weights = human_preference_weights or {2: 0.286, 3: 0.571, 4: 0.143}
    
    def _diversify_impl(self, items: List[ImageItem]) -> List[ImageItem]:
        if not items:
            return []
        
        # Separate by gender
        males = [item for item in items if item.gender == Gender.MALE]
        females = [item for item in items if item.gender == Gender.FEMALE]
        
        if not males or not females:
            return items
        
        # Determine minority and majority
        if len(females) <= len(males):
            minority, majority = females, males
        else:
            minority, majority = males, females
        
        # Choose first minority position based on human preferences
        positions = list(self.weights.keys())
        weights = list(self.weights.values())
        
        # Select position based on weights (deterministic for consistency)
        # Use position with highest weight as default
        first_minority_pos = max(positions, key=lambda x: self.weights[x]) - 1  # Convert to 0-indexed
        
        result = []
        
        # Place majority items before first minority
        maj_used = 0
        for i in range(min(first_minority_pos, len(majority))):
            result.append(majority[maj_used])
            maj_used += 1
        
        # Place first minority
        min_used = 0
        if min_used < len(minority):
            result.append(minority[min_used])
            min_used += 1
        
        # Continue with balanced distribution
        while maj_used < len(majority) or min_used < len(minority):
            # Prefer majority if we have more majority items left
            maj_remaining = len(majority) - maj_used
            min_remaining = len(minority) - min_used
            
            if maj_remaining > min_remaining and maj_used < len(majority):
                result.append(majority[maj_used])
                maj_used += 1
            elif min_used < len(minority):
                result.append(minority[min_used])
                min_used += 1
            elif maj_used < len(majority):
                result.append(majority[maj_used])
                maj_used += 1
        
        return result


class AdaptiveHumanAlignedAlgorithm(DiversificationAlgorithm):
    """
    Adaptive algorithm that adjusts strategy based on sequence characteristics
    and learned human preferences.
    """
    
    def __init__(self):
        super().__init__()
        # Learned patterns from human feedback
        self.preference_patterns = {
            # For sequences with high majority bias, prefer early diversity
            'high_bias': {'min_ratio_threshold': 0.2, 'preferred_first_pos': 2},
            'medium_bias': {'min_ratio_threshold': 0.4, 'preferred_first_pos': 3},
            'low_bias': {'min_ratio_threshold': 0.6, 'preferred_first_pos': 4}
        }
    
    def _diversify_impl(self, items: List[ImageItem]) -> List[ImageItem]:
        if not items:
            return []
        
        # Separate by gender
        males = [item for item in items if item.gender == Gender.MALE]
        females = [item for item in items if item.gender == Gender.FEMALE]
        
        if not males or not females:
            return items
        
        # Calculate minority ratio
        minority_count = min(len(males), len(females))
        minority_ratio = minority_count / len(items)
        
        # Determine strategy based on minority ratio
        if minority_ratio <= self.preference_patterns['high_bias']['min_ratio_threshold']:
            strategy = 'high_bias'
        elif minority_ratio <= self.preference_patterns['medium_bias']['min_ratio_threshold']:
            strategy = 'medium_bias'
        else:
            strategy = 'low_bias'
        
        preferred_pos = self.preference_patterns[strategy]['preferred_first_pos'] - 1  # 0-indexed
        
        # Determine minority and majority
        if len(females) <= len(males):
            minority, majority = females, males
        else:
            minority, majority = males, females
        
        # Apply strategy
        result = []
        
        # Place majority items before preferred position
        maj_used = 0
        for i in range(min(preferred_pos, len(majority))):
            result.append(majority[maj_used])
            maj_used += 1
        
        # Place first minority at preferred position
        min_used = 0
        if min_used < len(minority):
            result.append(minority[min_used])
            min_used += 1
        
        # Adaptive continuation based on remaining items
        while maj_used < len(majority) or min_used < len(minority):
            maj_remaining = len(majority) - maj_used
            min_remaining = len(minority) - min_used
            
            # Adaptive spacing - try to distribute minorities evenly
            if min_remaining > 0:
                positions_left = (maj_remaining + min_remaining)
                ideal_spacing = positions_left // (min_remaining + 1)
                
                # Place minority if it's time or if we're running out of positions
                if len(result) % max(2, ideal_spacing) == 0 and min_used < len(minority):
                    result.append(minority[min_used])
                    min_used += 1
                elif maj_used < len(majority):
                    result.append(majority[maj_used])
                    maj_used += 1
            else:
                # No more minorities, place majority
                if maj_used < len(majority):
                    result.append(majority[maj_used])
                    maj_used += 1
        
        return result