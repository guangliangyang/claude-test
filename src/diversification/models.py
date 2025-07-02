"""
Data models for diversification algorithms.

Defines the core data structures used across all diversification algorithms.
"""

from dataclasses import dataclass
from typing import List, Literal, Dict, Any, Optional
from enum import Enum


class Gender(Enum):
    """Gender categories for image classification."""
    MALE = "male"
    FEMALE = "female"
    UNKNOWN = "unknown"
    
    def __str__(self) -> str:
        return self.value
    
    @classmethod
    def from_char(cls, char: str) -> 'Gender':
        """Create Gender from character representation."""
        char_lower = char.lower()
        if char_lower == 'm':
            return cls.MALE
        elif char_lower == 'f':
            return cls.FEMALE
        else:
            return cls.UNKNOWN


@dataclass
class ImageItem:
    """Represents a single image item in search results."""
    id: str
    gender: Gender
    relevance_score: float = 1.0
    features: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize default values after object creation."""
        if self.features is None:
            self.features = {}
        if self.metadata is None:
            self.metadata = {}
    
    def similarity(self, other: 'ImageItem') -> float:
        """
        Calculate similarity with another image item.
        
        For this implementation, similarity is based on gender:
        - Same gender: high similarity (0.8)
        - Different gender: low similarity (0.2)
        """
        if self.gender == other.gender:
            return 0.8
        else:
            return 0.2
    
    def __str__(self) -> str:
        return f"ImageItem(id={self.id}, gender={self.gender}, score={self.relevance_score:.2f})"
    
    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class SearchResult:
    """Represents a collection of search results."""
    query: str
    items: List[ImageItem]
    total_count: int = None
    
    def __post_init__(self):
        """Initialize computed fields."""
        if self.total_count is None:
            self.total_count = len(self.items)
    
    @classmethod
    def from_string(cls, gender_string: str, query: str = "sample_query") -> 'SearchResult':
        """
        Create SearchResult from a string like 'mmmmmffffff'.
        
        Args:
            gender_string: String with 'm' for male, 'f' for female
            query: Search query string
            
        Returns:
            SearchResult with ImageItems created from the string
        """
        items = []
        for i, char in enumerate(gender_string):
            gender = Gender.from_char(char)
            item = ImageItem(
                id=f"img_{i:03d}",
                gender=gender,
                relevance_score=1.0 - (i * 0.01)  # Slightly decreasing relevance
            )
            items.append(item)
        
        return cls(query=query, items=items)
    
    def get_gender_distribution(self) -> Dict[Gender, int]:
        """Get count of items by gender."""
        distribution = {gender: 0 for gender in Gender}
        for item in self.items:
            distribution[item.gender] += 1
        return distribution
    
    def get_gender_sequence(self) -> str:
        """Get string representation of gender sequence."""
        return ''.join(
            'M' if item.gender == Gender.MALE else 
            'F' if item.gender == Gender.FEMALE else 
            '?' 
            for item in self.items
        )
    
    def __len__(self) -> int:
        return len(self.items)
    
    def __str__(self) -> str:
        gender_seq = self.get_gender_sequence()
        return f"SearchResult(query='{self.query}', sequence='{gender_seq}', count={len(self.items)})"


@dataclass
class DiversificationResult:
    """Result of applying a diversification algorithm."""
    algorithm_name: str
    original_result: SearchResult
    diversified_items: List[ImageItem]
    execution_time: float
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.parameters is None:
            self.parameters = {}
    
    def get_diversified_sequence(self) -> str:
        """Get string representation of diversified sequence."""
        return ''.join(
            'M' if item.gender == Gender.MALE else 
            'F' if item.gender == Gender.FEMALE else 
            '?' 
            for item in self.diversified_items
        )
    
    def to_search_result(self) -> SearchResult:
        """Convert to SearchResult object."""
        return SearchResult(
            query=self.original_result.query,
            items=self.diversified_items.copy(),
            total_count=len(self.diversified_items)
        )
    
    def __str__(self) -> str:
        diversified_seq = self.get_diversified_sequence()
        return f"DiversificationResult(algorithm='{self.algorithm_name}', sequence='{diversified_seq}', time={self.execution_time:.4f}s)"