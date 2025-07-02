"""
Human Alignment Analysis for Gender Diversification
==================================================

This module analyzes human expert feedback to extract patterns and preferences
for gender diversification in search results, then creates evaluation metrics
that align with human judgment.
"""

from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Any
import numpy as np
from scipy import stats

from .models import Gender, ImageItem, SearchResult, DiversificationResult


class HumanFeedbackAnalyzer:
    """Analyzes human expert feedback to understand diversification preferences."""
    
    def __init__(self):
        self.patterns = {}
        self.preference_rules = {}
    
    def analyze_survey_data(self, survey_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze survey data to extract human preferences and patterns.
        
        Args:
            survey_data: List of survey cases with question and responses
            
        Returns:
            Dictionary containing analysis results and extracted patterns
        """
        analysis = {
            'position_preferences': defaultdict(list),
            'diversity_patterns': [],
            'consensus_analysis': {},
            'extracted_rules': []
        }
        
        for case in survey_data:
            question = case['question'].replace(' ', '')
            responses = [r.replace(' ', '') for r in case['responses']]
            
            # Analyze where minority gender first appears
            first_f_positions = []
            for response in responses:
                first_f_pos = response.find('F')
                if first_f_pos != -1:
                    first_f_positions.append(first_f_pos + 1)  # 1-indexed
            
            analysis['position_preferences'][question] = first_f_positions
            
            # Analyze consensus
            consensus = self._analyze_consensus(responses)
            analysis['consensus_analysis'][question] = consensus
            
            # Extract diversity patterns
            patterns = self._extract_diversity_patterns(question, responses)
            analysis['diversity_patterns'].extend(patterns)
        
        # Extract rules from patterns
        analysis['extracted_rules'] = self._extract_rules(analysis)
        
        return analysis
    
    def _analyze_consensus(self, responses: List[str]) -> Dict[str, Any]:
        """Analyze consensus among respondents."""
        response_counts = Counter(responses)
        total_responses = len(responses)
        
        # Find most common response
        most_common = response_counts.most_common(1)[0] if response_counts else ("", 0)
        
        # Calculate consensus strength
        consensus_strength = most_common[1] / total_responses if total_responses > 0 else 0
        
        # Analyze first F position distribution
        first_f_positions = []
        for response in responses:
            pos = response.find('F')
            if pos != -1:
                first_f_positions.append(pos + 1)
        
        return {
            'most_common_response': most_common[0],
            'consensus_strength': consensus_strength,
            'response_distribution': dict(response_counts),
            'first_f_position_stats': {
                'mean': np.mean(first_f_positions) if first_f_positions else 0,
                'median': np.median(first_f_positions) if first_f_positions else 0,
                'mode': stats.mode(first_f_positions, keepdims=False).mode if first_f_positions else 0,
                'distribution': Counter(first_f_positions)
            }
        }
    
    def _extract_diversity_patterns(self, question: str, responses: List[str]) -> List[Dict[str, Any]]:
        """Extract diversity patterns from responses."""
        patterns = []
        
        for response in responses:
            pattern = {
                'original': question,
                'diversified': response,
                'first_f_position': response.find('F') + 1 if 'F' in response else -1,
                'alternation_score': self._calculate_alternation_score(response),
                'balance_score': self._calculate_balance_score(response)
            }
            patterns.append(pattern)
        
        return patterns
    
    def _extract_rules(self, analysis: Dict[str, Any]) -> List[str]:
        """Extract human preference rules from analysis."""
        rules = []
        
        # Analyze position preferences
        for question, positions in analysis['position_preferences'].items():
            if positions:
                avg_pos = np.mean(positions)
                median_pos = np.median(positions)
                mode_pos = stats.mode(positions, keepdims=False).mode if positions else 0
                
                rules.append(f"For sequence '{question}': majority prefer first F at position {mode_pos} (avg: {avg_pos:.1f})")
        
        # General rules based on patterns
        all_patterns = analysis['diversity_patterns']
        if all_patterns:
            first_f_positions = [p['first_f_position'] for p in all_patterns if p['first_f_position'] > 0]
            if first_f_positions:
                position_dist = Counter(first_f_positions)
                most_preferred = position_dist.most_common(1)[0][0] if position_dist else 0
                rules.append(f"Most preferred first minority position: {most_preferred}")
        
        return rules
    
    def _calculate_alternation_score(self, sequence: str) -> float:
        """Calculate how well sequence alternates between genders."""
        if len(sequence) < 2:
            return 1.0
        
        alternations = sum(1 for i in range(len(sequence)-1) 
                          if sequence[i] != sequence[i+1])
        max_alternations = len(sequence) - 1
        return alternations / max_alternations if max_alternations > 0 else 0
    
    def _calculate_balance_score(self, sequence: str) -> float:
        """Calculate gender balance in sequence."""
        if not sequence:
            return 0
        
        m_count = sequence.count('M')
        f_count = sequence.count('F')
        total = len(sequence)
        
        if total == 0:
            return 0
        
        # Perfect balance is 0.5 for each gender
        m_ratio = m_count / total
        f_ratio = f_count / total
        
        # Distance from perfect balance
        balance_distance = abs(m_ratio - 0.5) + abs(f_ratio - 0.5)
        return 1 - balance_distance


class HumanAlignedEvaluator:
    """Evaluator that aligns with human expert preferences."""
    
    def __init__(self, human_preferences: Dict[str, Any]):
        self.preferences = human_preferences
        self.weights = self._calculate_weights()
    
    def _calculate_weights(self) -> Dict[str, float]:
        """Calculate metric weights based on human preferences."""
        # Default weights - can be tuned based on analysis
        return {
            'early_diversity': 0.4,  # High weight for early minority appearance
            'alternation': 0.3,      # Medium weight for alternation
            'balance': 0.2,          # Lower weight for overall balance
            'consensus_alignment': 0.1  # Bonus for matching human consensus
        }
    
    def evaluate(self, original: str, diversified: str) -> float:
        """
        Evaluate diversified sequence against human preferences.
        
        Args:
            original: Original sequence
            diversified: Diversified sequence
            
        Returns:
            Score from 0 to 1, with 1 being perfect alignment with human preferences
        """
        scores = {}
        
        # Early diversity score - penalize late first minority appearance
        first_f_pos = diversified.find('F')
        if first_f_pos != -1:
            # Lower position numbers are better (earlier appearance)
            # Position 2 (index 1) gets score 1.0, position 3 gets 0.8, etc.
            scores['early_diversity'] = max(0, 1.0 - (first_f_pos * 0.2))
        else:
            scores['early_diversity'] = 0
        
        # Alternation score
        scores['alternation'] = self._calculate_alternation_score(diversified)
        
        # Balance score
        scores['balance'] = self._calculate_balance_score(diversified)
        
        # Consensus alignment score
        scores['consensus_alignment'] = self._calculate_consensus_alignment(original, diversified)
        
        # Weighted final score
        final_score = sum(scores[metric] * self.weights[metric] 
                         for metric in scores)
        
        return final_score
    
    def _calculate_alternation_score(self, sequence: str) -> float:
        """Calculate alternation score."""
        if len(sequence) < 2:
            return 1.0
        
        alternations = sum(1 for i in range(len(sequence)-1) 
                          if sequence[i] != sequence[i+1])
        max_alternations = len(sequence) - 1
        return alternations / max_alternations if max_alternations > 0 else 0
    
    def _calculate_balance_score(self, sequence: str) -> float:
        """Calculate balance score."""
        if not sequence:
            return 0
        
        m_count = sequence.count('M')
        f_count = sequence.count('F')
        total = len(sequence)
        
        if total == 0:
            return 0
        
        # Calculate how close to 50/50 split
        ideal_ratio = 0.5
        m_ratio = m_count / total
        f_ratio = f_count / total
        
        balance_error = abs(m_ratio - ideal_ratio) + abs(f_ratio - ideal_ratio)
        return max(0, 1 - balance_error)
    
    def _calculate_consensus_alignment(self, original: str, diversified: str) -> float:
        """Calculate how well this aligns with human consensus."""
        # For the survey case "MMMMMFFFFF", check if first F appears at preferred positions
        if original == "MMMMMFFFFF":
            first_f_pos = diversified.find('F') + 1  # 1-indexed
            
            # Based on survey data: position 2 (4 votes), position 3 (8 votes), position 4 (2 votes)
            # Weight by human preference frequency
            if first_f_pos == 2:
                return 0.8  # Strong preference shown by 4/14 experts
            elif first_f_pos == 3:
                return 1.0  # Highest preference shown by 8/14 experts
            elif first_f_pos == 4:
                return 0.6  # Some preference shown by 2/14 experts
            else:
                return 0.2  # Not aligned with expert preferences
        
        # For other cases, use general heuristics
        first_f_pos = diversified.find('F') + 1
        if first_f_pos <= 3:
            return 0.8  # Early diversity generally preferred
        else:
            return 0.4  # Later diversity less preferred
    
    def detailed_evaluation(self, original: str, diversified: str) -> Dict[str, float]:
        """Provide detailed breakdown of evaluation scores."""
        scores = {}
        
        # Early diversity
        first_f_pos = diversified.find('F')
        if first_f_pos != -1:
            scores['early_diversity'] = max(0, 1.0 - (first_f_pos * 0.2))
        else:
            scores['early_diversity'] = 0
        
        # Alternation
        scores['alternation'] = self._calculate_alternation_score(diversified)
        
        # Balance
        scores['balance'] = self._calculate_balance_score(diversified)
        
        # Consensus alignment
        scores['consensus_alignment'] = self._calculate_consensus_alignment(original, diversified)
        
        # Weighted components
        weighted_scores = {f"{k}_weighted": v * self.weights[k] 
                          for k, v in scores.items()}
        
        # Final score
        final_score = sum(weighted_scores.values())
        
        return {
            **scores,
            **weighted_scores,
            'final_score': final_score,
            'weights_used': self.weights.copy()
        }