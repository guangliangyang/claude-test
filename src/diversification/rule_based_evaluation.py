"""
Rule-Based Evaluation System
============================

This module extracts mix rules from human feedback and creates a scoring system
based on actual user preferences rather than theoretical metrics.
"""

from typing import List, Dict, Tuple, Any
from collections import Counter, defaultdict
import numpy as np
from .models import Gender, ImageItem, SearchResult, DiversificationResult


class MixRuleExtractor:
    """Extracts diversification rules from human expert feedback."""
    
    def __init__(self):
        self.extracted_rules = []
        self.rule_weights = {}
    
    def extract_rules_from_feedback(self, survey_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract concrete mix rules from human expert responses.
        
        Args:
            survey_data: Survey data with questions and expert responses
            
        Returns:
            Dictionary containing extracted rules and their weights
        """
        rules = {
            'position_rules': {},
            'pattern_rules': {},
            'sequence_rules': {},
            'extracted_patterns': []
        }
        
        for case in survey_data:
            question = case['question'].replace(' ', '')
            responses = [r.replace(' ', '') for r in case['responses']]
            
            # Extract position rules
            position_rules = self._extract_position_rules(question, responses)
            rules['position_rules'][question] = position_rules
            
            # Extract pattern rules
            pattern_rules = self._extract_pattern_rules(question, responses)
            rules['pattern_rules'][question] = pattern_rules
            
            # Extract sequence transformation rules
            sequence_rules = self._extract_sequence_rules(question, responses)
            rules['sequence_rules'][question] = sequence_rules
            
            # Extract general patterns
            patterns = self._extract_general_patterns(responses)
            rules['extracted_patterns'].extend(patterns)
        
        # Consolidate rules across all cases
        rules['consolidated_rules'] = self._consolidate_rules(rules)
        
        return rules
    
    def _extract_position_rules(self, original: str, responses: List[str]) -> Dict[str, Any]:
        """Extract rules about where minorities should be positioned."""
        rules = {
            'first_f_positions': [],
            'position_frequencies': Counter(),
            'preferred_positions': []
        }
        
        for response in responses:
            first_f_pos = response.find('F') + 1  # 1-indexed
            if first_f_pos > 0:
                rules['first_f_positions'].append(first_f_pos)
                rules['position_frequencies'][first_f_pos] += 1
        
        # Find most preferred positions
        if rules['position_frequencies']:
            most_common = rules['position_frequencies'].most_common()
            rules['preferred_positions'] = [pos for pos, count in most_common if count >= 2]
        
        return rules
    
    def _extract_pattern_rules(self, original: str, responses: List[str]) -> Dict[str, Any]:
        """Extract rules about diversification patterns."""
        rules = {
            'common_patterns': Counter(),
            'transition_patterns': Counter(),
            'spacing_patterns': []
        }
        
        for response in responses:
            # Count common sub-patterns
            for i in range(len(response) - 1):
                pattern = response[i:i+2]
                rules['common_patterns'][pattern] += 1
            
            # Analyze transitions
            for i in range(len(response) - 1):
                if response[i] != response[i+1]:
                    transition = f"{response[i]}->{response[i+1]}"
                    rules['transition_patterns'][transition] += 1
            
            # Analyze spacing between minorities
            f_positions = [i for i, char in enumerate(response) if char == 'F']
            if len(f_positions) > 1:
                spacings = [f_positions[i+1] - f_positions[i] for i in range(len(f_positions)-1)]
                rules['spacing_patterns'].extend(spacings)
        
        return rules
    
    def _extract_sequence_rules(self, original: str, responses: List[str]) -> Dict[str, Any]:
        """Extract rules about how sequences should be transformed."""
        rules = {
            'transformation_types': Counter(),
            'movement_patterns': [],
            'insertion_strategies': []
        }
        
        original_f_positions = [i for i, char in enumerate(original) if char == 'F']
        
        for response in responses:
            response_f_positions = [i for i, char in enumerate(response) if char == 'F']
            
            # Classify transformation type
            if len(response_f_positions) == len(original_f_positions):
                # Same number of F's, check if they moved
                if response_f_positions != original_f_positions:
                    rules['transformation_types']['repositioning'] += 1
                    
                    # Analyze movement pattern
                    movements = []
                    for orig_pos, new_pos in zip(original_f_positions, response_f_positions):
                        movement = new_pos - orig_pos
                        movements.append(movement)
                    rules['movement_patterns'].append(movements)
                else:
                    rules['transformation_types']['no_change'] += 1
            
            # Analyze insertion strategy (how F's are distributed)
            if response_f_positions:
                first_f = response_f_positions[0]
                if first_f < len(response) // 2:
                    rules['insertion_strategies'].append('early_insertion')
                else:
                    rules['insertion_strategies'].append('late_insertion')
        
        return rules
    
    def _extract_general_patterns(self, responses: List[str]) -> List[Dict[str, Any]]:
        """Extract general diversification patterns."""
        patterns = []
        
        for response in responses:
            pattern = {
                'sequence': response,
                'length': len(response),
                'f_count': response.count('F'),
                'm_count': response.count('M'),
                'first_f_position': response.find('F') + 1,
                'alternation_count': sum(1 for i in range(len(response)-1) if response[i] != response[i+1]),
                'max_consecutive_m': self._max_consecutive(response, 'M'),
                'max_consecutive_f': self._max_consecutive(response, 'F')
            }
            patterns.append(pattern)
        
        return patterns
    
    def _max_consecutive(self, sequence: str, char: str) -> int:
        """Find maximum consecutive occurrences of a character."""
        max_count = 0
        current_count = 0
        
        for c in sequence:
            if c == char:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        
        return max_count
    
    def _consolidate_rules(self, rules: Dict[str, Any]) -> Dict[str, Any]:
        """Consolidate rules across all survey cases."""
        consolidated = {
            'priority_rules': [],
            'scoring_weights': {},
            'violation_penalties': {}
        }
        
        # Rule 1: First F position preference
        all_first_f_positions = []
        for case_rules in rules['position_rules'].values():
            all_first_f_positions.extend(case_rules['first_f_positions'])
        
        if all_first_f_positions:
            position_freq = Counter(all_first_f_positions)
            most_preferred_pos = position_freq.most_common(1)[0][0]
            
            consolidated['priority_rules'].append({
                'rule_type': 'first_minority_position',
                'preferred_position': most_preferred_pos,
                'weight': 0.4,
                'description': f'First F should appear at position {most_preferred_pos}'
            })
            
            # Create scoring weights for different positions
            total_responses = sum(position_freq.values())
            for pos, count in position_freq.items():
                weight = count / total_responses
                consolidated['scoring_weights'][f'first_f_pos_{pos}'] = weight
        
        # Rule 2: Avoid excessive consecutive same gender
        all_patterns = rules['extracted_patterns']
        if all_patterns:
            max_consecutive_m_values = [p['max_consecutive_m'] for p in all_patterns]
            max_consecutive_f_values = [p['max_consecutive_f'] for p in all_patterns]
            
            avg_max_m = np.mean(max_consecutive_m_values)
            avg_max_f = np.mean(max_consecutive_f_values)
            
            consolidated['priority_rules'].append({
                'rule_type': 'max_consecutive_limit',
                'max_consecutive_m': int(np.ceil(avg_max_m)),
                'max_consecutive_f': int(np.ceil(avg_max_f)),
                'weight': 0.3,
                'description': f'Avoid more than {int(np.ceil(avg_max_m))} consecutive M or {int(np.ceil(avg_max_f))} consecutive F'
            })
        
        # Rule 3: Alternation preference
        alternation_scores = []
        for patterns in all_patterns:
            if patterns['length'] > 1:
                alternation_ratio = patterns['alternation_count'] / (patterns['length'] - 1)
                alternation_scores.append(alternation_ratio)
        
        if alternation_scores:
            avg_alternation = np.mean(alternation_scores)
            consolidated['priority_rules'].append({
                'rule_type': 'alternation_preference',
                'target_alternation_ratio': avg_alternation,
                'weight': 0.2,
                'description': f'Target alternation ratio: {avg_alternation:.2f}'
            })
        
        # Rule 4: Balance preference
        balance_ratios = []
        for patterns in all_patterns:
            if patterns['length'] > 0:
                f_ratio = patterns['f_count'] / patterns['length']
                balance_ratios.append(abs(f_ratio - 0.5))
        
        if balance_ratios:
            avg_balance_deviation = np.mean(balance_ratios)
            consolidated['priority_rules'].append({
                'rule_type': 'balance_preference',
                'max_balance_deviation': avg_balance_deviation,
                'weight': 0.1,
                'description': f'Maintain balance within {avg_balance_deviation:.2f} of 50/50'
            })
        
        return consolidated


class RuleBasedEvaluator:
    """Evaluator that scores based on extracted human preference rules."""
    
    def __init__(self, extracted_rules: Dict[str, Any]):
        self.rules = extracted_rules
        self.priority_rules = extracted_rules.get('consolidated_rules', {}).get('priority_rules', [])
        self.scoring_weights = extracted_rules.get('consolidated_rules', {}).get('scoring_weights', {})
    
    def evaluate(self, original: str, diversified: str) -> float:
        """
        Evaluate diversified sequence based on extracted human rules.
        
        Args:
            original: Original sequence
            diversified: Diversified sequence
            
        Returns:
            Score from 0 to 1 based on rule compliance
        """
        total_score = 0
        total_weight = 0
        
        for rule in self.priority_rules:
            rule_score = self._evaluate_rule(original, diversified, rule)
            weight = rule.get('weight', 1.0)
            
            total_score += rule_score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0
    
    def _evaluate_rule(self, original: str, diversified: str, rule: Dict[str, Any]) -> float:
        """Evaluate a specific rule."""
        rule_type = rule['rule_type']
        
        if rule_type == 'first_minority_position':
            return self._evaluate_first_position_rule(diversified, rule)
        elif rule_type == 'max_consecutive_limit':
            return self._evaluate_consecutive_rule(diversified, rule)
        elif rule_type == 'alternation_preference':
            return self._evaluate_alternation_rule(diversified, rule)
        elif rule_type == 'balance_preference':
            return self._evaluate_balance_rule(diversified, rule)
        
        return 0
    
    def _evaluate_first_position_rule(self, diversified: str, rule: Dict[str, Any]) -> float:
        """Evaluate first minority position rule."""
        preferred_pos = rule['preferred_position']
        actual_pos = diversified.find('F') + 1  # 1-indexed
        
        if actual_pos <= 0:
            return 0  # No F found
        
        # Score based on how close to preferred position
        if actual_pos == preferred_pos:
            return 1.0
        elif abs(actual_pos - preferred_pos) == 1:
            return 0.8  # One position off
        elif abs(actual_pos - preferred_pos) == 2:
            return 0.5  # Two positions off
        else:
            return 0.2  # Further away
    
    def _evaluate_consecutive_rule(self, diversified: str, rule: Dict[str, Any]) -> float:
        """Evaluate consecutive character limit rule."""
        max_m = rule['max_consecutive_m']
        max_f = rule['max_consecutive_f']
        
        actual_max_m = self._max_consecutive(diversified, 'M')
        actual_max_f = self._max_consecutive(diversified, 'F')
        
        # Penalty for exceeding limits
        m_penalty = max(0, actual_max_m - max_m)
        f_penalty = max(0, actual_max_f - max_f)
        
        total_penalty = m_penalty + f_penalty
        
        # Convert penalty to score (lower penalty = higher score)
        max_possible_penalty = len(diversified)  # Worst case: all same character
        score = max(0, 1 - (total_penalty / max_possible_penalty))
        
        return score
    
    def _evaluate_alternation_rule(self, diversified: str, rule: Dict[str, Any]) -> float:
        """Evaluate alternation preference rule."""
        target_ratio = rule['target_alternation_ratio']
        
        if len(diversified) <= 1:
            return 1.0
        
        alternations = sum(1 for i in range(len(diversified)-1) if diversified[i] != diversified[i+1])
        actual_ratio = alternations / (len(diversified) - 1)
        
        # Score based on how close to target ratio
        deviation = abs(actual_ratio - target_ratio)
        score = max(0, 1 - deviation)
        
        return score
    
    def _evaluate_balance_rule(self, diversified: str, rule: Dict[str, Any]) -> float:
        """Evaluate balance preference rule."""
        max_deviation = rule['max_balance_deviation']
        
        if len(diversified) == 0:
            return 0
        
        f_count = diversified.count('F')
        f_ratio = f_count / len(diversified)
        actual_deviation = abs(f_ratio - 0.5)
        
        if actual_deviation <= max_deviation:
            return 1.0
        else:
            # Penalty for exceeding acceptable deviation
            excess_deviation = actual_deviation - max_deviation
            score = max(0, 1 - (excess_deviation / 0.5))  # 0.5 is max possible deviation
            return score
    
    def _max_consecutive(self, sequence: str, char: str) -> int:
        """Find maximum consecutive occurrences of a character."""
        max_count = 0
        current_count = 0
        
        for c in sequence:
            if c == char:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        
        return max_count
    
    def detailed_evaluation(self, original: str, diversified: str) -> Dict[str, Any]:
        """Provide detailed rule-by-rule evaluation."""
        evaluation = {
            'overall_score': self.evaluate(original, diversified),
            'rule_scores': [],
            'rule_violations': [],
            'rule_compliance': []
        }
        
        for rule in self.priority_rules:
            rule_score = self._evaluate_rule(original, diversified, rule)
            weight = rule.get('weight', 1.0)
            
            rule_result = {
                'rule_type': rule['rule_type'],
                'description': rule.get('description', ''),
                'score': rule_score,
                'weight': weight,
                'weighted_score': rule_score * weight
            }
            
            evaluation['rule_scores'].append(rule_result)
            
            if rule_score < 0.5:
                evaluation['rule_violations'].append(rule_result)
            else:
                evaluation['rule_compliance'].append(rule_result)
        
        return evaluation
    
    def get_rule_summary(self) -> str:
        """Get a human-readable summary of extracted rules."""
        summary = "📋 Extracted Human Preference Rules:\\n"
        summary += "=" * 40 + "\\n\\n"
        
        for i, rule in enumerate(self.priority_rules, 1):
            summary += f"{i}. {rule.get('description', rule['rule_type'])}\\n"
            summary += f"   Weight: {rule.get('weight', 1.0)}\\n\\n"
        
        return summary