#!/usr/bin/env python3
"""
Quick Rule-Based Evaluation Test
===============================

Fast test of the rule-based evaluation system without heavy visualizations.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.survey_feedback import SURVEY_DATA
from src.diversification.rule_based_evaluation import MixRuleExtractor, RuleBasedEvaluator
from src.diversification.algorithms import (
    InterleavingAlgorithm,
    ProportionalDistributionAlgorithm,
    MMRAlgorithm,
    GreedyDiversificationAlgorithm
)
from src.diversification.improved_algorithms import (
    HumanAlignedInterleavingAlgorithm,
    EarlyDiversityOptimizedAlgorithm,
    ConsensusBasedAlgorithm,
    AdaptiveHumanAlignedAlgorithm
)
from src.diversification.models import SearchResult


def main():
    """Quick test of rule-based evaluation."""
    print("🎯 Quick Rule-Based Evaluation Test")
    print("=" * 50)
    
    # Extract rules
    print("🔍 Extracting mix rules from human feedback...")
    extractor = MixRuleExtractor()
    extracted_rules = extractor.extract_rules_from_feedback(SURVEY_DATA)
    
    # Show extracted rules
    priority_rules = extracted_rules['consolidated_rules']['priority_rules']
    print("\\n📋 Extracted Mix Rules:")
    for i, rule in enumerate(priority_rules, 1):
        print(f"  {i}. {rule['description']} (weight: {rule['weight']})")
    
    # Create evaluator
    evaluator = RuleBasedEvaluator(extracted_rules)
    
    # Test sequence
    original_sequence = "MMMMMFFFFF"
    search_result = SearchResult.from_string(original_sequence, "rule test")
    
    print(f"\\n🧪 Testing algorithms on: {original_sequence}")
    print("\\nRule-Based Scores:")
    print("-" * 70)
    
    # Test original algorithms
    original_algorithms = [
        ("Original Interleaving", InterleavingAlgorithm()),
        ("Original Proportional", ProportionalDistributionAlgorithm()),
        ("Original MMR", MMRAlgorithm()),
        ("Original Greedy", GreedyDiversificationAlgorithm())
    ]
    
    original_scores = []
    for name, algorithm in original_algorithms:
        result = algorithm.diversify(search_result)
        diversified_seq = ''.join([item.gender.value for item in result.diversified_items])
        rule_score = evaluator.evaluate(original_sequence, diversified_seq)
        first_f_pos = diversified_seq.find('F') + 1 if 'F' in diversified_seq else -1
        
        original_scores.append(rule_score)
        print(f"{name:25} | {diversified_seq} | {rule_score:.3f} | F@{first_f_pos}")
    
    print()
    
    # Test improved algorithms
    improved_algorithms = [
        ("Human-Aligned (Pos 2)", HumanAlignedInterleavingAlgorithm(preferred_first_minority_pos=2)),
        ("Human-Aligned (Pos 3)", HumanAlignedInterleavingAlgorithm(preferred_first_minority_pos=3)),
        ("Early Diversity Opt", EarlyDiversityOptimizedAlgorithm(target_first_minority_pos=3)),
        ("Consensus-Based", ConsensusBasedAlgorithm()),
        ("Adaptive Aligned", AdaptiveHumanAlignedAlgorithm())
    ]
    
    improved_scores = []
    for name, algorithm in improved_algorithms:
        result = algorithm.diversify(search_result)
        diversified_seq = ''.join([item.gender.value for item in result.diversified_items])
        rule_score = evaluator.evaluate(original_sequence, diversified_seq)
        first_f_pos = diversified_seq.find('F') + 1 if 'F' in diversified_seq else -1
        
        improved_scores.append(rule_score)
        print(f"{name:25} | {diversified_seq} | {rule_score:.3f} | F@{first_f_pos}")
    
    # Test human expert responses
    print("\\n👥 Human Expert Responses:")
    print("-" * 70)
    
    test_case = SURVEY_DATA[0]
    human_responses = [r.replace(' ', '') for r in test_case['responses']]
    
    human_scores = []
    for i, response in enumerate(human_responses[:5], 1):  # Show first 5
        rule_score = evaluator.evaluate(original_sequence, response)
        first_f_pos = response.find('F') + 1
        human_scores.append(rule_score)
        print(f"Expert {i:2d}              | {response} | {rule_score:.3f} | F@{first_f_pos}")
    
    print("...")
    
    # Summary
    print("\\n📊 Summary:")
    print("-" * 50)
    
    orig_avg = sum(original_scores) / len(original_scores)
    improved_avg = sum(improved_scores) / len(improved_scores)
    human_avg = sum(evaluator.evaluate(original_sequence, r.replace(' ', '')) 
                   for r in test_case['responses']) / len(test_case['responses'])
    
    print(f"Original algorithms average:  {orig_avg:.3f}")
    print(f"Improved algorithms average:  {improved_avg:.3f}")
    print(f"Human experts average:        {human_avg:.3f}")
    
    improvement = improved_avg - orig_avg
    print(f"\\n🎯 Improvement: +{improvement:.3f} points ({improvement/orig_avg*100:.1f}%)")
    
    if improved_avg > human_avg:
        print("✅ Improved algorithms exceed human expert average!")
    else:
        gap = human_avg - improved_avg
        print(f"⚠️  Human experts still outperform by {gap:.3f} points")
    
    # Best algorithm
    best_improved_idx = improved_scores.index(max(improved_scores))
    best_algo_name = improved_algorithms[best_improved_idx][0]
    print(f"\\n🏆 Best algorithm: {best_algo_name} (score: {max(improved_scores):.3f})")
    
    return {
        'extracted_rules': extracted_rules,
        'original_avg': orig_avg,
        'improved_avg': improved_avg,
        'human_avg': human_avg,
        'best_score': max(improved_scores)
    }


if __name__ == "__main__":
    results = main()