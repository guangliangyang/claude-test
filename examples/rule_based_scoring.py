#!/usr/bin/env python3
"""
Rule-Based Scoring System
=========================

This script demonstrates the rule-based evaluation system that scores algorithms
based on mix rules extracted from human expert feedback.
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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def extract_mix_rules():
    """Extract mix rules from human expert feedback."""
    print("🔍 Extracting Mix Rules from Human Expert Feedback")
    print("=" * 60)
    
    extractor = MixRuleExtractor()
    extracted_rules = extractor.extract_rules_from_feedback(SURVEY_DATA)
    
    # Display extracted rules
    print("\\n📋 Extracted Rules Summary:")
    print("-" * 40)
    
    # Position rules
    print("\\n1. Position Rules:")
    for question, rules in extracted_rules['position_rules'].items():
        print(f"   Question: {question}")
        print(f"   Preferred first F positions: {dict(rules['position_frequencies'])}")
        print(f"   Most preferred positions: {rules['preferred_positions']}")
    
    # Pattern rules
    print("\\n2. Pattern Rules:")
    for question, rules in extracted_rules['pattern_rules'].items():
        print(f"   Question: {question}")
        print(f"   Common patterns: {dict(rules['common_patterns'].most_common(3))}")
        print(f"   Common transitions: {dict(rules['transition_patterns'].most_common(3))}")
        if rules['spacing_patterns']:
            avg_spacing = sum(rules['spacing_patterns']) / len(rules['spacing_patterns'])
            print(f"   Average F spacing: {avg_spacing:.1f}")
    
    # Consolidated rules
    print("\\n3. Consolidated Priority Rules:")
    priority_rules = extracted_rules['consolidated_rules']['priority_rules']
    for i, rule in enumerate(priority_rules, 1):
        print(f"   {i}. {rule['description']} (weight: {rule['weight']})")
    
    return extracted_rules


def test_rule_based_scoring():
    """Test algorithms using rule-based scoring."""
    print("\\n🎯 Testing Rule-Based Scoring System")
    print("=" * 60)
    
    # Extract rules
    extractor = MixRuleExtractor()
    extracted_rules = extractor.extract_rules_from_feedback(SURVEY_DATA)
    
    # Create rule-based evaluator
    evaluator = RuleBasedEvaluator(extracted_rules)
    
    # Test sequence
    original_sequence = "MMMMMFFFFF"
    search_result = SearchResult.from_string(original_sequence, "rule-based test")
    
    # Test algorithms
    algorithms = [
        ("Original Interleaving", InterleavingAlgorithm()),
        ("Original Proportional", ProportionalDistributionAlgorithm()),
        ("Original MMR", MMRAlgorithm()),
        ("Original Greedy", GreedyDiversificationAlgorithm()),
        ("Human-Aligned Interleaving (Pos 2)", HumanAlignedInterleavingAlgorithm(preferred_first_minority_pos=2)),
        ("Human-Aligned Interleaving (Pos 3)", HumanAlignedInterleavingAlgorithm(preferred_first_minority_pos=3)),
        ("Early Diversity Optimized", EarlyDiversityOptimizedAlgorithm(target_first_minority_pos=3)),
        ("Consensus-Based", ConsensusBasedAlgorithm()),
        ("Adaptive Human-Aligned", AdaptiveHumanAlignedAlgorithm())
    ]
    
    results = []
    
    print(f"\\nOriginal sequence: {original_sequence}")
    print("\\nRule-Based Algorithm Evaluation:")
    print("-" * 80)
    
    for name, algorithm in algorithms:
        result = algorithm.diversify(search_result)
        diversified_seq = ''.join([item.gender.value for item in result.diversified_items])
        
        # Rule-based evaluation
        rule_score = evaluator.evaluate(original_sequence, diversified_seq)
        detailed_eval = evaluator.detailed_evaluation(original_sequence, diversified_seq)
        
        results.append({
            'algorithm': name,
            'sequence': diversified_seq,
            'rule_score': rule_score,
            'first_f_pos': diversified_seq.find('F') + 1 if 'F' in diversified_seq else -1,
            'detailed_evaluation': detailed_eval
        })
        
        print(f"{name:35} | {diversified_seq} | Score: {rule_score:.3f}")
    
    return results, evaluator


def compare_human_responses_with_rules():
    """Compare human expert responses using rule-based scoring."""
    print("\\n👥 Human Expert Responses - Rule-Based Evaluation")
    print("=" * 60)
    
    # Extract rules and create evaluator
    extractor = MixRuleExtractor()
    extracted_rules = extractor.extract_rules_from_feedback(SURVEY_DATA)
    evaluator = RuleBasedEvaluator(extracted_rules)
    
    # Get human responses
    test_case = SURVEY_DATA[0]
    original = test_case['question'].replace(' ', '')
    human_responses = [r.replace(' ', '') for r in test_case['responses']]
    
    print(f"Original sequence: {original}")
    print("\\nHuman Expert Rule Compliance:")
    print("-" * 50)
    
    human_scores = []
    for i, response in enumerate(human_responses, 1):
        rule_score = evaluator.evaluate(original, response)
        detailed_eval = evaluator.detailed_evaluation(original, response)
        
        human_scores.append({
            'expert': f'Expert_{i}',
            'response': response,
            'rule_score': rule_score,
            'first_f_pos': response.find('F') + 1,
            'detailed_evaluation': detailed_eval
        })
        
        print(f"Expert {i:2d} | {response} | Score: {rule_score:.3f} | First F: pos {response.find('F')+1}")
    
    # Summary statistics
    scores = [h['rule_score'] for h in human_scores]
    print(f"\\nHuman Expert Rule Compliance Summary:")
    print(f"Average score: {sum(scores)/len(scores):.3f}")
    print(f"Score range: {min(scores):.3f} - {max(scores):.3f}")
    print(f"Experts scoring > 0.8: {sum(1 for s in scores if s > 0.8)}/{len(scores)}")
    
    return human_scores


def detailed_rule_analysis():
    """Provide detailed analysis of rule compliance."""
    print("\\n🔍 Detailed Rule Compliance Analysis")
    print("=" * 60)
    
    # Get data
    algo_results, evaluator = test_rule_based_scoring()
    human_results = compare_human_responses_with_rules()
    
    print("\\n📊 Rule-by-Rule Performance Analysis:")
    print("-" * 50)
    
    # Analyze best performing algorithm
    best_algo = max(algo_results, key=lambda x: x['rule_score'])
    print(f"\\n🏆 Best Algorithm: {best_algo['algorithm']} (Score: {best_algo['rule_score']:.3f})")
    
    detailed = best_algo['detailed_evaluation']
    print("\\nRule Compliance Breakdown:")
    for rule_result in detailed['rule_scores']:
        print(f"  {rule_result['description']}: {rule_result['score']:.3f} (weight: {rule_result['weight']})")
    
    # Analyze best human expert
    best_human = max(human_results, key=lambda x: x['rule_score'])
    print(f"\\n👤 Best Human Expert: {best_human['expert']} (Score: {best_human['rule_score']:.3f})")
    
    detailed_human = best_human['detailed_evaluation']
    print("\\nRule Compliance Breakdown:")
    for rule_result in detailed_human['rule_scores']:
        print(f"  {rule_result['description']}: {rule_result['score']:.3f} (weight: {rule_result['weight']})")
    
    # Compare algorithm vs human performance
    algo_scores = [r['rule_score'] for r in algo_results]
    human_scores = [h['rule_score'] for h in human_results]
    
    print(f"\\n📈 Performance Comparison:")
    print(f"Best algorithm score: {max(algo_scores):.3f}")
    print(f"Best human score: {max(human_scores):.3f}")
    print(f"Average algorithm score: {sum(algo_scores)/len(algo_scores):.3f}")
    print(f"Average human score: {sum(human_scores)/len(human_scores):.3f}")
    
    if max(algo_scores) > max(human_scores):
        print("✅ Best algorithm exceeds best human expert!")
    elif sum(algo_scores)/len(algo_scores) > sum(human_scores)/len(human_scores):
        print("✅ Algorithms exceed human average performance!")
    else:
        gap = (sum(human_scores)/len(human_scores)) - (sum(algo_scores)/len(algo_scores))
        print(f"⚠️  Human experts outperform algorithms by {gap:.3f} points on average")


def visualize_rule_based_results():
    """Create visualizations for rule-based evaluation results."""
    print("\\n📊 Creating Rule-Based Evaluation Visualizations")
    print("=" * 60)
    
    # Get data
    algo_results, evaluator = test_rule_based_scoring()
    human_results = compare_human_responses_with_rules()
    
    # Create visualizations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Algorithm rule-based scores
    algo_df = pd.DataFrame(algo_results)
    algo_df['type'] = algo_df['algorithm'].apply(lambda x: 'Improved' if any(keyword in x for keyword in ['Human-Aligned', 'Early Diversity', 'Consensus', 'Adaptive']) else 'Original')
    
    # Sort by score for better visualization
    algo_df_sorted = algo_df.sort_values('rule_score')
    colors = ['#FF6B6B' if t == 'Original' else '#4ECDC4' for t in algo_df_sorted['type']]
    
    bars1 = ax1.barh(range(len(algo_df_sorted)), algo_df_sorted['rule_score'], color=colors)
    ax1.set_yticks(range(len(algo_df_sorted)))
    ax1.set_yticklabels([name.replace('Human-Aligned ', '').replace('Optimized', 'Opt.') for name in algo_df_sorted['algorithm']], fontsize=9)
    ax1.set_title('Algorithm Rule-Based Scores', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Rule Compliance Score')
    
    # Add score labels
    for i, (bar, score) in enumerate(zip(bars1, algo_df_sorted['rule_score'])):
        ax1.text(score + 0.01, bar.get_y() + bar.get_height()/2, f'{score:.3f}', 
                va='center', fontweight='bold', fontsize=8)
    
    # 2. Human expert rule compliance distribution
    human_df = pd.DataFrame(human_results)
    ax2.hist(human_df['rule_score'], bins=8, alpha=0.7, color='#74B9FF', edgecolor='black')
    ax2.axvline(human_df['rule_score'].mean(), color='red', linestyle='--', linewidth=2,
                label=f'Human Avg: {human_df["rule_score"].mean():.3f}')
    ax2.axvline(algo_df['rule_score'].mean(), color='green', linestyle='--', linewidth=2,
                label=f'Algo Avg: {algo_df["rule_score"].mean():.3f}')
    ax2.set_title('Human Expert Rule Compliance', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Rule Compliance Score')
    ax2.set_ylabel('Number of Experts')
    ax2.legend()
    
    # 3. Algorithm type comparison
    type_comparison = algo_df.groupby('type')['rule_score'].agg(['mean', 'max', 'min']).reset_index()
    x_pos = range(len(type_comparison))
    
    bars3 = ax3.bar(x_pos, type_comparison['mean'], 
                    yerr=[type_comparison['mean'] - type_comparison['min'], 
                          type_comparison['max'] - type_comparison['mean']], 
                    color=['#FF6B6B', '#4ECDC4'], alpha=0.8, capsize=5)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(type_comparison['type'])
    ax3.set_title('Original vs Improved Algorithms\\n(Rule-Based Scoring)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Rule Compliance Score')
    
    # Add value labels
    for i, (bar, mean_val) in enumerate(zip(bars3, type_comparison['mean'])):
        ax3.text(bar.get_x() + bar.get_width()/2, mean_val + 0.02, f'{mean_val:.3f}', 
                ha='center', fontweight='bold')
    
    # 4. First F position vs rule score
    all_data = []
    for result in algo_results:
        all_data.append({
            'type': 'Algorithm',
            'name': result['algorithm'],
            'first_f_pos': result['first_f_pos'],
            'rule_score': result['rule_score']
        })
    
    for result in human_results:
        all_data.append({
            'type': 'Human Expert',
            'name': result['expert'],
            'first_f_pos': result['first_f_pos'],
            'rule_score': result['rule_score']
        })
    
    all_df = pd.DataFrame(all_data)
    
    # Scatter plot
    for type_name, color in [('Algorithm', '#4ECDC4'), ('Human Expert', '#74B9FF')]:
        type_data = all_df[all_df['type'] == type_name]
        ax4.scatter(type_data['first_f_pos'], type_data['rule_score'], 
                   c=color, label=type_name, alpha=0.7, s=50)
    
    ax4.set_title('First F Position vs Rule Score', fontsize=14, fontweight='bold')
    ax4.set_xlabel('First F Position')
    ax4.set_ylabel('Rule Compliance Score')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/andy/workspace/claude-test/diversification_report/rule_based_evaluation.png', 
                dpi=300, bbox_inches='tight')
    # plt.show()  # Commented out to avoid timeout
    
    print("📊 Visualization saved as 'rule_based_evaluation.png'")


def main():
    """Main function to run rule-based evaluation."""
    print("🎯 Rule-Based Evaluation System for Gender Diversification")
    print("=" * 70)
    
    # Extract mix rules
    extracted_rules = extract_mix_rules()
    
    # Test rule-based scoring
    algo_results, evaluator = test_rule_based_scoring()
    human_results = compare_human_responses_with_rules()
    
    # Detailed analysis
    detailed_rule_analysis()
    
    # Create visualizations
    visualize_rule_based_results()
    
    # Print rule summary
    print("\\n" + evaluator.get_rule_summary())
    
    return {
        'extracted_rules': extracted_rules,
        'algorithm_results': algo_results,
        'human_results': human_results,
        'evaluator': evaluator
    }


if __name__ == "__main__":
    results = main()