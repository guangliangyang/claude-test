#!/usr/bin/env python3
"""
Test Improved Human-Aligned Algorithms
======================================

This script tests the new human-aligned algorithms against the survey data
and compares their performance with the original algorithms.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.survey_feedback import SURVEY_DATA
from src.diversification.human_alignment import HumanFeedbackAnalyzer, HumanAlignedEvaluator
from src.diversification.improved_algorithms import (
    HumanAlignedInterleavingAlgorithm,
    EarlyDiversityOptimizedAlgorithm,
    ConsensusBasedAlgorithm,
    AdaptiveHumanAlignedAlgorithm
)
from src.diversification.algorithms import (
    InterleavingAlgorithm,
    ProportionalDistributionAlgorithm,
    MMRAlgorithm,
    GreedyDiversificationAlgorithm
)
from src.diversification.models import SearchResult
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def test_improved_algorithms():
    """Test improved algorithms against human feedback."""
    print("🚀 Testing Improved Human-Aligned Algorithms")
    print("=" * 60)
    
    # Set up human-aligned evaluator
    analyzer = HumanFeedbackAnalyzer()
    analysis = analyzer.analyze_survey_data(SURVEY_DATA)
    evaluator = HumanAlignedEvaluator(analysis)
    
    # Test sequence from survey
    original_sequence = "MMMMMFFFFF"
    search_result = SearchResult.from_string(original_sequence, "survey test case")
    
    # Original algorithms
    original_algorithms = [
        ("Original Interleaving", InterleavingAlgorithm()),
        ("Original Proportional", ProportionalDistributionAlgorithm()),
        ("Original MMR", MMRAlgorithm()),
        ("Original Greedy", GreedyDiversificationAlgorithm())
    ]
    
    # Improved algorithms
    improved_algorithms = [
        ("Human-Aligned Interleaving", HumanAlignedInterleavingAlgorithm(preferred_first_minority_pos=2)),
        ("Human-Aligned Interleaving (Pos 3)", HumanAlignedInterleavingAlgorithm(preferred_first_minority_pos=3)),
        ("Early Diversity Optimized", EarlyDiversityOptimizedAlgorithm(target_first_minority_pos=3)),
        ("Consensus-Based", ConsensusBasedAlgorithm()),
        ("Adaptive Human-Aligned", AdaptiveHumanAlignedAlgorithm())
    ]
    
    all_algorithms = original_algorithms + improved_algorithms
    
    results = []
    
    print(f"Original sequence: {original_sequence}")
    print("\\nAlgorithm Performance:")
    print("-" * 80)
    
    for name, algorithm in all_algorithms:
        result = algorithm.diversify(search_result)
        diversified_seq = ''.join([item.gender.value for item in result.diversified_items])
        
        # Human-aligned evaluation
        human_score = evaluator.evaluate(original_sequence, diversified_seq)
        detailed_scores = evaluator.detailed_evaluation(original_sequence, diversified_seq)
        
        results.append({
            'algorithm': name,
            'type': 'Improved' if 'Human-Aligned' in name or 'Early Diversity' in name or 'Consensus' in name or 'Adaptive' in name else 'Original',
            'sequence': diversified_seq,
            'human_score': human_score,
            'early_diversity': detailed_scores['early_diversity'],
            'alternation': detailed_scores['alternation'],
            'balance': detailed_scores['balance'],
            'consensus_alignment': detailed_scores['consensus_alignment'],
            'first_f_pos': diversified_seq.find('F') + 1 if 'F' in diversified_seq else -1
        })
        
        print(f"{name:30} | {diversified_seq} | Score: {human_score:.3f} | First F: pos {diversified_seq.find('F')+1}")
    
    return results


def compare_with_human_experts():
    """Compare improved algorithms with human expert responses."""
    print("\\n👥 Comparison with Human Expert Responses")
    print("=" * 60)
    
    # Get human expert data
    test_case = SURVEY_DATA[0]
    original = test_case['question'].replace(' ', '')
    human_responses = [r.replace(' ', '') for r in test_case['responses']]
    
    # Set up evaluator
    analyzer = HumanFeedbackAnalyzer()
    analysis = analyzer.analyze_survey_data(SURVEY_DATA)
    evaluator = HumanAlignedEvaluator(analysis)
    
    # Test improved algorithms
    search_result = SearchResult.from_string(original, "survey comparison")
    
    improved_algorithms = [
        ("Human-Aligned Interleaving (Pos 2)", HumanAlignedInterleavingAlgorithm(preferred_first_minority_pos=2)),
        ("Human-Aligned Interleaving (Pos 3)", HumanAlignedInterleavingAlgorithm(preferred_first_minority_pos=3)),
        ("Early Diversity Optimized", EarlyDiversityOptimizedAlgorithm(target_first_minority_pos=3)),
        ("Consensus-Based", ConsensusBasedAlgorithm()),
        ("Adaptive Human-Aligned", AdaptiveHumanAlignedAlgorithm())
    ]
    
    # Get algorithm scores
    algo_scores = []
    for name, algorithm in improved_algorithms:
        result = algorithm.diversify(search_result)
        diversified_seq = ''.join([item.gender.value for item in result.diversified_items])
        score = evaluator.evaluate(original, diversified_seq)
        
        algo_scores.append({
            'type': 'Algorithm',
            'name': name,
            'sequence': diversified_seq,
            'score': score,
            'first_f_pos': diversified_seq.find('F') + 1
        })
    
    # Get human scores
    human_scores = []
    for i, response in enumerate(human_responses, 1):
        score = evaluator.evaluate(original, response)
        human_scores.append({
            'type': 'Human Expert',
            'name': f'Expert {i}',
            'sequence': response,
            'score': score,
            'first_f_pos': response.find('F') + 1
        })
    
    # Combine and analyze
    all_scores = algo_scores + human_scores
    
    # Summary statistics
    algo_avg = sum(s['score'] for s in algo_scores) / len(algo_scores)
    human_avg = sum(s['score'] for s in human_scores) / len(human_scores)
    best_algo = max(algo_scores, key=lambda x: x['score'])
    best_human = max(human_scores, key=lambda x: x['score'])
    
    print(f"Algorithm Average Score: {algo_avg:.3f}")
    print(f"Human Expert Average Score: {human_avg:.3f}")
    print(f"Best Algorithm: {best_algo['name']} (score: {best_algo['score']:.3f})")
    print(f"Best Human Expert: {best_human['name']} (score: {best_human['score']:.3f})")
    
    improvement = algo_avg - 0.320  # Original algorithm score was 0.320
    print(f"\\n🎯 Improvement over original algorithms: {improvement:.3f} points")
    
    if algo_avg > human_avg:
        print("✅ Improved algorithms now exceed human expert average!")
    else:
        gap = human_avg - algo_avg
        print(f"⚠️  Human experts still outperform by {gap:.3f} points")
    
    return all_scores


def visualize_improvements():
    """Create visualizations showing algorithm improvements."""
    print("\\n📊 Creating Improvement Visualizations")
    print("=" * 60)
    
    # Get data
    results = test_improved_algorithms()
    comparison_data = compare_with_human_experts()
    
    # Create visualizations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Algorithm type comparison
    df = pd.DataFrame(results)
    type_comparison = df.groupby('type')['human_score'].agg(['mean', 'max', 'min']).reset_index()
    
    x_pos = range(len(type_comparison))
    ax1.bar(x_pos, type_comparison['mean'], yerr=[type_comparison['mean'] - type_comparison['min'], 
                                                  type_comparison['max'] - type_comparison['mean']], 
            color=['#FF6B6B', '#4ECDC4'], alpha=0.8, capsize=5)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(type_comparison['type'])
    ax1.set_title('Original vs Improved Algorithms', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Human Alignment Score')
    ax1.set_ylim(0, 1)
    
    # Add value labels
    for i, (mean_val, max_val, min_val) in enumerate(zip(type_comparison['mean'], 
                                                         type_comparison['max'], 
                                                         type_comparison['min'])):
        ax1.text(i, mean_val + 0.05, f'{mean_val:.3f}', ha='center', fontweight='bold')
    
    # 2. Individual algorithm scores
    improved_algos = df[df['type'] == 'Improved'].sort_values('human_score', ascending=True)
    ax2.barh(range(len(improved_algos)), improved_algos['human_score'], 
             color=plt.cm.viridis(improved_algos['human_score']))
    ax2.set_yticks(range(len(improved_algos)))
    ax2.set_yticklabels([name.replace('Human-Aligned ', '').replace('Optimized', 'Opt.') 
                        for name in improved_algos['algorithm']], fontsize=10)
    ax2.set_title('Improved Algorithm Performance', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Human Alignment Score')
    
    # Add score labels
    for i, score in enumerate(improved_algos['human_score']):
        ax2.text(score + 0.01, i, f'{score:.3f}', va='center', fontweight='bold')
    
    # 3. Algorithm vs Human comparison
    comp_df = pd.DataFrame(comparison_data)
    algo_data = comp_df[comp_df['type'] == 'Algorithm']
    human_data = comp_df[comp_df['type'] == 'Human Expert']
    
    ax3.hist(algo_data['score'], bins=10, alpha=0.7, label='Algorithms', color='#74B9FF')
    ax3.hist(human_data['score'], bins=10, alpha=0.7, label='Human Experts', color='#55A3FF')
    ax3.axvline(algo_data['score'].mean(), color='#74B9FF', linestyle='--', linewidth=2, 
                label=f'Algo Avg: {algo_data["score"].mean():.3f}')
    ax3.axvline(human_data['score'].mean(), color='#55A3FF', linestyle='--', linewidth=2,
                label=f'Human Avg: {human_data["score"].mean():.3f}')
    ax3.set_title('Algorithm vs Human Expert Scores', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Human Alignment Score')
    ax3.set_ylabel('Count')
    ax3.legend()
    
    # 4. First F position preferences
    position_data = df.groupby(['type', 'first_f_pos']).size().unstack(fill_value=0)
    position_data.plot(kind='bar', ax=ax4, color=['#E17055', '#74B9FF', '#00B894'])
    ax4.set_title('First F Position Distribution', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Algorithm Type')
    ax4.set_ylabel('Count')
    ax4.legend(title='First F Position')
    ax4.tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.savefig('/Users/andy/workspace/claude-test/diversification_report/algorithm_improvements.png', 
                dpi=300, bbox_inches='tight')
    # plt.show()  # Commented out to avoid display timeout
    
    print("📊 Visualization saved as 'algorithm_improvements.png'")


def detailed_analysis():
    """Provide detailed analysis of improvements."""
    print("\\n🔍 Detailed Analysis of Improvements")
    print("=" * 60)
    
    results = test_improved_algorithms()
    comparison_data = compare_with_human_experts()
    
    df = pd.DataFrame(results)
    
    # Best performing algorithm
    best_algo = df.loc[df['human_score'].idxmax()]
    print(f"🏆 Best Performing Algorithm:")
    print(f"   Name: {best_algo['algorithm']}")
    print(f"   Score: {best_algo['human_score']:.3f}")
    print(f"   Sequence: {best_algo['sequence']}")
    print(f"   First F Position: {best_algo['first_f_pos']}")
    
    # Improvement analysis
    original_avg = df[df['type'] == 'Original']['human_score'].mean()
    improved_avg = df[df['type'] == 'Improved']['human_score'].mean()
    improvement = improved_avg - original_avg
    
    print(f"\\n📈 Performance Improvement:")
    print(f"   Original algorithms average: {original_avg:.3f}")
    print(f"   Improved algorithms average: {improved_avg:.3f}")
    print(f"   Improvement: +{improvement:.3f} points ({improvement/original_avg*100:.1f}%)")
    
    # Human alignment analysis
    comp_df = pd.DataFrame(comparison_data)
    algo_scores = comp_df[comp_df['type'] == 'Algorithm']['score']
    human_scores = comp_df[comp_df['type'] == 'Human Expert']['score']
    
    print(f"\\n👥 Human Alignment Analysis:")
    print(f"   Improved algorithms average: {algo_scores.mean():.3f}")
    print(f"   Human experts average: {human_scores.mean():.3f}")
    
    if algo_scores.mean() > human_scores.mean():
        print("   ✅ Algorithms now exceed human performance!")
    else:
        gap = human_scores.mean() - algo_scores.mean()
        print(f"   Gap to human performance: {gap:.3f} points")
    
    # Position preference analysis
    print(f"\\n📍 Position Preference Analysis:")
    for algo_type in ['Original', 'Improved']:
        type_data = df[df['type'] == algo_type]
        avg_pos = type_data['first_f_pos'].mean()
        print(f"   {algo_type} algorithms - Average first F position: {avg_pos:.1f}")
    
    human_pos_avg = comp_df[comp_df['type'] == 'Human Expert']['first_f_pos'].mean()
    print(f"   Human experts - Average first F position: {human_pos_avg:.1f}")


def main():
    """Main function to run improved algorithm testing."""
    print("🎯 Testing Human-Aligned Algorithm Improvements")
    print("=" * 70)
    
    # Run tests
    results = test_improved_algorithms()
    comparison_data = compare_with_human_experts()
    
    # Create visualizations
    visualize_improvements()
    
    # Detailed analysis
    detailed_analysis()
    
    return {
        'algorithm_results': results,
        'comparison_data': comparison_data
    }


if __name__ == "__main__":
    results = main()