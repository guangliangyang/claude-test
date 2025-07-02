#!/usr/bin/env python3
"""
Human Feedback Analysis and Evaluation
======================================

This script analyzes human expert feedback for gender diversification
and demonstrates how to align algorithmic evaluation with human preferences.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.survey_feedback import SURVEY_DATA
from src.diversification.human_alignment import HumanFeedbackAnalyzer, HumanAlignedEvaluator
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
from collections import Counter


def analyze_human_feedback():
    """Analyze human expert feedback to understand preferences."""
    print("🔍 Analyzing Human Expert Feedback")
    print("=" * 50)
    
    analyzer = HumanFeedbackAnalyzer()
    analysis = analyzer.analyze_survey_data(SURVEY_DATA)
    
    # Print analysis results
    print("\n📊 Position Preferences Analysis:")
    for question, positions in analysis['position_preferences'].items():
        position_dist = Counter(positions)
        print(f"\nSequence: {question}")
        print(f"First F position distribution: {dict(position_dist)}")
        print(f"Most preferred position: {position_dist.most_common(1)[0][0]}")
        print(f"Average position: {sum(positions)/len(positions):.1f}")
    
    print("\n📋 Consensus Analysis:")
    for question, consensus in analysis['consensus_analysis'].items():
        print(f"\nSequence: {question}")
        print(f"Consensus strength: {consensus['consensus_strength']:.2f}")
        print(f"Most common response: {consensus['most_common_response']}")
        print(f"First F position stats: {consensus['first_f_position_stats']}")
    
    print("\n📏 Extracted Rules:")
    for rule in analysis['extracted_rules']:
        print(f"• {rule}")
    
    return analysis


def compare_algorithms_with_human_alignment():
    """Compare algorithms using human-aligned evaluation."""
    print("\n🤖 Algorithm Comparison with Human Alignment")
    print("=" * 50)
    
    # Analyze feedback first
    analyzer = HumanFeedbackAnalyzer()
    analysis = analyzer.analyze_survey_data(SURVEY_DATA)
    
    # Create human-aligned evaluator
    evaluator = HumanAlignedEvaluator(analysis)
    
    # Test sequence from survey
    original_sequence = "MMMMMFFFFF"
    search_result = SearchResult.from_string(original_sequence, "survey test case")
    
    # Test different algorithms
    algorithms = [
        ("Interleaving", InterleavingAlgorithm()),
        ("Proportional", ProportionalDistributionAlgorithm()),
        ("MMR", MMRAlgorithm()),
        ("Greedy", GreedyDiversificationAlgorithm())
    ]
    
    results = []
    
    print(f"\nOriginal sequence: {original_sequence}")
    print("\nAlgorithm Results:")
    
    for name, algorithm in algorithms:
        result = algorithm.diversify(search_result)
        diversified_seq = ''.join([item.gender.value for item in result.diversified_items])
        
        # Human-aligned evaluation
        human_score = evaluator.evaluate(original_sequence, diversified_seq)
        detailed_scores = evaluator.detailed_evaluation(original_sequence, diversified_seq)
        
        results.append({
            'algorithm': name,
            'sequence': diversified_seq,
            'human_score': human_score,
            'early_diversity': detailed_scores['early_diversity'],
            'alternation': detailed_scores['alternation'],
            'balance': detailed_scores['balance'],
            'consensus_alignment': detailed_scores['consensus_alignment']
        })
        
        print(f"\n{name:12} | {diversified_seq} | Score: {human_score:.3f}")
        print(f"             | Early: {detailed_scores['early_diversity']:.3f} | "
              f"Alt: {detailed_scores['alternation']:.3f} | "
              f"Bal: {detailed_scores['balance']:.3f} | "
              f"Cons: {detailed_scores['consensus_alignment']:.3f}")
    
    return results


def evaluate_against_human_responses():
    """Evaluate how well each human response scores with our metrics."""
    print("\n👥 Human Response Evaluation")
    print("=" * 50)
    
    # Analyze feedback
    analyzer = HumanFeedbackAnalyzer()
    analysis = analyzer.analyze_survey_data(SURVEY_DATA)
    evaluator = HumanAlignedEvaluator(analysis)
    
    # Get human responses for the main test case
    test_case = SURVEY_DATA[0]
    original = test_case['question'].replace(' ', '')
    responses = [r.replace(' ', '') for r in test_case['responses']]
    
    print(f"Original sequence: {original}")
    print("\nHuman Expert Responses Evaluation:")
    
    response_scores = []
    for i, response in enumerate(responses, 1):
        score = evaluator.evaluate(original, response)
        detailed = evaluator.detailed_evaluation(original, response)
        
        response_scores.append({
            'respondent': f'Expert_{i}',
            'response': response,
            'score': score,
            'first_f_pos': response.find('F') + 1
        })
        
        print(f"Expert {i:2d} | {response} | Score: {score:.3f} | First F: pos {response.find('F')+1}")
    
    # Analyze response quality distribution
    scores = [r['score'] for r in response_scores]
    positions = [r['first_f_pos'] for r in response_scores]
    
    print(f"\nResponse Quality Stats:")
    print(f"Average score: {sum(scores)/len(scores):.3f}")
    print(f"Score range: {min(scores):.3f} - {max(scores):.3f}")
    print(f"Position preferences: {Counter(positions)}")
    
    return response_scores


def visualize_human_alignment():
    """Create visualizations for human alignment analysis."""
    print("\n📈 Creating Human Alignment Visualizations")
    print("=" * 50)
    
    # Get data
    analyzer = HumanFeedbackAnalyzer()
    analysis = analyzer.analyze_survey_data(SURVEY_DATA)
    evaluator = HumanAlignedEvaluator(analysis)
    
    # Algorithm comparison
    algo_results = compare_algorithms_with_human_alignment()
    human_scores = evaluate_against_human_responses()
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Algorithm human-alignment scores
    algo_df = pd.DataFrame(algo_results)
    bars1 = ax1.bar(algo_df['algorithm'], algo_df['human_score'], 
                    color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax1.set_title('Algorithm Human-Alignment Scores', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Human Alignment Score')
    ax1.set_ylim(0, 1)
    for bar, score in zip(bars1, algo_df['human_score']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', fontweight='bold')
    
    # 2. Human response score distribution
    human_df = pd.DataFrame(human_scores)
    ax2.hist(human_df['score'], bins=10, alpha=0.7, color='#74B9FF', edgecolor='black')
    ax2.axvline(human_df['score'].mean(), color='red', linestyle='--', 
                label=f'Mean: {human_df["score"].mean():.3f}')
    ax2.set_title('Human Expert Response Score Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Human Alignment Score')
    ax2.set_ylabel('Number of Responses')
    ax2.legend()
    
    # 3. First F position preferences
    position_counts = Counter(human_df['first_f_pos'])
    positions = list(position_counts.keys())
    counts = list(position_counts.values())
    colors = ['#FF7675', '#74B9FF', '#00B894', '#FDCB6E']
    ax3.pie(counts, labels=[f'Position {p}' for p in positions], autopct='%1.1f%%',
            colors=colors[:len(positions)])
    ax3.set_title('Human Preference: First F Position', fontsize=14, fontweight='bold')
    
    # 4. Metric breakdown comparison
    metrics = ['early_diversity', 'alternation', 'balance', 'consensus_alignment']
    algo_names = algo_df['algorithm'].tolist()
    
    x = range(len(algo_names))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = algo_df[metric].tolist()
        ax4.bar([xi + i*width for xi in x], values, width, 
                label=metric.replace('_', ' ').title(), alpha=0.8)
    
    ax4.set_title('Algorithm Performance by Metric', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Algorithm')
    ax4.set_ylabel('Score')
    ax4.set_xticks([xi + width*1.5 for xi in x])
    ax4.set_xticklabels(algo_names)
    ax4.legend()
    ax4.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('/Users/andy/workspace/claude-test/diversification_report/human_alignment_analysis.png', 
                dpi=300, bbox_inches='tight')
    # plt.show()  # Commented out to avoid display issues in CLI
    
    print("📊 Visualization saved as 'human_alignment_analysis.png'")


def main():
    """Main function to run complete human alignment analysis."""
    print("🎯 Human Feedback Analysis for Gender Diversification")
    print("=" * 60)
    
    # Run analysis
    analysis = analyze_human_feedback()
    algo_results = compare_algorithms_with_human_alignment()
    human_scores = evaluate_against_human_responses()
    
    # Key insights
    print("\n🎯 Key Insights:")
    print("=" * 50)
    
    # Best algorithm for human alignment
    best_algo = max(algo_results, key=lambda x: x['human_score'])
    print(f"• Best algorithm for human alignment: {best_algo['algorithm']} (score: {best_algo['human_score']:.3f})")
    
    # Human expert consensus
    human_df = pd.DataFrame(human_scores)
    avg_human_score = human_df['score'].mean()
    position_consensus = Counter(human_df['first_f_pos']).most_common(1)[0]
    print(f"• Human expert average score: {avg_human_score:.3f}")
    print(f"• Most preferred first F position: {position_consensus[0]} ({position_consensus[1]}/14 experts)")
    
    # Algorithm vs human comparison
    algo_scores = [r['human_score'] for r in algo_results]
    best_algo_score = max(algo_scores)
    print(f"• Best algorithm vs human average: {best_algo_score:.3f} vs {avg_human_score:.3f}")
    
    if best_algo_score > avg_human_score:
        print("✅ Our best algorithm exceeds average human performance!")
    else:
        gap = avg_human_score - best_algo_score
        print(f"⚠️  Human experts outperform algorithms by {gap:.3f} points")
    
    # Create visualizations
    visualize_human_alignment()
    
    return {
        'analysis': analysis,
        'algorithm_results': algo_results,
        'human_scores': human_scores
    }


if __name__ == "__main__":
    results = main()