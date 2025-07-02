#!/usr/bin/env python3
"""
Image Search Diversification Example

This script demonstrates various diversification algorithms for the
interview question: "How to diversify search results like 'mmmmmffffff'?"

Usage:
    python examples/image_search_diversity.py
"""

import sys
import os
import time
from typing import List, Dict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from diversification import (
    SearchResult, DiversificationResult,
    InterleavingAlgorithm, ProportionalDistributionAlgorithm,
    MMRAlgorithm, GreedyDiversificationAlgorithm,
    RandomizedDiversificationAlgorithm, SlidingWindowAlgorithm,
    EntropyMaximizationAlgorithm, DiversityEvaluator, DiversityVisualizer
)


def demonstrate_basic_example():
    """Demonstrate basic diversification with the example from the interview."""
    print("🎯 Image Search Diversification - Interview Question Demo")
    print("=" * 70)
    print("Original search result: 'mmmmmffffff' (5 males, 6 females)")
    print()
    
    # Create the search result from the interview question
    original_sequence = "mmmmmffffff"
    search_result = SearchResult.from_string(original_sequence, "image search query")
    
    # Initialize all algorithms
    algorithms = [
        InterleavingAlgorithm(prioritize_minority=True),
        ProportionalDistributionAlgorithm(window_size=4),
        MMRAlgorithm(lambda_param=0.6),
        GreedyDiversificationAlgorithm(diversity_weight=1.0),
        RandomizedDiversificationAlgorithm(randomness=0.3, seed=42),
        SlidingWindowAlgorithm(window_size=4, max_same_gender=2),
        EntropyMaximizationAlgorithm(look_ahead=3)
    ]
    
    print("🔄 Running all diversification algorithms...")
    print()
    
    results = []
    for algorithm in algorithms:
        result = algorithm.diversify(search_result)
        results.append(result)
        
        # Display result
        diversified_sequence = result.get_diversified_sequence()
        print(f"📊 {algorithm.name:<30} → {diversified_sequence}")
        print(f"   ⏱️  Execution time: {result.execution_time*1000:.2f}ms")
        print()
    
    return results


def evaluate_algorithms(results: List[DiversificationResult]):
    """Evaluate and compare all algorithms."""
    print("📈 Algorithm Performance Evaluation")
    print("=" * 50)
    
    evaluator = DiversityEvaluator()
    
    # Evaluate each algorithm
    all_scores = {}
    for result in results:
        scores = evaluator.evaluate_all(result)
        all_scores[result.algorithm_name] = scores
        
        print(f"🎯 {result.algorithm_name}")
        print(f"   Overall Score:     {scores['overall_score']:.3f}")
        print(f"   Alternation:       {scores['alternation_score']:.3f}")
        print(f"   Gender Balance:    {scores['gender_balance_score']:.3f}")
        print(f"   Entropy:           {scores['entropy_score']:.3f}")
        print(f"   Relevance Preserved: {scores['relevance_preservation']:.3f}")
        print()
    
    # Compare algorithms
    comparison = evaluator.compare_algorithms(results)
    
    print("🏆 Algorithm Rankings")
    print("-" * 30)
    print(f"Best Overall: {comparison['best_algorithm']}")
    print()
    
    # Show top 3 for key metrics
    key_metrics = ['overall_score', 'alternation_score', 'gender_balance_score']
    for metric in key_metrics:
        if metric in comparison['rankings']:
            top_3 = comparison['rankings'][metric][:3]
            print(f"{metric.replace('_', ' ').title()}: {' > '.join(top_3)}")
    
    return comparison


def demonstrate_different_scenarios():
    """Test algorithms on different gender distribution scenarios."""
    print("\n🔬 Testing Different Scenarios")
    print("=" * 40)
    
    scenarios = [
        ("Balanced", "mfmfmfmf"),
        ("Male Heavy", "mmmmmmff"),
        ("Female Heavy", "fffffff"),
        ("Mixed", "mmmffffmm"),
        ("Interview Question", "mmmmmffffff"),
        ("Large Balanced", "mfmfmfmfmfmfmfmf"),
        ("Extreme Imbalance", "mmmmmmmmmf")
    ]
    
    # Test with best performing algorithm
    algorithm = ProportionalDistributionAlgorithm(window_size=4)
    evaluator = DiversityEvaluator()
    
    print(f"Testing with: {algorithm.name}")
    print()
    
    for scenario_name, sequence in scenarios:
        search_result = SearchResult.from_string(sequence, f"{scenario_name} scenario")
        result = algorithm.diversify(search_result)
        scores = evaluator.evaluate_all(result)
        
        original = search_result.get_gender_sequence()
        diversified = result.get_diversified_sequence()
        
        print(f"📊 {scenario_name:<15} | {original:<15} → {diversified:<15} | Score: {scores['overall_score']:.3f}")
    
    print()


def create_visualizations(results: List[DiversificationResult]):
    """Create visualizations for the results."""
    print("📊 Creating Visualizations...")
    
    visualizer = DiversityVisualizer()
    
    try:
        # Create comprehensive report
        saved_plots = visualizer.create_comprehensive_report(
            results, 
            save_dir="diversification_report"
        )
        
        print("✅ Visualizations created successfully!")
        print("📁 Saved plots:")
        for plot_name, path in saved_plots.items():
            print(f"   • {plot_name}: {path}")
        
        print("\n💡 Open the PNG files to see detailed algorithm comparisons")
        
    except Exception as e:
        print(f"⚠️  Visualization creation failed: {e}")
        print("   (This might be due to missing matplotlib/seaborn dependencies)")


def benchmark_performance():
    """Benchmark algorithm performance on different data sizes."""
    print("\n⚡ Performance Benchmarking")
    print("=" * 35)
    
    sizes = [10, 50, 100, 500]
    algorithms = [
        InterleavingAlgorithm(),
        ProportionalDistributionAlgorithm(),
        MMRAlgorithm(),
        EntropyMaximizationAlgorithm()
    ]
    
    print(f"{'Algorithm':<25} | {'10 items':<10} | {'50 items':<10} | {'100 items':<11} | {'500 items':<11}")
    print("-" * 80)
    
    for algorithm in algorithms:
        times = []
        
        for size in sizes:
            # Create test sequence (alternating for consistency)
            sequence = ''.join(['m' if i % 2 == 0 else 'f' for i in range(size)])
            search_result = SearchResult.from_string(sequence, f"test_{size}")
            
            # Measure time
            start_time = time.time()
            result = algorithm.diversify(search_result)
            execution_time = time.time() - start_time
            
            times.append(execution_time * 1000)  # Convert to ms
        
        times_str = " | ".join([f"{t:8.2f}ms" for t in times])
        print(f"{algorithm.name:<25} | {times_str}")
    
    print()


def interview_answer_summary():
    """Provide a concise answer to the interview question."""
    print("\n💡 Interview Question Answer Summary")
    print("=" * 45)
    print()
    print("Question: 我们有一个图片搜索程序，返回男女图片，我们希望尽可能排序多样化。")
    print("例如：mmmmmffffff 有多少种推荐算法？")
    print()
    print("Answer: 有多种推荐算法可以实现多样化排序，主要分为以下几类：")
    print()
    print("🎯 核心算法类型：")
    print("1. 交替插入算法 (Interleaving) - 简单有效")
    print("2. 比例分配算法 (Proportional) - 维持全局比例")
    print("3. MMR算法 - 平衡相关性和多样性")
    print("4. 贪心多样化算法 - 最大化差异性")
    print("5. 随机化算法 - 增加不可预测性")
    print("6. 滑动窗口算法 - 局部多样性保证")
    print("7. 熵最大化算法 - 信息论优化")
    print()
    print("🏆 推荐方案：")
    print("• 生产环境：比例分配算法 (最佳平衡)")
    print("• 简单快速：交替插入算法")
    print("• 高质量：MMR算法")
    print("• 可定制：滑动窗口算法")
    print()
    print("📊 对于 'mmmmmffffff' 的最优结果：'MFMFMFMFMFM'")


def main():
    """Main demonstration function."""
    print("🚀 Image Search Diversification - Complete Demo")
    print("=" * 60)
    print("This demo shows how to solve the interview question about")
    print("diversifying image search results with gender balance.")
    print()
    
    # 1. Basic demonstration
    results = demonstrate_basic_example()
    
    # 2. Evaluation
    comparison = evaluate_algorithms(results)
    
    # 3. Different scenarios
    demonstrate_different_scenarios()
    
    # 4. Performance benchmarking
    benchmark_performance()
    
    # 5. Create visualizations
    create_visualizations(results)
    
    # 6. Interview answer summary
    interview_answer_summary()
    
    print("\n🎉 Demo completed! You now have 7 different algorithms")
    print("   to answer the interview question about diversification.")


if __name__ == "__main__":
    main()