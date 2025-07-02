"""
Visualization tools for diversification algorithm results.

This module provides comprehensive visualization capabilities to analyze
and compare different diversification algorithms.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

from .models import DiversificationResult, Gender
from .evaluators import DiversityEvaluator


class DiversityVisualizer:
    """Visualization toolkit for diversification algorithm analysis."""
    
    def __init__(self, style: str = 'whitegrid', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer.
        
        Args:
            style: Seaborn style for plots
            figsize: Default figure size
        """
        self.style = style
        self.figsize = figsize
        self.evaluator = DiversityEvaluator()
        
        # Set style
        sns.set_style(style)
        plt.rcParams['figure.figsize'] = figsize
        
        # Color palette for genders
        self.gender_colors = {
            Gender.MALE: '#3498db',      # Blue
            Gender.FEMALE: '#e74c3c',    # Red
            Gender.UNKNOWN: '#95a5a6'    # Gray
        }
    
    def plot_sequence_comparison(
        self, 
        results: List[DiversificationResult], 
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot gender sequences for multiple algorithms side by side.
        
        Args:
            results: List of diversification results
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        n_algorithms = len(results)
        fig, axes = plt.subplots(n_algorithms, 1, figsize=(14, 2 * n_algorithms))
        
        if n_algorithms == 1:
            axes = [axes]
        
        for i, result in enumerate(results):
            ax = axes[i]
            sequence = [item.gender for item in result.diversified_items]
            
            # Create color sequence
            colors = [self.gender_colors.get(gender, '#95a5a6') for gender in sequence]
            
            # Plot as horizontal bar
            positions = list(range(len(sequence)))
            bars = ax.barh(0, [1] * len(sequence), left=positions, height=0.8, color=colors)
            
            # Customize axis
            ax.set_xlim(-0.5, len(sequence) - 0.5)
            ax.set_ylim(-0.5, 0.5)
            ax.set_xlabel('Position in Result List')
            ax.set_title(f'{result.algorithm_name}')
            ax.set_yticks([])
            
            # Add gender labels
            gender_str = ''.join('M' if g == Gender.MALE else 'F' if g == Gender.FEMALE else '?' for g in sequence)
            ax.text(len(sequence)/2, -0.3, gender_str, ha='center', va='top', 
                   fontfamily='monospace', fontsize=10)
        
        # Add legend
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor=self.gender_colors[Gender.MALE], label='Male'),
            plt.Rectangle((0,0),1,1, facecolor=self.gender_colors[Gender.FEMALE], label='Female')
        ]
        fig.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_metrics_comparison(
        self, 
        results: List[DiversificationResult],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot radar chart comparing algorithms across all metrics.
        
        Args:
            results: List of diversification results
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        # Calculate metrics for all results
        metrics_data = []
        for result in results:
            scores = self.evaluator.evaluate_all(result)
            metrics_data.append({
                'Algorithm': result.algorithm_name,
                **scores
            })
        
        df = pd.DataFrame(metrics_data)
        
        # Select key metrics for radar chart
        key_metrics = [
            'alternation_score', 'gender_balance_score', 'entropy_score',
            'prefix_diversity', 'consecutive_penalty', 'relevance_preservation'
        ]
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(key_metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(results)))
        
        for i, result in enumerate(results):
            scores = self.evaluator.evaluate_all(result)
            values = [scores.get(metric, 0) for metric in key_metrics]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=result.algorithm_name, color=colors[i])
            ax.fill(angles, values, alpha=0.1, color=colors[i])
        
        # Customize the chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([metric.replace('_', ' ').title() for metric in key_metrics])
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.grid(True)
        
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        plt.title('Algorithm Performance Comparison\n(Higher values are better)', 
                 size=16, weight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_metric_heatmap(
        self, 
        results: List[DiversificationResult],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot heatmap of all metrics for all algorithms.
        
        Args:
            results: List of diversification results
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        # Calculate metrics for all results
        metrics_data = []
        for result in results:
            scores = self.evaluator.evaluate_all(result)
            metrics_data.append(scores)
        
        # Create DataFrame
        df = pd.DataFrame(metrics_data, index=[r.algorithm_name for r in results])
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sns.heatmap(
            df, 
            annot=True, 
            cmap='RdYlGn', 
            center=0.5,
            vmin=0, 
            vmax=1,
            fmt='.3f',
            cbar_kws={'label': 'Score (0-1)'},
            ax=ax
        )
        
        plt.title('Algorithm Performance Heatmap\n(Higher values are better)', 
                 size=16, weight='bold')
        plt.xlabel('Metrics')
        plt.ylabel('Algorithms')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_position_analysis(
        self, 
        result: DiversificationResult,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Analyze gender distribution by position in the result.
        
        Args:
            result: Single diversification result to analyze
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        sequence = [item.gender for item in result.diversified_items]
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # 1. Position-by-position gender plot
        colors = [self.gender_colors.get(gender, '#95a5a6') for gender in sequence]
        positions = list(range(len(sequence)))
        
        ax1.bar(positions, [1] * len(sequence), color=colors, alpha=0.8)
        ax1.set_xlabel('Position')
        ax1.set_ylabel('Gender')
        ax1.set_title(f'Gender Sequence - {result.algorithm_name}')
        ax1.set_ylim(0, 1.2)
        
        # Add text labels
        for i, gender in enumerate(sequence):
            label = 'M' if gender == Gender.MALE else 'F'
            ax1.text(i, 0.5, label, ha='center', va='center', fontweight='bold', color='white')
        
        # 2. Cumulative gender balance
        male_cumsum = np.cumsum([1 if g == Gender.MALE else 0 for g in sequence])
        female_cumsum = np.cumsum([1 if g == Gender.FEMALE else 0 for g in sequence])
        total_cumsum = np.arange(1, len(sequence) + 1)
        
        male_ratio = male_cumsum / total_cumsum
        female_ratio = female_cumsum / total_cumsum
        
        ax2.plot(positions, male_ratio, label='Male Ratio', color=self.gender_colors[Gender.MALE], linewidth=2)
        ax2.plot(positions, female_ratio, label='Female Ratio', color=self.gender_colors[Gender.FEMALE], linewidth=2)
        ax2.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Perfect Balance')
        ax2.set_xlabel('Position')
        ax2.set_ylabel('Cumulative Ratio')
        ax2.set_title('Cumulative Gender Balance')
        ax2.legend()
        ax2.set_ylim(0, 1)
        
        # 3. Local diversity (sliding window)
        window_size = 5
        local_diversity = []
        
        for i in range(len(sequence) - window_size + 1):
            window = sequence[i:i + window_size]
            unique_genders = len(set(window))
            diversity = unique_genders / min(window_size, 2)  # Normalize by max possible
            local_diversity.append(diversity)
        
        window_positions = list(range(window_size//2, len(sequence) - window_size//2))
        ax3.plot(window_positions, local_diversity, marker='o', linewidth=2, markersize=4)
        ax3.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Maximum Diversity')
        ax3.set_xlabel('Window Center Position')
        ax3.set_ylabel('Local Diversity')
        ax3.set_title(f'Local Diversity (Window Size: {window_size})')
        ax3.legend()
        ax3.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_algorithm_ranking(
        self, 
        results: List[DiversificationResult],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot ranking of algorithms across different metrics.
        
        Args:
            results: List of diversification results
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        comparison = self.evaluator.compare_algorithms(results)
        
        # Prepare data for plotting
        algorithms = comparison['algorithms']
        metrics = [m for m in self.evaluator.metrics if m != 'overall_score']
        
        ranking_matrix = np.zeros((len(algorithms), len(metrics)))
        
        for j, metric in enumerate(metrics):
            if metric in comparison['rankings']:
                ranking = comparison['rankings'][metric]
                for i, algorithm in enumerate(algorithms):
                    rank = ranking.index(algorithm) + 1  # 1-based ranking
                    ranking_matrix[i, j] = rank
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create heatmap with inverted colormap (lower rank = better = darker)
        im = ax.imshow(ranking_matrix, cmap='RdYlGn_r', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_yticks(np.arange(len(algorithms)))
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax.set_yticklabels(algorithms)
        
        # Add text annotations
        for i in range(len(algorithms)):
            for j in range(len(metrics)):
                rank = int(ranking_matrix[i, j])
                if rank > 0:
                    text = ax.text(j, i, str(rank), ha="center", va="center",
                                 color="white" if rank > len(algorithms)/2 else "black",
                                 fontweight='bold')
        
        # Customize plot
        ax.set_title('Algorithm Rankings by Metric\n(Lower numbers = better performance)', 
                    size=16, weight='bold')
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Algorithms')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Rank (1=best)', rotation=270, labelpad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_comprehensive_report(
        self, 
        results: List[DiversificationResult],
        save_dir: str = "diversification_report"
    ) -> Dict[str, str]:
        """
        Create a comprehensive visual report of all algorithms.
        
        Args:
            results: List of diversification results
            save_dir: Directory to save all plots
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        saved_plots = {}
        
        # 1. Sequence comparison
        fig1 = self.plot_sequence_comparison(results)
        path1 = os.path.join(save_dir, "sequence_comparison.png")
        fig1.savefig(path1, dpi=300, bbox_inches='tight')
        saved_plots['sequence_comparison'] = path1
        plt.close(fig1)
        
        # 2. Metrics comparison (radar)
        fig2 = self.plot_metrics_comparison(results)
        path2 = os.path.join(save_dir, "metrics_radar.png")
        fig2.savefig(path2, dpi=300, bbox_inches='tight')
        saved_plots['metrics_radar'] = path2
        plt.close(fig2)
        
        # 3. Metrics heatmap
        fig3 = self.plot_metric_heatmap(results)
        path3 = os.path.join(save_dir, "metrics_heatmap.png")
        fig3.savefig(path3, dpi=300, bbox_inches='tight')
        saved_plots['metrics_heatmap'] = path3
        plt.close(fig3)
        
        # 4. Algorithm ranking
        fig4 = self.plot_algorithm_ranking(results)
        path4 = os.path.join(save_dir, "algorithm_ranking.png")
        fig4.savefig(path4, dpi=300, bbox_inches='tight')
        saved_plots['algorithm_ranking'] = path4
        plt.close(fig4)
        
        # 5. Position analysis for best algorithm
        comparison = self.evaluator.compare_algorithms(results)
        best_algo = comparison['best_algorithm']
        best_result = next(r for r in results if r.algorithm_name == best_algo)
        
        fig5 = self.plot_position_analysis(best_result)
        path5 = os.path.join(save_dir, f"position_analysis_{best_algo}.png")
        fig5.savefig(path5, dpi=300, bbox_inches='tight')
        saved_plots['position_analysis'] = path5
        plt.close(fig5)
        
        return saved_plots