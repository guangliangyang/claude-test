#!/usr/bin/env python3
"""
Canva Image Processing Performance Benchmark

This script provides comprehensive performance testing and benchmarking
for the sanitize_image function to ensure it meets Canva's production requirements.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Dict, List, Tuple
import psutil
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import gc

# Add src to path (go up one level from benchmarks/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from image_processing import sanitize_image, benchmark_sanitize_image
from utils.image_utils import create_test_images, create_gradient_image, get_image_stats


class PerformanceBenchmark:
    """Comprehensive performance benchmark for image processing."""
    
    def __init__(self):
        self.results = {}
        self.memory_usage = []
        self.lock = threading.Lock()
    
    def run_full_benchmark(self, save_plots: bool = True) -> Dict:
        """
        Run complete benchmark suite.
        
        Args:
            save_plots: Whether to save performance plots
            
        Returns:
            Dictionary with all benchmark results
        """
        print("🚀 Starting Canva Image Processing Benchmark")
        print("=" * 60)
        
        # 1. Basic performance tests
        print("\n📊 Running basic performance tests...")
        self.results['basic_performance'] = self._benchmark_basic_performance()
        
        # 2. Memory usage tests
        print("\n🧠 Running memory usage tests...")
        self.results['memory_usage'] = self._benchmark_memory_usage()
        
        # 3. Concurrency tests
        print("\n🔄 Running concurrency tests...")
        self.results['concurrency'] = self._benchmark_concurrency()
        
        # 4. Edge case performance
        print("\n⚡ Running edge case tests...")
        self.results['edge_cases'] = self._benchmark_edge_cases()
        
        # 5. Quality vs performance trade-offs
        print("\n🎯 Running interpolation method comparison...")
        self.results['interpolation'] = self._benchmark_interpolation_methods()
        
        # 6. Generate summary report
        print("\n📋 Generating summary report...")
        self.results['summary'] = self._generate_summary()
        
        # 7. Save plots if requested
        if save_plots:
            print("\n📈 Generating performance plots...")
            self._generate_plots()
        
        print("\n✅ Benchmark completed successfully!")
        return self.results
    
    def _benchmark_basic_performance(self) -> Dict:
        """Benchmark basic performance across different image sizes."""
        test_cases = [
            ('VGA', (480, 640, 3)),
            ('HD', (720, 1280, 3)),
            ('Full HD', (1080, 1920, 3)),
            ('4K', (2160, 3840, 3)),
        ]
        
        results = {}
        
        for name, shape in test_cases:
            print(f"  Testing {name} ({shape[1]}x{shape[0]})...")
            
            # Create test image
            test_image = np.random.randint(0, 256, shape, dtype=np.uint8)
            
            # Warm up
            _ = sanitize_image(test_image, max_memory_mb=1000)
            
            # Benchmark
            times = []
            for i in range(10):
                start_time = time.time()
                result = sanitize_image(test_image, max_memory_mb=1000)
                end_time = time.time()
                times.append(end_time - start_time)
                
                # Validate result
                assert result.shape == (224, 224, 3)
                assert 0 <= result.min() <= result.max() <= 1
            
            results[name] = {
                'input_shape': shape,
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'input_size_mb': test_image.nbytes / (1024 * 1024),
                'meets_100ms_requirement': np.mean(times) < 0.1
            }
            
            print(f"    Mean time: {np.mean(times)*1000:.1f}ms ({'✅' if np.mean(times) < 0.1 else '❌'})")
        
        return results
    
    def _benchmark_memory_usage(self) -> Dict:
        """Benchmark memory usage patterns."""
        test_cases = [
            ('Small', (224, 224, 3)),
            ('Medium', (512, 512, 3)),
            ('Large', (1024, 1024, 3)),
            ('XLarge', (2048, 2048, 3)),
        ]
        
        results = {}
        
        for name, shape in test_cases:
            print(f"  Testing memory usage for {name} ({shape[1]}x{shape[0]})...")
            
            # Monitor memory before
            process = psutil.Process()
            memory_before = process.memory_info().rss / (1024 * 1024)  # MB
            
            try:
                # Create test image
                test_image = np.random.randint(0, 256, shape, dtype=np.uint8)
                
                # Process image
                result = sanitize_image(test_image, max_memory_mb=1000)
                
                # Monitor memory after
                memory_after = process.memory_info().rss / (1024 * 1024)  # MB
                memory_used = memory_after - memory_before
                
                results[name] = {
                    'input_shape': shape,
                    'input_size_mb': test_image.nbytes / (1024 * 1024),
                    'memory_used_mb': memory_used,
                    'memory_efficiency': test_image.nbytes / (memory_used * 1024 * 1024) if memory_used > 0 else float('inf'),
                    'meets_500mb_requirement': memory_used < 500
                }
                
                print(f"    Memory used: {memory_used:.1f}MB ({'✅' if memory_used < 500 else '❌'})")
                
                # Clean up
                del test_image, result
                gc.collect()
                
            except Exception as e:
                print(f"    ❌ Failed: {e}")
                results[name] = {
                    'input_shape': shape,
                    'error': str(e)
                }
        
        return results
    
    def _benchmark_concurrency(self) -> Dict:
        """Benchmark thread safety and concurrent processing."""
        print("  Testing thread safety...")
        
        def process_image(image_id: int) -> Tuple[int, float, bool]:
            """Process a single image and return timing info."""
            # Create unique test image
            np.random.seed(image_id)
            test_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            
            start_time = time.time()
            try:
                result = sanitize_image(test_image)
                processing_time = time.time() - start_time
                
                # Validate result
                success = (result.shape == (224, 224, 3) and 
                          0 <= result.min() <= result.max() <= 1)
                
                return image_id, processing_time, success
            except Exception as e:
                return image_id, -1, False
        
        # Test with different thread counts
        thread_counts = [1, 2, 4, 8]
        results = {}
        
        for num_threads in thread_counts:
            print(f"    Testing with {num_threads} threads...")
            
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                # Submit 20 tasks
                futures = [executor.submit(process_image, i) for i in range(20)]
                
                # Collect results
                thread_results = []
                for future in as_completed(futures):
                    image_id, proc_time, success = future.result()
                    thread_results.append({
                        'image_id': image_id,
                        'processing_time': proc_time,
                        'success': success
                    })
            
            total_time = time.time() - start_time
            successful_processes = sum(1 for r in thread_results if r['success'])
            avg_processing_time = np.mean([r['processing_time'] for r in thread_results if r['success']])
            
            results[f'{num_threads}_threads'] = {
                'num_threads': num_threads,
                'total_time': total_time,
                'successful_processes': successful_processes,
                'success_rate': successful_processes / 20,
                'avg_processing_time': avg_processing_time,
                'throughput_imgs_per_sec': successful_processes / total_time
            }
            
            print(f"      Success rate: {successful_processes}/20 ({successful_processes/20*100:.1f}%)")
            print(f"      Throughput: {successful_processes/total_time:.1f} images/sec")
        
        return results
    
    def _benchmark_edge_cases(self) -> Dict:
        """Benchmark performance on edge cases."""
        edge_cases = {
            'single_pixel': np.array([[255]], dtype=np.uint8),
            'very_wide': np.random.randint(0, 256, (100, 2000, 3), dtype=np.uint8),
            'very_tall': np.random.randint(0, 256, (2000, 100, 3), dtype=np.uint8),
            'all_black': np.zeros((500, 500, 3), dtype=np.uint8),
            'all_white': np.full((500, 500, 3), 255, dtype=np.uint8),
            'float32_input': np.random.random((300, 400, 3)).astype(np.float32),
            'uint16_input': np.random.randint(0, 65536, (200, 300), dtype=np.uint16),
        }
        
        results = {}
        
        for case_name, test_image in edge_cases.items():
            print(f"  Testing {case_name}...")
            
            try:
                # Time the processing
                start_time = time.time()
                result = sanitize_image(test_image)
                processing_time = time.time() - start_time
                
                # Validate result
                if test_image.ndim == 2:
                    expected_shape = (224, 224)
                else:
                    expected_shape = (224, 224, test_image.shape[2])
                
                success = (result.shape == expected_shape and 
                          0 <= result.min() <= result.max() <= 1)
                
                results[case_name] = {
                    'input_shape': test_image.shape,
                    'input_dtype': str(test_image.dtype),
                    'processing_time': processing_time,
                    'output_shape': result.shape,
                    'success': success,
                    'fast_enough': processing_time < 0.1
                }
                
                print(f"    Time: {processing_time*1000:.1f}ms ({'✅' if success else '❌'})")
                
            except Exception as e:
                print(f"    ❌ Failed: {e}")
                results[case_name] = {
                    'input_shape': test_image.shape,
                    'input_dtype': str(test_image.dtype),
                    'error': str(e),
                    'success': False
                }
        
        return results
    
    def _benchmark_interpolation_methods(self) -> Dict:
        """Benchmark different interpolation methods."""
        interpolation_methods = ['nearest', 'bilinear', 'bicubic', 'lanczos']
        test_image = create_gradient_image((300, 400), channels=3)
        
        results = {}
        
        for method in interpolation_methods:
            print(f"  Testing {method} interpolation...")
            
            times = []
            for _ in range(10):
                start_time = time.time()
                result = sanitize_image(test_image, interpolation=method)
                times.append(time.time() - start_time)
            
            results[method] = {
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times)
            }
            
            print(f"    Mean time: {np.mean(times)*1000:.1f}ms")
        
        return results
    
    def _generate_summary(self) -> Dict:
        """Generate overall performance summary."""
        summary = {
            'total_tests_run': 0,
            'tests_passed': 0,
            'performance_requirements_met': {},
            'recommendations': []
        }
        
        # Count tests and passes
        for category, results in self.results.items():
            if category == 'summary':
                continue
            
            if isinstance(results, dict):
                for test_name, test_result in results.items():
                    summary['total_tests_run'] += 1
                    if isinstance(test_result, dict) and test_result.get('success', True):
                        summary['tests_passed'] += 1
        
        # Check performance requirements
        basic_perf = self.results.get('basic_performance', {})
        memory_usage = self.results.get('memory_usage', {})
        
        # 100ms requirement for VGA
        vga_time = basic_perf.get('VGA', {}).get('mean_time', float('inf'))
        summary['performance_requirements_met']['100ms_vga'] = vga_time < 0.1
        
        # 500MB memory requirement for 4K
        memory_4k = memory_usage.get('XLarge', {}).get('memory_used_mb', float('inf'))
        summary['performance_requirements_met']['500mb_memory'] = memory_4k < 500
        
        # Thread safety
        concurrency = self.results.get('concurrency', {})
        thread_success = all(result.get('success_rate', 0) >= 0.9 
                           for result in concurrency.values())
        summary['performance_requirements_met']['thread_safety'] = thread_success
        
        # Generate recommendations
        if vga_time >= 0.1:
            summary['recommendations'].append("Optimize processing for VGA images to meet 100ms requirement")
        
        if memory_4k >= 500:
            summary['recommendations'].append("Reduce memory usage for large images")
        
        if not thread_success:
            summary['recommendations'].append("Improve thread safety and concurrent processing")
        
        # Success rate
        if summary['total_tests_run'] > 0:
            summary['success_rate'] = summary['tests_passed'] / summary['total_tests_run']
        else:
            summary['success_rate'] = 0
        
        return summary
    
    def _generate_plots(self):
        """Generate performance visualization plots."""
        try:
            # Set style
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Canva Image Processing Performance Benchmark', fontsize=16, fontweight='bold')
            
            # Plot 1: Processing time vs image size
            basic_perf = self.results.get('basic_performance', {})
            if basic_perf:
                sizes = []
                times = []
                labels = []
                
                for name, data in basic_perf.items():
                    if 'mean_time' in data:
                        shape = data['input_shape']
                        size_mb = data['input_size_mb']
                        time_ms = data['mean_time'] * 1000
                        
                        sizes.append(size_mb)
                        times.append(time_ms)
                        labels.append(name)
                
                axes[0, 0].scatter(sizes, times, s=100, alpha=0.7)
                for i, label in enumerate(labels):
                    axes[0, 0].annotate(label, (sizes[i], times[i]), 
                                       xytext=(5, 5), textcoords='offset points')
                
                axes[0, 0].axhline(y=100, color='r', linestyle='--', label='100ms target')
                axes[0, 0].set_xlabel('Input Size (MB)')
                axes[0, 0].set_ylabel('Processing Time (ms)')
                axes[0, 0].set_title('Processing Time vs Image Size')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Memory usage
            memory_usage = self.results.get('memory_usage', {})
            if memory_usage:
                names = []
                memory_used = []
                
                for name, data in memory_usage.items():
                    if 'memory_used_mb' in data:
                        names.append(name)
                        memory_used.append(data['memory_used_mb'])
                
                bars = axes[0, 1].bar(names, memory_used, alpha=0.7)
                axes[0, 1].axhline(y=500, color='r', linestyle='--', label='500MB limit')
                axes[0, 1].set_ylabel('Memory Usage (MB)')
                axes[0, 1].set_title('Memory Usage by Image Size')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3, axis='y')
                
                # Color bars based on whether they meet requirement
                for bar, memory in zip(bars, memory_used):
                    if memory > 500:
                        bar.set_color('red')
                    else:
                        bar.set_color('green')
            
            # Plot 3: Concurrency performance
            concurrency = self.results.get('concurrency', {})
            if concurrency:
                threads = []
                throughput = []
                
                for name, data in concurrency.items():
                    if 'num_threads' in data:
                        threads.append(data['num_threads'])
                        throughput.append(data['throughput_imgs_per_sec'])
                
                axes[1, 0].plot(threads, throughput, 'o-', linewidth=2, markersize=8)
                axes[1, 0].set_xlabel('Number of Threads')
                axes[1, 0].set_ylabel('Throughput (images/sec)')
                axes[1, 0].set_title('Concurrent Processing Performance')
                axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Interpolation method comparison
            interpolation = self.results.get('interpolation', {})
            if interpolation:
                methods = list(interpolation.keys())
                times = [data['mean_time'] * 1000 for data in interpolation.values()]
                
                bars = axes[1, 1].bar(methods, times, alpha=0.7)
                axes[1, 1].set_ylabel('Processing Time (ms)')
                axes[1, 1].set_title('Interpolation Method Performance')
                axes[1, 1].grid(True, alpha=0.3, axis='y')
                
                # Rotate x-axis labels
                axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # Save plot
            output_file = 'image_processing_benchmark.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"  📈 Performance plots saved to {output_file}")
            
            # Show plot if in interactive environment
            if hasattr(plt, 'show'):
                plt.show()
                
        except Exception as e:
            print(f"  ⚠️  Failed to generate plots: {e}")
    
    def print_summary_report(self):
        """Print a formatted summary report."""
        summary = self.results.get('summary', {})
        
        print("\n" + "="*60)
        print("📋 CANVA IMAGE PROCESSING BENCHMARK SUMMARY")
        print("="*60)
        
        print(f"\n📊 Overall Results:")
        print(f"  Tests Run: {summary.get('total_tests_run', 0)}")
        print(f"  Tests Passed: {summary.get('tests_passed', 0)}")
        print(f"  Success Rate: {summary.get('success_rate', 0)*100:.1f}%")
        
        print(f"\n🎯 Performance Requirements:")
        requirements = summary.get('performance_requirements_met', {})
        for req, met in requirements.items():
            status = "✅" if met else "❌"
            print(f"  {req}: {status}")
        
        recommendations = summary.get('recommendations', [])
        if recommendations:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        print("\n" + "="*60)


def main():
    """Run the complete benchmark suite."""
    benchmark = PerformanceBenchmark()
    
    try:
        # Run benchmark
        results = benchmark.run_full_benchmark(save_plots=True)
        
        # Print summary
        benchmark.print_summary_report()
        
        # Save detailed results
        output_file = 'benchmark_results.txt'
        with open(output_file, 'w') as f:
            f.write("Canva Image Processing Benchmark Results\n")
            f.write("="*50 + "\n\n")
            
            for category, data in results.items():
                f.write(f"{category.upper()}:\n")
                f.write(str(data))
                f.write("\n\n")
        
        print(f"\n📄 Detailed results saved to {output_file}")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())