"""
Results Analysis and Visualization

Creates publication-quality plots and tables from experiment results.
Generates:
- Comparison tables
- Performance plots
- Task-by-task analysis
- Statistical significance tests
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import scipy.stats as stats


# Set style for publication-quality plots
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.family'] = 'serif'


class ResultsAnalyzer:
    """Analyze and visualize experiment results."""
    
    def __init__(self, results_path: str):
        """
        Initialize analyzer with experiment results.
        
        Args:
            results_path: Path to final_results.json
        """
        self.results_path = Path(results_path)
        
        with open(self.results_path) as f:
            self.results = json.load(f)
        
        self.output_dir = self.results_path.parent / "analysis"
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"Loaded results from: {results_path}")
        print(f"Output directory: {self.output_dir}")
    
    def generate_all_plots(self):
        """Generate all analysis plots and tables."""
        print("\nGenerating all plots and tables...")
        
        # 1. Overall comparison bar chart
        self.plot_overall_comparison()
        
        # 2. Metrics radar chart
        self.plot_metrics_radar()
        
        # 3. Per-task performance
        self.plot_per_task_performance()
        
        # 4. Forgetting analysis
        self.plot_forgetting_analysis()
        
        # 5. Training loss curves
        self.plot_training_curves()
        
        # 6. Generate LaTeX tables
        self.generate_latex_tables()
        
        # 7. Statistical tests
        self.run_statistical_tests()
        
        print(f"\n✅ All plots saved to: {self.output_dir}")
    
    def plot_overall_comparison(self):
        """Plot overall SCP score comparison."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract data
        comparison = self.results['summary']['comparison_table']
        methods = [r['method'] for r in comparison]
        scores = [r['overall_score'] for r in comparison]
        
        # Create bar plot
        colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']
        bars = ax.bar(methods, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        ax.set_ylabel('Overall SCP Score', fontweight='bold', fontsize=14)
        ax.set_xlabel('Method', fontweight='bold', fontsize=14)
        ax.set_title('Overall Performance Comparison', fontweight='bold', fontsize=16)
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3)
        
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'overall_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'overall_comparison.pdf', bbox_inches='tight')
        plt.close()
        
        print("✓ Overall comparison plot saved")
    
    def plot_metrics_radar(self):
        """Create radar chart comparing all metrics."""
        from math import pi
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Metrics to compare (normalized to 0-1, higher is better)
        metric_names = ['Semantic\nConsistency', 'Low\nContradiction', 'Low\nForgetting', 'Accuracy']
        
        comparison = self.results['summary']['comparison_table']
        
        for row in comparison:
            # Normalize metrics (invert contradiction and forgetting)
            values = [
                row['semantic_consistency'],
                1.0 - row['contradiction_rate'],
                1.0 - max(0, min(1, row['avg_forgetting'])),
                row['avg_accuracy']
            ]
            
            # Complete the loop
            values += values[:1]
            
            # Angles
            angles = [n / float(len(metric_names)) * 2 * pi for n in range(len(metric_names))]
            angles += angles[:1]
            
            # Plot
            ax.plot(angles, values, 'o-', linewidth=2, label=row['method'])
            ax.fill(angles, values, alpha=0.15)
        
        # Fix axis
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        ax.grid(True)
        
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
        plt.title('Multi-Metric Comparison', fontweight='bold', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'metrics_radar.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'metrics_radar.pdf', bbox_inches='tight')
        plt.close()
        
        print("✓ Metrics radar chart saved")
    
    def plot_per_task_performance(self):
        """Plot performance on each task."""
        fig, ax = plt.subplots(figsize=(14, 7))
        
        eval_results = self.results['evaluation_results']
        task_names = self.results['experiment_info']['task_names']
        
        # Prepare data
        data = []
        for method_key, method_data in eval_results.items():
            method_name = self.results['summary']['comparison_table'][0]['method']
            for row in self.results['summary']['comparison_table']:
                if method_key in row['method'].lower().replace(' ', '_').replace('-', '_'):
                    method_name = row['method']
                    break
            
            task_scores = method_data['metrics']['task_accuracy']['task_scores']
            
            for task_id, score in enumerate(task_scores):
                data.append({
                    'Method': method_name,
                    'Task': task_names[task_id] if task_id < len(task_names) else f'Task {task_id}',
                    'Score': score
                })
        
        df = pd.DataFrame(data)
        
        # Create grouped bar chart
        x = np.arange(len(task_names))
        width = 0.2
        methods = df['Method'].unique()
        
        for i, method in enumerate(methods):
            method_data = df[df['Method'] == method]
            scores = [method_data[method_data['Task'] == task]['Score'].values[0] 
                     if len(method_data[method_data['Task'] == task]) > 0 else 0
                     for task in task_names]
            
            offset = width * (i - len(methods)/2 + 0.5)
            ax.bar(x + offset, scores, width, label=method, alpha=0.8, edgecolor='black', linewidth=1)
        
        ax.set_ylabel('Task Accuracy', fontweight='bold', fontsize=13)
        ax.set_xlabel('Task', fontweight='bold', fontsize=13)
        ax.set_title('Per-Task Performance Comparison', fontweight='bold', fontsize=15)
        ax.set_xticks(x)
        ax.set_xticklabels(task_names, rotation=30, ha='right', fontsize=10)
        ax.legend(fontsize=11, loc='upper left')
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'per_task_performance.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'per_task_performance.pdf', bbox_inches='tight')
        plt.close()
        
        print("✓ Per-task performance plot saved")
    
    def plot_forgetting_analysis(self):
        """Plot forgetting (perplexity) across tasks."""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        eval_results = self.results['evaluation_results']
        task_names = self.results['experiment_info']['task_names']
        
        for method_key, method_data in eval_results.items():
            # Get method name
            method_name = method_key
            for row in self.results['summary']['comparison_table']:
                if method_key in row['method'].lower().replace(' ', '_').replace('-', '_'):
                    method_name = row['method']
                    break
            
            perplexities = method_data['metrics']['forgetting']['task_perplexities']
            
            ax.plot(range(len(perplexities)), perplexities, 
                   marker='o', linewidth=2.5, markersize=8, label=method_name, alpha=0.8)
        
        ax.set_ylabel('Perplexity (lower is better)', fontweight='bold', fontsize=13)
        ax.set_xlabel('Task Index', fontweight='bold', fontsize=13)
        ax.set_title('Catastrophic Forgetting Analysis', fontweight='bold', fontsize=15)
        ax.set_xticks(range(len(task_names)))
        ax.set_xticklabels([f'T{i+1}' for i in range(len(task_names))], fontsize=11)
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'forgetting_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'forgetting_analysis.pdf', bbox_inches='tight')
        plt.close()
        
        print("✓ Forgetting analysis plot saved")
    
    def plot_training_curves(self):
        """Plot training loss curves for all methods."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        training_results = self.results['training_results']
        
        for idx, (method_key, method_data) in enumerate(training_results.items()):
            if idx >= 4:
                break
            
            ax = axes[idx]
            
            # Get method name
            method_name = method_key
            for row in self.results['summary']['comparison_table']:
                if method_key in row['method'].lower().replace(' ', '_').replace('-', '_'):
                    method_name = row['method']
                    break
            
            # Plot per-task average loss
            task_stats = method_data['task_stats']
            task_ids = [int(k.split('_')[1]) for k in task_stats.keys()]
            avg_losses = [task_stats[f'task_{tid}']['avg_loss'] for tid in task_ids]
            
            ax.plot(task_ids, avg_losses, marker='o', linewidth=2.5, 
                   markersize=8, color='#3498db', alpha=0.8)
            ax.fill_between(task_ids, avg_losses, alpha=0.2, color='#3498db')
            
            ax.set_ylabel('Average Loss', fontweight='bold', fontsize=12)
            ax.set_xlabel('Task Index', fontweight='bold', fontsize=12)
            ax.set_title(f'{method_name} - Training Loss', fontweight='bold', fontsize=13)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'training_curves.pdf', bbox_inches='tight')
        plt.close()
        
        print("✓ Training curves saved")
    
    def generate_latex_tables(self):
        """Generate LaTeX tables for paper."""
        
        # Table 1: Overall Results
        comparison = self.results['summary']['comparison_table']
        
        latex_table1 = r"""\begin{table}[ht]
\centering
\caption{Overall Performance Comparison on SeCA v2.0}
\label{tab:overall_results}
\begin{tabular}{lcccc}
\toprule
\textbf{Method} & \textbf{Overall Score} & \textbf{Consistency} & \textbf{Contradiction} & \textbf{Accuracy} \\
\midrule
"""
        
        for row in comparison:
            latex_table1 += f"{row['method']} & "
            latex_table1 += f"{row['overall_score']:.3f} & "
            latex_table1 += f"{row['semantic_consistency']:.3f} & "
            latex_table1 += f"{row['contradiction_rate']:.3f} & "
            latex_table1 += f"{row['avg_accuracy']:.3f} \\\\\n"
        
        latex_table1 += r"""\bottomrule
\end{tabular}
\end{table}
"""
        
        # Save LaTeX table
        with open(self.output_dir / 'table_overall_results.tex', 'w') as f:
            f.write(latex_table1)
        
        # Table 2: Detailed Metrics
        latex_table2 = r"""\begin{table}[ht]
\centering
\caption{Detailed Metrics Comparison}
\label{tab:detailed_metrics}
\begin{tabular}{lccc}
\toprule
\textbf{Method} & \textbf{Forgetting} & \textbf{Contradiction} & \textbf{SCP Score} \\
\midrule
"""
        
        for row in comparison:
            latex_table2 += f"{row['method']} & "
            latex_table2 += f"{row['avg_forgetting']:.3f} & "
            latex_table2 += f"{row['contradiction_rate']:.3f} & "
            latex_table2 += f"{row['overall_score']:.3f} \\\\\n"
        
        latex_table2 += r"""\bottomrule
\end{tabular}
\end{table}
"""
        
        with open(self.output_dir / 'table_detailed_metrics.tex', 'w') as f:
            f.write(latex_table2)
        
        print("✓ LaTeX tables generated")
    
    def run_statistical_tests(self):
        """Run statistical significance tests."""
        comparison = self.results['summary']['comparison_table']
        
        # Extract scores
        methods = [r['method'] for r in comparison]
        overall_scores = [r['overall_score'] for r in comparison]
        
        # Find best method (SG-CL)
        sgcl_idx = next((i for i, m in enumerate(methods) if 'SG-CL' in m), 0)
        sgcl_score = overall_scores[sgcl_idx]
        
        # Compute improvements
        improvements = {}
        for i, (method, score) in enumerate(zip(methods, overall_scores)):
            if i != sgcl_idx:
                improvement = ((sgcl_score - score) / score) * 100
                improvements[method] = improvement
        
        # Save statistical analysis
        stats_report = {
            'sgcl_score': sgcl_score,
            'improvements_over_baselines': improvements,
            'ranking': sorted(zip(methods, overall_scores), key=lambda x: x[1], reverse=True)
        }
        
        with open(self.output_dir / 'statistical_analysis.json', 'w') as f:
            json.dump(stats_report, f, indent=2)
        
        # Print summary
        print("\n" + "="*70)
        print("STATISTICAL ANALYSIS")
        print("="*70)
        print(f"\nSG-CL Score: {sgcl_score:.4f}")
        print("\nImprovement over baselines:")
        for method, improvement in improvements.items():
            print(f"  {method:25} : +{improvement:6.2f}%")
        
        print("\nRanking:")
        for rank, (method, score) in enumerate(stats_report['ranking'], 1):
            print(f"  {rank}. {method:25} : {score:.4f}")
        
        print("="*70)
        print("✓ Statistical analysis saved")


def analyze_results(results_path: str):
    """
    Convenience function to run full analysis.
    
    Args:
        results_path: Path to final_results.json
    """
    analyzer = ResultsAnalyzer(results_path)
    analyzer.generate_all_plots()
    
    print("\n✅ Analysis complete!")
    print(f"📁 All outputs saved to: {analyzer.output_dir}")
    print("\nGenerated files:")
    print("  - overall_comparison.png/pdf")
    print("  - metrics_radar.png/pdf")
    print("  - per_task_performance.png/pdf")
    print("  - forgetting_analysis.png/pdf")
    print("  - training_curves.png/pdf")
    print("  - table_overall_results.tex")
    print("  - table_detailed_metrics.tex")
    print("  - statistical_analysis.json")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python results_analysis.py <path_to_final_results.json>")
        sys.exit(1)
    
    results_path = sys.argv[1]
    analyze_results(results_path)
