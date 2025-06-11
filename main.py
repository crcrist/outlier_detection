import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class StoreOutlierDetector:
    """
    Simple and efficient outlier detection for test stores based on metric values.
    
    Uses statistical thresholds (mean ± n*std) to identify stores that consistently
    perform outside normal ranges across different metrics.
    """
    
    def __init__(self, std_threshold: float = 2.5, min_outlier_metrics: int = 1):
        """
        Initialize the outlier detector.
        
        Args:
            std_threshold: Number of standard deviations from mean to consider outlier
            min_outlier_metrics: Minimum number of metrics a store must be outlier in
        """
        self.std_threshold = std_threshold
        self.min_outlier_metrics = min_outlier_metrics
        self.outlier_results = None
    
    def detect_outliers(self, store_metrics: pd.DataFrame) -> pd.DataFrame:
        """
        Detect outlier stores based on metric values.
        
        Args:
            store_metrics: DataFrame with columns [store_nbr, wm_yr_wk_nbr, metric, 
                          store_type, metric_value]
        
        Returns:
            DataFrame with outlier stores and details about which metrics triggered detection
        """
        # Filter for test stores only
        test_stores = store_metrics[store_metrics['store_type'] == 'test'].copy()
        
        if test_stores.empty:
            print("Warning: No test stores found in the data")
            return pd.DataFrame()
        
        # Calculate aggregate metrics per store (mean across all weeks)
        store_aggregates = test_stores.groupby(['store_nbr', 'metric'])['metric_value'].agg([
            'mean', 'std', 'count'
        ]).reset_index()
        
        # Calculate outlier thresholds for each metric
        metric_stats = store_aggregates.groupby('metric')['mean'].agg([
            'mean', 'std'
        ]).reset_index()
        metric_stats.columns = ['metric', 'global_mean', 'global_std']
        
        # Merge stats back to store data
        store_with_stats = store_aggregates.merge(metric_stats, on='metric')
        
        # Calculate z-scores and identify outliers
        store_with_stats['z_score'] = (
            (store_with_stats['mean'] - store_with_stats['global_mean']) / 
            store_with_stats['global_std']
        )
        store_with_stats['is_outlier'] = (
            np.abs(store_with_stats['z_score']) > self.std_threshold
        )
        store_with_stats['outlier_direction'] = np.where(
            store_with_stats['z_score'] > self.std_threshold, 'high',
            np.where(store_with_stats['z_score'] < -self.std_threshold, 'low', 'normal')
        )
        
        # Find stores that are outliers in multiple metrics
        outlier_summary = store_with_stats[store_with_stats['is_outlier']].groupby('store_nbr').agg({
            'metric': lambda x: list(x),
            'z_score': lambda x: list(x),
            'outlier_direction': lambda x: list(x),
            'is_outlier': 'count'
        }).reset_index()
        outlier_summary.columns = ['store_nbr', 'outlier_metrics', 'z_scores', 'directions', 'outlier_count']
        
        # Filter by minimum outlier metrics threshold
        final_outliers = outlier_summary[
            outlier_summary['outlier_count'] >= self.min_outlier_metrics
        ].copy()
        
        # Add summary statistics
        if not final_outliers.empty:
            final_outliers['max_abs_z_score'] = final_outliers['z_scores'].apply(
                lambda x: max([abs(z) for z in x])
            )
            final_outliers = final_outliers.sort_values('max_abs_z_score', ascending=False)
        
        self.outlier_results = final_outliers
        return final_outliers
    
    def get_detailed_report(self, store_metrics: pd.DataFrame, store_nbr: int) -> Dict:
        """
        Get detailed report for a specific store.
        
        Args:
            store_metrics: Original DataFrame
            store_nbr: Store number to analyze
        
        Returns:
            Dictionary with detailed store analysis
        """
        store_data = store_metrics[
            (store_metrics['store_nbr'] == store_nbr) & 
            (store_metrics['store_type'] == 'test')
        ].copy()
        
        if store_data.empty:
            return {'error': f'No data found for test store {store_nbr}'}
        
        # Calculate store metrics
        store_summary = store_data.groupby('metric')['metric_value'].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).round(2)
        
        # Calculate population statistics for comparison
        test_stores = store_metrics[store_metrics['store_type'] == 'test']
        population_stats = test_stores.groupby('metric')['metric_value'].agg([
            'mean', 'std'
        ]).round(2)
        population_stats.columns = ['pop_mean', 'pop_std']
        
        # Merge and calculate z-scores
        comparison = store_summary.merge(population_stats, left_index=True, right_index=True)
        comparison['z_score'] = ((comparison['mean'] - comparison['pop_mean']) / 
                                comparison['pop_std']).round(2)
        
        return {
            'store_nbr': store_nbr,
            'metrics_summary': comparison.to_dict('index'),
            'total_weeks': store_data['wm_yr_wk_nbr'].nunique(),
            'metrics_tracked': store_data['metric'].nunique()
        }
    
    def print_summary(self):
        """Print a summary of outlier detection results."""
        if self.outlier_results is None:
            print("No outlier detection has been run yet.")
            return
        
        if self.outlier_results.empty:
            print("No outlier stores detected with current thresholds.")
            return
        
        print(f"\n🚨 OUTLIER STORES DETECTED: {len(self.outlier_results)} stores")
        print(f"Threshold: {self.std_threshold} standard deviations")
        print(f"Minimum outlier metrics: {self.min_outlier_metrics}")
        print("-" * 60)
        
        for _, row in self.outlier_results.iterrows():
            print(f"Store {row['store_nbr']}:")
            print(f"  • Outlier in {row['outlier_count']} metric(s): {', '.join(row['outlier_metrics'])}")
            print(f"  • Max Z-score: {row['max_abs_z_score']:.2f}")
            print(f"  • Directions: {', '.join(row['directions'])}")
            print()
    
    def create_scatter_plot_matrix(self, store_metrics: pd.DataFrame, 
                                  figsize: Tuple[int, int] = (12, 10),
                                  save_path: Optional[str] = None) -> None:
        """
        Create a multi-metric scatter plot matrix showing outlier stores.
        
        Args:
            store_metrics: Original DataFrame
            figsize: Figure size (width, height)
            save_path: Optional path to save the plot
        """
        if self.outlier_results is None:
            print("Please run detect_outliers() first.")
            return
        
        # Prepare data for plotting
        test_stores = store_metrics[store_metrics['store_type'] == 'test'].copy()
        
        # Calculate store averages for each metric
        store_averages = test_stores.groupby(['store_nbr', 'metric'])['metric_value'].mean().reset_index()
        
        # Pivot to get metrics as columns
        plot_data = store_averages.pivot(index='store_nbr', columns='metric', values='metric_value')
        
        # Add outlier status
        outlier_stores = set(self.outlier_results['store_nbr'].tolist()) if not self.outlier_results.empty else set()
        plot_data['is_outlier'] = plot_data.index.isin(outlier_stores)
        plot_data['outlier_type'] = plot_data['is_outlier'].map({True: 'Outlier', False: 'Normal'})
        
        # Get metrics (exclude our added columns)
        metrics = [col for col in plot_data.columns if col not in ['is_outlier', 'outlier_type']]
        
        if len(metrics) < 2:
            print("Need at least 2 metrics to create scatter plot matrix.")
            return
        
        # Create the plot matrix
        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, n_metrics, figsize=figsize)
        fig.suptitle('Store Performance: Multi-Metric Scatter Plot Matrix', fontsize=16, y=0.98)
        
        # Define colors
        colors = {'Normal': '#1f77b4', 'Outlier': '#d62728'}
        
        for i, metric_y in enumerate(metrics):
            for j, metric_x in enumerate(metrics):
                ax = axes[i, j] if n_metrics > 1 else axes
                
                if i == j:
                    # Diagonal: histogram
                    for outlier_type in ['Normal', 'Outlier']:
                        data_subset = plot_data[plot_data['outlier_type'] == outlier_type]
                        if not data_subset.empty:
                            ax.hist(data_subset[metric_x], alpha=0.7, 
                                   label=outlier_type, color=colors[outlier_type],
                                   bins=15, density=True)
                    ax.set_xlabel(self._format_metric_name(metric_x))
                    ax.set_ylabel('Density')
                    
                else:
                    # Off-diagonal: scatter plot
                    for outlier_type in ['Normal', 'Outlier']:
                        data_subset = plot_data[plot_data['outlier_type'] == outlier_type]
                        if not data_subset.empty:
                            size = 60 if outlier_type == 'Outlier' else 30
                            alpha = 0.8 if outlier_type == 'Outlier' else 0.6
                            ax.scatter(data_subset[metric_x], data_subset[metric_y], 
                                     c=colors[outlier_type], label=outlier_type,
                                     s=size, alpha=alpha, edgecolors='white', linewidth=0.5)
                    
                    ax.set_xlabel(self._format_metric_name(metric_x))
                    ax.set_ylabel(self._format_metric_name(metric_y))
                
                # Add grid
                ax.grid(True, alpha=0.3)
                
                # Only add legend to top-right plot
                if i == 0 and j == n_metrics - 1:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Adjust layout
        plt.tight_layout()
        
        # Add summary text
        n_outliers = len(outlier_stores)
        n_total = len(plot_data)
        summary_text = f"Outliers: {n_outliers}/{n_total} stores ({n_outliers/n_total*100:.1f}%)"
        fig.text(0.02, 0.02, summary_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
        
        # Print outlier store details
        if not self.outlier_results.empty:
            print(f"\n📍 OUTLIER STORES HIGHLIGHTED IN RED:")
            for _, row in self.outlier_results.iterrows():
                metrics_str = ", ".join([f"{m} ({d})" for m, d in 
                                       zip(row['outlier_metrics'], row['directions'])])
                print(f"  Store {row['store_nbr']}: {metrics_str}")
    
    def _format_metric_name(self, metric: str) -> str:
        """Format metric names for display."""
        return metric.replace('_', ' ').title()
    
    def create_z_score_heatmap(self, store_metrics: pd.DataFrame,
                              figsize: Optional[Tuple[int, int]] = None,
                              save_path: Optional[str] = None) -> None:
        """
        Create a heatmap showing Z-scores for each store and metric.
        
        Args:
            store_metrics: Original DataFrame
            figsize: Figure size (width, height). If None, auto-calculates based on data size
            save_path: Optional path to save the plot
        """
        if self.outlier_results is None:
            print("Please run detect_outliers() first.")
            return
        
        # Prepare Z-score data
        test_stores = store_metrics[store_metrics['store_type'] == 'test'].copy()
        
        # Calculate store averages
        store_averages = test_stores.groupby(['store_nbr', 'metric'])['metric_value'].mean().reset_index()
        
        # Calculate global stats for Z-scores
        global_stats = store_averages.groupby('metric')['metric_value'].agg(['mean', 'std']).reset_index()
        
        # Merge and calculate Z-scores
        store_with_stats = store_averages.merge(global_stats, on='metric')
        store_with_stats['z_score'] = ((store_with_stats['metric_value'] - store_with_stats['mean']) / 
                                      store_with_stats['std'])
        
        # Handle division by zero (when std = 0)
        store_with_stats['z_score'] = store_with_stats['z_score'].fillna(0)
        
        # Pivot for heatmap
        z_score_matrix = store_with_stats.pivot(index='store_nbr', columns='metric', values='z_score')
        
        # Ensure all values are numeric and handle any remaining NaN values
        z_score_matrix = z_score_matrix.astype(float).fillna(0)
        
        # Auto-calculate figure size if not provided
        if figsize is None:
            n_stores = len(z_score_matrix.index)
            n_metrics = len(z_score_matrix.columns)
            
            # Calculate width and height based on content
            # Width: base + extra for each metric
            width = max(8, 2 + n_metrics * 1.5)
            
            # Height: base + extra for each store (minimum 0.4 inches per store)
            height = max(6, 3 + n_stores * 0.4)
            
            # Cap maximum size to prevent huge plots
            width = min(width, 20)
            height = min(height, 30)
            
            figsize = (width, height)
            print(f"Auto-sizing heatmap: {figsize[0]:.1f}x{figsize[1]:.1f} inches for {n_stores} stores x {n_metrics} metrics")
        
        # Create the heatmap
        plt.figure(figsize=figsize)
        
        # Create custom colormap (blue-white-red)
        cmap = sns.diverging_palette(220, 20, as_cmap=True)
        
        # Adjust annotation font size based on matrix size
        n_cells = len(z_score_matrix.index) * len(z_score_matrix.columns)
        if n_cells > 100:
            annot_size = 8
        elif n_cells > 50:
            annot_size = 10
        else:
            annot_size = 12
        
        # Plot heatmap with better error handling
        try:
            sns.heatmap(z_score_matrix, 
                       cmap=cmap, center=0, 
                       annot=True, fmt='.1f',
                       annot_kws={'size': annot_size},
                       cbar_kws={'label': 'Z-Score'},
                       linewidths=0.5,
                       square=False)  # Allow rectangular cells
        except Exception as e:
            print(f"Error creating detailed heatmap: {e}")
            print("Creating simplified heatmap...")
            # Fallback to simpler heatmap
            sns.heatmap(z_score_matrix, 
                       cmap='RdBu_r', center=0,
                       cbar_kws={'label': 'Z-Score'},
                       square=False)
        
        plt.title('Store Performance Z-Scores by Metric', fontsize=14, pad=20)
        plt.xlabel('Metrics', fontsize=12)
        plt.ylabel('Store Number', fontsize=12)
        
        # Highlight outlier stores
        outlier_stores = set(self.outlier_results['store_nbr'].tolist()) if not self.outlier_results.empty else set()
        
        # Add red borders around outlier stores
        for i, store in enumerate(z_score_matrix.index):
            if store in outlier_stores:
                plt.gca().add_patch(plt.Rectangle((0, i), len(z_score_matrix.columns), 1, 
                                                fill=False, edgecolor='red', lw=2))
        
        # Improve layout
        plt.tight_layout()
        
        # Rotate x-axis labels if needed
        if len(z_score_matrix.columns) > 5:
            plt.xticks(rotation=45, ha='right')
        
        # Adjust y-axis labels for readability
        plt.yticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to: {save_path}")
        
        plt.show()
        
        # Print some debug info
        print(f"Z-score matrix shape: {z_score_matrix.shape}")
        print(f"Data types: {z_score_matrix.dtypes.unique()}")
        print(f"Contains NaN: {z_score_matrix.isnull().any().any()}")
        if not z_score_matrix.empty:
            print(f"Z-score range: {z_score_matrix.min().min():.2f} to {z_score_matrix.max().max():.2f}")
        """Print a summary of outlier detection results."""
        if self.outlier_results is None:
            print("No outlier detection has been run yet.")
            return
        
        if self.outlier_results.empty:
            print("No outlier stores detected with current thresholds.")
            return
        
        print(f"\n🚨 OUTLIER STORES DETECTED: {len(self.outlier_results)} stores")
        print(f"Threshold: {self.std_threshold} standard deviations")
        print(f"Minimum outlier metrics: {self.min_outlier_metrics}")
        print("-" * 60)
        
        for _, row in self.outlier_results.iterrows():
            print(f"Store {row['store_nbr']}:")
            print(f"  • Outlier in {row['outlier_count']} metric(s): {', '.join(row['outlier_metrics'])}")
            print(f"  • Max Z-score: {row['max_abs_z_score']:.2f}")
            print(f"  • Directions: {', '.join(row['directions'])}")
            print()


# Example usage and testing functions
def create_sample_data(n_stores: int = 20, n_weeks: int = 10) -> pd.DataFrame:
    """Create sample data for testing the outlier detector."""
    np.random.seed(42)
    
    data = []
    metrics = ['sales', 'traffic', 'conversion']
    
    for store in range(1, n_stores + 1):
        # Create some outlier stores (stores 18, 19, 20)
        is_outlier = store > 17
        
        for week in range(6501, 6501 + n_weeks):
            for metric in metrics:
                # Base values
                if metric == 'sales':
                    base_value = np.random.normal(10000, 1500)
                elif metric == 'traffic':
                    base_value = np.random.normal(500, 75)
                else:  # conversion
                    base_value = np.random.normal(0.15, 0.02)
                
                # Make outlier stores significantly different
                if is_outlier:
                    if metric == 'sales':
                        base_value *= (2.5 if store == 18 else 0.4)  # High and low outliers
                    elif metric == 'traffic':
                        base_value *= (1.8 if store == 19 else 0.6)
                    else:  # conversion
                        base_value *= (1.5 if store == 20 else 0.5)
                
                data.append({
                    'store_nbr': store,
                    'wm_yr_wk_nbr': week,
                    'metric': metric,
                    'store_type': 'test',
                    'metric_value': max(0, base_value)  # Ensure non-negative
                })
    
    return pd.DataFrame(data)


def main():
    """Main function to demonstrate the outlier detector with visualizations."""
    # Create sample data
    print("Creating sample data...")
    store_metrics = create_sample_data()
    
    # Initialize detector
    detector = StoreOutlierDetector(std_threshold=2.0, min_outlier_metrics=1)
    
    # Detect outliers
    print("Detecting outliers...")
    outliers = detector.detect_outliers(store_metrics)
    
    # Print summary
    detector.print_summary()
    
    # Create visualizations
    print("\nGenerating scatter plot matrix...")
    detector.create_scatter_plot_matrix(store_metrics)
    
    print("\nGenerating Z-score heatmap...")
    detector.create_z_score_heatmap(store_metrics)
    
    # Show detailed report for first outlier (if any)
    if not outliers.empty:
        first_outlier = outliers.iloc[0]['store_nbr']
        print(f"\n📊 DETAILED REPORT FOR STORE {first_outlier}:")
        print("-" * 50)
        report = detector.get_detailed_report(store_metrics, first_outlier)
        
        for metric, stats in report['metrics_summary'].items():
            print(f"{metric.upper()}:")
            print(f"  Store avg: {stats['mean']:.2f} (Z-score: {stats['z_score']:.2f})")
            print(f"  Population avg: {stats['pop_mean']:.2f}")
            print()


if __name__ == "__main__":
    main()
