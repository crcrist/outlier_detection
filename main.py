import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

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
    """Main function to demonstrate the outlier detector."""
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
