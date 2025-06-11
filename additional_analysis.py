import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class TestStoreAnalyzer:
    """
    Dynamic analysis framework for test store impact analysis with staggered rollouts.
    
    Handles multiple metrics, multiple test start dates, and provides comprehensive
    statistical analysis with validation checks.
    """
    
    def __init__(self, df):
        """
        Initialize the analyzer with store data including test start dates.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with columns: store_nbr, wm_yr_wk_nbr, metric, store_type, metric_value, test_start_date
            Note: test_start_date should be populated for test stores, can be null/NaN for control stores
        """
        self.df = df.copy()
        self.results = {}
        self.validation_flags = []
        
        # Extract test start dates from DataFrame
        self.test_start_dates = self._extract_test_start_dates()
        
        # Process the data
        self._prepare_data()
        
    def _extract_test_start_dates(self):
        """Extract test start dates from the DataFrame."""
        # Get unique test start dates for each store
        test_dates = self.df[self.df['store_type'] == 'test'].groupby('store_nbr')['test_start_date'].first()
        
        # Convert to dictionary for easier lookup
        test_start_dict = test_dates.dropna().to_dict()
        
        print(f"Extracted test start dates for {len(test_start_dict)} test stores")
        if test_start_dict:
            print(f"Test start date range: {min(test_start_dict.values())} to {max(test_start_dict.values())}")
        
        return test_start_dict
        
    def _prepare_data(self):
        """Prepare data for analysis by adding relative time periods."""
        print("Preparing data for analysis...")
        
        # Add test start date to each row (using the extracted test start dates)
        self.df['test_start_week'] = self.df['store_nbr'].map(self.test_start_dates)
        
        # For control stores, we don't need test_start_week, but let's handle it gracefully
        # Control stores will have NaN for test_start_week, which is fine
        
        # Create relative week (weeks since/before implementation) - only for test stores
        self.df['weeks_relative_to_start'] = np.where(
            self.df['store_type'] == 'test',
            self.df['wm_yr_wk_nbr'] - self.df['test_start_week'],
            np.nan  # Control stores don't have relative weeks in the same way
        )
        
        # For control stores, we'll calculate relative weeks based on each test store's timeline during analysis
        
        # Create treatment indicator (1 if post-implementation for test stores)
        self.df['post_treatment'] = (
            (self.df['store_type'] == 'test') & 
            (self.df['weeks_relative_to_start'] >= 0)
        ).astype(int)
        
        # Create test store indicator
        self.df['is_test_store'] = (self.df['store_type'] == 'test').astype(int)
        
        print(f"Data prepared: {len(self.df)} rows processed")
        print(f"Metrics found: {sorted(self.df['metric'].unique())}")
        print(f"Week range: {self.df['wm_yr_wk_nbr'].min()} to {self.df['wm_yr_wk_nbr'].max()}")
        print(f"Test stores: {self.df[self.df['store_type'] == 'test']['store_nbr'].nunique()}")
        print(f"Control stores: {self.df[self.df['store_type'] == 'non-test-store']['store_nbr'].nunique()}")
        
    def run_validation_checks(self, min_pre_weeks=4):
        """
        Run validation checks on the data quality and control matching.
        
        Parameters:
        -----------
        min_pre_weeks : int
            Minimum number of pre-period weeks required
        """
        print("\nRunning validation checks...")
        
        for metric in self.df['metric'].unique():
            metric_data = self.df[self.df['metric'] == metric].copy()
            
            # Check 1: Sufficient pre-period data
            min_relative_week = metric_data['weeks_relative_to_start'].min()
            if min_relative_week > -min_pre_weeks:
                self.validation_flags.append(
                    f"WARNING - {metric}: Only {abs(min_relative_week)} pre-period weeks available (recommended: {min_pre_weeks}+)"
                )
            
            # Check 2: Pre-period trend similarity
            pre_period = metric_data[metric_data['weeks_relative_to_start'] < 0]
            if len(pre_period) > 0:
                pre_trends = self._calculate_trend_similarity(pre_period, metric)
                if pre_trends['trend_diff_pvalue'] < 0.05:
                    self.validation_flags.append(
                        f"WARNING - {metric}: Significant pre-period trend difference detected (p={pre_trends['trend_diff_pvalue']:.3f})"
                    )
            
            # Check 3: Sample size check
            test_stores = metric_data[metric_data['store_type'] == 'test']['store_nbr'].nunique()
            control_stores = metric_data[metric_data['store_type'] == 'non-test-store']['store_nbr'].nunique()
            
            if test_stores < 5:
                self.validation_flags.append(f"WARNING - {metric}: Low test store count ({test_stores})")
            if control_stores < test_stores:
                self.validation_flags.append(f"WARNING - {metric}: Fewer control stores ({control_stores}) than test stores ({test_stores})")
        
        # Print validation results
        if self.validation_flags:
            print("⚠️  Validation Flags:")
            for flag in self.validation_flags:
                print(f"  • {flag}")
        else:
            print("✅ All validation checks passed!")
            
    def _calculate_trend_similarity(self, pre_period_data, metric):
        """Calculate trend similarity between test and control stores in pre-period."""
        try:
            test_trend = pre_period_data[pre_period_data['store_type'] == 'test'].groupby('weeks_relative_to_start')['metric_value'].mean()
            control_trend = pre_period_data[pre_period_data['store_type'] == 'non-test-store'].groupby('weeks_relative_to_start')['metric_value'].mean()
            
            # Calculate correlation and trend difference
            if len(test_trend) > 2 and len(control_trend) > 2:
                correlation = test_trend.corr(control_trend)
                
                # Simple trend slope comparison
                test_slope = np.polyfit(range(len(test_trend)), test_trend.values, 1)[0]
                control_slope = np.polyfit(range(len(control_trend)), control_trend.values, 1)[0]
                slope_diff = abs(test_slope - control_slope)
                
                # T-test for trend difference (simplified)
                _, trend_p = stats.ttest_ind(test_trend.values, control_trend.values)
                
                return {
                    'correlation': correlation,
                    'slope_difference': slope_diff,
                    'trend_diff_pvalue': trend_p
                }
        except:
            pass
        
        return {'correlation': np.nan, 'slope_difference': np.nan, 'trend_diff_pvalue': 1.0}
    
    def analyze_impact(self, metrics=None):
        """
        Run comprehensive impact analysis for specified metrics.
        
        Parameters:
        -----------
        metrics : list or None
            List of metrics to analyze. If None, analyzes all metrics.
        """
        if metrics is None:
            metrics = self.df['metric'].unique()
        
        print(f"\nAnalyzing impact for metrics: {list(metrics)}")
        
        for metric in metrics:
            print(f"\nProcessing {metric}...")
            metric_results = self._analyze_single_metric(metric)
            self.results[metric] = metric_results
            
    def _analyze_single_metric(self, metric):
        """Analyze impact for a single metric."""
        metric_data = self.df[self.df['metric'] == metric].copy()
        
        # For this analysis, we need to handle control stores properly
        # We'll create a unified timeline for comparison
        
        # Get all test start dates to understand the timeline
        all_test_starts = list(self.test_start_dates.values())
        earliest_start = min(all_test_starts)
        latest_start = max(all_test_starts)
        
        # For control stores, create relative weeks based on the median test start date
        # This gives us a fair comparison timeline
        median_start = np.median(all_test_starts)
        
        # Add relative weeks for control stores
        control_mask = metric_data['store_type'] == 'non-test-store'
        metric_data.loc[control_mask, 'weeks_relative_to_start'] = (
            metric_data.loc[control_mask, 'wm_yr_wk_nbr'] - median_start
        )
        
        # Calculate baseline (pre-period averages) - use weeks before earliest test start
        pre_period = metric_data[metric_data['wm_yr_wk_nbr'] < earliest_start]
        
        test_baseline = pre_period[pre_period['store_type'] == 'test']['metric_value'].mean()
        control_baseline = pre_period[pre_period['store_type'] == 'non-test-store']['metric_value'].mean()
        
        # Calculate post-period averages - use weeks after latest test start to ensure all tests are active
        post_period = metric_data[metric_data['wm_yr_wk_nbr'] > latest_start]
        
        test_post = post_period[post_period['store_type'] == 'test']['metric_value'].mean()
        control_post = post_period[post_period['store_type'] == 'non-test-store']['metric_value'].mean()
        
        # Difference-in-Differences calculation
        test_change = test_post - test_baseline
        control_change = control_post - control_baseline
        did_effect = test_change - control_change
        
        # Statistical significance test
        test_post_values = post_period[post_period['store_type'] == 'test']['metric_value']
        control_post_values = post_period[post_period['store_type'] == 'non-test-store']['metric_value']
        
        if len(test_post_values) > 0 and len(control_post_values) > 0:
            t_stat, p_value = stats.ttest_ind(test_post_values, control_post_values)
        else:
            t_stat, p_value = np.nan, 1.0
        
        # Effect size calculations
        percent_change = (did_effect / test_baseline * 100) if test_baseline != 0 else 0
        
        # Weekly progression analysis
        weekly_analysis = self._analyze_weekly_progression(metric_data, metric)
        
        # Cohort analysis (by test start date)
        cohort_analysis = self._analyze_cohorts(metric_data, metric)
        
        return {
            'baseline_test': test_baseline,
            'baseline_control': control_baseline,
            'post_test': test_post,
            'post_control': control_post,
            'test_change': test_change,
            'control_change': control_change,
            'did_effect': did_effect,
            'percent_change': percent_change,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'weekly_progression': weekly_analysis,
            'cohort_analysis': cohort_analysis,
            'sample_sizes': {
                'test_stores': metric_data[metric_data['store_type'] == 'test']['store_nbr'].nunique(),
                'control_stores': metric_data[metric_data['store_type'] == 'non-test-store']['store_nbr'].nunique(),
                'total_observations': len(metric_data)
            },
            'timeline_info': {
                'earliest_test_start': earliest_start,
                'latest_test_start': latest_start,
                'median_test_start': median_start
            }
        }
    
    def _analyze_weekly_progression(self, metric_data, metric):
        """Analyze how the effect develops week by week."""
        weekly_summary = metric_data.groupby(['weeks_relative_to_start', 'store_type'])['metric_value'].agg(['mean', 'count']).reset_index()
        weekly_pivot = weekly_summary.pivot(index='weeks_relative_to_start', columns='store_type', values='mean')
        
        if 'test' in weekly_pivot.columns and 'non-test-store' in weekly_pivot.columns:
            weekly_pivot['difference'] = weekly_pivot['test'] - weekly_pivot['non-test-store']
            
            # Detect when effect stabilizes (simplified approach)
            post_weeks = weekly_pivot[weekly_pivot.index >= 0]
            if len(post_weeks) >= 3:
                # Calculate rolling standard deviation of differences
                rolling_std = post_weeks['difference'].rolling(3).std()
                stabilization_week = None
                if not rolling_std.empty:
                    # Effect is "stable" when rolling std is below 10% of mean difference
                    mean_diff = post_weeks['difference'].mean()
                    threshold = abs(mean_diff * 0.1) if mean_diff != 0 else 0.1
                    stable_weeks = rolling_std[rolling_std < threshold]
                    if not stable_weeks.empty:
                        stabilization_week = stable_weeks.index[0]
            
            return {
                'weekly_data': weekly_pivot,
                'stabilization_week': stabilization_week,
                'weeks_analyzed': len(post_weeks)
            }
        
        return {'weekly_data': weekly_pivot, 'stabilization_week': None, 'weeks_analyzed': 0}
    
    def _analyze_cohorts(self, metric_data, metric):
        """Analyze results by test start date cohorts."""
        cohort_results = {}
        
        for start_week in sorted(self.test_start_dates.values()):
            cohort_stores = [store for store, week in self.test_start_dates.items() if week == start_week]
            cohort_data = metric_data[metric_data['store_nbr'].isin(cohort_stores)]
            
            if len(cohort_data) > 0:
                # Calculate effect for this cohort
                pre_cohort = cohort_data[cohort_data['weeks_relative_to_start'] < 0]
                post_cohort = cohort_data[cohort_data['weeks_relative_to_start'] >= 0]
                
                if len(pre_cohort) > 0 and len(post_cohort) > 0:
                    pre_mean = pre_cohort['metric_value'].mean()
                    post_mean = post_cohort['metric_value'].mean()
                    change = post_mean - pre_mean
                    percent_change = (change / pre_mean * 100) if pre_mean != 0 else 0
                    
                    cohort_results[start_week] = {
                        'stores_count': len(cohort_stores),
                        'pre_mean': pre_mean,
                        'post_mean': post_mean,
                        'change': change,
                        'percent_change': percent_change
                    }
        
        return cohort_results
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report of all results."""
        if not self.results:
            print("No results available. Please run analyze_impact() first.")
            return
        
        print("\n" + "="*80)
        print("TEST STORE IMPACT ANALYSIS SUMMARY REPORT")
        print("="*80)
        
        # Overall summary table
        summary_data = []
        for metric, results in self.results.items():
            summary_data.append({
                'Metric': metric,
                'DiD Effect': f"{results['did_effect']:.2f}",
                'Percent Change': f"{results['percent_change']:.1f}%",
                'P-Value': f"{results['p_value']:.3f}",
                'Significant': "✓" if results['significant'] else "✗",
                'Test Stores': results['sample_sizes']['test_stores'],
                'Control Stores': results['sample_sizes']['control_stores']
            })
        
        summary_df = pd.DataFrame(summary_data)
        print("\nOVERALL RESULTS SUMMARY:")
        print(summary_df.to_string(index=False))
        
        # Detailed results for each metric
        for metric, results in self.results.items():
            print(f"\n{'='*50}")
            print(f"DETAILED ANALYSIS: {metric.upper()}")
            print(f"{'='*50}")
            
            print(f"\nBaseline Comparison:")
            print(f"  Test Stores Average:    {results['baseline_test']:.2f}")
            print(f"  Control Stores Average: {results['baseline_control']:.2f}")
            
            print(f"\nPost-Implementation:")
            print(f"  Test Stores Average:    {results['post_test']:.2f}")
            print(f"  Control Stores Average: {results['post_control']:.2f}")
            
            print(f"\nImpact Analysis:")
            print(f"  Test Store Change:      {results['test_change']:+.2f}")
            print(f"  Control Store Change:   {results['control_change']:+.2f}")
            print(f"  Net Effect (DiD):       {results['did_effect']:+.2f}")
            print(f"  Percent Change:         {results['percent_change']:+.1f}%")
            print(f"  Statistical Significance: {'Yes' if results['significant'] else 'No'} (p={results['p_value']:.3f})")
            
            # Weekly progression insights
            if results['weekly_progression']['stabilization_week'] is not None:
                print(f"\nTiming Insights:")
                print(f"  Effect Stabilization:   Week {results['weekly_progression']['stabilization_week']}")
                print(f"  Weeks Analyzed:         {results['weekly_progression']['weeks_analyzed']}")
            
            # Cohort analysis summary
            if results['cohort_analysis']:
                print(f"\nCohort Analysis:")
                for start_week, cohort_data in results['cohort_analysis'].items():
                    print(f"  Start Week {start_week}: {cohort_data['percent_change']:+.1f}% ({cohort_data['stores_count']} stores)")
    
    def create_visualizations(self, metrics=None, figsize=(15, 10)):
        """Create comprehensive visualizations for the analysis."""
        if not self.results:
            print("No results available. Please run analyze_impact() first.")
            return
        
        if metrics is None:
            metrics = list(self.results.keys())
        
        # Create subplot layout
        n_metrics = len(metrics)
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=figsize)
        if n_metrics == 1:
            axes = axes.reshape(2, 1)
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            # Time series plot
            ax1 = axes[i * 2]
            self._plot_time_series(metric, ax1)
            
            # Effect summary plot
            ax2 = axes[i * 2 + 1]
            self._plot_effect_summary(metric, ax2)
        
        # Hide empty subplots
        for j in range(len(metrics) * 2, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_time_series(self, metric, ax):
        """Plot time series comparison for a metric."""
        metric_data = self.df[self.df['metric'] == metric]
        weekly_data = self.results[metric]['weekly_progression']['weekly_data']
        
        if 'test' in weekly_data.columns and 'non-test-store' in weekly_data.columns:
            ax.plot(weekly_data.index, weekly_data['test'], 'b-o', label='Test Stores', linewidth=2, markersize=4)
            ax.plot(weekly_data.index, weekly_data['non-test-store'], 'r-s', label='Control Stores', linewidth=2, markersize=4)
            
            # Add vertical line at implementation
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7, label='Implementation Start')
            
            ax.set_xlabel('Weeks Relative to Implementation')
            ax.set_ylabel(f'{metric.title()} Value')
            ax.set_title(f'{metric.title()} - Time Series Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _plot_effect_summary(self, metric, ax):
        """Plot effect summary with confidence indication."""
        results = self.results[metric]
        
        categories = ['Test\nChange', 'Control\nChange', 'Net Effect\n(DiD)']
        values = [results['test_change'], results['control_change'], results['did_effect']]
        colors = ['lightblue', 'lightcoral', 'green' if results['significant'] else 'orange']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (0.01 * max(abs(v) for v in values)),
                    f'{value:+.2f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel(f'{metric.title()} Change')
        ax.set_title(f'{metric.title()} - Effect Summary\n({"Significant" if results["significant"] else "Not Significant"}, p={results["p_value"]:.3f})')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add horizontal line at zero
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

# Example usage function
def run_analysis_example():
    """
    Example of how to use the TestStoreAnalyzer class.
    Replace this with your actual data loading.
    """
    
    # Your DataFrame should now include a 'test_start_date' column
    # Example structure:
    # store_nbr | wm_yr_wk_nbr | metric | store_type | metric_value | test_start_date
    # 101       | 6508         | sales  | test       | 1000         | 6510
    # 101       | 6509         | sales  | test       | 1050         | 6510  
    # 102       | 6508         | sales  | non-test-store | 950      | NaN
    # etc.
    
    # Load your data (replace with your actual DataFrame)
    # df = pd.read_csv('your_store_data.csv')  # or however you load your data
    
    # Initialize analyzer - much simpler now!
    # analyzer = TestStoreAnalyzer(df)
    
    # Run validation checks
    # analyzer.run_validation_checks(min_pre_weeks=6)
    
    # Analyze impact for all metrics
    # analyzer.analyze_impact()
    
    # Generate summary report
    # analyzer.generate_summary_report()
    
    # Create visualizations
    # analyzer.create_visualizations()
    
    print("Replace the example with your actual DataFrame that includes 'test_start_date' column!")
    print("\nYour DataFrame should have these columns:")
    print("- store_nbr: Store identifier")
    print("- wm_yr_wk_nbr: Week numbers")
    print("- metric: Metric name")
    print("- store_type: 'test' or 'non-test-store'") 
    print("- metric_value: Numeric value")
    print("- test_start_date: Week when test started (for test stores only)")

if __name__ == "__main__":
    run_analysis_example()
