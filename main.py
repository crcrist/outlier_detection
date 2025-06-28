import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import for BigQuery (will need to be installed: pip install google-cloud-bigquery pandas-gbq)
try:
    from google.cloud import bigquery
    import pandas_gbq
    BIGQUERY_AVAILABLE = True
except ImportError:
    BIGQUERY_AVAILABLE = False
    st.warning("BigQuery libraries not installed. Run: pip install google-cloud-bigquery pandas-gbq")

# Set page config
st.set_page_config(
    page_title="Store Outlier Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .outlier-high { color: #d62728; font-weight: bold; }
    .outlier-low { color: #1f77b4; font-weight: bold; }
    .metric-card { 
        background-color: #f0f2f6; 
        padding: 20px; 
        border-radius: 10px; 
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

class StoreOutlierDetector:
    """
    Simple and efficient outlier detection for test stores based on metric values.
    """
    
    def __init__(self, std_threshold: float = 2.5, min_outlier_metrics: int = 1):
        self.std_threshold = std_threshold
        self.min_outlier_metrics = min_outlier_metrics
        self.outlier_results = None
    
    def detect_outliers(self, store_metrics: pd.DataFrame) -> pd.DataFrame:
        """Detect outlier stores based on metric values."""
        test_stores = store_metrics[store_metrics['store_type'] == 'test'].copy()
        
        if test_stores.empty:
            return pd.DataFrame()
        
        # Calculate aggregate metrics per store
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
        self.store_with_stats = store_with_stats  # Store for later use
        return final_outliers
    
    def get_detailed_report(self, store_metrics: pd.DataFrame, store_nbr: int) -> Dict:
        """Get detailed report for a specific store."""
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

@st.cache_data
def create_sample_data(n_stores: int = 50, n_weeks: int = 20, outlier_percentage: float = 0.1) -> pd.DataFrame:
    """Create sample data for testing the outlier detector."""
    np.random.seed(42)
    
    data = []
    metrics = ['sales', 'traffic', 'conversion', 'basket_size', 'customer_satisfaction']
    
    # Determine which stores will be outliers
    n_outliers = int(n_stores * outlier_percentage)
    outlier_stores = list(range(n_stores - n_outliers + 1, n_stores + 1))
    
    for store in range(1, n_stores + 1):
        is_outlier = store in outlier_stores
        
        for week in range(6501, 6501 + n_weeks):
            for metric in metrics:
                # Base values with some random variation
                if metric == 'sales':
                    base_value = np.random.normal(10000, 1500)
                elif metric == 'traffic':
                    base_value = np.random.normal(500, 75)
                elif metric == 'conversion':
                    base_value = np.random.normal(0.15, 0.02)
                elif metric == 'basket_size':
                    base_value = np.random.normal(45, 8)
                else:  # customer_satisfaction
                    base_value = np.random.normal(4.2, 0.3)
                
                # Make outlier stores significantly different
                if is_outlier:
                    # Different outlier patterns for different stores
                    outlier_idx = outlier_stores.index(store)
                    if outlier_idx % 3 == 0:
                        # High performer
                        multiplier = np.random.uniform(1.5, 2.5)
                    elif outlier_idx % 3 == 1:
                        # Low performer
                        multiplier = np.random.uniform(0.3, 0.6)
                    else:
                        # Mixed - high in some metrics, low in others
                        if metrics.index(metric) % 2 == 0:
                            multiplier = np.random.uniform(1.5, 2.0)
                        else:
                            multiplier = np.random.uniform(0.5, 0.7)
                    
                    base_value *= multiplier
                
                # Add some weekly variation
                weekly_variation = np.random.normal(1.0, 0.05)
                base_value *= weekly_variation
                
                data.append({
                    'store_nbr': store,
                    'wm_yr_wk_nbr': week,
                    'metric': metric,
                    'store_type': 'test' if np.random.random() > 0.2 else 'control',
                    'metric_value': max(0, base_value)
                })
    
    df = pd.DataFrame(data)
    # Ensure our outlier stores are test stores
    df.loc[df['store_nbr'].isin(outlier_stores), 'store_type'] = 'test'
    
    return df

def create_scatter_plot(data: pd.DataFrame, metric_x: str, metric_y: str, outlier_stores: set):
    """Create a scatter plot for two metrics."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data
    plot_data = data[data['store_type'] == 'test'].groupby(['store_nbr', 'metric'])['metric_value'].mean().reset_index()
    plot_pivot = plot_data.pivot(index='store_nbr', columns='metric', values='metric_value')
    
    # Separate normal and outlier stores
    normal_stores = plot_pivot[~plot_pivot.index.isin(outlier_stores)]
    outlier_stores_data = plot_pivot[plot_pivot.index.isin(outlier_stores)]
    
    # Plot
    if len(normal_stores) > 0:
        ax.scatter(normal_stores[metric_x], normal_stores[metric_y], 
                  c='#1f77b4', label='Normal', s=60, alpha=0.6, edgecolors='white')
    
    if len(outlier_stores_data) > 0:
        ax.scatter(outlier_stores_data[metric_x], outlier_stores_data[metric_y], 
                  c='#d62728', label='Outlier', s=100, alpha=0.8, edgecolors='black', linewidth=1)
        
        # Add store labels for outliers
        for idx in outlier_stores_data.index:
            ax.annotate(f'Store {idx}', 
                       (outlier_stores_data.loc[idx, metric_x], 
                        outlier_stores_data.loc[idx, metric_y]),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax.set_xlabel(metric_x.replace('_', ' ').title())
    ax.set_ylabel(metric_y.replace('_', ' ').title())
    ax.set_title(f'{metric_x.replace("_", " ").title()} vs {metric_y.replace("_", " ").title()}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def create_optimized_heatmap(detector: StoreOutlierDetector, selected_metrics: List[str], max_stores: int = 100):
    """Create an optimized heatmap for large datasets."""
    if detector.store_with_stats is None:
        return None
    
    # Filter for selected metrics
    filtered_data = detector.store_with_stats[detector.store_with_stats['metric'].isin(selected_metrics)]
    
    # Pivot for heatmap
    z_score_matrix = filtered_data.pivot(index='store_nbr', columns='metric', values='z_score')
    
    # Limit number of stores shown for performance
    if len(z_score_matrix) > max_stores:
        # Show only outlier stores if we have them
        if detector.outlier_results is not None and not detector.outlier_results.empty:
            outlier_stores = detector.outlier_results['store_nbr'].tolist()[:max_stores]
            z_score_matrix = z_score_matrix.loc[z_score_matrix.index.isin(outlier_stores)]
            title_suffix = f" (Top {len(z_score_matrix)} Outlier Stores)"
        else:
            # Show random sample if no outliers
            z_score_matrix = z_score_matrix.sample(n=max_stores)
            title_suffix = f" (Sample of {max_stores} Stores)"
    else:
        title_suffix = ""
    
    # Create figure with appropriate size
    fig_height = max(6, len(z_score_matrix) * 0.3)
    fig_height = min(fig_height, 20)  # Cap maximum height
    
    fig, ax = plt.subplots(figsize=(10, fig_height))
    
    # Create heatmap
    sns.heatmap(z_score_matrix, 
                cmap='RdBu_r', center=0, 
                annot=True, fmt='.1f',
                cbar_kws={'label': 'Z-Score'},
                linewidths=0.5,
                ax=ax)
    
    ax.set_title(f'Store Performance Z-Scores by Metric{title_suffix}')
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Store Number')
    
    # Highlight outlier stores
    if detector.outlier_results is not None and not detector.outlier_results.empty:
        outlier_stores = set(detector.outlier_results['store_nbr'].tolist())
        for i, store in enumerate(z_score_matrix.index):
            if store in outlier_stores:
                ax.add_patch(plt.Rectangle((0, i), len(z_score_matrix.columns), 1, 
                                         fill=False, edgecolor='red', lw=2))
    
    plt.tight_layout()
    return fig

def main():
    st.title("🔍 Store Outlier Detection System")
    st.markdown("Identify stores that are performing significantly different from the norm across multiple metrics.")
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Data source selection
        data_source = st.radio(
            "Data Source",
            ["Use Sample Data", "Upload CSV", "Connect to BigQuery"],
            help="Choose your data source"
        )
        
        if data_source == "Use Sample Data":
            st.subheader("Sample Data Parameters")
            n_stores = st.slider("Number of Stores", 10, 100, 50)
            n_weeks = st.slider("Number of Weeks", 5, 52, 20)
            outlier_pct = st.slider("Outlier Percentage", 0.05, 0.30, 0.10, 0.05)
            
            if st.button("Generate Sample Data"):
                st.session_state.data = create_sample_data(n_stores, n_weeks, outlier_pct)
                st.success(f"Generated data for {n_stores} stores over {n_weeks} weeks")
        
        elif data_source == "Upload CSV":
            uploaded_file = st.file_uploader(
                "Upload Store Data CSV",
                type=['csv'],
                help="CSV should contain columns: store_nbr, wm_yr_wk_nbr, metric, store_type, metric_value"
            )
            
            if uploaded_file is not None:
                st.session_state.data = pd.read_csv(uploaded_file)
                st.success("Data uploaded successfully!")
        
        else:  # Connect to BigQuery
            st.subheader("BigQuery Configuration")
            
            if not BIGQUERY_AVAILABLE:
                st.error("Please install BigQuery libraries first:")
                st.code("pip install google-cloud-bigquery pandas-gbq")
            else:
                # Configuration inputs
                with st.expander("🔑 Authentication Setup", expanded=True):
                    st.info("""
                    **Setup Instructions:**
                    1. Create a service account in Google Cloud Console
                    2. Download the JSON key file
                    3. Either:
                       - Set environment variable: `GOOGLE_APPLICATION_CREDENTIALS`
                       - Or upload the JSON key file below
                    """)
                    
                    auth_method = st.radio(
                        "Authentication Method",
                        ["Environment Variable", "Upload Key File", "Default Credentials"]
                    )
                    
                    if auth_method == "Upload Key File":
                        key_file = st.file_uploader(
                            "Upload Service Account JSON Key",
                            type=['json'],
                            help="Your Google Cloud service account key file"
                        )
                        if key_file is not None:
                            # In production, you'd save this securely
                            st.success("Key file uploaded!")
                            st.warning("Note: In production, store credentials securely!")
                
                # BigQuery settings
                st.subheader("Query Configuration")
                
                project_id = st.text_input(
                    "Project ID",
                    placeholder="your-project-id",
                    help="Your Google Cloud Project ID"
                )
                
                dataset_id = st.text_input(
                    "Dataset ID",
                    placeholder="your_dataset",
                    help="BigQuery dataset containing your data"
                )
                
                table_id = st.text_input(
                    "Table ID",
                    placeholder="store_metrics",
                    help="BigQuery table name"
                )
                
                # Advanced query options
                with st.expander("Advanced Query Options"):
                    use_custom_query = st.checkbox("Use Custom SQL Query")
                    
                    if use_custom_query:
                        custom_query = st.text_area(
                            "SQL Query",
                            value=f"""
SELECT 
    store_nbr,
    wm_yr_wk_nbr,
                    metric,
    store_type,
    metric_value
FROM `{project_id}.{dataset_id}.{table_id}`
WHERE store_type = 'test'
    AND wm_yr_wk_nbr >= 6500  -- Adjust date range as needed
ORDER BY store_nbr, wm_yr_wk_nbr, metric
                            """,
                            height=200,
                            help="Customize your query. Must return required columns."
                        )
                    else:
                        # Date range filters
                        col1, col2 = st.columns(2)
                        with col1:
                            weeks_back = st.number_input(
                                "Weeks of History",
                                min_value=1,
                                max_value=104,
                                value=26,
                                help="How many weeks of data to load"
                            )
                        with col2:
                            limit_stores = st.number_input(
                                "Limit Stores (0 = all)",
                                min_value=0,
                                max_value=5000,
                                value=0,
                                help="Limit number of stores for testing"
                            )
                
                # Load data button
                if st.button("Load Data from BigQuery", type="primary"):
                    if not project_id or not dataset_id or not table_id:
                        st.error("Please fill in all BigQuery configuration fields")
                    else:
                        try:
                            with st.spinner("Connecting to BigQuery and loading data..."):
                                # Initialize BigQuery client
                                if auth_method == "Upload Key File" and key_file is not None:
                                    # In production, you'd handle authentication more securely
                                    # For now, we'll show the structure
                                    st.info("Using uploaded key file for authentication")
                                    # client = bigquery.Client.from_service_account_json(key_file)
                                    client = bigquery.Client(project=project_id)
                                else:
                                    # Use default credentials or environment variable
                                    client = bigquery.Client(project=project_id)
                                
                                # Build query
                                if use_custom_query:
                                    query = custom_query
                                else:
                                    # Build dynamic query based on parameters
                                    query = f"""
                                    SELECT 
                                        store_nbr,
                                        wm_yr_wk_nbr,
                                        metric,
                                        store_type,
                                        metric_value
                                    FROM `{project_id}.{dataset_id}.{table_id}`
                                    WHERE store_type = 'test'
                                        AND wm_yr_wk_nbr >= (
                                            SELECT MAX(wm_yr_wk_nbr) - {weeks_back}
                                            FROM `{project_id}.{dataset_id}.{table_id}`
                                        )
                                    """
                                    
                                    if limit_stores > 0:
                                        query += f"""
                                        AND store_nbr IN (
                                            SELECT DISTINCT store_nbr 
                                            FROM `{project_id}.{dataset_id}.{table_id}`
                                            WHERE store_type = 'test'
                                            LIMIT {limit_stores}
                                        )
                                        """
                                    
                                    query += "\nORDER BY store_nbr, wm_yr_wk_nbr, metric"
                                
                                # Execute query
                                st.session_state.data = client.query(query).to_dataframe()
                                
                                # Show data summary
                                n_stores = st.session_state.data['store_nbr'].nunique()
                                n_metrics = st.session_state.data['metric'].nunique()
                                n_weeks = st.session_state.data['wm_yr_wk_nbr'].nunique()
                                
                                st.success(f"""
                                ✅ Data loaded successfully!
                                - Stores: {n_stores:,}
                                - Metrics: {n_metrics}
                                - Weeks: {n_weeks}
                                - Total rows: {len(st.session_state.data):,}
                                """)
                                
                        except Exception as e:
                            st.error(f"Error loading data from BigQuery: {str(e)}")
                            st.info("""
                            Common issues:
                            - Check your project ID, dataset, and table names
                            - Ensure you have proper authentication set up
                            - Verify your service account has BigQuery Data Viewer permissions
                            - Check that the table has the required columns
                            """)
                
                # Query helper
                with st.expander("💡 BigQuery Tips"):
                    st.markdown("""
                    **For optimal performance with 4,000+ stores:**
                    
                    1. **Create a materialized view** for faster queries:
                    ```sql
                    CREATE MATERIALIZED VIEW `project.dataset.store_metrics_summary` AS
                    SELECT 
                        store_nbr,
                        metric,
                        wm_yr_wk_nbr,
                        AVG(metric_value) as metric_value,
                        store_type
                    FROM `project.dataset.raw_metrics`
                    GROUP BY store_nbr, metric, wm_yr_wk_nbr, store_type
                    ```
                    
                    2. **Partition your table** by week:
                    ```sql
                    CREATE TABLE `project.dataset.store_metrics_partitioned`
                    PARTITION BY RANGE_BUCKET(wm_yr_wk_nbr, GENERATE_ARRAY(6500, 6600, 1))
                    AS SELECT * FROM `project.dataset.store_metrics`
                    ```
                    
                    3. **Use clustering** for better performance:
                    ```sql
                    CREATE TABLE `project.dataset.store_metrics_clustered`
                    CLUSTER BY store_nbr, metric
                    AS SELECT * FROM `project.dataset.store_metrics`
                    ```
                    
                    4. **Pre-aggregate** data for outlier detection:
                    ```sql
                    CREATE TABLE `project.dataset.store_weekly_aggregates` AS
                    SELECT 
                        store_nbr,
                        metric,
                        AVG(metric_value) as avg_value,
                        STDDEV(metric_value) as std_value,
                        COUNT(*) as num_weeks
                    FROM `project.dataset.store_metrics`
                    WHERE store_type = 'test'
                    GROUP BY store_nbr, metric
                    ```
                    """)
        
        st.divider()
        
        # Outlier detection parameters
        st.subheader("Detection Parameters")
        std_threshold = st.slider(
            "Standard Deviation Threshold",
            1.0, 4.0, 2.0, 0.1,
            help="Number of standard deviations from mean to consider as outlier"
        )
        
        min_outlier_metrics = st.slider(
            "Minimum Outlier Metrics",
            1, 5, 1,
            help="Minimum number of metrics a store must be outlier in"
        )
    
    # Main content area
    if 'data' in st.session_state:
        data = st.session_state.data
        
        # Display data overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Stores", data['store_nbr'].nunique())
        with col2:
            st.metric("Test Stores", data[data['store_type'] == 'test']['store_nbr'].nunique())
        with col3:
            st.metric("Metrics", data['metric'].nunique())
        with col4:
            st.metric("Time Period", f"{data['wm_yr_wk_nbr'].nunique()} weeks")
        
        # Metric selection
        st.subheader("📊 Select Metrics for Analysis")
        available_metrics = sorted(data['metric'].unique())
        selected_metrics = st.multiselect(
            "Choose metrics to include in outlier detection:",
            available_metrics,
            default=available_metrics[:3] if len(available_metrics) >= 3 else available_metrics
        )
        
        if selected_metrics:
            # Filter data for selected metrics
            filtered_data = data[data['metric'].isin(selected_metrics)]
            
            # Run outlier detection
            detector = StoreOutlierDetector(std_threshold=std_threshold, min_outlier_metrics=min_outlier_metrics)
            outliers = detector.detect_outliers(filtered_data)
            
            # Display results
            st.header("🚨 Outlier Detection Results")
            
            if outliers.empty:
                st.info("No outlier stores detected with current parameters. Try adjusting the thresholds.")
            else:
                st.success(f"Found {len(outliers)} outlier stores")
                
                # Create tabs for different views
                tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Detailed Analysis", "Visualizations", "Store Deep Dive"])
                
                with tab1:
                    st.subheader("Outlier Stores Summary")
                    
                    # Format the outliers dataframe for display
                    display_df = outliers[['store_nbr', 'outlier_count', 'max_abs_z_score']].copy()
                    display_df['outlier_metrics'] = outliers['outlier_metrics'].apply(lambda x: ', '.join(x))
                    display_df['directions'] = outliers['directions'].apply(lambda x: ', '.join(x))
                    display_df = display_df.round(2)
                    
                    st.dataframe(
                        display_df,
                        column_config={
                            "store_nbr": st.column_config.NumberColumn("Store Number", format="%d"),
                            "outlier_count": st.column_config.NumberColumn("# Outlier Metrics", format="%d"),
                            "max_abs_z_score": st.column_config.NumberColumn("Max |Z-Score|", format="%.2f"),
                            "outlier_metrics": "Outlier Metrics",
                            "directions": "Direction"
                        },
                        hide_index=True
                    )
                
                with tab2:
                    st.subheader("Metric-wise Outlier Breakdown")
                    
                    # Show outliers by metric
                    for metric in selected_metrics:
                        metric_outliers = detector.store_with_stats[
                            (detector.store_with_stats['metric'] == metric) & 
                            (detector.store_with_stats['is_outlier'])
                        ].sort_values('z_score', key=abs, ascending=False)
                        
                        if not metric_outliers.empty:
                            st.write(f"**{metric.replace('_', ' ').title()}**")
                            
                            # Create a simple bar chart
                            fig, ax = plt.subplots(figsize=(10, 4))
                            colors = ['red' if d == 'high' else 'blue' for d in metric_outliers['outlier_direction']]
                            bars = ax.bar(metric_outliers['store_nbr'].astype(str), 
                                         metric_outliers['z_score'], 
                                         color=colors, alpha=0.7)
                            ax.axhline(y=std_threshold, color='red', linestyle='--', alpha=0.5)
                            ax.axhline(y=-std_threshold, color='blue', linestyle='--', alpha=0.5)
                            ax.set_xlabel('Store Number')
                            ax.set_ylabel('Z-Score')
                            ax.set_title(f'{metric.replace("_", " ").title()} - Outlier Stores')
                            ax.grid(True, alpha=0.3)
                            plt.xticks(rotation=45)
                            st.pyplot(fig)
                            plt.close()
                
                with tab3:
                    st.subheader("Interactive Visualizations")
                    
                    # Scatter plot matrix
                    if len(selected_metrics) >= 2:
                        st.write("**Scatter Plot Analysis**")
                        col1, col2 = st.columns(2)
                        with col1:
                            metric_x = st.selectbox("X-axis metric", selected_metrics, index=0)
                        with col2:
                            metric_y = st.selectbox("Y-axis metric", selected_metrics, index=1)
                        
                        outlier_store_set = set(outliers['store_nbr'].tolist())
                        fig = create_scatter_plot(filtered_data, metric_x, metric_y, outlier_store_set)
                        st.pyplot(fig)
                        plt.close()
                    
                    # Z-score heatmap
                    st.write("**Z-Score Heatmap**")
                    
                    # Add control for heatmap size
                    if len(data[data['store_type'] == 'test']['store_nbr'].unique()) > 100:
                        max_stores = st.slider(
                            "Maximum stores to display in heatmap",
                            min_value=50,
                            max_value=500,
                            value=100,
                            step=50,
                            help="Limit the number of stores shown for better performance"
                        )
                    else:
                        max_stores = 100
                    
                    heatmap_fig = create_optimized_heatmap(detector, selected_metrics, max_stores)
                    if heatmap_fig:
                        st.pyplot(heatmap_fig)
                        plt.close()
                
                with tab4:
                    st.subheader("Store Deep Dive")
                    
                    # Store selector
                    all_test_stores = sorted(data[data['store_type'] == 'test']['store_nbr'].unique())
                    selected_store = st.selectbox(
                        "Select a store to analyze:",
                        all_test_stores,
                        index=0 if outliers.empty else all_test_stores.index(outliers.iloc[0]['store_nbr'])
                    )
                    
                    # Get detailed report
                    report = detector.get_detailed_report(filtered_data, selected_store)
                    
                    if 'error' not in report:
                        # Check if this is an outlier store
                        is_outlier = selected_store in outliers['store_nbr'].values if not outliers.empty else False
                        
                        if is_outlier:
                            st.warning(f"⚠️ Store {selected_store} is flagged as an OUTLIER")
                            outlier_info = outliers[outliers['store_nbr'] == selected_store].iloc[0]
                            st.write(f"Outlier in metrics: {', '.join(outlier_info['outlier_metrics'])}")
                        else:
                            st.success(f"✅ Store {selected_store} is within normal range")
                        
                        # Display metrics comparison
                        st.write("**Metric Performance vs Population**")
                        
                        metric_data = []
                        for metric, stats in report['metrics_summary'].items():
                            metric_data.append({
                                'Metric': metric,
                                'Store Average': stats['mean'],
                                'Population Average': stats['pop_mean'],
                                'Z-Score': stats['z_score'],
                                'Status': '🔴 High' if stats['z_score'] > std_threshold else 
                                         '🔵 Low' if stats['z_score'] < -std_threshold else '🟢 Normal'
                            })
                        
                        metric_df = pd.DataFrame(metric_data)
                        st.dataframe(metric_df, hide_index=True)
                        
                        # Time series for selected store
                        st.write("**Store Performance Over Time**")
                        store_time_data = filtered_data[filtered_data['store_nbr'] == selected_store]
                        
                        fig, ax = plt.subplots(figsize=(12, 6))
                        for metric in selected_metrics:
                            metric_data = store_time_data[store_time_data['metric'] == metric]
                            if not metric_data.empty:
                                ax.plot(metric_data['wm_yr_wk_nbr'], metric_data['metric_value'], 
                                       marker='o', label=metric, linewidth=2)
                        
                        ax.set_xlabel('Week')
                        ax.set_ylabel('Metric Value')
                        ax.set_title(f'Store {selected_store} - Performance Trends')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                        plt.close()
        else:
            st.warning("Please select at least one metric for analysis.")
    
    else:
        # No data loaded
        st.info("👈 Please load data using the sidebar controls to begin analysis.")
        
        # Show example of expected data format
        with st.expander("Expected Data Format"):
            st.write("Your CSV should contain the following columns:")
            example_data = pd.DataFrame({
                'store_nbr': [101, 101, 102, 102],
                'wm_yr_wk_nbr': [6501, 6502, 6501, 6502],
                'metric': ['sales', 'sales', 'sales', 'sales'],
                'store_type': ['test', 'test', 'test', 'test'],
                'metric_value': [10000, 10500, 9500, 9800]
            })
            st.dataframe(example_data)

if __name__ == "__main__":
    main()
