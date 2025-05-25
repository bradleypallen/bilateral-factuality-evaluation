import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Any
import warnings
warnings.filterwarnings('ignore')

def subsample_statistic_standard_error(data: np.ndarray,
                                     statistic_func: Callable[[np.ndarray], float],
                                     subsample_size: int = None,
                                     n_subsamples: int = 1000,
                                     random_state: int = None) -> Tuple[float, float, List[float]]:
    """
    Estimate standard error of any statistic using subsampling method from Politis & Romano (1994).
    
    Args:
        data: Input data array (n,)
        statistic_func: Function that computes the statistic on a data array
        subsample_size: Size of each subsample (b). If None, uses b = n^(2/3)
        n_subsamples: Number of subsamples to draw
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (original_statistic, estimated_std_error, subsample_statistics)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n = len(data)
    
    # Choose subsample size according to theory: b → ∞ and b/n → 0
    # Common choice is b = n^(2/3) for optimal rate
    if subsample_size is None:
        subsample_size = int(np.power(n, 2/3))
    
    # Ensure subsample size is valid
    subsample_size = min(subsample_size, n-1)
    subsample_size = max(subsample_size, 1)
    
    # print(f"Sample size n = {n}")
    # print(f"Subsample size b = {subsample_size}")
    # print(f"Ratio b/n = {subsample_size/n:.3f}")
    # print(f"Number of subsamples = {n_subsamples}")
    
    # Compute original statistic
    original_statistic = statistic_func(data)
    
    # Generate subsamples and compute statistics
    subsample_statistics = []
    
    for i in range(n_subsamples):
        # Sample without replacement
        indices = np.random.choice(n, size=subsample_size, replace=False)
        data_sub = data[indices]
        
        # Compute statistic on subsample
        try:
            stat_sub = statistic_func(data_sub)
            if not np.isnan(stat_sub) and not np.isinf(stat_sub):
                subsample_statistics.append(stat_sub)
        except Exception as e:
            # Skip if statistic cannot be computed
            continue
    
    subsample_statistics = np.array(subsample_statistics)
    
    if len(subsample_statistics) == 0:
        raise ValueError("No valid subsample statistics could be computed")
    
    # Estimate standard error
    # This approximates the standard error of the original statistic
    estimated_std_error = np.sqrt(subsample_size / n) * np.std(subsample_statistics, ddof=1)
    
    # print(f"\nResults:")
    # print(f"Original statistic: {original_statistic:.4f}")
    # print(f"Mean subsample statistic: {np.mean(subsample_statistics):.4f}")
    # print(f"Std of subsample statistics: {np.std(subsample_statistics, ddof=1):.4f}")
    # print(f"Estimated standard error: {estimated_std_error:.4f}")
    # print(f"Number of valid subsamples: {len(subsample_statistics)}")
    
    return original_statistic, estimated_std_error, subsample_statistics.tolist()

def analyze_column_statistic(df: pd.DataFrame,
                           column: str,
                           statistic_func: Callable[[np.ndarray], float],
                           statistic_name: str = "Statistic",
                           subsample_size: int = None,
                           n_subsamples: int = 1000,
                           random_state: int = 42) -> dict:
    """
    Apply subsampling standard error estimation to any column and statistic.
    
    Args:
        df: pandas DataFrame
        column: name of the numeric column to analyze
        statistic_func: function that computes the statistic (e.g., np.mean, np.median)
        statistic_name: name for reporting purposes
        subsample_size: size of subsamples (if None, uses n^(2/3))
        n_subsamples: number of subsamples to draw
        random_state: random seed
        
    Returns:
        Dictionary with results
    """
    
    # Extract data from column
    data = df[column].values
    
    # Remove any NaN values
    data = data[~np.isnan(data)]
    
    if len(data) == 0:
        raise ValueError(f"Column '{column}' contains no valid numeric data")
    
    # print(f"Analyzing column '{column}' with {len(data)} valid values...")
    # print(f"Data range: [{np.min(data):.3f}, {np.max(data):.3f}]")
    # print(f"Computing: {statistic_name}")
    
    # Apply subsampling method
    original_stat, std_error, subsample_stats = subsample_statistic_standard_error(
        data,
        statistic_func,
        subsample_size=subsample_size,
        n_subsamples=n_subsamples,
        random_state=random_state
    )
    
    # Calculate confidence interval (assuming normality)
    from scipy.stats import norm
    margin_of_error_95 = norm.ppf(0.975) * std_error
    ci_lower = original_stat - margin_of_error_95
    ci_upper = original_stat + margin_of_error_95
    
    results = {
        'statistic_name': statistic_name,
        'column': column,
        'value': original_stat,
        'standard_error': std_error,
        'confidence_interval_95': (ci_lower, ci_upper),
        'subsample_values': subsample_stats,
        'n_valid_subsamples': len(subsample_stats),
        'n_data_points': len(data)
    }
    
    return results

# Predefined common statistics
def create_common_statistics():
    """Returns a dictionary of common statistical functions"""
    return {
        'mean': np.mean,
        'median': np.median,
        'std': lambda x: np.std(x, ddof=1),
        'var': lambda x: np.var(x, ddof=1),
        'min': np.min,
        'max': np.max,
        'q25': lambda x: np.percentile(x, 25),
        'q75': lambda x: np.percentile(x, 75),
        'iqr': lambda x: np.percentile(x, 75) - np.percentile(x, 25),
        'range': lambda x: np.max(x) - np.min(x),
        'skewness': lambda x: pd.Series(x).skew(),
        'kurtosis': lambda x: pd.Series(x).kurtosis()
    }

def analyze_column_multiple_statistics(df: pd.DataFrame,
                                     column: str,
                                     statistics: dict = None,
                                     subsample_size: int = None,
                                     n_subsamples: int = 1000,
                                     random_state: int = 42) -> pd.DataFrame:
    """
    Analyze a column with multiple statistics at once.
    
    Args:
        df: pandas DataFrame
        column: name of the numeric column to analyze
        statistics: dict of {name: function} pairs. If None, uses common statistics
        subsample_size: size of subsamples
        n_subsamples: number of subsamples
        random_state: random seed
        
    Returns:
        DataFrame with results for all statistics
    """
    
    if statistics is None:
        statistics = create_common_statistics()
    
    results = []
    
    for stat_name, stat_func in statistics.items():
        # print(f"\n{'='*50}")
        # print(f"Computing {stat_name.upper()}")
        # print(f"{'='*50}")
        
        try:
            result = analyze_column_statistic(
                df, column, stat_func, stat_name,
                subsample_size, n_subsamples, random_state
            )
            
            results.append({
                'statistic': stat_name,
                'value': result['value'],
                'std_error': result['standard_error'],
                'ci_lower': result['confidence_interval_95'][0],
                'ci_upper': result['confidence_interval_95'][1],
                'plus_minus_notation': f"{result['value']:.3f} ± {result['standard_error']:.3f}"
            })
            
        except Exception as e:
            print(f"Error computing {stat_name}: {e}")
            continue
    
    return pd.DataFrame(results)

# Example usage functions
def example_usage():
    """Demonstrate usage with sample data"""
    
    # Create sample data
    np.random.seed(42)
    n = 400
    
    # Some example numeric data (could be model scores, measurements, etc.)
    sample_data = {
        'model_scores': np.random.beta(2, 5, n),  # Scores between 0 and 1
        'response_times': np.random.exponential(2, n),  # Response times
        'ratings': np.random.normal(7, 1.5, n)  # Ratings around 7
    }
    df = pd.DataFrame(sample_data)
    
    print("=== EXAMPLE: Analyzing Model Scores ===")
    
    # Analyze mean of model scores
    result = analyze_column_statistic(
        df, 
        column='model_scores',
        statistic_func=np.mean,
        statistic_name='Mean Model Score',
        n_subsamples=1000
    )
    
    print(f"\n=== RESULTS ===")
    print(f"Mean Model Score = {result['value']:.3f} ± {result['standard_error']:.3f}")
    print(f"95% CI: [{result['confidence_interval_95'][0]:.3f}, {result['confidence_interval_95'][1]:.3f}]")
    
    # Analyze multiple statistics for one column
    print(f"\n=== MULTIPLE STATISTICS FOR MODEL SCORES ===")
    stats_df = analyze_column_multiple_statistics(
        df, 
        column='model_scores',
        statistics={'mean': np.mean, 'median': np.median, 'std': lambda x: np.std(x, ddof=1)},
        n_subsamples=500  # Fewer subsamples for demo
    )
    
    print("\nSummary Table:")
    print(stats_df[['statistic', 'plus_minus_notation']].to_string(index=False))
    
    return result, stats_df

# Custom statistic example
def custom_statistic_example():
    """Example with a custom statistic function"""
    
    # Create sample data
    np.random.seed(42)
    df = pd.DataFrame({
        'values': np.random.normal(100, 15, 400)
    })
    
    # Define a custom statistic (e.g., coefficient of variation)
    def coefficient_of_variation(x):
        return np.std(x, ddof=1) / np.mean(x)
    
    result = analyze_column_statistic(
        df,
        column='values',
        statistic_func=coefficient_of_variation,
        statistic_name='Coefficient of Variation',
        n_subsamples=1000
    )
    
    print(f"Coefficient of Variation = {result['value']:.4f} ± {result['standard_error']:.4f}")
    
    return result

if __name__ == "__main__":
    # Run examples
    basic_result, multi_stats = example_usage()
    print(f"\n" + "="*60)
    custom_result = custom_statistic_example()