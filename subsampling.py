import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from typing import Tuple, List, Callable, Any
import warnings
warnings.filterwarnings('ignore')

def macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute macro F1 score.
    
    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted binary labels
        
    Returns:
        Macro F1 score
    """
    return f1_score(y_true, y_pred, average='macro')

def subsample_f1_standard_error(y_true: np.ndarray, 
                           y_pred: np.ndarray,
                           subsample_size: int = None,
                           n_subsamples: int = 1000,
                           random_state: int = None) -> Tuple[float, float, List[float]]:
    """
    Estimate standard error of macro F1 using subsampling method from Politis & Romano (1994).
    
    Args:
        y_true: Ground truth binary labels (n,)
        y_pred: Predicted binary labels (n,)
        subsample_size: Size of each subsample (b). If None, uses b = n^(2/3)
        n_subsamples: Number of subsamples to draw
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (original_f1, estimated_std_error, subsample_f1_scores)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n = len(y_true)
    
    # Choose subsample size according to theory: b → ∞ and b/n → 0
    # Common choice is b = n^(2/3) for optimal rate
    if subsample_size is None:
        subsample_size = int(np.power(n, 2/3))
    
    # Ensure subsample size is valid
    subsample_size = min(subsample_size, n-1)
    subsample_size = max(subsample_size, 2)  # Need at least 2 samples for F1
    
    # print(f"Sample size n = {n}")
    # print(f"Subsample size b = {subsample_size}")
    # print(f"Ratio b/n = {subsample_size/n:.3f}")
    # print(f"Number of subsamples = {n_subsamples}")
    
    # Compute original statistic
    original_f1 = macro_f1_score(y_true, y_pred)
    
    # Generate subsamples and compute F1 scores
    subsample_f1_scores = []
    
    for i in range(n_subsamples):
        # Sample without replacement
        indices = np.random.choice(n, size=subsample_size, replace=False)
        y_true_sub = y_true[indices]
        y_pred_sub = y_pred[indices]
        
        # Compute F1 on subsample
        try:
            f1_sub = macro_f1_score(y_true_sub, y_pred_sub)
            subsample_f1_scores.append(f1_sub)
        except:
            # Skip if F1 cannot be computed (e.g., if one class is missing)
            continue
    
    subsample_f1_scores = np.array(subsample_f1_scores)
    
    # According to the paper, we need to properly normalize
    # The standard error is estimated from the variance of the subsample statistics
    # For the empirical variance, we use the sample variance of subsample F1 scores
    
    # Estimate standard error
    # This approximates the standard error of the original F1 score
    estimated_std_error = np.sqrt(subsample_size / n) * np.std(subsample_f1_scores, ddof=1)
    
    # print(f"\nResults:")
    # print(f"Original macro F1: {original_f1:.4f}")
    # print(f"Mean subsample F1: {np.mean(subsample_f1_scores):.4f}")
    # print(f"Std of subsample F1: {np.std(subsample_f1_scores, ddof=1):.4f}")
    # print(f"Estimated standard error: {estimated_std_error:.4f}")
    # print(f"Number of valid subsamples: {len(subsample_f1_scores)}")
    
    return original_f1, estimated_std_error, subsample_f1_scores.tolist()

def confidence_interval(f1_score: float, 
                       std_error: float, 
                       confidence_level: float = 0.95) -> Tuple[float, float]:
    """
    Construct confidence interval for F1 score using normal approximation.
    
    Args:
        f1_score: Original F1 score
        std_error: Estimated standard error
        confidence_level: Confidence level (default 0.95 for 95% CI)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    from scipy.stats import norm
    
    alpha = 1 - confidence_level
    z_score = norm.ppf(1 - alpha/2)
    
    margin_of_error = z_score * std_error
    lower_bound = max(0, f1_score - margin_of_error)  # F1 is bounded by 0
    upper_bound = min(1, f1_score + margin_of_error)  # F1 is bounded by 1
    
    return lower_bound, upper_bound

def analyze_classification_results(df, 
                                 true_label_col='ground_truth', 
                                 pred_label_col='predictions',
                                 subsample_size=None,
                                 n_subsamples=1000,
                                 random_state=42):
    """
    Apply subsampling standard error estimation to your classification DataFrame.
    
    Args:
        df: pandas DataFrame with your classification results
        true_label_col: name of column containing ground truth labels
        pred_label_col: name of column containing predictions
        subsample_size: size of subsamples (if None, uses n^(2/3))
        n_subsamples: number of subsamples to draw
        random_state: random seed
        
    Returns:
        Dictionary with results
    """
    
    # Extract arrays from DataFrame
    y_true = df[true_label_col].values
    y_pred = df[pred_label_col].values
    
    # Ensure binary values
    assert set(np.unique(y_true)).issubset({0, 1}), "Ground truth must be binary (0/1)"
    assert set(np.unique(y_pred)).issubset({0, 1}), "Predictions must be binary (0/1)"
    
    # print(f"Analyzing {len(df)} classification results...")
    # print(f"Class distribution in ground truth: {np.bincount(y_true)}")
    # print(f"Class distribution in predictions: {np.bincount(y_pred)}")
    
    # Apply subsampling method
    f1_score, std_error, subsample_scores = subsample_f1_standard_error(
        y_true, y_pred,
        subsample_size=subsample_size,
        n_subsamples=n_subsamples,
        random_state=random_state
    )
    
    # Get confidence interval
    ci_lower, ci_upper = confidence_interval(f1_score, std_error, 0.95)
    
    results = {
        'macro_f1': f1_score,
        'standard_error': std_error,
        'confidence_interval_95': (ci_lower, ci_upper),
        'subsample_scores': subsample_scores,
        'n_valid_subsamples': len(subsample_scores)
    }
    
    return results

# Example usage with your DataFrame:
def example_usage():
    """
    Example of how to use with your DataFrame
    """
    # If your DataFrame looks like this:
    #   ground_truth  predictions
    # 0            1            1
    # 1            0            1
    # 2            1            0
    # ... etc for 400 rows

    # Load your data (replace with your actual loading method)
    # df = pd.read_csv('your_data.csv')
    
    # Create sample data for demonstration
    np.random.seed(42)
    n = 400
    sample_data = {
        'ground_truth': np.random.choice([0, 1], size=n, p=[0.3, 0.7]),
        'predictions': np.random.choice([0, 1], size=n, p=[0.4, 0.6])
    }
    df = pd.DataFrame(sample_data)
    
    # Run the analysis
    results = analyze_classification_results(
        df, 
        true_label_col='ground_truth',    # adjust column name as needed
        pred_label_col='predictions',     # adjust column name as needed
        n_subsamples=1000,               # as you requested
        random_state=42                  # for reproducibility
    )
    
    # Print results
    print(f"\n=== FINAL RESULTS ===")
    print(f"Macro F1 Score: {results['macro_f1']:.4f}")
    print(f"Standard Error: {results['standard_error']:.4f}")
    print(f"95% CI: [{results['confidence_interval_95'][0]:.4f}, {results['confidence_interval_95'][1]:.4f}]")
    
    return results

# Alternative: if you want to experiment with different subsample sizes
def compare_subsample_sizes(df, true_label_col, pred_label_col, sizes=None):
    """
    Compare results using different subsample sizes to see sensitivity.
    """
    if sizes is None:
        n = len(df)
        sizes = [
            int(n**0.5),     # n^(1/2) 
            int(n**(2/3)),   # n^(2/3) - theoretical optimum
            int(n**0.8),     # n^(4/5)
            n//4,            # n/4
            n//3             # n/3
        ]
    
    results = {}
    y_true = df[true_label_col].values
    y_pred = df[pred_label_col].values
    
    for size in sizes:
        if size >= len(df) or size < 10:
            continue
            
        print(f"\n--- Subsample size: {size} (ratio: {size/len(df):.3f}) ---")
        f1, se, _ = subsample_f1_standard_error(
            y_true, y_pred, 
            subsample_size=size, 
            n_subsamples=1000,
            random_state=42
        )
        results[size] = {'f1': f1, 'std_error': se}
    
    return results

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