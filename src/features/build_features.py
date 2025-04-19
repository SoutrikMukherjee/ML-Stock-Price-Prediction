import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import pickle
import os

def create_time_features(df, date_column='Date'):
    """
    Create time-based features from a date column
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    date_column : str, optional
        Name of the date column
    
    Returns:
    --------
    pandas.DataFrame
        Dataframe with added time features
    """
    # Create a copy to avoid modifying the original
    df_new = df.copy()
    
    # Convert to datetime if not already
    df_new[date_column] = pd.to_datetime(df_new[date_column])
    
    # Extract various date components
    df_new['day_of_week'] = df_new[date_column].dt.dayofweek
    df_new['day_of_month'] = df_new[date_column].dt.day
    df_new['week_of_year'] = df_new[date_column].dt.isocalendar().week
    df_new['month'] = df_new[date_column].dt.month
    df_new['quarter'] = df_new[date_column].dt.quarter
    df_new['year'] = df_new[date_column].dt.year
    df_new['is_month_start'] = df_new[date_column].dt.is_month_start.astype(int)
    df_new['is_month_end'] = df_new[date_column].dt.is_month_end.astype(int)
    
    # Create cyclical features for day of week, month, etc.
    df_new['day_of_week_sin'] = np.sin(2 * np.pi * df_new['day_of_week'] / 7)
    df_new['day_of_week_cos'] = np.cos(2 * np.pi * df_new['day_of_week'] / 7)
    df_new['month_sin'] = np.sin(2 * np.pi * df_new['month'] / 12)
    df_new['month_cos'] = np.cos(2 * np.pi * df_new['month'] / 12)
    
    return df_new

def create_lagged_features(df, target_col, lag_periods, group_col=None):
    """
    Create lagged features for time series data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    target_col : str
        Column to create lags for
    lag_periods : list
        List of lag periods
    group_col : str, optional
        Column to group by before creating lags
    
    Returns:
    --------
    pandas.DataFrame
        Dataframe with added lag features
    """
    # Create a copy to avoid modifying the original
    df_new = df.copy()
    
    # If grouping column is provided, create lags within each group
    if group_col:
        for lag in lag_periods:
            new_col_name = f"{target_col}_lag_{lag}"
            df_new[new_col_name] = df_new.groupby(group_col)[target_col].shift(lag)
    else:
        for lag in lag_periods:
            new_col_name = f"{target_col}_lag_{lag}"
            df_new[new_col_name] = df_new[target_col].shift(lag)
    
    return df_new

def create_rolling_features(df, target_col, windows, functions, group_col=None):
    """
    Create rolling window features for time series data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    target_col : str
        Column to create rolling features for
    windows : list
        List of window sizes
    functions : list
        List of functions to apply (e.g., 'mean', 'std', 'min', 'max')
    group_col : str, optional
        Column to group by before creating rolling features
    
    Returns:
    --------
    pandas.DataFrame
        Dataframe with added rolling features
    """
    # Create a copy to avoid modifying the original
    df_new = df.copy()
    
    # Define function mapping
    func_map = {
        'mean': np.mean,
        'std': np.std,
        'min': np.min,
        'max': np.max,
        'median': np.median
    }
    
    # If grouping column is provided, create rolling features within each group
    if group_col:
        for window in windows:
            for func_name in functions:
                func = func_map.get(func_name)
                if func is None:
                    print(f"Warning: Function '{func_name}' not recognized, skipping.")
                    continue
                
                new_col_name = f"{target_col}_roll_{window}_{func_name}"
                df_new[new_col_name] = df_new.groupby(group_col)[target_col].rolling(window=window).apply(func, raw=True).reset_index(0, drop=True)
    else:
        for window in windows:
            for func_name in functions:
                func = func_map.get(func_name)
                if func is None:
                    print(f"Warning: Function '{func_name}' not recognized, skipping.")
                    continue
                
                new_col_name = f"{target_col}_roll_{window}_{func_name}"
                df_new[new_col_name] = df_new[target_col].rolling(window=window).apply(func, raw=True)
    
    return df_new

def extract_text_features(df, text_column, max_features=100, output_dir=None):
    """
    Extract TF-IDF features from text
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    text_column : str
        Column containing text
    max_features : int, optional
        Maximum number of features to extract
    output_dir : str, optional
        Directory to save the vectorizer
    
    Returns:
    --------
    pandas.DataFrame, scipy.sparse.csr_matrix
        Original dataframe and TF-IDF features
    """
    # Initialize TF-IDF vectorizer
    tfidf = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        min_df=5,  # Minimum document frequency
        max_df=0.8  # Maximum document frequency
    )
    
    # Fill NaN values
    texts = df[text_column].fillna('').values
    
    # Fit and transform the texts
    tfidf_features = tfidf.fit_transform(texts)
    
    # Get feature names
    feature_names = tfidf.get_feature_names_out()
    
    # Save the vectorizer if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
            pickle.dump(tfidf, f)
    
    return df, tfidf_features, feature_names

def prepare_features_for_modeling(df, numeric_features, text_features=None, target_col='Target'):
    """
    Prepare features for modeling
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    numeric_features : list
        List of numeric feature columns
    text_features : scipy.sparse.csr_matrix, optional
        Text features
    target_col : str, optional
        Target column name
    
    Returns:
    --------
    scipy.sparse.csr_matrix or numpy.ndarray, numpy.ndarray
        Features and target arrays
    """
    # Extract numeric features
    X_numeric = df[numeric_features].values
    
    # Extract target
    y = df[target_col].values
    
    # Combine numeric and text features if provided
    if text_features is not None:
        X = hstack([X_numeric, text_features])
    else:
        X = X_numeric
    
    return X, y
