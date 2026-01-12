"""
Data preparation for Delta-based COPD Incidence Prediction.

This module builds delta features from consecutive years and stacks
historical deltas as separate features (lag1, lag2, etc.).

Functions:
- load_and_prepare_base: Load data and apply basic imputation
- compute_delta_features: Calculate delta for all feature columns
- build_stacked_delta_dataset: Stack historical deltas as separate features
- normalize_features: Z-score normalization per feature
- prepare_delta_data: Full pipeline
"""
import sys
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Add parent directory to import from original data_prep
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_prep import load_data, impute_target, impute_features
from delta_config import BASE_PATH, VERSION, METADATA_COLS, get_config


def load_and_prepare_base(
    base_path: str = BASE_PATH,
    version: str = VERSION,
    config: dict = None
) -> pd.DataFrame:
    """
    Load data and apply basic imputation (target + features).
    
    Args:
        base_path: Root path to data folder
        version: Dataset version
        config: Configuration dict
        
    Returns:
        DataFrame with imputed values (no lag construction yet)
    """
    if config is None:
        config = get_config()
    
    print("=" * 60)
    print("LOADING AND IMPUTING BASE DATA")
    print("=" * 60)
    
    # Load data
    df = load_data(base_path, version)
    
    # Filter excluded countries
    if config.get('exclude_countries'):
        excluded = config['exclude_countries']
        df = df[~df['Country Code'].isin(excluded)]
        print(f"✓ Excluded {len(excluded)} countries")
    
    # Impute target (Value column)
    df = impute_target(df)
    
    # Impute features
    nan_threshold = config.get('nan_threshold', 0.5)
    df = impute_features(df, nan_threshold=nan_threshold)
    
    return df
    

def get_base_feature_columns(df: pd.DataFrame) -> list:
    """
    Get list of base feature columns (excluding metadata).
    
    Args:
        df: DataFrame
        
    Returns:
        List of feature column names
    """
    metadata = set(METADATA_COLS) | {'target', 'delta_incidence'}
    feature_cols = [col for col in df.columns if col not in metadata]
    
    # Keep only numeric
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    return numeric_cols


def compute_delta_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Compute delta (change) for each feature column per country.
    
    delta_feature[Y] = feature[Y] - feature[Y-1]
    
    Args:
        df: DataFrame sorted by Country Code, Year
        feature_cols: List of feature columns to compute deltas for
        
    Returns:
        DataFrame with additional delta_* columns
    """
    result = df.copy()
    result = result.sort_values(['Country Code', 'Year']).reset_index(drop=True)
    
    # Also compute delta_incidence (Value change)
    result['delta_incidence'] = result.groupby('Country Code')['Value'].diff()
    
    # Compute delta for each feature
    for col in feature_cols:
        delta_col = f"delta_{col}"
        result[delta_col] = result.groupby('Country Code')[col].diff()
    
    print(f"✓ Computed deltas for {len(feature_cols)} features + incidence")
    
    return result


def build_stacked_delta_dataset(
    df: pd.DataFrame,
    base_feature_cols: list,
    target_year: int,
    drop_na: bool = True
) -> tuple[pd.DataFrame, list, dict]:
    """
    Build dataset for predicting delta_incidence of target_year.
    
    For target_year Y predicting delta_incidence[Y→Y+1]:
    - Include all historical delta features: delta_feature_lag1 (Y-1→Y), 
      delta_feature_lag2 (Y-2→Y-1), etc.
    
    Args:
        df: DataFrame with delta_* columns computed
        base_feature_cols: Original feature column names
        target_year: The year Y for which we predict delta_incidence[Y→Y+1]
        drop_na: If True, drop rows with NaN in ANY feature. If False, keep all rows
                 (caller should filter on selected features only)
        
    Returns:
        (dataset_df, feature_list, feature_family_map)
        - dataset_df: DataFrame ready for modeling
        - feature_list: List of all delta feature columns
        - feature_family_map: Dict mapping feature name -> family name
    """
    result_df = df.copy()
    result_df = result_df.sort_values(['Country Code', 'Year']).reset_index(drop=True)
    
    # Get all available years
    all_years = sorted(result_df['Year'].unique())
    
    # Find the minimum year we have deltas for (need Y-1 to compute delta at Y)
    min_delta_year = all_years[1] if len(all_years) > 1 else all_years[0]
    
    # target_year must have a next year for target
    if target_year not in all_years or (target_year + 1) not in [y for y in all_years]:
        # Check if we have delta_incidence for target_year (which is delta to next year)
        pass
    
    # Compute how many lag years we can use
    # For target_year Y, we can use deltas from years: min_delta_year to Y
    available_lag_years = [y for y in all_years if y <= target_year and y >= min_delta_year]
    n_lags = len(available_lag_years)
    
    if n_lags == 0:
        print(f"⚠ No delta features available for target year {target_year}")
        return pd.DataFrame(), [], {}
    
    print(f"  Target year {target_year}: {n_lags} lag years available ({available_lag_years})")
    
    # Build delta columns names (delta_FEATURE)
    delta_cols = [f"delta_{col}" for col in base_feature_cols]
    
    # For each country, build a single row with all stacked delta features
    rows = []
    feature_names = []
    feature_family_map = {}
    
    countries = result_df['Country Code'].unique()
    
    for country in countries:
        country_df = result_df[result_df['Country Code'] == country].copy()
        country_df = country_df.sort_values('Year')
        
        # Get target: delta_incidence for target_year (Value[Y+1] - Value[Y])
        target_row = country_df[country_df['Year'] == target_year]
        if target_row.empty:
            continue
            
        # Get next year's value for target
        next_year_row = country_df[country_df['Year'] == target_year + 1]
        if next_year_row.empty:
            continue
            
        target_value = next_year_row['Value'].values[0] - target_row['Value'].values[0]
        
        row_data = {
            'Country Code': country,
            'Country Name': target_row['Country Name'].values[0],
            'target_year': target_year,
            'target': target_value,
        }
        
        # Stack historical delta features
        for lag_idx, lag_year in enumerate(reversed(available_lag_years)):
            lag_num = lag_idx + 1  # lag1 is most recent
            
            lag_row = country_df[country_df['Year'] == lag_year]
            if lag_row.empty:
                continue
                
            for delta_col in delta_cols:
                if delta_col in lag_row.columns:
                    feature_name = f"{delta_col}_lag{lag_num}"
                    base_feature = delta_col  # e.g., "delta_GDP"
                    
                    row_data[feature_name] = lag_row[delta_col].values[0]
                    
                    if feature_name not in feature_family_map:
                        feature_family_map[feature_name] = base_feature
        
        rows.append(row_data)
    
    if not rows:
        return pd.DataFrame(), [], {}
    
    dataset_df = pd.DataFrame(rows)
    
    # Get all feature columns (exclude metadata)
    feature_cols = [col for col in dataset_df.columns 
                   if col not in ['Country Code', 'Country Name', 'target_year', 'target']]
    
    # Drop rows with NaN in features or target (only if drop_na=True)
    if drop_na:
        initial = len(dataset_df)
        dataset_df = dataset_df.dropna(subset=feature_cols + ['target'])
        dropped = initial - len(dataset_df)
        if dropped > 0:
            print(f"  Dropped {dropped} rows with NaN")
    else:
        # Only drop rows where target is NaN (we need the target)
        initial = len(dataset_df)
        dataset_df = dataset_df.dropna(subset=['target'])
        dropped = initial - len(dataset_df)
        if dropped > 0:
            print(f"  Dropped {dropped} rows with NaN target (kept NaN features)")
    
    print(f"  Built dataset: {len(dataset_df)} countries, {len(feature_cols)} features")
    
    return dataset_df, feature_cols, feature_family_map


def normalize_features(
    df: pd.DataFrame,
    feature_cols: list,
    return_scaler: bool = False
) -> tuple[pd.DataFrame, StandardScaler] | pd.DataFrame:
    """
    Z-score normalize all feature columns.
    
    Args:
        df: DataFrame with features
        feature_cols: List of feature column names
        return_scaler: If True, return the fitted scaler
        
    Returns:
        DataFrame with normalized features (and optionally the scaler)
    """
    result = df.copy()
    
    scaler = StandardScaler()
    result[feature_cols] = scaler.fit_transform(result[feature_cols])
    
    if return_scaler:
        return result, scaler
    return result


def prepare_delta_data_for_year(
    df_base: pd.DataFrame,
    base_feature_cols: list,
    target_year: int,
    normalize: bool = True
) -> tuple[pd.DataFrame, list, dict]:
    """
    Prepare delta dataset for a specific target year.
    
    Args:
        df_base: Base DataFrame with computed deltas
        base_feature_cols: Original feature column names
        target_year: Year to predict delta_incidence for
        normalize: Whether to normalize features
        
    Returns:
        (dataset_df, feature_cols, feature_family_map)
    """
    # Build stacked dataset
    dataset_df, feature_cols, family_map = build_stacked_delta_dataset(
        df_base, base_feature_cols, target_year
    )
    
    if dataset_df.empty:
        return pd.DataFrame(), [], {}
    
    # Normalize if requested
    if normalize and feature_cols:
        dataset_df = normalize_features(dataset_df, feature_cols)
    
    return dataset_df, feature_cols, family_map


def get_available_target_years(df: pd.DataFrame) -> list:
    """
    Get years where we can predict delta_incidence.
    
    Requirements:
    - Year Y must have delta features available (need Y-1 data)
    - Year Y+1 must exist (for target calculation)
    
    Args:
        df: DataFrame with Year column
        
    Returns:
        List of valid target years
    """
    all_years = sorted(df['Year'].unique())
    
    # Need at least 2 years to compute deltas
    if len(all_years) < 2:
        return []
    
    # First year with delta = all_years[1] (delta = year[1] - year[0])
    # Last valid target year = all_years[-2] (need Y+1 for target)
    min_target = all_years[1]  # First year with delta available
    max_target = all_years[-2]  # Last year where Y+1 exists
    
    valid_years = [y for y in all_years if min_target <= y <= max_target]
    
    return valid_years


def prepare_full_pipeline(
    base_path: str = BASE_PATH,
    version: str = VERSION,
    config: dict = None
) -> tuple[pd.DataFrame, list, list]:
    """
    Full data preparation pipeline.
    
    Returns base DataFrame with deltas computed, base feature columns,
    and list of available target years.
    
    Args:
        base_path: Path to data
        version: Data version
        config: Configuration dict
        
    Returns:
        (df_with_deltas, base_feature_cols, available_target_years)
    """
    if config is None:
        config = get_config()
    
    # Load and impute
    df = load_and_prepare_base(base_path, version, config)
    
    # Get base feature columns
    base_feature_cols = get_base_feature_columns(df)
    print(f"✓ {len(base_feature_cols)} base features identified")
    
    # Compute deltas
    df = compute_delta_features(df, base_feature_cols)
    
    # Get available target years
    target_years = get_available_target_years(df)
    print(f"✓ {len(target_years)} target years available: {target_years}")
    
    print("=" * 60)
    
    return df, base_feature_cols, target_years
