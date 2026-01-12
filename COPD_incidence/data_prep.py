"""
Data preparation for COPD Incidence Prediction.

Functions:
- load_data: Load CSV file
- impute_target: Linear regression imputation per country for Value column
- impute_features: Forward-fill then backward-fill per country + IterativeImputer
- build_lagged_dataset: Create (Features[Y], Target[Y+1]) structure
- prepare_data: Full pipeline
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from config import BASE_PATH, VERSION, COLS_TO_DROP, get_config


def load_data(base_path: str = BASE_PATH, version: str = VERSION) -> pd.DataFrame:
    """
    Load COPD incidence rate CSV.
    
    Args:
        base_path: Root path to data folder
        version: Dataset version ("1021" or "9019")
    
    Returns:
        Raw DataFrame
    """
    file_path = f"{base_path}/{version}/COPD_incidence_rate.csv"
    df = pd.read_csv(file_path)
    
    # Replace '..' and 'nan' strings with real NaN
    df = df.replace(['..', 'nan', ''], np.nan)
    
    # Ensure Year is integer
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
    
    # Ensure Value is float
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    
    print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"  Years: {df['Year'].min()} - {df['Year'].max()}")
    print(f"  Countries: {df['Country Code'].nunique()}")
    
    return df


def impute_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing Value (incidence) using linear regression per country.
    
    For each country:
    - Fit linear regression: Value = a*Year + b
    - Use it to fill missing years
    - If only 1 observation: use constant imputation
    - If 0 observations: drop the country
    
    Args:
        df: DataFrame with Country Code, Year, Value columns
    
    Returns:
        DataFrame with imputed Value column
    """
    result = df.copy()
    countries_to_drop = []
    
    for country_code in result['Country Code'].unique():
        mask = result['Country Code'] == country_code
        country_data = result.loc[mask, ['Year', 'Value']].dropna()
        
        n_obs = len(country_data)
        
        if n_obs == 0:
            # No observations: mark for removal
            countries_to_drop.append(country_code)
            continue
        
        if n_obs == 1:
            # Single observation: constant imputation
            constant_value = country_data['Value'].iloc[0]
            result.loc[mask, 'Value'] = result.loc[mask, 'Value'].fillna(constant_value)
        else:
            # Multiple observations: linear regression
            X = country_data['Year'].values.reshape(-1, 1)
            y = country_data['Value'].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Predict for all years of this country
            all_years = result.loc[mask, 'Year'].values.reshape(-1, 1)
            predicted = model.predict(all_years)
            
            # Fill only missing values
            missing_mask = mask & result['Value'].isna()
            result.loc[missing_mask, 'Value'] = predicted[result.loc[mask, 'Value'].isna()]
    
    # Drop countries with no observations
    if countries_to_drop:
        print(f"⚠ Dropping {len(countries_to_drop)} countries with no Value observations")
        result = result[~result['Country Code'].isin(countries_to_drop)]
    
    print(f"✓ Target imputation complete. Remaining NaN in Value: {result['Value'].isna().sum()}")
    
    return result


def impute_features(df: pd.DataFrame, nan_threshold: float = 0.5) -> pd.DataFrame:
    """
    Impute missing feature values:
    1. Drop columns with >nan_threshold NaN ratio
    2. Forward-fill then backward-fill per country
    3. Use IterativeImputer for remaining NaN
    
    Args:
        df: DataFrame with Country Code and feature columns
        nan_threshold: Drop columns with NaN ratio above this (default 0.5)
    
    Returns:
        DataFrame with imputed features
    """
    result = df.copy()
    
    # Identify feature columns (exclude metadata)
    metadata_cols = {'Country Code', 'Country Name', 'Year', 'Value', 'Disease', 'Measure', 'Metric'}
    feature_cols = [col for col in result.columns if col not in metadata_cols]
    
    # Convert feature columns to numeric
    for col in feature_cols:
        result[col] = pd.to_numeric(result[col], errors='coerce')
    
    # Step 1: Drop columns with too many NaN (>threshold)
    nan_ratios = result[feature_cols].isna().mean()
    cols_to_drop = nan_ratios[nan_ratios > nan_threshold].index.tolist()
    if cols_to_drop:
        print(f"⚠ Dropping {len(cols_to_drop)} columns with >{nan_threshold*100:.0f}% NaN:")
        for col in cols_to_drop:
            print(f"    - {col} ({nan_ratios[col]*100:.1f}% NaN)")
        result = result.drop(columns=cols_to_drop)
        feature_cols = [c for c in feature_cols if c not in cols_to_drop]
    
    # Sort by country and year for proper ffill/bfill
    result = result.sort_values(['Country Code', 'Year']).reset_index(drop=True)
    
    # Step 2: Forward-fill then backward-fill within each country
    result[feature_cols] = (
        result.groupby('Country Code')[feature_cols]
        .transform(lambda g: g.ffill().bfill())
    )
    
    remaining_nans = result[feature_cols].isna().sum().sum()
    print(f"✓ After ffill/bfill: {remaining_nans} NaN remaining in features")
    
    # Step 3: Use IterativeImputer for remaining NaN (cross-country imputation)
    if remaining_nans > 0:
        print("  Applying IterativeImputer for remaining NaN...")
        
        # Prepare numeric data for imputer
        imputer = IterativeImputer(random_state=42, max_iter=10, n_nearest_features=15)
        
        # Fit and transform only feature columns
        imputed_values = imputer.fit_transform(result[feature_cols])
        result[feature_cols] = imputed_values
        
        final_nans = result[feature_cols].isna().sum().sum()
        print(f"✓ After IterativeImputer: {final_nans} NaN remaining")
    
    return result


def build_lagged_dataset(df: pd.DataFrame, target_mode: str = "incidence") -> pd.DataFrame:
    """
    Build dataset with lagged structure: Features[Y] → Target[Y+1].
    
    Each row contains:
    - Country Code, Country Name
    - Year (the feature year Y)
    - Features from year Y
    - Target: Value[Y+1] (incidence mode) or Value[Y+1] - Value[Y] (delta mode)
    
    Args:
        df: DataFrame after imputation
        target_mode: "incidence" or "delta_incidence"
    
    Returns:
        DataFrame with lagged structure, target column added
    """
    result = df.copy()
    result = result.sort_values(['Country Code', 'Year']).reset_index(drop=True)
    
    # Create next year's value for each country
    result['Value_next'] = result.groupby('Country Code')['Value'].shift(-1)
    
    if target_mode == "incidence":
        result['target'] = result['Value_next']
    elif target_mode == "delta_incidence":
        result['target'] = result['Value_next'] - result['Value']
    else:
        raise ValueError(f"Unknown target_mode: {target_mode}. Use 'incidence' or 'delta_incidence'")
    
    # Drop rows without target (last year of each country)
    initial_rows = len(result)
    result = result.dropna(subset=['target'])
    dropped = initial_rows - len(result)
    
    print(f"✓ Lagged dataset built ({target_mode} mode)")
    print(f"  Dropped {dropped} rows (no Y+1 available)")
    print(f"  Final: {len(result)} prediction rows")
    
    # Clean up helper column
    result = result.drop(columns=['Value_next'])
    
    return result


def get_feature_columns(df: pd.DataFrame, config: dict = None) -> list:
    """
    Get list of feature columns (excluding metadata, target, and config exclusions).
    
    Args:
        df: DataFrame
        config: Optional config dict with 'exclude_features'
    
    Returns:
        List of feature column names
    """
    if config is None:
        config = get_config()
    
    # Start with all columns
    all_cols = set(df.columns)
    
    # Remove standard drops
    cols_to_drop = set(COLS_TO_DROP)
    
    # Remove target column
    cols_to_drop.add('target')
    
    # Remove config exclusions
    cols_to_drop.update(config.get('exclude_features', []))
    
    # Get remaining columns
    feature_cols = [col for col in df.columns if col not in cols_to_drop]
    
    # Keep only numeric columns
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    return numeric_cols


def prepare_data(
    base_path: str = BASE_PATH,
    version: str = VERSION,
    config: dict = None
) -> tuple[pd.DataFrame, list]:
    """
    Full data preparation pipeline.
    
    Args:
        base_path: Root path to data folder
        version: Dataset version
        config: Configuration dict
    
    Returns:
        (prepared_df, feature_columns)
    """
    if config is None:
        config = get_config()
    
    print("=" * 60)
    print("DATA PREPARATION")
    print("=" * 60)
    
    # Step 1: Load data
    df = load_data(base_path, version)
    
    # Step 2: Filter excluded countries
    if config.get('exclude_countries'):
        excluded = config['exclude_countries']
        df = df[~df['Country Code'].isin(excluded)]
        print(f"✓ Excluded {len(excluded)} countries")
    
    # Step 3: Impute target
    df = impute_target(df)
    
    # Step 4: Impute features (with column dropping for >50% NaN)
    nan_threshold = config.get('nan_threshold', 0.5)
    df = impute_features(df, nan_threshold=nan_threshold)
    
    # Step 5: Build lagged dataset
    target_mode = config.get('target_mode', 'incidence')
    df = build_lagged_dataset(df, target_mode)
    
    # Step 6: Filter excluded years (feature years)
    if config.get('exclude_years'):
        excluded_years = config['exclude_years']
        df = df[~df['Year'].isin(excluded_years)]
        print(f"✓ Excluded feature years: {excluded_years}")
    
    # Step 7: Get feature columns
    feature_cols = get_feature_columns(df, config)
    print(f"✓ {len(feature_cols)} feature columns selected")
    
    # Step 8: Drop rows only if they still have NaN (should be rare after IterativeImputer)
    initial = len(df)
    df = df.dropna(subset=feature_cols + ['target'])
    dropped = initial - len(df)
    if dropped > 0:
        print(f"⚠ Dropped {dropped} rows with remaining NaN values")
    
    print("=" * 60)
    print(f"FINAL: {len(df)} rows, {len(feature_cols)} features")
    print(f"Years: {sorted(df['Year'].unique())}")
    print(f"Countries: {df['Country Code'].nunique()}")
    print("=" * 60)
    
    return df, feature_cols

