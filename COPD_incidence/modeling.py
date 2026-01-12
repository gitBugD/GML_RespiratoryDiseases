"""
Modeling for COPD Incidence Prediction.

Functions:
- run_loco_year: LOCO cross-validation for a single year
- run_full_loco: Full LOCO evaluation across all years
- compute_metrics: Calculate RMSE and worst-case error
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from typing import Any

from config import get_config, MODELS


def run_loco_year(
    df: pd.DataFrame,
    year: int,
    model: Any,
    feature_cols: list,
    min_countries: int = 5
) -> pd.DataFrame:
    """
    Run Leave-One-Country-Out cross-validation for a single year.
    
    For each country in this year:
    - Train on all OTHER countries (same year)
    - Test on this single country
    - Record prediction
    
    Args:
        df: Prepared DataFrame with 'target' column
        year: The feature year to evaluate
        model: Sklearn-compatible model (will be cloned per fold)
        feature_cols: List of feature column names
        min_countries: Minimum countries required to run
    
    Returns:
        DataFrame with columns: Year, Country, y_true, y_pred, error
    """
    # Filter to this year
    year_df = df[df['Year'] == year].copy()
    countries = year_df['Country Code'].unique()
    
    if len(countries) < min_countries:
        print(f"  ⚠ Year {year}: only {len(countries)} countries, skipping (min={min_countries})")
        return pd.DataFrame()
    
    results = []
    
    for country in countries:
        # Split: test = this country, train = others
        test_mask = year_df['Country Code'] == country
        train_mask = ~test_mask
        
        X_train = year_df.loc[train_mask, feature_cols].values
        y_train = year_df.loc[train_mask, 'target'].values
        X_test = year_df.loc[test_mask, feature_cols].values
        y_test = year_df.loc[test_mask, 'target'].values
        
        # Fit scaler on train only
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Clone and fit model
        model_clone = clone(model)
        model_clone.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model_clone.predict(X_test_scaled)
        
        # Record result (should be single row)
        for i, (yt, yp) in enumerate(zip(y_test, y_pred)):
            results.append({
                'Year': year,
                'Country': country,
                'y_true': yt,
                'y_pred': yp,
                'error': yp - yt,
                'abs_error': abs(yp - yt),
            })
    
    return pd.DataFrame(results)


def run_full_loco(
    df: pd.DataFrame,
    feature_cols: list,
    config: dict = None,
    model_name: str = None,
    verbose: bool = True
) -> tuple[pd.DataFrame, dict]:
    """
    Run full LOCO evaluation across all years for one model.
    
    Args:
        df: Prepared DataFrame
        feature_cols: List of feature column names
        config: Configuration dict
        model_name: Specific model to run (default: all models)
        verbose: Print progress
    
    Returns:
        (predictions_df, metrics_dict)
    """
    if config is None:
        config = get_config()
    
    models = config.get('models', MODELS)
    min_countries = config.get('min_countries_per_year', 5)
    
    # If specific model requested
    if model_name:
        models = {model_name: models[model_name]}
    
    years = sorted(df['Year'].unique())
    
    all_results = []
    
    for name, model in models.items():
        if verbose:
            print(f"\n▶ Running LOCO for {name}...")
        
        for year in years:
            year_results = run_loco_year(
                df, year, model, feature_cols, min_countries
            )
            if not year_results.empty:
                year_results['model'] = name
                all_results.append(year_results)
    
    if not all_results:
        return pd.DataFrame(), {}
    
    predictions_df = pd.concat(all_results, ignore_index=True)
    
    # Compute metrics
    metrics = compute_metrics(predictions_df)
    
    if verbose:
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        for model_name, m in metrics.items():
            print(f"{model_name}:")
            print(f"  RMSE: {m['rmse']:.4f}")
            print(f"  Worst-case |error|: {m['worst_case']:.4f}")
            print(f"  MAE: {m['mae']:.4f}")
        print("=" * 60)
    
    return predictions_df, metrics


def compute_metrics(predictions_df: pd.DataFrame) -> dict:
    """
    Compute RMSE and worst-case error per model.
    
    Args:
        predictions_df: DataFrame with model, y_true, y_pred, error columns
    
    Returns:
        Dict: {model_name: {rmse, worst_case, mae, n_predictions}}
    """
    metrics = {}
    
    for model_name in predictions_df['model'].unique():
        model_df = predictions_df[predictions_df['model'] == model_name]
        errors = model_df['error'].values
        abs_errors = np.abs(errors)
        
        metrics[model_name] = {
            'rmse': np.sqrt(np.mean(errors ** 2)),
            'worst_case': np.max(abs_errors),
            'mae': np.mean(abs_errors),
            'n_predictions': len(model_df),
        }
    
    return metrics


def compute_metrics_by_year(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute metrics aggregated by year.
    
    Returns:
        DataFrame with Year, model, rmse, worst_case, mae columns
    """
    results = []
    
    for (year, model), group in predictions_df.groupby(['Year', 'model']):
        errors = group['error'].values
        abs_errors = np.abs(errors)
        
        results.append({
            'Year': year,
            'model': model,
            'rmse': np.sqrt(np.mean(errors ** 2)),
            'worst_case': np.max(abs_errors),
            'mae': np.mean(abs_errors),
            'n_countries': len(group),
        })
    
    return pd.DataFrame(results)


def compute_metrics_by_country(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute metrics aggregated by country.
    
    Returns:
        DataFrame with Country, model, rmse, worst_case, mae columns
    """
    results = []
    
    for (country, model), group in predictions_df.groupby(['Country', 'model']):
        errors = group['error'].values
        abs_errors = np.abs(errors)
        
        results.append({
            'Country': country,
            'model': model,
            'rmse': np.sqrt(np.mean(errors ** 2)),
            'worst_case': np.max(abs_errors),
            'mae': np.mean(abs_errors),
            'n_years': len(group),
        })
    
    return pd.DataFrame(results)


def quick_loco_eval(
    df: pd.DataFrame,
    feature_cols: list,
    model: Any,
    config: dict = None
) -> tuple[float, float]:
    """
    Quick LOCO evaluation returning only (RMSE, worst_case).
    Useful for feature selection loops.
    
    Args:
        df: Prepared DataFrame
        feature_cols: Feature columns to use
        model: Model to evaluate
        config: Configuration dict
    
    Returns:
        (rmse, worst_case)
    """
    if config is None:
        config = get_config()
    
    min_countries = config.get('min_countries_per_year', 5)
    years = sorted(df['Year'].unique())
    
    all_errors = []
    
    for year in years:
        year_results = run_loco_year(df, year, model, feature_cols, min_countries)
        if not year_results.empty:
            all_errors.extend(year_results['error'].tolist())
    
    if not all_errors:
        return float('inf'), float('inf')
    
    errors = np.array(all_errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    worst_case = np.max(np.abs(errors))
    
    return rmse, worst_case