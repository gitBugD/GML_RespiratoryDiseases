"""
LOCO Modeling for Delta-based COPD Incidence Prediction.

Functions:
- run_loco_single_year: LOCO CV for a single target year
- compute_metrics: Calculate RMSE, worst-case, MAE
- compute_metrics_by_country: Metrics per country
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from typing import Any

from delta_config import get_config, get_models


def run_loco_single_year(
    df: pd.DataFrame,
    feature_cols: list,
    model: Any,
    model_name: str,
    min_countries: int = 5
) -> pd.DataFrame:
    """
    Run Leave-One-Country-Out cross-validation for a single year dataset.
    
    For each country:
    - Train on all OTHER countries
    - Test on this single country
    - Record prediction
    
    Args:
        df: Prepared DataFrame with 'target' and 'target_year' columns
        feature_cols: List of feature column names
        model: Sklearn-compatible model
        model_name: Name of the model
        min_countries: Minimum countries required
        
    Returns:
        DataFrame with columns: Year, Country, y_true, y_pred, error, model
    """
    if len(df) < min_countries:
        print(f"  ⚠ Only {len(df)} countries, skipping (min={min_countries})")
        return pd.DataFrame()
    
    countries = df['Country Code'].unique()
    target_year = df['target_year'].iloc[0] if 'target_year' in df.columns else None
    
    results = []
    
    for country in countries:
        # Split: test = this country, train = others
        test_mask = df['Country Code'] == country
        train_mask = ~test_mask
        
        if train_mask.sum() < 2:
            continue
        
        X_train = df.loc[train_mask, feature_cols].values
        y_train = df.loc[train_mask, 'target'].values
        X_test = df.loc[test_mask, feature_cols].values
        y_test = df.loc[test_mask, 'target'].values
        
        # Standardize features (fit on train only)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Clone and fit model
        model_clone = clone(model)
        model_clone.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model_clone.predict(X_test_scaled)
        
        # Record results
        for yt, yp in zip(y_test, y_pred):
            results.append({
                'Year': target_year,
                'Country': country,
                'y_true': yt,
                'y_pred': yp,
                'error': yp - yt,
                'abs_error': abs(yp - yt),
                'model': model_name,
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
    Run full LOCO evaluation for all models.
    
    Args:
        df: Prepared DataFrame
        feature_cols: Feature column names
        config: Configuration dict
        model_name: Specific model to run (or all if None)
        verbose: Print progress
        
    Returns:
        (predictions_df, metrics_dict)
    """
    if config is None:
        config = get_config()
    
    models = config.get('models', get_models())
    min_countries = config.get('min_countries_per_year', 5)
    
    if model_name:
        models = {model_name: models[model_name]}
    
    all_results = []
    
    for name, model in models.items():
        if verbose:
            print(f"  ▶ Running LOCO for {name}...")
        
        results = run_loco_single_year(
            df, feature_cols, model, name, min_countries
        )
        if not results.empty:
            all_results.append(results)
    
    if not all_results:
        return pd.DataFrame(), {}
    
    predictions_df = pd.concat(all_results, ignore_index=True)
    metrics = compute_metrics(predictions_df)
    
    if verbose:
        print("\n  Results:")
        for mname, m in metrics.items():
            print(f"    {mname}: RMSE={m['rmse']:.4f}, Worst={m['worst_case']:.4f}")
    
    return predictions_df, metrics


def compute_metrics(predictions_df: pd.DataFrame) -> dict:
    """
    Compute RMSE and worst-case error per model.
    
    Args:
        predictions_df: DataFrame with model, error columns
        
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
            'n_predictions': len(group),
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
    
    if len(df) < min_countries or not feature_cols:
        return float('inf'), float('inf')
    
    countries = df['Country Code'].unique()
    all_errors = []
    
    for country in countries:
        test_mask = df['Country Code'] == country
        train_mask = ~test_mask
        
        if train_mask.sum() < 2:
            continue
        
        X_train = df.loc[train_mask, feature_cols].values
        y_train = df.loc[train_mask, 'target'].values
        X_test = df.loc[test_mask, feature_cols].values
        y_test = df.loc[test_mask, 'target'].values
        
        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Clone, fit, predict
        model_clone = clone(model)
        model_clone.fit(X_train_scaled, y_train)
        y_pred = model_clone.predict(X_test_scaled)
        
        # Collect errors
        all_errors.extend((y_pred - y_test).tolist())
    
    if not all_errors:
        return float('inf'), float('inf')
    
    errors = np.array(all_errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    worst_case = np.max(np.abs(errors))
    
    return rmse, worst_case
