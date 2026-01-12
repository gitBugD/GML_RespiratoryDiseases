"""
Visualization utilities for Delta Analysis.

Functions for plotting RMSE, worst-case errors per country/year,
and exporting results.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from delta_config import ensure_output_dir


def plot_metrics_by_year(
    all_predictions: pd.DataFrame,
    metric: str = "rmse",
    figsize: tuple = (14, 6),
    save_path: str = None
):
    """
    Plot metric across all target years for each model.
    
    Args:
        all_predictions: DataFrame with all predictions (all years)
        metric: "rmse" or "worst_case"
        figsize: Figure size
        save_path: Optional save path
    """
    # Compute metrics by year and model
    results = []
    for (year, model), group in all_predictions.groupby(['Year', 'model']):
        errors = group['error'].values
        abs_errors = np.abs(errors)
        
        results.append({
            'Year': year,
            'model': model,
            'rmse': np.sqrt(np.mean(errors ** 2)),
            'worst_case': np.max(abs_errors),
            'mae': np.mean(abs_errors),
        })
    
    metrics_df = pd.DataFrame(results)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for model in metrics_df['model'].unique():
        model_data = metrics_df[metrics_df['model'] == model]
        ax.plot(model_data['Year'], model_data[metric], marker='o', 
                label=model, linewidth=2, markersize=8)
    
    ax.set_xlabel('Target Year')
    ax.set_ylabel(metric.upper() if metric != 'worst_case' else 'Worst Case |Error|')
    ax.set_title(f'{metric.upper()} per Target Year')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.show()
    
    return metrics_df


def plot_worst_case_by_country(
    predictions_df: pd.DataFrame,
    top_n: int = 20,
    figsize: tuple = (12, 10),
    save_path: str = None
):
    """
    Plot worst-case errors by country.
    
    Args:
        predictions_df: Predictions DataFrame
        top_n: Number of top countries to show
        figsize: Figure size
        save_path: Optional save path
    """
    # Compute worst-case per country and model
    results = []
    for (country, model), group in predictions_df.groupby(['Country', 'model']):
        abs_errors = np.abs(group['error'].values)
        
        results.append({
            'Country': country,
            'model': model,
            'worst_case': np.max(abs_errors),
            'rmse': np.sqrt(np.mean(group['error'].values ** 2)),
        })
    
    metrics_df = pd.DataFrame(results)
    
    # Get unique models
    models = metrics_df['model'].unique()
    n_models = len(models)
    
    fig, axes = plt.subplots(1, n_models, figsize=(figsize[0], figsize[1]))
    if n_models == 1:
        axes = [axes]
    
    for ax, model in zip(axes, models):
        model_data = metrics_df[metrics_df['model'] == model].nlargest(top_n, 'worst_case')
        
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(model_data)))
        ax.barh(range(len(model_data)), model_data['worst_case'].values, color=colors)
        ax.set_yticks(range(len(model_data)))
        ax.set_yticklabels(model_data['Country'].values)
        ax.set_xlabel('Worst Case |Error|')
        ax.set_title(f'{model} - Top {top_n} Countries')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.show()
    
    return metrics_df


def plot_actual_vs_predicted(
    predictions_df: pd.DataFrame,
    model_name: str = None,
    figsize: tuple = (10, 10),
    save_path: str = None
):
    """
    Scatter plot of actual vs predicted values.
    
    Args:
        predictions_df: Predictions DataFrame
        model_name: Specific model (or best if None)
        figsize: Figure size
        save_path: Optional save path
    """
    if model_name is None:
        # Select model with lowest RMSE
        model_rmse = predictions_df.groupby('model')['error'].apply(
            lambda x: np.sqrt(np.mean(x**2))
        )
        model_name = model_rmse.idxmin()
    
    model_preds = predictions_df[predictions_df['model'] == model_name]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    scatter = ax.scatter(
        model_preds['y_true'], 
        model_preds['y_pred'],
        c=model_preds['Year'],
        cmap='viridis',
        alpha=0.6,
        s=50
    )
    
    # Perfect prediction line
    min_val = min(model_preds['y_true'].min(), model_preds['y_pred'].min())
    max_val = max(model_preds['y_true'].max(), model_preds['y_pred'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
    
    ax.set_xlabel('Actual Δ Incidence')
    ax.set_ylabel('Predicted Δ Incidence')
    ax.set_title(f'Actual vs Predicted - {model_name}')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.colorbar(scatter, ax=ax, label='Target Year')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.show()


def plot_error_distribution(
    predictions_df: pd.DataFrame,
    figsize: tuple = (14, 5),
    save_path: str = None
):
    """
    Plot error distribution by model.
    
    Args:
        predictions_df: Predictions DataFrame
        figsize: Figure size
        save_path: Optional save path
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram
    for model in predictions_df['model'].unique():
        model_data = predictions_df[predictions_df['model'] == model]
        axes[0].hist(model_data['error'], bins=30, alpha=0.5, 
                    label=model, edgecolor='black')
    
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Error (Predicted - Actual)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Error Distribution')
    axes[0].legend()
    
    # Box plot
    predictions_df.boxplot(column='error', by='model', ax=axes[1])
    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('Error')
    axes[1].set_title('Error by Model')
    plt.suptitle('')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.show()


def export_predictions(
    predictions_df: pd.DataFrame,
    output_dir: str = None,
    filename: str = "predictions.csv"
) -> str:
    """
    Export predictions to CSV.
    
    Args:
        predictions_df: Predictions DataFrame
        output_dir: Output directory (default: from config)
        filename: Output filename
        
    Returns:
        Path to saved file
    """
    if output_dir is None:
        output_dir = ensure_output_dir()
    
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    predictions_df.to_csv(filepath, index=False)
    print(f"✓ Exported predictions: {filepath}")
    
    return filepath


def export_year_predictions(
    predictions_df: pd.DataFrame,
    target_year: int,
    output_dir: str = None
) -> str:
    """
    Export predictions for a specific target year.
    
    Args:
        predictions_df: Predictions DataFrame
        target_year: Year to export
        output_dir: Output directory
        
    Returns:
        Path to saved file
    """
    if output_dir is None:
        output_dir = ensure_output_dir()
    
    os.makedirs(output_dir, exist_ok=True)
    
    year_preds = predictions_df[predictions_df['Year'] == target_year]
    filename = f"predictions_year_{target_year}.csv"
    filepath = os.path.join(output_dir, filename)
    
    year_preds.to_csv(filepath, index=False)
    print(f"✓ Exported year {target_year} predictions: {filepath}")
    
    return filepath


def export_metrics_summary(
    all_predictions: pd.DataFrame,
    output_dir: str = None,
    filename: str = "metrics_summary.csv"
) -> str:
    """
    Export metrics summary across all years and models.
    
    Args:
        all_predictions: All predictions DataFrame
        output_dir: Output directory
        filename: Output filename
        
    Returns:
        Path to saved file
    """
    if output_dir is None:
        output_dir = ensure_output_dir()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute metrics
    results = []
    for (year, model), group in all_predictions.groupby(['Year', 'model']):
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
    
    metrics_df = pd.DataFrame(results)
    filepath = os.path.join(output_dir, filename)
    metrics_df.to_csv(filepath, index=False)
    
    print(f"✓ Exported metrics summary: {filepath}")
    
    return filepath


def create_full_report(
    all_predictions: pd.DataFrame,
    selection_results: dict = None,
    output_dir: str = None
):
    """
    Create full analysis report with all plots and CSVs.
    
    Args:
        all_predictions: All predictions DataFrame
        selection_results: Feature selection results (optional)
        output_dir: Output directory
    """
    if output_dir is None:
        output_dir = ensure_output_dir()
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("CREATING FULL REPORT")
    print("=" * 60)
    
    # Export CSVs
    export_predictions(all_predictions, output_dir, "all_predictions.csv")
    export_metrics_summary(all_predictions, output_dir, "metrics_summary.csv")
    
    # Export per-year predictions
    for year in all_predictions['Year'].unique():
        export_year_predictions(all_predictions, year, output_dir)
    
    # Generate plots
    print("\nGenerating plots...")
    
    plot_metrics_by_year(
        all_predictions, metric='rmse',
        save_path=os.path.join(output_dir, 'rmse_by_year.png')
    )
    
    plot_metrics_by_year(
        all_predictions, metric='worst_case',
        save_path=os.path.join(output_dir, 'worst_case_by_year.png')
    )
    
    plot_worst_case_by_country(
        all_predictions,
        save_path=os.path.join(output_dir, 'worst_case_by_country.png')
    )
    
    plot_actual_vs_predicted(
        all_predictions,
        save_path=os.path.join(output_dir, 'actual_vs_predicted.png')
    )
    
    plot_error_distribution(
        all_predictions,
        save_path=os.path.join(output_dir, 'error_distribution.png')
    )
    
    # Feature selection plots
    if selection_results:
        from delta_feature_selection import plot_selection_history, plot_selected_features
        
        plot_selection_history(
            selection_results,
            save_path=os.path.join(output_dir, 'feature_selection_history.png')
        )
        
        plot_selected_features(
            selection_results,
            save_path=os.path.join(output_dir, 'selected_features.png')
        )
    
    print("=" * 60)
    print(f"Report complete! Files saved to: {output_dir}")
    print("=" * 60)


# =============================================================================
# NEW EXPORT FUNCTIONS
# =============================================================================

def export_per_country_errors(
    predictions_df: pd.DataFrame,
    output_dir: str = None,
    filename: str = 'errors_by_country.csv'
) -> pd.DataFrame:
    """
    Export per-country error summary across all years.
    
    For each country and model:
    - Mean absolute error
    - RMSE
    - Worst-case |error|
    - Number of predictions
    - List of years
    
    Args:
        predictions_df: Predictions DataFrame
        output_dir: Output directory
        filename: Output filename
        
    Returns:
        DataFrame with per-country errors
    """
    if predictions_df.empty:
        return pd.DataFrame()
    
    if output_dir is None:
        output_dir = ensure_output_dir()
    
    os.makedirs(output_dir, exist_ok=True)
    
    rows = []
    for model in predictions_df['model'].unique():
        model_df = predictions_df[predictions_df['model'] == model]
        
        for country in model_df['Country'].unique():
            country_df = model_df[model_df['Country'] == country]
            
            errors = country_df['error'].values
            abs_errors = np.abs(errors)
            
            rows.append({
                'Country': country,
                'model': model,
                'n_predictions': len(country_df),
                'years': ', '.join(map(str, sorted(country_df['Year'].unique()))),
                'mae': np.mean(abs_errors),
                'rmse': np.sqrt(np.mean(errors ** 2)),
                'worst_case': np.max(abs_errors),
                'mean_error': np.mean(errors),  # signed, shows bias
                'std_error': np.std(errors),
            })
    
    df = pd.DataFrame(rows)
    df = df.sort_values(['model', 'worst_case'], ascending=[True, False])
    
    path = os.path.join(output_dir, filename)
    df.to_csv(path, index=False)
    print(f"✓ Exported per-country errors: {path}")
    
    return df


def export_per_year_per_country_errors(
    predictions_df: pd.DataFrame,
    output_dir: str = None,
    filename: str = 'errors_by_year_country.csv'
) -> pd.DataFrame:
    """
    Export detailed error for each year-country-model combination.
    
    Args:
        predictions_df: Predictions DataFrame
        output_dir: Output directory
        filename: Output filename
        
    Returns:
        DataFrame with detailed errors
    """
    if predictions_df.empty:
        return pd.DataFrame()
    
    if output_dir is None:
        output_dir = ensure_output_dir()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Already have this info, just reorganize
    cols = ['Year', 'Country', 'model', 'y_true', 'y_pred', 'error', 'abs_error']
    available_cols = [c for c in cols if c in predictions_df.columns]
    df = predictions_df[available_cols].copy()
    
    if 'abs_error' in df.columns:
        df = df.rename(columns={'abs_error': 'absolute_error'})
    
    df = df.sort_values(['model', 'Year', 'Country'])
    
    path = os.path.join(output_dir, filename)
    df.to_csv(path, index=False)
    print(f"✓ Exported year-country errors: {path}")
    
    return df


def export_selection_results(
    all_selection_results: dict,
    output_dir: str = None
) -> None:
    """
    Export detailed feature selection results for each year.
    
    Creates:
    - selected_features_by_year.csv: Which features were selected for each year/model/metric
    - selection_history_full.csv: Complete history of all selections
    
    Args:
        all_selection_results: Dict {year: {selection_key: {features, history, ...}}}
        output_dir: Output directory
    """
    if output_dir is None:
        output_dir = ensure_output_dir()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Selected features summary with metrics
    selected_features_rows = []
    for year, results in all_selection_results.items():
        for key, data in results.items():
            # key is like "Ridge_rmse" or "XGBoost_worst_case"
            parts = key.rsplit('_', 1)
            if len(parts) == 2:
                model, metric = parts[0], parts[1]
            else:
                model, metric = key, "unknown"
            
            features = data.get('features', [])
            history = data.get('history', pd.DataFrame())
            
            # Get final metrics from history
            final_rmse = np.nan
            final_worst = np.nan
            if not history.empty:
                final_rmse = history['rmse'].iloc[-1] if 'rmse' in history.columns else np.nan
                final_worst = history['worst_case'].iloc[-1] if 'worst_case' in history.columns else np.nan
            
            selected_features_rows.append({
                'target_year': year,
                'model': model,
                'optimize_metric': metric,
                'n_features_selected': len(features),
                'final_rmse': final_rmse,
                'final_worst_case': final_worst,
                'selected_features': ' | '.join(features),
            })
    
    if selected_features_rows:
        df = pd.DataFrame(selected_features_rows)
        path = os.path.join(output_dir, 'selected_features_by_year.csv')
        df.to_csv(path, index=False)
        print(f"✓ Exported selected features: {path}")
    
    # 2. Selection history (step by step)
    history_rows = []
    for year, results in all_selection_results.items():
        for key, data in results.items():
            parts = key.rsplit('_', 1)
            if len(parts) == 2:
                model, metric = parts[0], parts[1]
            else:
                model, metric = key, "unknown"
            
            history = data.get('history', pd.DataFrame())
            if not history.empty:
                for _, row in history.iterrows():
                    history_rows.append({
                        'target_year': year,
                        'model': model,
                        'optimize_metric': metric,
                        'step': row.get('step', np.nan),
                        'selected_feature': row.get('selected_feature', ''),
                        'family_removed': row.get('family', ''),
                        'n_selected': row.get('n_selected', np.nan),
                        'rmse': row.get('rmse', np.nan),
                        'worst_case': row.get('worst_case', np.nan),
                    })
    
    if history_rows:
        df = pd.DataFrame(history_rows)
        path = os.path.join(output_dir, 'selection_history_full.csv')
        df.to_csv(path, index=False)
        print(f"✓ Exported selection history: {path}")


def export_comprehensive_summary(
    year_summaries: list,
    all_selection_results: dict,
    output_dir: str = None
) -> pd.DataFrame:
    """
    Export comprehensive metrics summary combining LOCO results and feature selection.
    
    Args:
        year_summaries: List of dicts with year summary data
        all_selection_results: Feature selection results
        output_dir: Output directory
        
    Returns:
        DataFrame with comprehensive summary
    """
    if output_dir is None:
        output_dir = ensure_output_dir()
    
    os.makedirs(output_dir, exist_ok=True)
    
    rows = []
    
    for summary in year_summaries:
        year = summary['target_year']
        model = summary['model']
        
        row = {
            'target_year': year,
            'model': model,
            'n_countries': summary.get('n_countries', np.nan),
            'n_all_features': summary.get('n_features', np.nan),
            'all_features_rmse': summary.get('rmse', np.nan),
            'all_features_worst_case': summary.get('worst_case', np.nan),
            'all_features_mae': summary.get('mae', np.nan),
        }
        
        # Add feature selection results if available
        if year in all_selection_results:
            results = all_selection_results[year]
            
            # RMSE-optimized selection
            rmse_key = f"{model}_rmse"
            if rmse_key in results:
                history = results[rmse_key].get('history', pd.DataFrame())
                row['fs_rmse_n_features'] = len(results[rmse_key].get('features', []))
                if not history.empty:
                    row['fs_rmse_final_rmse'] = history['rmse'].iloc[-1] if 'rmse' in history.columns else np.nan
                    row['fs_rmse_final_worst'] = history['worst_case'].iloc[-1] if 'worst_case' in history.columns else np.nan
            
            # Worst-case-optimized selection
            worst_key = f"{model}_worst_case"
            if worst_key in results:
                history = results[worst_key].get('history', pd.DataFrame())
                row['fs_worst_n_features'] = len(results[worst_key].get('features', []))
                if not history.empty:
                    row['fs_worst_final_rmse'] = history['rmse'].iloc[-1] if 'rmse' in history.columns else np.nan
                    row['fs_worst_final_worst'] = history['worst_case'].iloc[-1] if 'worst_case' in history.columns else np.nan
        
        rows.append(row)
    
    if rows:
        df = pd.DataFrame(rows)
        path = os.path.join(output_dir, 'comprehensive_metrics_summary.csv')
        df.to_csv(path, index=False)
        print(f"✓ Exported comprehensive summary: {path}")
        return df
    
    return pd.DataFrame()
