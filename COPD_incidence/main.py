"""
Main orchestration script for COPD Incidence Prediction.

Usage:
    python main.py                    # Run with default config
    python main.py --feature-select   # Also run backward elimination
"""
import json
import os
from datetime import datetime
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import BASE_PATH, VERSION, get_config, get_cols_to_drop
from data_prep import prepare_data, get_feature_columns
from modeling import run_full_loco, compute_metrics_by_year, compute_metrics_by_country
from feature_selection import run_both_eliminations


def create_output_dir(base_dir: str = "outputs") -> str:
    """Create timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    return output_dir


def save_config(config: dict, output_dir: str):
    """Save configuration to JSON (excluding non-serializable objects)."""
    # Create serializable version
    config_serializable = {k: v for k, v in config.items() if k != 'models'}
    config_serializable['models'] = list(config.get('models', {}).keys())
    config_serializable['cols_dropped'] = get_cols_to_drop()
    config_serializable['base_path'] = BASE_PATH
    config_serializable['version'] = VERSION
    config_serializable['timestamp'] = datetime.now().isoformat()
    
    with open(os.path.join(output_dir, "config.json"), 'w') as f:
        json.dump(config_serializable, f, indent=2)


def save_metrics(metrics: dict, output_dir: str):
    """Save metrics summary to JSON."""
    # Convert numpy types to Python types
    metrics_clean = {}
    for model, m in metrics.items():
        metrics_clean[model] = {k: float(v) for k, v in m.items()}
    
    with open(os.path.join(output_dir, "metrics_summary.json"), 'w') as f:
        json.dump(metrics_clean, f, indent=2)


def plot_error_heatmap(predictions_df: pd.DataFrame, output_dir: str, model_name: str = None):
    """Plot heatmap of |error| per (year, country)."""
    if model_name:
        df = predictions_df[predictions_df['model'] == model_name]
        title = f"Absolute Error Heatmap - {model_name}"
        filename = f"heatmap_errors_{model_name}.png"
    else:
        # Use first model
        model_name = predictions_df['model'].iloc[0]
        df = predictions_df[predictions_df['model'] == model_name]
        title = f"Absolute Error Heatmap - {model_name}"
        filename = "heatmap_errors.png"
    
    # Pivot for heatmap
    pivot = df.pivot_table(values='abs_error', index='Country', columns='Year', aggfunc='mean')
    
    # Limit to reasonable size
    if len(pivot) > 30:
        # Show top 30 countries by mean error
        top_countries = pivot.mean(axis=1).nlargest(30).index
        pivot = pivot.loc[top_countries]
    
    plt.figure(figsize=(14, max(8, len(pivot) * 0.3)))
    sns.heatmap(pivot, cmap='YlOrRd', annot=False, fmt='.2f', cbar_kws={'label': '|Error|'})
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Year')
    plt.ylabel('Country')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", filename), dpi=150)
    plt.close()


def plot_rmse_per_year(predictions_df: pd.DataFrame, output_dir: str):
    """Plot RMSE per year for all models."""
    metrics_by_year = compute_metrics_by_year(predictions_df)
    
    plt.figure(figsize=(12, 6))
    for model in metrics_by_year['model'].unique():
        model_data = metrics_by_year[metrics_by_year['model'] == model]
        plt.plot(model_data['Year'], model_data['rmse'], marker='o', label=model, linewidth=2)
    
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.title('RMSE per Year by Model', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", "rmse_per_year.png"), dpi=150)
    plt.close()


def plot_worstcase_per_country(predictions_df: pd.DataFrame, output_dir: str, model_name: str = None):
    """Plot worst-case error per country."""
    if model_name is None:
        model_name = predictions_df['model'].iloc[0]
    
    df = predictions_df[predictions_df['model'] == model_name]
    metrics_by_country = compute_metrics_by_country(df)
    
    # Sort by worst-case
    metrics_by_country = metrics_by_country.sort_values('worst_case', ascending=False)
    
    # Show top 20
    top_20 = metrics_by_country.head(20)
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(top_20)), top_20['worst_case'].values, color='coral', edgecolor='black')
    plt.yticks(range(len(top_20)), top_20['Country'].values)
    plt.xlabel('Worst-Case |Error|', fontsize=12)
    plt.ylabel('Country', fontsize=12)
    plt.title(f'Top 20 Countries by Worst-Case Error - {model_name}', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", "worstcase_per_country.png"), dpi=150)
    plt.close()


def plot_elimination_history(history_df: pd.DataFrame, optimize: str, output_dir: str):
    """Plot feature elimination history."""
    metric_col = optimize if optimize in ['rmse', 'worst_case'] else 'rmse'
    
    plt.figure(figsize=(12, 6))
    plt.plot(history_df['step'], history_df[metric_col], marker='o', linewidth=2, color='steelblue')
    
    # Mark optimal point
    optimal_idx = history_df[metric_col].idxmin()
    optimal = history_df.loc[optimal_idx]
    plt.scatter([optimal['step']], [optimal[metric_col]], color='red', s=200, zorder=5, label='Optimal')
    
    plt.xlabel('Elimination Step', fontsize=12)
    plt.ylabel(metric_col.upper(), fontsize=12)
    plt.title(f'Backward Elimination - Optimizing {optimize.upper()}', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", f"elimination_{optimize}.png"), dpi=150)
    plt.close()


def main(run_feature_selection: bool = False):
    """Main execution pipeline."""
    print("=" * 70)
    print("COPD INCIDENCE PREDICTION - LOCO EVALUATION")
    print("=" * 70)
    
    # Get config
    config = get_config()
    
    # Create output directory
    output_dir = create_output_dir(config.get('output_dir', 'outputs'))
    print(f"Output directory: {output_dir}")
    
    # Save config
    save_config(config, output_dir)
    
    # Prepare data
    df, feature_cols = prepare_data(BASE_PATH, VERSION, config)
    
    # Save feature list
    with open(os.path.join(output_dir, "features_used.txt"), 'w') as f:
        for col in feature_cols:
            f.write(f"{col}\n")
    
    # Run full LOCO evaluation
    predictions_df, metrics = run_full_loco(df, feature_cols, config, verbose=True)
    
    if predictions_df.empty:
        print("ERROR: No predictions generated. Check data and config.")
        return
    
    # Save predictions
    predictions_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
    
    # Save metrics
    save_metrics(metrics, output_dir)
    
    # Generate plots
    print("\nGenerating plots...")
    
    for model_name in predictions_df['model'].unique():
        plot_error_heatmap(predictions_df, output_dir, model_name)
    
    plot_rmse_per_year(predictions_df, output_dir)
    plot_worstcase_per_country(predictions_df, output_dir)
    
    # Feature selection (optional)
    if run_feature_selection:
        print("\n" + "=" * 70)
        print("FEATURE SELECTION (Backward Elimination)")
        print("=" * 70)
        
        results = run_both_eliminations(df, feature_cols, config, verbose=True)
        
        # Save histories
        results['rmse']['history'].to_csv(
            os.path.join(output_dir, "elimination_rmse_history.csv"), index=False
        )
        results['worst_case']['history'].to_csv(
            os.path.join(output_dir, "elimination_worstcase_history.csv"), index=False
        )
        
        # Save optimal features
        with open(os.path.join(output_dir, "optimal_features_rmse.txt"), 'w') as f:
            for col in results['rmse']['features']:
                f.write(f"{col}\n")
        
        with open(os.path.join(output_dir, "optimal_features_worstcase.txt"), 'w') as f:
            for col in results['worst_case']['features']:
                f.write(f"{col}\n")
        
        # Plot elimination histories
        plot_elimination_history(results['rmse']['history'], 'rmse', output_dir)
        plot_elimination_history(results['worst_case']['history'], 'worst_case', output_dir)
    
    print("\n" + "=" * 70)
    print(f"✓ Results saved to: {output_dir}")
    print("=" * 70)
