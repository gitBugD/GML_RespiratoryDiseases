"""
Feature Selection for Delta-based COPD Incidence Prediction.

Custom forward selection with family removal:
1. Test each feature individually
2. Select best feature
3. Remove all features from the same family (same delta, any lag)
4. Repeat until no improvement

Functions:
- forward_selection_with_family: Main selection algorithm
- run_all_selections: Run 4 selections (2 models × 2 metrics)
- plot_selection_history: Visualize selection process
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.base import clone
from typing import Any

from delta_config import get_config, get_models
from delta_modeling import quick_loco_eval


def forward_selection_with_family(
    df: pd.DataFrame,
    feature_cols: list,
    feature_family_map: dict,
    model: Any,
    config: dict = None,
    optimize: str = "rmse",
    verbose: bool = True
) -> tuple[list, pd.DataFrame]:
    """
    Forward selection with family removal.
    
    At each step:
    1. Test each remaining feature INDIVIDUALLY (not cumulatively)
    2. Select the feature with best metric
    3. Remove ALL features from the same family
    4. Record history
    5. Stop when no improvement or no features left
    
    Args:
        df: Prepared DataFrame
        feature_cols: List of all available feature columns
        feature_family_map: Dict mapping feature_name -> family_name (e.g., delta_GDP)
        model: Model to use
        config: Configuration dict
        optimize: "rmse" or "worst_case"
        verbose: Print progress
        
    Returns:
        (selected_features, history_df)
    """
    if config is None:
        config = get_config()
    
    remaining = list(feature_cols)
    selected = []
    history = []
    
    step = 0
    best_metric_overall = float('inf')
    
    if verbose:
        print(f"Forward Selection (optimizing {optimize}, {len(remaining)} features)")
        print("-" * 60)
    
    while remaining:
        step += 1
        
        # Test each remaining feature individually
        candidates = []
        for feature in remaining:
            # Test with ONLY this feature (forward selection starts from scratch each step)
            test_features = selected + [feature]
            rmse, worst_case = quick_loco_eval(df, test_features, model, config)
            metric = rmse if optimize == "rmse" else worst_case
            
            candidates.append({
                'feature': feature,
                'family': feature_family_map.get(feature, feature),
                'rmse': rmse,
                'worst_case': worst_case,
                'metric': metric,
            })
        
        # Sort by metric (ascending = better)
        candidates.sort(key=lambda x: x['metric'])
        best = candidates[0]
        
        # Check for improvement
        if best['metric'] >= best_metric_overall and step > 1:
            if verbose:
                print(f"Step {step}: No improvement. Stopping.")
            break
        
        # Select best feature
        selected_feature = best['feature']
        selected_family = best['family']
        selected.append(selected_feature)
        best_metric_overall = best['metric']
        
        # Record history
        history.append({
            'step': step,
            'selected_feature': selected_feature,
            'family': selected_family,
            'n_selected': len(selected),
            'n_remaining': len(remaining),
            'rmse': best['rmse'],
            'worst_case': best['worst_case'],
        })
        
        if verbose:
            print(f"Step {step}: Selected '{selected_feature}'")
            print(f"         Family: {selected_family}")
            print(f"         RMSE={best['rmse']:.4f}, Worst={best['worst_case']:.4f}")
        
        # Remove all features from same family
        family_features = [f for f in remaining if feature_family_map.get(f, f) == selected_family]
        removed_count = len(family_features)
        remaining = [f for f in remaining if f not in family_features]
        
        if verbose and removed_count > 1:
            print(f"         Removed {removed_count} features from family '{selected_family}'")
        
        if not remaining:
            if verbose:
                print("No more features remaining.")
            break
    
    history_df = pd.DataFrame(history)
    
    if verbose:
        print("-" * 60)
        print(f"Final: {len(selected)} features selected")
        print(f"Best {optimize}: {best_metric_overall:.4f}")
    
    return selected, history_df


def run_all_selections(
    df: pd.DataFrame,
    feature_cols: list,
    feature_family_map: dict,
    config: dict = None,
    verbose: bool = True,
    export_dir: str = None,
    target_year: int = None
) -> dict:
    """
    Run 4 feature selections: (Ridge, XGBoost) × (RMSE, worst_case).
    
    Args:
        df: Prepared DataFrame
        feature_cols: All available features
        feature_family_map: Feature to family mapping
        config: Configuration dict
        verbose: Print progress
        export_dir: If provided, export results immediately to this directory
        target_year: Year being processed (for export filenames)
        
    Returns:
        Dict with results for each combination:
        {
            'Ridge_rmse': {'features': [...], 'history': DataFrame},
            'Ridge_worst_case': {...},
            'XGBoost_rmse': {...},
            'XGBoost_worst_case': {...},
        }
    """
    import os
    
    if config is None:
        config = get_config()
    
    # Use faster XGBoost for feature selection (fewer trees, shallower)
    models = {
        'Ridge': Ridge(random_state=config.get('random_state', 42)),
        'XGBoost': XGBRegressor(n_estimators=30, max_depth=3, random_state=42, verbosity=0),
    }
    
    metrics = ['rmse', 'worst_case']
    
    results = {}
    
    for model_name, model in models.items():
        for metric in metrics:
            key = f"{model_name}_{metric}"
            
            if verbose:
                print("\n" + "=" * 60)
                print(f"{key.upper()}")
                print("=" * 60)
            
            selected, history = forward_selection_with_family(
                df=df,
                feature_cols=feature_cols,
                feature_family_map=feature_family_map,
                model=model,
                config=config,
                optimize=metric,
                verbose=verbose
            )
            
            results[key] = {
                'features': selected,
                'history': history,
                'model_name': model_name,
                'metric': metric,
            }
            
            # Export immediately if export_dir provided
            if export_dir and target_year is not None:
                os.makedirs(export_dir, exist_ok=True)
                
                # Export history for this selection
                if not history.empty:
                    history_path = os.path.join(export_dir, f'fs_history_{target_year}_{key}.csv')
                    history.to_csv(history_path, index=False)
                    if verbose:
                        print(f"    ✓ Exported: {history_path}")
    
    # Export summary for this year
    if export_dir and target_year is not None:
        summary_rows = []
        for key, data in results.items():
            history = data.get('history', pd.DataFrame())
            final_rmse = history['rmse'].iloc[-1] if not history.empty and 'rmse' in history.columns else np.nan
            final_worst = history['worst_case'].iloc[-1] if not history.empty and 'worst_case' in history.columns else np.nan
            
            summary_rows.append({
                'target_year': target_year,
                'selection': key,
                'model': data['model_name'],
                'optimize_metric': data['metric'],
                'n_features': len(data['features']),
                'final_rmse': final_rmse,
                'final_worst_case': final_worst,
                'selected_features': ' | '.join(data['features']),
            })
        
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(export_dir, f'fs_summary_{target_year}.csv')
        summary_df.to_csv(summary_path, index=False)
        if verbose:
            print(f"  ✓ Exported year summary: {summary_path}")
    
    return results


def get_selection_summary(results: dict) -> pd.DataFrame:
    """
    Create summary DataFrame of all selection results.
    
    Args:
        results: Output from run_all_selections
        
    Returns:
        Summary DataFrame
    """
    rows = []
    for key, data in results.items():
        history = data['history']
        if not history.empty:
            final = history.iloc[-1]
            rows.append({
                'selection': key,
                'model': data['model_name'],
                'optimized_for': data['metric'],
                'n_features': len(data['features']),
                'final_rmse': final['rmse'],
                'final_worst_case': final['worst_case'],
                'features': ', '.join(data['features']),
            })
    
    return pd.DataFrame(rows)


def plot_selection_history(
    results: dict,
    figsize: tuple = (14, 10),
    save_path: str = None
):
    """
    Plot feature selection history for all 4 selections.
    
    Args:
        results: Output from run_all_selections
        figsize: Figure size
        save_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    for idx, (key, data) in enumerate(results.items()):
        ax = axes[idx]
        history = data['history']
        
        if history.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(key)
            continue
        
        # Plot both metrics
        ax.plot(history['step'], history['rmse'], 'b-o', label='RMSE', linewidth=2)
        ax.plot(history['step'], history['worst_case'], 'r-s', label='Worst Case', linewidth=2)
        
        # Highlight optimized metric
        metric = data['metric']
        if metric == 'rmse':
            ax.fill_between(history['step'], 0, history['rmse'], alpha=0.2, color='blue')
        else:
            ax.fill_between(history['step'], 0, history['worst_case'], alpha=0.2, color='red')
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Error')
        ax.set_title(f"{key}\n(Final: {len(data['features'])} features)")
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved plot to {save_path}")
    
    plt.show()


def plot_selected_features(
    results: dict,
    figsize: tuple = (12, 8),
    save_path: str = None
):
    """
    Plot which features were selected by each method.
    
    Args:
        results: Output from run_all_selections
        figsize: Figure size
        save_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt
    
    # Collect all unique features selected
    all_features = set()
    for data in results.values():
        all_features.update(data['features'])
    
    all_features = sorted(list(all_features))
    methods = list(results.keys())
    
    # Create selection matrix
    matrix = np.zeros((len(all_features), len(methods)))
    for j, method in enumerate(methods):
        for i, feature in enumerate(all_features):
            if feature in results[method]['features']:
                matrix[i, j] = 1
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(matrix, cmap='Blues', aspect='auto')
    
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_yticks(range(len(all_features)))
    ax.set_yticklabels([f[:40] + '...' if len(f) > 40 else f for f in all_features])
    
    ax.set_xlabel('Selection Method')
    ax.set_ylabel('Feature')
    ax.set_title('Feature Selection Results')
    
    plt.colorbar(im, ax=ax, label='Selected')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved plot to {save_path}")
    
    plt.show()
