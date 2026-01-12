"""
Feature Selection for COPD Incidence Prediction.

Backward elimination with two parallel optimization tracks:
1. Optimizing RMSE
2. Optimizing Worst-Case |error|
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.base import clone
from typing import Any

from config import get_config, MODELS
from modeling import quick_loco_eval


def backward_elimination(
    df: pd.DataFrame,
    feature_cols: list,
    config: dict = None,
    optimize: str = "rmse",
    model: Any = None,
    verbose: bool = True
) -> tuple[list, pd.DataFrame]:
    """
    Perform backward elimination to find optimal feature subset.
    
    At each step:
    1. Evaluate metric with each feature removed
    2. Remove the feature that improves metric the most
    3. If no improvement, remove the feature that hurts least
    4. Record history
    
    Args:
        df: Prepared DataFrame
        feature_cols: Starting list of feature columns
        config: Configuration dict
        optimize: "rmse" or "worst_case"
        model: Model to use (default: Ridge for speed)
        verbose: Print progress
    
    Returns:
        (optimal_features, history_df)
        history_df columns: step, removed_feature, n_features, rmse, worst_case
    """
    if config is None:
        config = get_config()
    
    if model is None:
        # Use Ridge for speed during feature selection
        model = Ridge(random_state=config.get('random_state', 42))
    
    remaining = list(feature_cols)
    history = []
    
    # Initial evaluation
    rmse, worst_case = quick_loco_eval(df, remaining, model, config)
    current_metric = rmse if optimize == "rmse" else worst_case
    
    history.append({
        'step': 0,
        'removed_feature': None,
        'n_features': len(remaining),
        'rmse': rmse,
        'worst_case': worst_case,
    })
    
    if verbose:
        print(f"Backward Elimination (optimizing {optimize})")
        print(f"Initial: {len(remaining)} features, RMSE={rmse:.4f}, Worst={worst_case:.4f}")
        print("-" * 60)
    
    best_metric_seen = current_metric
    best_features_seen = remaining.copy()
    
    step = 0
    while len(remaining) > 1:
        step += 1
        
        # Evaluate removing each feature
        candidates = []
        for feature in remaining:
            temp_features = [f for f in remaining if f != feature]
            rmse_temp, worst_temp = quick_loco_eval(df, temp_features, model, config)
            metric_temp = rmse_temp if optimize == "rmse" else worst_temp
            
            candidates.append({
                'feature': feature,
                'rmse': rmse_temp,
                'worst_case': worst_temp,
                'metric': metric_temp,
                'delta': current_metric - metric_temp,  # positive = improvement
            })
        
        # Sort by improvement (descending)
        candidates.sort(key=lambda x: x['delta'], reverse=True)
        best_candidate = candidates[0]
        
        # Remove the best candidate (most improvement or least degradation)
        removed = best_candidate['feature']
        remaining.remove(removed)
        
        rmse = best_candidate['rmse']
        worst_case = best_candidate['worst_case']
        new_metric = best_candidate['metric']
        
        history.append({
            'step': step,
            'removed_feature': removed,
            'n_features': len(remaining),
            'rmse': rmse,
            'worst_case': worst_case,
        })
        
        # Track best seen
        if new_metric < best_metric_seen:
            best_metric_seen = new_metric
            best_features_seen = remaining.copy()
        
        if verbose:
            improvement = "✓" if best_candidate['delta'] > 0 else "✗"
            print(f"Step {step}: Removed '{removed[:30]}...' {improvement}")
            print(f"         {len(remaining)} features, RMSE={rmse:.4f}, Worst={worst_case:.4f}")
        
        current_metric = new_metric
    
    # Convert history to DataFrame
    history_df = pd.DataFrame(history)
    
    if verbose:
        print("-" * 60)
        print(f"Best {optimize}: {best_metric_seen:.4f} with {len(best_features_seen)} features")
    
    return best_features_seen, history_df


def run_both_eliminations(
    df: pd.DataFrame,
    feature_cols: list,
    config: dict = None,
    model: Any = None,
    verbose: bool = True
) -> dict:
    """
    Run backward elimination for both RMSE and worst-case optimization.
    
    Args:
        df: Prepared DataFrame
        feature_cols: Starting feature columns
        config: Configuration dict
        model: Model to use
        verbose: Print progress
    
    Returns:
        {
            'rmse': {'features': [...], 'history': DataFrame},
            'worst_case': {'features': [...], 'history': DataFrame},
        }
    """
    results = {}
    
    if verbose:
        print("=" * 60)
        print("RMSE OPTIMIZATION")
        print("=" * 60)
    
    rmse_features, rmse_history = backward_elimination(
        df, feature_cols, config, optimize="rmse", model=model, verbose=verbose
    )
    results['rmse'] = {'features': rmse_features, 'history': rmse_history}
    
    if verbose:
        print("\n" + "=" * 60)
        print("WORST-CASE OPTIMIZATION")
        print("=" * 60)
    
    worst_features, worst_history = backward_elimination(
        df, feature_cols, config, optimize="worst_case", model=model, verbose=verbose
    )
    results['worst_case'] = {'features': worst_features, 'history': worst_history}
    
    return results


def find_optimal_subset(history_df: pd.DataFrame, optimize: str = "rmse") -> tuple[int, float]:
    """
    Find the step with optimal metric value from elimination history.
    
    Args:
        history_df: History from backward_elimination
        optimize: "rmse" or "worst_case"
    
    Returns:
        (optimal_step, optimal_metric_value)
    """
    metric_col = optimize if optimize in history_df.columns else 'rmse'
    optimal_idx = history_df[metric_col].idxmin()
    optimal_row = history_df.loc[optimal_idx]
    
    return int(optimal_row['step']), float(optimal_row[metric_col])


def forward_selection(
    df: pd.DataFrame,
    feature_cols: list,
    config: dict = None,
    optimize: str = "rmse",
    model: Any = None,
    verbose: bool = True,
    max_steps: int = 5
) -> tuple[list, pd.DataFrame]:
    """
    Perform forward selection to find optimal feature subset.
    
    At each step:
    1. Start with no features
    2. Add the feature that improves metric the most
    3. Continue until max_steps reached or all features are added
    4. Record history
    
    Args:
        df: Prepared DataFrame
        feature_cols: Available feature columns
        config: Configuration dict
        optimize: "rmse" or "worst_case"
        model: Model to use (default: Ridge for speed)
        verbose: Print progress
        max_steps: Maximum number of features to add (default: 5). Set to None for no limit.
    
    Returns:
        (optimal_features, history_df)
        history_df columns: step, added_feature, n_features, rmse, worst_case
    """
    if config is None:
        config = get_config()
    
    if model is None:
        model = Ridge(random_state=config.get('random_state', 42))
    
    available = list(feature_cols)
    selected = []
    history = []
    
    current_metric = float('inf')
    best_metric_seen = float('inf')
    best_features_seen = []
    
    if verbose:
        print(f"Forward Selection (optimizing {optimize})")
        max_info = f", max_steps={max_steps}" if max_steps else ""
        print(f"Starting: 0 features, {len(available)} available{max_info}")
        print("-" * 60)
    
    step = 0
    while available and (max_steps is None or step < max_steps):
        step += 1
        
        # Evaluate adding each feature
        candidates = []
        for feature in available:
            temp_features = selected + [feature]
            rmse_temp, worst_temp = quick_loco_eval(df, temp_features, model, config)
            metric_temp = rmse_temp if optimize == "rmse" else worst_temp
            
            candidates.append({
                'feature': feature,
                'rmse': rmse_temp,
                'worst_case': worst_temp,
                'metric': metric_temp,
                'improvement': current_metric - metric_temp,  # positive = improvement
            })
        
        # Sort by improvement (descending) - best improvement first
        candidates.sort(key=lambda x: x['improvement'], reverse=True)
        best_candidate = candidates[0]
        
        # Add the best candidate
        added = best_candidate['feature']
        selected.append(added)
        available.remove(added)
        
        rmse = best_candidate['rmse']
        worst_case = best_candidate['worst_case']
        new_metric = best_candidate['metric']
        
        history.append({
            'step': step,
            'added_feature': added,
            'n_features': len(selected),
            'rmse': rmse,
            'worst_case': worst_case,
        })
        
        # Track best seen
        if new_metric < best_metric_seen:
            best_metric_seen = new_metric
            best_features_seen = selected.copy()
        
        if verbose:
            improvement = "✓" if best_candidate['improvement'] > 0 else "✗"
            print(f"Step {step}: Added '{added[:30]}...' {improvement}")
            print(f"         {len(selected)} features, RMSE={rmse:.4f}, Worst={worst_case:.4f}")
        
        current_metric = new_metric
    
    # Convert history to DataFrame
    history_df = pd.DataFrame(history)
    
    if verbose:
        print("-" * 60)
        print(f"Best {optimize}: {best_metric_seen:.4f} with {len(best_features_seen)} features")
    
    return best_features_seen, history_df


def run_forward_selections(
    df: pd.DataFrame,
    feature_cols: list,
    config: dict = None,
    model: Any = None,
    verbose: bool = True
) -> dict:
    """
    Run forward selection for both RMSE and worst-case optimization.
    
    Returns:
        {
            'rmse': {'features': [...], 'history': DataFrame},
            'worst_case': {'features': [...], 'history': DataFrame},
        }
    """
    results = {}
    
    if verbose:
        print("=" * 60)
        print("FORWARD SELECTION - RMSE OPTIMIZATION")
        print("=" * 60)
    
    rmse_features, rmse_history = forward_selection(
        df, feature_cols, config, optimize="rmse", model=model, verbose=verbose
    )
    results['rmse'] = {'features': rmse_features, 'history': rmse_history}
    
    if verbose:
        print("\n" + "=" * 60)
        print("FORWARD SELECTION - WORST-CASE OPTIMIZATION")
        print("=" * 60)
    
    worst_features, worst_history = forward_selection(
        df, feature_cols, config, optimize="worst_case", model=model, verbose=verbose
    )
    results['worst_case'] = {'features': worst_features, 'history': worst_history}
    
    return results


def plot_forward_selection_history(results: dict, figsize: tuple = (14, 6)):
    """
    Plot forward selection history for both RMSE and worst-case optimization.
    
    Args:
        results: Output from run_forward_selections()
        figsize: Figure size tuple
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot RMSE optimization
    rmse_history = results['rmse']['history']
    
    axes[0].plot(range(len(rmse_history)), rmse_history['rmse'].values, 
                 marker='o', linewidth=2, color='steelblue')
    axes[0].set_xticks(range(len(rmse_history)))
    labels = [f[:15] + '...' if len(f) > 15 else f for f in rmse_history['added_feature'].values]
    axes[0].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    axes[0].set_xlabel('Feature Added', fontsize=11)
    axes[0].set_ylabel('RMSE', fontsize=11)
    axes[0].set_title('Forward Selection - RMSE Optimization', fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    # Mark optimal point
    optimal_idx = rmse_history['rmse'].idxmin()
    axes[0].scatter([optimal_idx], [rmse_history.loc[optimal_idx, 'rmse']], 
                    color='red', s=150, zorder=5, label='Optimal')
    axes[0].legend()
    
    # Plot Worst-case optimization
    worst_history = results['worst_case']['history']
    
    axes[1].plot(range(len(worst_history)), worst_history['worst_case'].values, 
                 marker='o', linewidth=2, color='coral')
    axes[1].set_xticks(range(len(worst_history)))
    labels = [f[:15] + '...' if len(f) > 15 else f for f in worst_history['added_feature'].values]
    axes[1].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    axes[1].set_xlabel('Feature Added', fontsize=11)
    axes[1].set_ylabel('Worst-Case |Error|', fontsize=11)
    axes[1].set_title('Forward Selection - Worst-Case Optimization', fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    # Mark optimal point
    optimal_idx = worst_history['worst_case'].idxmin()
    axes[1].scatter([optimal_idx], [worst_history.loc[optimal_idx, 'worst_case']], 
                    color='red', s=150, zorder=5, label='Optimal')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    return fig


def plot_elimination_history(results: dict, figsize: tuple = (14, 6)):
    """
    Plot feature elimination history for both RMSE and worst-case optimization.
    
    Args:
        results: Output from run_both_eliminations()
        figsize: Figure size tuple
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot RMSE optimization
    rmse_history = results['rmse']['history']
    # Skip step 0 (no feature removed)
    rmse_plot = rmse_history[rmse_history['step'] > 0].copy()
    
    axes[0].plot(range(len(rmse_plot)), rmse_plot['rmse'].values, 
                 marker='o', linewidth=2, color='steelblue')
    axes[0].set_xticks(range(len(rmse_plot)))
    # Truncate feature names for readability
    labels = [f[:15] + '...' if len(f) > 15 else f for f in rmse_plot['removed_feature'].values]
    axes[0].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    axes[0].set_xlabel('Feature Removed', fontsize=11)
    axes[0].set_ylabel('RMSE', fontsize=11)
    axes[0].set_title('Backward Elimination - RMSE Optimization', fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    # Mark optimal point
    optimal_idx = rmse_plot['rmse'].idxmin()
    optimal_step = rmse_plot.index.get_loc(optimal_idx)
    axes[0].scatter([optimal_step], [rmse_plot.loc[optimal_idx, 'rmse']], 
                    color='red', s=150, zorder=5, label='Optimal')
    axes[0].legend()
    
    # Plot Worst-case optimization
    worst_history = results['worst_case']['history']
    worst_plot = worst_history[worst_history['step'] > 0].copy()
    
    axes[1].plot(range(len(worst_plot)), worst_plot['worst_case'].values, 
                 marker='o', linewidth=2, color='coral')
    axes[1].set_xticks(range(len(worst_plot)))
    labels = [f[:15] + '...' if len(f) > 15 else f for f in worst_plot['removed_feature'].values]
    axes[1].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    axes[1].set_xlabel('Feature Removed', fontsize=11)
    axes[1].set_ylabel('Worst-Case |Error|', fontsize=11)
    axes[1].set_title('Backward Elimination - Worst-Case Optimization', fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    # Mark optimal point
    optimal_idx = worst_plot['worst_case'].idxmin()
    optimal_step = worst_plot.index.get_loc(optimal_idx)
    axes[1].scatter([optimal_step], [worst_plot.loc[optimal_idx, 'worst_case']], 
                    color='red', s=150, zorder=5, label='Optimal')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    return fig


# =============================================================================
# MULTI-MODEL FEATURE SELECTION
# =============================================================================

def multi_model_backward_elimination(
    df: pd.DataFrame,
    feature_cols: list,
    config: dict = None,
    optimize: str = "rmse",
    verbose: bool = True
) -> dict:
    """
    Run backward elimination for all models in MODELS.
    
    Args:
        df: Prepared DataFrame
        feature_cols: Starting list of feature columns
        config: Configuration dict
        optimize: "rmse" or "worst_case"
        verbose: Print progress
    
    Returns:
        Dictionary with model names as keys:
        {
            'LinearRegression': {'features': [...], 'history': DataFrame},
            'Ridge': {'features': [...], 'history': DataFrame},
            ...
        }
    """
    if config is None:
        config = get_config()
    
    results = {}
    
    for model_name, model_instance in MODELS.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f"BACKWARD ELIMINATION - {model_name} ({optimize} optimization)")
            print(f"{'='*60}")
        
        # Clone the model instance to get a fresh copy
        model = clone(model_instance)
        
        features, history = backward_elimination(
            df=df,
            feature_cols=feature_cols,
            config=config,
            optimize=optimize,
            model=model,
            verbose=verbose
        )
        
        results[model_name] = {
            'features': features,
            'history': history
        }
    
    return results


def multi_model_forward_selection(
    df: pd.DataFrame,
    feature_cols: list,
    config: dict = None,
    optimize: str = "rmse",
    verbose: bool = True,
    max_steps: int = 5
) -> dict:
    """
    Run forward selection for all models in MODELS.
    
    Args:
        df: Prepared DataFrame
        feature_cols: Available feature columns
        config: Configuration dict
        optimize: "rmse" or "worst_case"
        verbose: Print progress
        max_steps: Maximum number of features to add (default: 5). Set to None for no limit.
    
    Returns:
        Dictionary with model names as keys:
        {
            'LinearRegression': {'features': [...], 'history': DataFrame},
            'Ridge': {'features': [...], 'history': DataFrame},
            ...
        }
    """
    if config is None:
        config = get_config()
    
    results = {}
    
    for model_name, model_instance in MODELS.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f"FORWARD SELECTION - {model_name} ({optimize} optimization)")
            print(f"{'='*60}")
        
        # Clone the model instance to get a fresh copy
        model = clone(model_instance)
        
        features, history = forward_selection(
            df=df,
            feature_cols=feature_cols,
            config=config,
            optimize=optimize,
            model=model,
            verbose=verbose,
            max_steps=max_steps
        )
        
        results[model_name] = {
            'features': features,
            'history': history
        }
    
    return results


def plot_multi_model_elimination(
    results: dict,
    metric: str = "rmse",
    title: str = "Multi-Model Backward Elimination",
    figsize: tuple = (12, 6)
):
    """
    Plot backward elimination history for all models on same plot.
    
    Args:
        results: Output from multi_model_backward_elimination()
        metric: "rmse" or "worst_case"
        title: Plot title
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['steelblue', 'coral', 'forestgreen', 'mediumpurple', 'goldenrod']
    markers = ['o', 's', '^', 'D', 'v']
    
    for idx, (model_name, data) in enumerate(results.items()):
        history = data['history']
        # Skip step 0
        plot_data = history[history['step'] > 0].copy()
        
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        x = range(len(plot_data))
        y = plot_data[metric].values
        
        ax.plot(x, y, marker=marker, linewidth=2, color=color, 
                label=model_name, markersize=6, alpha=0.8)
        
        # Find and mark optimal point for this model
        optimal_idx = plot_data[metric].idxmin()
        optimal_step = plot_data.index.get_loc(optimal_idx)
        optimal_val = plot_data.loc[optimal_idx, metric]
        
        ax.scatter([optimal_step], [optimal_val], color=color, s=200, 
                   zorder=5, edgecolor='black', linewidth=2)
        ax.annotate(f'{optimal_val:.4f}', (optimal_step, optimal_val),
                   textcoords="offset points", xytext=(0, 10),
                   ha='center', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Elimination Step', fontsize=11)
    ax.set_ylabel('RMSE' if metric == 'rmse' else 'Worst-Case |Error|', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    return fig


def plot_multi_model_forward(
    results: dict,
    metric: str = "rmse",
    title: str = "Multi-Model Forward Selection",
    figsize: tuple = (12, 6)
):
    """
    Plot forward selection history for all models on same plot.
    
    Args:
        results: Output from multi_model_forward_selection()
        metric: "rmse" or "worst_case"
        title: Plot title
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['steelblue', 'coral', 'forestgreen', 'mediumpurple', 'goldenrod']
    markers = ['o', 's', '^', 'D', 'v']
    
    for idx, (model_name, data) in enumerate(results.items()):
        history = data['history']
        
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        x = range(len(history))
        y = history[metric].values
        
        ax.plot(x, y, marker=marker, linewidth=2, color=color, 
                label=model_name, markersize=6, alpha=0.8)
        
        # Find and mark optimal point for this model
        optimal_idx = history[metric].idxmin()
        optimal_val = history.loc[optimal_idx, metric]
        
        ax.scatter([optimal_idx], [optimal_val], color=color, s=200, 
                   zorder=5, edgecolor='black', linewidth=2)
        ax.annotate(f'{optimal_val:.4f}', (optimal_idx, optimal_val),
                   textcoords="offset points", xytext=(0, 10),
                   ha='center', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Selection Step (Features Added)', fontsize=11)
    ax.set_ylabel('RMSE' if metric == 'rmse' else 'Worst-Case |Error|', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    return fig


def compare_selection_methods(
    df: pd.DataFrame,
    feature_cols: list,
    config: dict = None,
    optimize: str = "rmse",
    verbose: bool = False
) -> tuple:
    """
    Compare forward selection vs backward elimination for all models.
    
    Returns a summary DataFrame with optimal metrics for each method/model.
    """
    if config is None:
        config = get_config()
    
    print("Running multi-model backward elimination...")
    backward_results = multi_model_backward_elimination(
        df, feature_cols, config, optimize=optimize, verbose=verbose
    )
    
    print("\nRunning multi-model forward selection...")
    forward_results = multi_model_forward_selection(
        df, feature_cols, config, optimize=optimize, verbose=verbose
    )
    
    # Build comparison table
    rows = []
    for model_name in MODELS.keys():
        # Backward
        bh = backward_results[model_name]['history']
        b_opt_idx = bh[optimize].idxmin()
        b_opt_metric = bh.loc[b_opt_idx, optimize]
        b_n_features = bh.loc[b_opt_idx, 'n_features']
        
        # Forward
        fh = forward_results[model_name]['history']
        f_opt_idx = fh[optimize].idxmin()
        f_opt_metric = fh.loc[f_opt_idx, optimize]
        f_n_features = fh.loc[f_opt_idx, 'n_features']
        
        rows.append({
            'Model': model_name,
            'Backward - Optimal Metric': b_opt_metric,
            'Backward - N Features': b_n_features,
            'Forward - Optimal Metric': f_opt_metric,
            'Forward - N Features': f_n_features,
            'Best Method': 'Backward' if b_opt_metric < f_opt_metric else 'Forward'
        })
    
    comparison_df = pd.DataFrame(rows)
    
    return comparison_df, backward_results, forward_results