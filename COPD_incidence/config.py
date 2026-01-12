"""
Configuration for COPD Incidence Prediction.
"""
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# =============================================================================
# PATHS
# =============================================================================
import os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(_THIS_DIR, "..", "Data")  # Points to Data folder
VERSION = "9019"  # "1021" or "9019"

# =============================================================================
# MODELS
# =============================================================================
MODELS = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
}

# =============================================================================
# FEATURES TO DROP (Data Leakage Prevention)
# =============================================================================
# Metadata columns (identifiers, target)
METADATA_COLS = ["Country Code", "Country Name", "Year", "Value", "Disease", "Measure", "Metric"]

# Proxy identifiers (unique per country = leakage)
PROXY_IDENTIFIER_COLS = [
    '''
    "Surface area (sq. km)",
    "Total area (Square Km)",
    "Population, total",
    "GDP (current US$)",
    '''
]

# Redundant/correlated features
REDUNDANT_COLS = [
    '''
    "d2m",   # ~1.0 correlation with t2m
    "skt",   # ~1.0 correlation with t2m
    '''
]

# All columns to drop before training
COLS_TO_DROP = METADATA_COLS + PROXY_IDENTIFIER_COLS + REDUNDANT_COLS

# =============================================================================
# MAIN CONFIGURATION
# =============================================================================
CONFIG = {
    # Target mode: "incidence" or "delta_incidence"
    "target_mode": "delta_incidence",
    
    # Exclusions
    "exclude_years": [],           # e.g., [2020, 2021]
    "exclude_countries": [],       # e.g., ["USA", "CHN"]
    "exclude_features": [],        # Additional features to exclude
    
    # Minimum sample requirements
    "min_countries_per_year": 5,
    
    # NaN handling: drop columns with NaN ratio above this threshold
    "nan_threshold": 0.5,
    
    # Random seed for reproducibility
    "random_state": 42,
    
    # Models to evaluate
    "models": MODELS,
    
    # Output directory
    "output_dir": "outputs",
}


def get_config():
    """Return a copy of the config dict (safe for modification)."""
    return CONFIG.copy()


def get_cols_to_drop():
    """Return list of all columns to drop before training."""
    return COLS_TO_DROP.copy()
