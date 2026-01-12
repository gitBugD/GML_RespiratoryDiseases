"""
Configuration for Delta-based COPD Incidence Prediction.
"""
import os
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

# =============================================================================
# PATHS
# =============================================================================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(_THIS_DIR, "..", "..", "Data")  # Points to Data folder
VERSION = "9019"  # "1021" or "9019"
OUTPUT_DIR = os.path.join(_THIS_DIR, "outputs")

# =============================================================================
# MODELS (Only Ridge and XGBoost as per specs)
# =============================================================================
MODELS = {
    "Ridge": Ridge(random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
}

# =============================================================================
# FEATURES TO DROP (Data Leakage Prevention)
# =============================================================================
METADATA_COLS = ["Country Code", "Country Name", "Year", "Value", "Disease", "Measure", "Metric"]

# =============================================================================
# MAIN CONFIGURATION
# =============================================================================
CONFIG = {
    # Target mode: always delta_incidence for this analysis
    "target_mode": "delta_incidence",
    
    # Exclusions
    "exclude_years": [],
    "exclude_countries": [],
    "exclude_features": [],
    
    # Minimum sample requirements
    "min_countries_per_year": 5,
    
    # NaN handling
    "nan_threshold": 0.5,
    
    # Random seed
    "random_state": 42,
    
    # Models to evaluate
    "models": MODELS,
    
    # Output directory
    "output_dir": OUTPUT_DIR,
}


def get_config():
    """Return a copy of the config dict."""
    return CONFIG.copy()


def get_models():
    """Return models dict."""
    return MODELS.copy()


def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR
