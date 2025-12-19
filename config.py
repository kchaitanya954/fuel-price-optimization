"""
Configuration file for fuel price optimization system.
"""
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Data files
HISTORICAL_DATA_PATH = PROJECT_ROOT / "oil_retail_history.csv"
MODEL_PATH = MODELS_DIR / "price_optimization_model.pkl"
SCALER_PATH = MODELS_DIR / "feature_scaler.pkl"

# Model Parameters (tuned for better accuracy)
RANDOM_FOREST_N_ESTIMATORS = 500
RANDOM_FOREST_MAX_DEPTH = None
RANDOM_FOREST_MIN_SAMPLES_SPLIT = 4
RANDOM_FOREST_MIN_SAMPLES_LEAF = 2
RANDOM_FOREST_MAX_FEATURES = "sqrt"
RANDOM_FOREST_RANDOM_STATE = 16

# Price Search Range
# Slightly narrower range to avoid wild swings
PRICE_SEARCH_MIN_MULTIPLIER = 0.95  # 5% below current price
PRICE_SEARCH_MAX_MULTIPLIER = 1.05  # 5% above current price
PRICE_SEARCH_STEPS = 40  # Number of price points to evaluate

# Business Guardrails
MAX_PRICE_CHANGE_PERCENT = 5.0     # hard cap on daily change
MIN_PROFIT_MARGIN_PERCENT = 1.0    # minimum margin vs cost
COMPETITIVENESS_THRESHOLD = 0.5    # keep within 1% of avg competitor (soft)
