"""
Random Forest model for price optimization.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
from typing import Dict, List, Optional
import logging
from config import (
    MODEL_PATH, SCALER_PATH,
    RANDOM_FOREST_N_ESTIMATORS, RANDOM_FOREST_MAX_DEPTH,
    RANDOM_FOREST_MIN_SAMPLES_SPLIT, RANDOM_FOREST_RANDOM_STATE,
    PRICE_SEARCH_MIN_MULTIPLIER, PRICE_SEARCH_MAX_MULTIPLIER, PRICE_SEARCH_STEPS,
    MAX_PRICE_CHANGE_PERCENT, MIN_PROFIT_MARGIN_PERCENT, COMPETITIVENESS_THRESHOLD,
    RANDOM_FOREST_MIN_SAMPLES_LEAF, RANDOM_FOREST_MAX_FEATURES,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PriceOptimizationModel:
    """Random Forest model for predicting volume and optimizing price."""
    
    def __init__(self):
        self.model: Optional[RandomForestRegressor] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_columns: List[str] = []
        self.is_trained = False
    
    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Dict:
        """Train the Random Forest model using a time-based split (no shuffling)."""
        logger.info("Training model...")
        
        self.feature_columns = list(X.columns)
        
        # Time-series split: oldest data for training, most recent for testing
        n_samples = len(X)
        n_test = int(n_samples * test_size)
        n_train = n_samples - n_test
        if n_train <= 0 or n_test <= 0:
            raise ValueError("Not enough samples to split into train and test sets.")

        X_train = X.iloc[:n_train].copy()
        y_train = y.iloc[:n_train].copy()
        X_test = X.iloc[n_train:].copy()
        y_test = y.iloc[n_train:].copy()
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=RANDOM_FOREST_N_ESTIMATORS,
            max_depth=RANDOM_FOREST_MAX_DEPTH,
            min_samples_split=RANDOM_FOREST_MIN_SAMPLES_SPLIT,
            min_samples_leaf=RANDOM_FOREST_MIN_SAMPLES_LEAF,
            max_features=RANDOM_FOREST_MAX_FEATURES,
            random_state=RANDOM_FOREST_RANDOM_STATE,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Evaluate
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        metrics = {
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred))
        }
        
        logger.info("Training completed. Test R²: %.4f", metrics['test_r2'])
        return metrics
    
    def predict_volume(self, X: pd.DataFrame) -> np.ndarray:
        """Predict volume for given features."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_aligned = X[self.feature_columns].copy()
        X_scaled = self.scaler.transform(X_aligned)
        predictions = self.model.predict(X_scaled)
        return np.maximum(predictions, 0)  # Ensure non-negative
    
    def optimize_price(
        self,
        features_base: pd.DataFrame,
        current_price: float,
        cost: float,
        comp1_price: float,
        comp2_price: float,
        comp3_price: float
    ) -> Dict:
        """Find optimal price by maximizing profit."""
        if not self.is_trained:
            raise ValueError("Model must be trained before optimization")
        
        logger.info("Finding optimal price...")
        
        # Price search range
        min_price = current_price * PRICE_SEARCH_MIN_MULTIPLIER
        max_price = current_price * PRICE_SEARCH_MAX_MULTIPLIER
        price_candidates = np.linspace(min_price, max_price, PRICE_SEARCH_STEPS)
        
        best_price = current_price
        best_profit = -np.inf
        best_volume = 0
        
        avg_comp_price = np.mean([comp1_price, comp2_price, comp3_price])
        max_change_up = current_price * (1 + MAX_PRICE_CHANGE_PERCENT / 100)
        max_change_down = current_price * (1 - MAX_PRICE_CHANGE_PERCENT / 100)
        min_price_for_margin = cost * (1 + MIN_PROFIT_MARGIN_PERCENT / 100)
        comp_ceiling = avg_comp_price * (1 + COMPETITIVENESS_THRESHOLD / 100)
        comp_floor = avg_comp_price * (1 - COMPETITIVENESS_THRESHOLD / 100)
        
        for price in price_candidates:
            # Apply guardrails (hard caps)
            price = min(price, max_change_up)
            price = max(price, max_change_down)
            price = max(price, min_price_for_margin)
            # Soft competitiveness: clamp to within threshold of avg competitor
            price = min(price, comp_ceiling)
            price = max(price, comp_floor)

            # Create features with this price
            features = features_base.copy()
            features['price'] = price
            features['price_vs_avg_comp'] = price - avg_comp_price
            features['profit_margin'] = price - cost
            
            # Predict volume and calculate profit
            try:
                volume = self.predict_volume(features)[0]
                profit = volume * (price - cost)
                
                if profit > best_profit:
                    best_profit = profit
                    best_price = price
                    best_volume = volume
            except Exception as e:
                logger.warning("Error for price %.2f: %s", price, e)
                continue
        
        if best_profit == -np.inf:
            # Fallback to current price
            features = features_base.copy()
            features['price'] = current_price
            best_volume = self.predict_volume(features)[0]
            best_profit = best_volume * (current_price - cost)
        
        logger.info("Optimal price: %.2f, Expected profit: %.2f", best_price, best_profit)

        # Final business-rule checks for the chosen price
        price_change_pct = ((best_price - current_price) / current_price) * 100
        margin_pct = ((best_price - cost) / cost) * 100
        comp_diff_pct = abs((best_price - avg_comp_price) / avg_comp_price) * 100

        within_max_change = abs(price_change_pct) <= MAX_PRICE_CHANGE_PERCENT + 1e-6
        meets_min_margin = margin_pct >= MIN_PROFIT_MARGIN_PERCENT - 1e-6
        within_competitive_band = comp_diff_pct <= COMPETITIVENESS_THRESHOLD + 1e-6

        return {
            'recommended_price': best_price,
            'predicted_volume': best_volume,
            'expected_profit': best_profit,
            'profit_per_liter': best_price - cost,
            'price_change_pct': price_change_pct,
            'within_max_change': within_max_change,
            'meets_min_margin': meets_min_margin,
            'within_competitive_band': within_competitive_band
        }
    
    def save(self):
        """Save model and scaler."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(self.model, f)
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        logger.info("Model saved")
    
    def load(self):
        """Load model and scaler."""
        with open(MODEL_PATH, 'rb') as f:
            self.model = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            self.scaler = pickle.load(f)
        
        self.is_trained = True
        logger.info("Model loaded")
