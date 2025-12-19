"""
Simple feature engineering module.
"""
import pandas as pd
import numpy as np
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Creates features for the model."""
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all features."""
        df = df.copy()
        
        # Price differentials
        df['avg_comp_price'] = df[['comp1_price', 'comp2_price', 'comp3_price']].mean(axis=1)
        df['price_vs_avg_comp'] = df['price'] - df['avg_comp_price']
        df['price_vs_avg_comp_pct'] = (df['price'] - df['avg_comp_price']) / df['avg_comp_price'] * 100
        df['profit_margin'] = df['price'] - df['cost']
        df['profit_margin_pct'] = (df['price'] - df['cost']) / df['cost'] * 100
        
        # Lag features (1 day, 7 days)
        df = df.sort_values('date').reset_index(drop=True)
        for lag in [1, 3, 7, 14, 30]:
            df[f'price_lag_{lag}'] = df['price'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag) if 'volume' in df.columns else np.nan
            df[f'avg_comp_price_lag_{lag}'] = df['avg_comp_price'].shift(lag)
        
        # Moving averages
        for window in [7, 14, 30]:
            df[f'price_ma_{window}'] = df['price'].rolling(window=window, min_periods=1).mean()
            df[f'volume_ma_{window}'] = df['volume'].rolling(window=window, min_periods=1).mean() if 'volume' in df.columns else np.nan
            df[f'avg_comp_price_ma_{window}'] = df['avg_comp_price'].rolling(window=window, min_periods=1).mean()
            df[f'price_std_{window}'] = df['price'].rolling(window=window, min_periods=1).std()
            df[f'volume_std_{window}'] = df['volume'].rolling(window=window, min_periods=1).std() if 'volume' in df.columns else np.nan
        
        # Temporal features
        if 'date' in df.columns:
            df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
            df['month'] = pd.to_datetime(df['date']).dt.month
            df['day_of_year'] = pd.to_datetime(df['date']).dt.dayofyear
            # cyclical encodings
            df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
            df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # Fill NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get feature columns (exclude target and metadata)."""
        exclude = ['date', 'volume']
        return [col for col in df.columns if col not in exclude]
