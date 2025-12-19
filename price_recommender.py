"""
Main price recommendation system.
"""
import pandas as pd
import json
from pathlib import Path
from typing import Dict
import logging

from data_pipeline import DataPipeline
from feature_engineering import FeatureEngineer
from model import PriceOptimizationModel
from config import HISTORICAL_DATA_PATH, MODEL_PATH, SCALER_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PriceRecommender:
    """Main system for price recommendation."""
    
    def __init__(self):
        self.data_pipeline = DataPipeline(HISTORICAL_DATA_PATH)
        self.feature_engineer = FeatureEngineer()
        # Model will be trained via train_model() or /train API
        self.model = PriceOptimizationModel()
    
    def train_model(self) -> Dict:
        """Train the model using historical data."""
        logger.info("Training model...")
        
        # Load and process historical data
        df = self.data_pipeline.process()
        df_features = self.feature_engineer.create_features(df)
        
        # Prepare features and target
        feature_columns = self.feature_engineer.get_feature_columns(df_features)
        X = df_features[feature_columns]
        y = df_features['volume']
        
        # Remove rows with NaN
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        # Train model
        metrics = self.model.train(X, y)
        self.model.save()
        
        return metrics
    
    def recommend_price(self, today_data: Dict) -> Dict:
        """Recommend optimal price for today."""
        logger.info("Generating recommendation for %s", today_data.get('date', 'today'))
        
        # Load historical data for context
        df_historical = self.data_pipeline.process()
        df_historical_features = self.feature_engineer.create_features(df_historical)
        
        # Create today's data
        df_today = pd.DataFrame([today_data])
        df_today['date'] = pd.to_datetime(df_today['date'])
        
        # Combine with historical data
        df_combined = pd.concat([df_historical_features, df_today], ignore_index=True)
        df_combined = df_combined.sort_values('date').reset_index(drop=True)
        
        # Create features
        df_combined_features = self.feature_engineer.create_features(df_combined)
        
        # Get today's features
        today_features = df_combined_features.iloc[[-1]].copy()
        feature_columns = self.feature_engineer.get_feature_columns(today_features)
        features_base = today_features[feature_columns].copy()
        
        # Optimize price
        result = self.model.optimize_price(
            features_base=features_base,
            current_price=today_data['price'],
            cost=today_data['cost'],
            comp1_price=today_data['comp1_price'],
            comp2_price=today_data['comp2_price'],
            comp3_price=today_data['comp3_price']
        )
        
        result['input_data'] = today_data
        return result
    
    def recommend_from_json(self, json_path: Path) -> Dict:
        """Load JSON and generate recommendation."""
        with open(json_path, 'r', encoding='utf-8') as f:
            today_data = json.load(f)
        return self.recommend_price(today_data)


def main():
    """Main function."""
    recommender = PriceRecommender()
    
    # Train if needed
    if not recommender.model.is_trained:
        logger.info("Training model...")
        metrics = recommender.train_model()
        logger.info("Training metrics: %s", metrics)
    
    # Generate recommendation
    today_example_path = Path("today_example.json")
    if today_example_path.exists():
        result = recommender.recommend_from_json(today_example_path)
        
        print("\n" + "="*60)
        print("PRICE RECOMMENDATION")
        print("="*60)
        print(f"Date: {result['input_data']['date']}")
        print(f"Current Price: {result['input_data']['price']:.2f}")
        print(f"Cost: {result['input_data']['cost']:.2f}")
        print(f"\nRecommended Price: {result['recommended_price']:.2f}")
        print(f"Price Change: {result['price_change_pct']:.2f}%")
        print(f"\nPredicted Volume: {result['predicted_volume']:.0f} liters")
        print(f"Expected Profit: {result['expected_profit']:.2f}")
        print(f"Profit per Liter: {result['profit_per_liter']:.2f}")
        print("="*60)
    else:
        logger.warning("File %s not found", today_example_path)


if __name__ == "__main__":
    main()
