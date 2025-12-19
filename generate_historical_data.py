"""
Script to generate extended historical data for model training.
This creates a more realistic dataset with ~2 years of daily data.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

def generate_historical_data(n_days=730, output_path="data/oil_retail_history.csv"):
    """
    Generate synthetic historical data for fuel prices.
    
    Args:
        n_days: Number of days to generate (default: 730 = ~2 years)
        output_path: Output CSV file path
    """
    np.random.seed(42)
    
    # Start date
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    
    # Base prices (realistic fuel price ranges)
    base_cost = 85.0
    base_price = 95.0
    base_comp1 = 96.0
    base_comp2 = 96.5
    base_comp3 = 95.5
    
    # Generate time series with trends and seasonality
    days = np.arange(n_days)
    
    # Long-term trend (slight upward)
    trend = 0.01 * days / 365
    
    # Seasonal component (higher in summer)
    seasonal = 2 * np.sin(2 * np.pi * days / 365.25 + np.pi/2)
    
    # Weekly pattern (slightly higher on weekends)
    weekly = 0.5 * np.sin(2 * np.pi * days / 7)
    
    # Random walk component
    random_walk = np.cumsum(np.random.randn(n_days) * 0.3)
    
    # Generate costs (less volatile)
    costs = base_cost + trend * 5 + seasonal * 0.5 + np.random.randn(n_days) * 1.5
    costs = np.maximum(costs, 80)  # Floor at 80
    
    # Generate competitor prices (correlated but with variation)
    comp1_prices = base_comp1 + trend * 5 + seasonal + weekly + random_walk + np.random.randn(n_days) * 1.2
    comp2_prices = base_comp2 + trend * 5 + seasonal + weekly + random_walk + np.random.randn(n_days) * 1.2
    comp3_prices = base_comp3 + trend * 5 + seasonal + weekly + random_walk + np.random.randn(n_days) * 1.2
    
    # Ensure competitor prices are reasonable
    comp1_prices = np.clip(comp1_prices, 90, 110)
    comp2_prices = np.clip(comp2_prices, 90, 110)
    comp3_prices = np.clip(comp3_prices, 90, 110)
    
    # Generate company prices (correlated with competitors but with some lag/strategy)
    # Company price tends to be close to average competitor, with some strategic variation
    avg_comp = (comp1_prices + comp2_prices + comp3_prices) / 3
    company_prices = avg_comp + np.random.randn(n_days) * 1.5 - 0.3  # Slightly below average on average
    company_prices = np.clip(company_prices, 90, 110)
    
    # Generate volumes (inversely related to price differential, with seasonality)
    price_diff = company_prices - avg_comp
    base_volume = 14000
    
    # Volume decreases when price is higher than competitors
    volume_effect = -50 * price_diff
    
    # Seasonal volume (higher in summer, holidays)
    volume_seasonal = 1000 * np.sin(2 * np.pi * days / 365.25 + np.pi/2)
    
    # Weekend effect (higher volume)
    volume_weekly = 500 * (np.sin(2 * np.pi * days / 7) > 0.5).astype(int)
    
    # Random variation
    volume_noise = np.random.randn(n_days) * 800
    
    volumes = base_volume + volume_effect + volume_seasonal + volume_weekly + volume_noise
    volumes = np.maximum(volumes, 10000)  # Floor at 10000
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': [d.strftime('%Y-%m-%d') for d in dates],
        'price': np.round(company_prices, 2),
        'cost': np.round(costs, 2),
        'comp1_price': np.round(comp1_prices, 2),
        'comp2_price': np.round(comp2_prices, 2),
        'comp3_price': np.round(comp3_prices, 2),
        'volume': np.round(volumes, 0).astype(int)
    })
    
    # Save to CSV
    df.to_csv(output_path, sep='\t', index=False)
    print(f"Generated {n_days} days of historical data")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Saved to {output_path}")
    
    return df


if __name__ == "__main__":
    # Generate 2 years of data
    df = generate_historical_data(n_days=730, output_path="data/oil_retail_history.csv")
    print(f"\nFirst few rows:")
    print(df.head(10))
    print(f"\nLast few rows:")
    print(df.tail(10))
    print(f"\nSummary statistics:")
    print(df.describe())

