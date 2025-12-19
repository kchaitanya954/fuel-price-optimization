"""
Data exploration and analysis script.
Understands demand dynamics, seasonality, and price relationships.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from data_pipeline import DataPipeline
from feature_engineering import FeatureEngineer
from config import HISTORICAL_DATA_PATH

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_and_prepare_data():
    """Load and prepare data for exploration."""
    pipeline = DataPipeline(HISTORICAL_DATA_PATH)
    df = pipeline.process()
    return df


def explore_basic_stats(df):
    """Explore basic statistics of the dataset."""
    print("="*60)
    print("BASIC STATISTICS")
    print("="*60)
    print(f"\nDataset shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nSummary Statistics:")
    print(df[['price', 'cost', 'comp1_price', 'comp2_price', 'comp3_price', 'volume']].describe())
    
    print("\n" + "="*60)
    print("CORRELATION MATRIX")
    print("="*60)
    corr = df[['price', 'cost', 'comp1_price', 'comp2_price', 'comp3_price', 'volume']].corr()
    print(corr)
    
    # Visualize correlation
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=150)
    print("\nCorrelation matrix saved to correlation_matrix.png")


def explore_price_relationships(df):
    """Explore relationships between prices and volume."""
    print("\n" + "="*60)
    print("PRICE-VOLUME RELATIONSHIPS")
    print("="*60)
    
    # Price vs Volume
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Scatter: Price vs Volume
    axes[0, 0].scatter(df['price'], df['volume'], alpha=0.5)
    axes[0, 0].set_xlabel('Price')
    axes[0, 0].set_ylabel('Volume')
    axes[0, 0].set_title('Price vs Volume')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Price differential vs Volume
    df['price_vs_avg_comp'] = df['price'] - df[['comp1_price', 'comp2_price', 'comp3_price']].mean(axis=1)
    axes[0, 1].scatter(df['price_vs_avg_comp'], df['volume'], alpha=0.5)
    axes[0, 1].set_xlabel('Price vs Average Competitor')
    axes[0, 1].set_ylabel('Volume')
    axes[0, 1].set_title('Price Differential vs Volume')
    axes[0, 1].axvline(x=0, color='r', linestyle='--', alpha=0.5)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Profit margin vs Volume
    df['profit_margin'] = df['price'] - df['cost']
    axes[1, 0].scatter(df['profit_margin'], df['volume'], alpha=0.5)
    axes[1, 0].set_xlabel('Profit Margin')
    axes[1, 0].set_ylabel('Volume')
    axes[1, 0].set_title('Profit Margin vs Volume')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Time series: Price and Volume
    axes[1, 1].plot(df['date'], df['price'], label='Price', alpha=0.7)
    ax2 = axes[1, 1].twinx()
    ax2.plot(df['date'], df['volume'], label='Volume', color='orange', alpha=0.7)
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Price', color='blue')
    ax2.set_ylabel('Volume', color='orange')
    axes[1, 1].set_title('Price and Volume Over Time')
    axes[1, 1].tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='orange')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('price_volume_relationships.png', dpi=150)
    print("Price-volume relationships saved to price_volume_relationships.png")
    
    # Calculate price elasticity proxy
    price_elasticity = -np.corrcoef(df['price'], df['volume'])[0, 1]
    print(f"\nPrice-Volume Correlation (Elasticity Proxy): {price_elasticity:.4f}")


def explore_seasonality(df):
    """Explore seasonality patterns."""
    print("\n" + "="*60)
    print("SEASONALITY ANALYSIS")
    print("="*60)
    
    # Extract temporal features
    df['month'] = pd.to_datetime(df['date']).dt.month
    df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
    df['quarter'] = pd.to_datetime(df['date']).dt.quarter
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Volume by month
    monthly_volume = df.groupby('month')['volume'].mean()
    axes[0, 0].bar(monthly_volume.index, monthly_volume.values)
    axes[0, 0].set_xlabel('Month')
    axes[0, 0].set_ylabel('Average Volume')
    axes[0, 0].set_title('Average Volume by Month')
    axes[0, 0].set_xticks(range(1, 13))
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Volume by day of week
    daily_volume = df.groupby('day_of_week')['volume'].mean()
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    axes[0, 1].bar(range(7), daily_volume.values)
    axes[0, 1].set_xlabel('Day of Week')
    axes[0, 1].set_ylabel('Average Volume')
    axes[0, 1].set_title('Average Volume by Day of Week')
    axes[0, 1].set_xticks(range(7))
    axes[0, 1].set_xticklabels(day_names)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Price by month
    monthly_price = df.groupby('month')['price'].mean()
    axes[1, 0].plot(monthly_price.index, monthly_price.values, marker='o')
    axes[1, 0].set_xlabel('Month')
    axes[1, 0].set_ylabel('Average Price')
    axes[1, 0].set_title('Average Price by Month')
    axes[1, 0].set_xticks(range(1, 13))
    axes[1, 0].grid(True, alpha=0.3)
    
    # Volume by quarter
    quarterly_volume = df.groupby('quarter')['volume'].mean()
    axes[1, 1].bar(quarterly_volume.index, quarterly_volume.values)
    axes[1, 1].set_xlabel('Quarter')
    axes[1, 1].set_ylabel('Average Volume')
    axes[1, 1].set_title('Average Volume by Quarter')
    axes[1, 1].set_xticks(range(1, 5))
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('seasonality_analysis.png', dpi=150)
    print("Seasonality analysis saved to seasonality_analysis.png")
    
    print(f"\nMonthly Volume Statistics:")
    print(monthly_volume)
    print(f"\nDay of Week Volume Statistics:")
    print(daily_volume)


def explore_competitor_relationships(df):
    """Explore relationships with competitor prices."""
    print("\n" + "="*60)
    print("COMPETITOR PRICE ANALYSIS")
    print("="*60)
    
    df['avg_comp_price'] = df[['comp1_price', 'comp2_price', 'comp3_price']].mean(axis=1)
    df['price_position'] = df.apply(
        lambda row: 'Cheapest' if row['price'] < row[['comp1_price', 'comp2_price', 'comp3_price']].min()
        else 'Most Expensive' if row['price'] > row[['comp1_price', 'comp2_price', 'comp3_price']].max()
        else 'Middle',
        axis=1
    )
    
    # Volume by price position
    position_volume = df.groupby('price_position')['volume'].mean()
    print(f"\nAverage Volume by Price Position:")
    print(position_volume)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Price position distribution
    position_counts = df['price_position'].value_counts()
    axes[0].bar(position_counts.index, position_counts.values)
    axes[0].set_xlabel('Price Position')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of Price Positions')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Volume by price position
    axes[1].bar(position_volume.index, position_volume.values)
    axes[1].set_xlabel('Price Position')
    axes[1].set_ylabel('Average Volume')
    axes[1].set_title('Average Volume by Price Position')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('competitor_analysis.png', dpi=150)
    print("Competitor analysis saved to competitor_analysis.png")


def main():
    """Main exploration function."""
    print("Loading data...")
    df = load_and_prepare_data()
    
    explore_basic_stats(df)
    explore_price_relationships(df)
    explore_seasonality(df)
    explore_competitor_relationships(df)
    
    print("\n" + "="*60)
    print("EXPLORATION COMPLETE")
    print("="*60)
    print("Visualizations saved to:")
    print("  - correlation_matrix.png")
    print("  - price_volume_relationships.png")
    print("  - seasonality_analysis.png")
    print("  - competitor_analysis.png")


if __name__ == "__main__":
    main()

