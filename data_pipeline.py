"""
Simple data ingestion and transformation pipeline.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPipeline:
    """Handles data ingestion and basic cleaning."""
    
    def __init__(self, data_path: Path):
        self.data_path = data_path
    
    def process(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """Load and clean data."""
        if file_path is None:
            file_path = self.data_path
        
        logger.info("Loading data from %s", file_path)
        
        # Read data
        if file_path.suffix == '.csv':
            # Try comma first, then tab
            try:
                df = pd.read_csv(file_path, sep=',')
            except Exception:
                df = pd.read_csv(file_path, sep='\t')
        elif file_path.suffix == '.json':
            df = pd.read_json(file_path, orient='records')
            if 'volume' not in df.columns:
                df['volume'] = np.nan
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # Convert date to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
        
        # Fill missing values
        price_cols = ['price', 'cost', 'comp1_price', 'comp2_price', 'comp3_price']
        for col in price_cols:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        
        logger.info("Loaded %d records", len(df))
        return df
