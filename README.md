# Fuel Price Optimization System

A simple machine learning system for optimizing daily retail fuel prices to maximize profit using Random Forest.

## Features

- **Data Pipeline**: Simple ingestion and cleaning of CSV/JSON data
- **Feature Engineering**: Basic features (price differentials, lags, moving averages)
- **ML Model**: Random Forest regressor for volume prediction
- **Price Optimization**: Searches price space to maximize profit
- **API**: FastAPI endpoint for recommendations

## Project Structure

```
fuel-price-optimization/
├── config.py                 # Configuration
├── data_pipeline.py          # Data loading and cleaning
├── feature_engineering.py    # Feature creation
├── model.py                  # Random Forest model
├── price_recommender.py      # Main recommendation system
├── api.py                    # FastAPI endpoint
├── oil_retail_history.csv    # Historical training data
├── today_example.json       # Example daily input
└── requirements.txt          # Dependencies
```

## Environment Setup

### 1. Create and activate a virtual environment (recommended)

```bash
# From the project root
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### API

Start the API:
```bash
python api.py
```

### Train the Model

Model training is performed via the API (see below).  
There is no command-line or manual script to train the model outside of the API.  
To trigger training, send a POST request to the `/train` endpoint:

```bash
curl -X POST "http://localhost:8000/train"
```

The API will train the model using the historical data and return training metrics.

### Get Price Recommendation

You can also use the API to get recommendations by sending a POST request to `/recommend` (see API section below for an example).

If you want to use the recommender directly from Python (requires that a trained model already exists):

Make a request:
```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "date": "2024-12-31",
    "price": 94.45,
    "cost": 85.77,
    "comp1_price": 95.01,
    "comp2_price": 95.7,
    "comp3_price": 95.21
  }'
```

Visit `http://localhost:8000/docs` for interactive API documentation.

## How It Works

1. **Training**: Model learns volume patterns from historical data (`oil_retail_history.csv`) using engineered features\n   (lags, moving stats, temporal features, price vs competitors, margin, etc.).\n2. **Time-Based Split**: Training uses the earliest ~80% of days as train and the most recent ~20% as test (no shuffling),\n   which is appropriate for time-series data.\n3. **Prediction & Optimization**: For a given day, the system:\n   - Builds today’s feature row using historical context.\n   - Searches a price range (default ±5% of current price).\n   - Predicts volume for each candidate price.\n   - Computes expected profit = volume × (price − cost).\n   - Applies business guardrails (see below).\n   - Picks the price with maximum expected profit that respects the constraints.\n4. **Business Guardrails** (configured in `config.py`):\n   - `MAX_PRICE_CHANGE_PERCENT`: maximum allowed daily change vs current price.\n   - `MIN_PROFIT_MARGIN_PERCENT`: minimum profit margin vs cost.\n   - `COMPETITIVENESS_THRESHOLD`: keep price within a small % band around average competitor price.\n\nThe API response also includes booleans indicating whether the recommended price satisfies each guardrail.

## Configuration

Edit `config.py` to adjust:
- **Model parameters** (e.g. `RANDOM_FOREST_N_ESTIMATORS`, `RANDOM_FOREST_MAX_DEPTH`, `RANDOM_FOREST_MIN_SAMPLES_SPLIT`,\n  `RANDOM_FOREST_MIN_SAMPLES_LEAF`, `RANDOM_FOREST_MAX_FEATURES`).\n+- **Price search range** (`PRICE_SEARCH_MIN_MULTIPLIER`, `PRICE_SEARCH_MAX_MULTIPLIER`, `PRICE_SEARCH_STEPS`).\n+- **Business rules**:\n+  - `MAX_PRICE_CHANGE_PERCENT`\n+  - `MIN_PROFIT_MARGIN_PERCENT`\n+  - `COMPETITIVENESS_THRESHOLD`
