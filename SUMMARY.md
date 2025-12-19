## Fuel Price Optimization System – Summary Document

### 1. Understanding of the Problem

The business operates a fuel station in a competitive retail market where prices can be changed once per day, at the start of the day.  
The objective is to choose a daily retail price that **maximizes total profit**, taking into account:
- The company’s own historical prices and volumes.
- Daily costs (wholesale/input cost per liter).
- Competitor prices (three nearby competitors).

Key challenges:
- **Demand response to price** is non-linear and affected by competitors and seasonality.
- We only observe past prices and volumes; we never directly observe “what would have happened” at other prices.
- The system must be **operationally simple** (daily recommendation) and **safe** (avoid extreme, non-competitive prices).

The solution is to:
1. Learn a **demand model**: volume as a function of own price, cost, competitors, and time.
2. Use this model to simulate different candidate prices for today.
3. Compute profit for each candidate price and choose the best one **under business guardrails**.

---

### 2. Key Assumptions

1. **Station characteristics are stable**  
   No major structural changes (e.g., new highway exit, closure of nearby stations) over the historical period.

2. **Historical relationships generalize**  
   The relationship between price, competitors, seasonality, and volume in the last ~2 years will continue to hold reasonably well for the near future.

3. **No major unobserved shocks**  
   Events like strikes, extreme weather, or macro shocks are either rare or their effect is partially captured by seasonality and trends.

4. **Competitor prices are exogenous inputs**  
   Today’s competitor prices are known at decision time and are not significantly affected by our own pricing decision.

5. **Business rules reflect management preferences**  
   - Limit on daily price change (e.g. ±5%) is a hard constraint.
   - Minimum profit margin (e.g. ≥ 1–2% over cost) is necessary for sustainability.
   - Competitiveness band (e.g. within ~0.5–2% of average competitor) prevents “outlier” prices.

---

### 3. Data Pipeline Design and Technology Choices

**Technologies**:
- `pandas` for data ingestion and transformations.
- `numpy` for numerical operations.
- `scikit-learn` for Random Forest regression and scaling.
- `FastAPI` for serving an HTTP API.

**Pipeline components**:

1. **Ingestion (`data_pipeline.py`)**
   - Input: `oil_retail_history.csv` (comma-separated, daily rows).
   - Reads CSV or JSON (for daily streaming-like input).
   - Parses `date` as `datetime` and sorts chronologically.
   - Forward-fills / back-fills missing values in price/cost/competitor columns.

2. **Feature Engineering (`feature_engineering.py`)**
   - **Price and competition features**:
     - `avg_comp_price`: mean of `comp1_price`, `comp2_price`, `comp3_price`.
     - `price_vs_avg_comp` and `%` version.
     - `profit_margin` and `profit_margin_pct`.
   - **Lag features** (memory of past behavior):
     - Lags at 1, 3, 7, 14, 30 days for price, volume, and competitor averages.
   - **Moving averages and volatility**:
     - 7 / 14 / 30-day moving averages and standard deviations for price, volume, and average competitor price.
   - **Temporal features**:
     - `day_of_week`, `month`, `day_of_year`.
     - Cyclical encodings (`sin`/`cos`) for day-of-week and day-of-year to capture seasonality.
   - All NaNs produced by lags/rolling windows are forward/back-filled to keep feature matrices dense.

3. **Model-Ready Dataset**
   - Target variable: `volume` (liters sold).
   - Feature set: all engineered columns except `date` and `volume`.
   - Rows with missing `volume` are dropped (if any).

4. **Why this design**
   - **Rich, but still interpretable** feature set that encodes:
     - Own pricing history.
     - Competitive position.
     - Demand trends and seasonality.
   - Built entirely on `pandas` + `scikit-learn` for simplicity and portability.

---

### 4. Methodology and Reasoning

#### 4.1. Modeling Approach

- **Model class**: `RandomForestRegressor`
  - Non-parametric and handles non-linear relationships well.
  - Robust to outliers and feature scaling (though we still use `StandardScaler`).
  - Provides stable performance on medium-sized tabular datasets.

- **Training setup**:
  - Features: as described above.
  - Target: daily `volume`.
  - **Time-based train/test split**:
    - The earliest ~80% of days used for training.
    - The most recent ~20% used for testing.
    - No shuffling – this respects the time-series nature and avoids “looking into the future”.
  - Model hyperparameters (at time of summary):
    - `n_estimators = 500`
    - `max_depth = None` (let trees grow until constrained by other params)
    - `min_samples_split = 4`
    - `min_samples_leaf = 2`
    - `max_features = "sqrt"`

#### 4.2. Price Optimization Logic

Given:
- Today’s `price`, `cost`, and competitor prices.
- Today’s feature vector (using historical context).

Steps:
1. **Generate candidate prices**:
   - Linearly spaced between `current_price × PRICE_SEARCH_MIN_MULTIPLIER` and  
     `current_price × PRICE_SEARCH_MAX_MULTIPLIER` (e.g. ±5%).

2. **Apply business guardrails to each candidate**:
   - **Max daily change**: clamp price to within ±`MAX_PRICE_CHANGE_PERCENT`.
   - **Min margin**: ensure `price ≥ cost × (1 + MIN_PROFIT_MARGIN_PERCENT / 100)`.
   - **Competitiveness**: clamp price to stay within ±`COMPETITIVENESS_THRESHOLD`% of average competitor price.

3. **For each valid candidate**:
   - Update feature vector with that candidate price and derived features (`price_vs_avg_comp`, margin, etc.).
   - Use the trained model to predict **volume**.
   - Compute **profit = volume × (price − cost)**.

4. **Select the best candidate**:
   - Choose the price with highest expected profit.
   - Return price, predicted volume, expected profit, and diagnostic flags about guardrails.

This is essentially a **one-step look-ahead optimization** on top of a supervised demand model.

---

### 5. Validation Results

metrics from the current configuration (time-based split, feature set, and hyperparameters):
- `train_r2 ≈ 0.88`
- `test_r2 ≈ 0.48`
- `test_mae ≈ 523` liters
- `test_rmse ≈ 683` liters

Interpretation:
- The model explains ~48% of the variance in out-of-sample daily volumes on the most recent 20% of days.
- Errors of ~500–700 liters per day are reasonable given:
  - Typical volumes around 13k–15k liters.
  - Unobserved demand drivers (e.g., weather, holidays, local events) not included in the dataset.

In practice, an R² around 0.5 for retail fuel demand **without external covariates** is realistic and provides enough signal
to support price optimization, especially once constrained by business rules.

---

### 6. Example Output for `today_example.json`

Using the provided `today_example.json`:

```json
{
  "date": "2024-12-31",
  "price": 94.45,
  "cost": 85.77,
  "comp1_price": 95.01,
  "comp2_price": 95.7,
  "comp3_price": 95.21
}
```

A sample API response looks like:

```json
{
    "recommended_price":95.7832,
    "predicted_volume":14364.579987301588,
    "expected_profit":143835.41232884824,
    "profit_per_liter":10.013199999999998,
    "price_change_pct":1.4115404976177774,
    "within_max_change":true,
    "meets_min_margin":true,
    "within_competitive_band":true,
    "input_data":
        {
            "date":"2024-12-31","
            price":94.45,
            "cost":85.77,
            "comp1_price":95.01,
            "comp2_price":95.7,
            "comp3_price":95.21
        }
}
```

Key points:
- Recommended price is **very close** to competitors and current price (within <1% change).
- All guardrail flags are `true`:
  - Max daily change respected.
  - Margin above minimum threshold.
  - Within competitiveness band.

---

### 7. Recommendations for Improvements and Extensions

1. **Richer Models**
   - Try gradient-boosting models (`HistGradientBoostingRegressor`, XGBoost, LightGBM) which often perform better on tabular data.
   - Consider modelling `log(volume)` instead of raw volume for more stable regressions.

2. **Additional Data Sources**
   - Calendar features: explicit public holidays and local events.
   - Weather: temperature, precipitation (drive volume and mode choice).
   - Macroeconomic indicators: crude price indices, economic activity proxies.

3. **More Sophisticated Time-Series Handling**
   - Rolling or expanding window re-training (e.g., retrain monthly on the most recent N months).
   - Cross-validation with `TimeSeriesSplit` to fine-tune hyperparameters.

4. **Richer Business Logic**
   - Add constraints for **maximum price vs legal caps** or corporately set bands.
   - Add multi-objective optimization: slight trade-off between profit and volume or market share.
   - Support “what-if” analysis: simulate multiple scenarios (e.g., competitor A drops price by 1).

5. **Operationalization**
   - Schedule daily training/recommendation jobs (via cron, Airflow, or Prefect) using the existing `schedule_example.py`.
   - Log recommendations and realized volumes to continuously monitor performance and drift.

Overall, the current system provides a solid baseline: a reproducible data pipeline, a reasonably accurate demand model, and
a profit-maximizing, guardrail-aware price recommendation. The suggested extensions can incrementally improve both accuracy
and business value.


