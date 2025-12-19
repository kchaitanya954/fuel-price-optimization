"""
FastAPI endpoint for price recommendation.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import logging
from price_recommender import PriceRecommender

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fuel Price Optimization API",
    description="API for recommending optimal daily retail prices",
    version="1.0.0"
)

recommender = None


class PriceRequest(BaseModel):
    """Request model."""
    date: str
    price: float = Field(..., gt=0)
    cost: float = Field(..., gt=0)
    comp1_price: float = Field(..., gt=0)
    comp2_price: float = Field(..., gt=0)
    comp3_price: float = Field(..., gt=0)


class PriceResponse(BaseModel):
    """Response model."""
    recommended_price: float
    predicted_volume: float
    expected_profit: float
    profit_per_liter: float
    price_change_pct: float
    within_max_change: bool
    meets_min_margin: bool
    within_competitive_band: bool
    input_data: dict


@app.on_event("startup")
async def startup_event():
    """Initialize recommender."""
    global recommender  # noqa: PLW0603
    try:
        recommender = PriceRecommender()
        if not recommender.model.is_trained:
            logger.warning("Model not trained. Please train first.")
    except Exception as e:
        logger.error("Error initializing: %s", e)


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Fuel Price Optimization API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    """Health check."""
    is_ready = recommender is not None and recommender.model.is_trained
    return {"status": "healthy" if is_ready else "not ready", "model_trained": recommender.model.is_trained if recommender else False}


@app.post("/recommend", response_model=PriceResponse)
async def recommend_price(request: PriceRequest):
    """Get price recommendation."""
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender not initialized")
    
    if not recommender.model.is_trained:
        raise HTTPException(status_code=503, detail="Model not trained. Use /train endpoint first.")
    
    try:
        today_data = {
            "date": request.date,
            "price": request.price,
            "cost": request.cost,
            "comp1_price": request.comp1_price,
            "comp2_price": request.comp2_price,
            "comp3_price": request.comp3_price
        }
        
        result = recommender.recommend_price(today_data)
        return PriceResponse(**result)
    except Exception as e:
        logger.error("Error: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/train")
async def train_model():
    """Train the model."""
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender not initialized")
    
    try:
        metrics = recommender.train_model()
        return {"status": "success", "metrics": metrics}
    except Exception as e:
        logger.error("Error: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
