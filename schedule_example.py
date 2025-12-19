"""
Example scheduling configuration for daily price recommendations.
Can be adapted for Airflow, Prefect, or Cron.
"""
from datetime import datetime
import json
from pathlib import Path
from price_recommender import PriceRecommender
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def daily_price_recommendation_job(input_json_path: Path, output_path: Path = None):
    """
    Scheduled job function for daily price recommendation.
    
    This function can be called by:
    - Cron: Add to crontab
    - Airflow: Use as PythonOperator task
    - Prefect: Use as @task decorator
    - Celery: Use as scheduled task
    
    Args:
        input_json_path: Path to JSON file with today's data
        output_path: Optional path to save recommendation result
    """
    logger.info(f"Starting daily price recommendation job at {datetime.now()}")
    
    try:
        # Initialize recommender
        recommender = PriceRecommender()
        
        if not recommender.model.is_trained:
            logger.warning("Model not trained. Training now...")
            recommender.train_model()
        
        # Load today's data
        with open(input_json_path, 'r') as f:
            today_data = json.load(f)
        
        # Get recommendation
        result = recommender.recommend_price(today_data)
        
        # Log result
        logger.info(f"Recommended Price: {result['recommended_price']:.2f}")
        logger.info(f"Predicted Volume: {result['predicted_volume']:.0f} liters")
        logger.info(f"Expected Profit: {result['expected_profit']:.2f}")
        
        # Save result if output path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            logger.info(f"Result saved to {output_path}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error in daily price recommendation job: {e}")
        raise


# Example for Prefect
def prefect_example():
    """Example using Prefect for scheduling."""
    try:
        from prefect import task, flow
        from prefect.schedules import CronSchedule
        
        @task
        def get_price_recommendation():
            return daily_price_recommendation_job(
                Path("today_example.json"),
                Path("recommendation_output.json")
            )
        
        @flow(schedule=CronSchedule(cron="0 6 * * *"))  # Daily at 6 AM
        def daily_price_optimization_flow():
            return get_price_recommendation()
        
        if __name__ == "__main__":
            daily_price_optimization_flow()
    except ImportError:
        logger.warning("Prefect not installed. Install with: pip install prefect")



if __name__ == "__main__":
    # Example: Run manually
    daily_price_recommendation_job(
        Path("today_example.json"),
        Path("recommendation_output.json")
    )

