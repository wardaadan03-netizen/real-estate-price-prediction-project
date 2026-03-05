import os
import logging
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

from src.data_preprocessing import DataPreprocessor


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    try:
        # Path to dataset (adjust if needed)
        data_path = os.path.join("data", "Melbourne_housing_FULL.csv")

        # Initialize preprocessor
        logger.info("Initializing data preprocessor...")
        preprocessor = DataPreprocessor(data_path)

        # Run full preprocessing pipeline
        X_train, X_test, y_train, y_test, scaler = \
            preprocessor.run_preprocessing_pipeline()

        # Train model
        logger.info("Training Linear Regression model...")
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate model
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        logger.info(f"Model Evaluation Results:")
        logger.info(f"RMSE: {rmse}")
        logger.info(f"R² Score: {r2}")

        print("\nModel Performance:")
        print(f"RMSE: {rmse}")
        print(f"R² Score: {r2}")

    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")
        raise


if __name__ == "__main__":
    main()
