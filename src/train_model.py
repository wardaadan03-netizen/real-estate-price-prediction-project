import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_score = -np.inf
        self.results = {}
        
    def define_models(self):
        """Define all models to be trained"""
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Random Forest': RandomForestRegressor(
                n_estimators=100, 
                max_depth=10, 
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
            'LightGBM': lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        }
        logger.info(f"Defined {len(self.models)} models for training")
        
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Train all models and evaluate their performance"""
        self.define_models()
        
        for name, model in self.models.items():
            try:
                logger.info(f"Training {name}...")
                
                # Train the model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Calculate metrics
                train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                train_mae = mean_absolute_error(y_train, y_pred_train)
                test_mae = mean_absolute_error(y_test, y_pred_test)
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                
                # Store results
                self.results[name] = {
                    'model': model,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'train_r2': train_r2,
                    'test_r2': test_r2
                }
                
                # Check if this is the best model
                if test_r2 > self.best_score:
                    self.best_score = test_r2
                    self.best_model = model
                    self.best_model_name = name
                
                logger.info(f"{name} - Test RMSE: {test_rmse:.2f}, Test R2: {test_r2:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
        
        return self.results
    
    def get_feature_importance(self, feature_names, model_name='Random Forest'):
        """Get feature importance from tree-based models"""
        if model_name in self.results:
            model = self.results[model_name]['model']
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                logger.info(f"Top 10 important features:\n{importance_df.head(10)}")
                return importance_df
        
        return None
    
    def save_model(self, model_path='models/model.pkl'):
        """Save the best model"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        model_info = {
            'model': self.best_model,
            'name': self.best_model_name,
            'score': self.best_score,
            'results': self.results
        }
        
        joblib.dump(model_info, model_path)
        logger.info(f"Best model saved to {model_path}")
        
        return model_path
    
    def load_model(self, model_path='models/model.pkl'):
        """Load a saved model"""
        if os.path.exists(model_path):
            model_info = joblib.load(model_path)
            self.best_model = model_info['model']
            self.best_model_name = model_info['name']
            self.best_score = model_info['score']
            self.results = model_info['results']
            logger.info(f"Model loaded from {model_path}")
            return model_info
        else:
            logger.error(f"Model file not found: {model_path}")
            return None