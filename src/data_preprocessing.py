import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.label_encoders = {}
        
    def load_data(self):
        """Load the Melbourne housing dataset"""
        try:
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            return self.df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def clean_data(self):
        """Clean the dataset - Your simplified approach"""
        # Drop duplicates
        initial_shape = self.df.shape
        self.df = self.df.drop_duplicates()
        logger.info(f"Dropped {initial_shape[0] - self.df.shape[0]} duplicates")
        
        # Drop rows where Price is missing (target variable)
        price_missing = self.df['Price'].isnull().sum()
        self.df = self.df.dropna(subset=["Price"])
        logger.info(f"Dropped {price_missing} rows with missing Price")
        
        # Fill missing numeric values with median
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'Price':  # Don't fill Price as we already dropped missing
                missing_count = self.df[col].isnull().sum()
                if missing_count > 0:
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                    logger.info(f"Filled {missing_count} missing values in {col} with median")
        
        # Fill categorical with "Unknown"
        cat_cols = self.df.select_dtypes(include="object").columns
        for col in cat_cols:
            missing_count = self.df[col].isnull().sum()
            if missing_count > 0:
                self.df[col] = self.df[col].fillna("Unknown")
                logger.info(f"Filled {missing_count} missing values in {col} with 'Unknown'")
        
        logger.info(f"Data cleaning completed. New shape: {self.df.shape}")
        return self.df
    
    def remove_outliers(self, columns, n_std=3):
        """Remove outliers using z-score method (optional)"""
        for col in columns:
            if col in self.df.columns:
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                self.df = self.df[z_scores < n_std]
        
        logger.info(f"Outliers removed. New shape: {self.df.shape}")
        return self.df
    
    def encode_categorical_variables(self):
        """Encode categorical variables"""
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        encoded_cols = []
        
        for col in categorical_cols:
            if col != 'Address' and col != 'Suburb':  # Skip high cardinality columns
                le = LabelEncoder()
                self.df[col + '_encoded'] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le
                encoded_cols.append(col + '_encoded')
        
        logger.info(f"Encoded categorical variables: {len(encoded_cols)} new features created")
        return self.df
    
    def prepare_features_and_target(self, target_col='Price'):
        """Prepare features and target variable"""
        # Drop columns that shouldn't be features
        cols_to_drop = [target_col, 'Address']  # Address is too unique
        feature_cols = [col for col in self.df.columns if col not in cols_to_drop]
        
        X = self.df[feature_cols].copy()
        y = self.df[target_col].copy()
        
        # One-hot encode remaining categorical columns
        X = pd.get_dummies(X, drop_first=True)
        
        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
        return X, y
    
    def split_and_scale_data(self, X, y, test_size=0.2, random_state=42):
        """Split data and scale features"""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert back to DataFrame for easier interpretation
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
        
        logger.info(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler
    
    def run_preprocessing_pipeline(self, remove_outliers_flag=False):
        """Run the complete preprocessing pipeline"""
        logger.info("Starting preprocessing pipeline...")
        
        # Load data
        self.load_data()
        
        # Clean data (your simplified approach)
        self.clean_data()
        
        # Optional: Remove outliers
        if remove_outliers_flag:
            numerical_cols = ['Rooms', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea']
            self.remove_outliers([col for col in numerical_cols if col in self.df.columns])
        
        # Encode categorical variables
        self.encode_categorical_variables()
        
        # Prepare features and target
        X, y = self.prepare_features_and_target()
        
        # Split and scale
        X_train, X_test, y_train, y_test, scaler = self.split_and_scale_data(X, y)
        
        logger.info("Preprocessing pipeline completed successfully!")
        
        return X_train, X_test, y_train, y_test, scaler
