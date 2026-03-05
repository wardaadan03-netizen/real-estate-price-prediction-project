import pandas as pd
import numpy as np
from geopy.distance import geodesic
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, df):
        self.df = df
        
    def create_price_per_room(self):
        """Create price per room feature"""
        if 'Price' in self.df.columns and 'Rooms' in self.df.columns:
            self.df['Price_per_Room'] = self.df['Price'] / self.df['Rooms'].replace(0, 1)
            logger.info("Created Price_per_Room feature")
    
    def create_age_feature(self):
        """Create age of property feature"""
        if 'YearBuilt' in self.df.columns:
            current_year = 2024
            self.df['Property_Age'] = current_year - self.df['YearBuilt']
            self.df['Property_Age'] = self.df['Property_Age'].clip(lower=0)
            logger.info("Created Property_Age feature")
    
    def create_land_efficiency(self):
        """Create land efficiency ratio"""
        if 'Landsize' in self.df.columns and 'BuildingArea' in self.df.columns:
            # Avoid division by zero
            mask = self.df['BuildingArea'] > 0
            self.df.loc[mask, 'Land_Efficiency'] = self.df.loc[mask, 'BuildingArea'] / self.df.loc[mask, 'Landsize']
            self.df['Land_Efficiency'].fillna(0, inplace=True)
            logger.info("Created Land_Efficiency feature")
    
    def create_property_density(self):
        """Create property density feature"""
        if 'Propertycount' in self.df.columns and 'Landsize' in self.df.columns:
            mask = self.df['Landsize'] > 0
            self.df.loc[mask, 'Property_Density'] = self.df.loc[mask, 'Propertycount'] / self.df.loc[mask, 'Landsize']
            self.df['Property_Density'].fillna(0, inplace=True)
            logger.info("Created Property_Density feature")
    
    def create_price_category(self):
        """Create price categories for classification"""
        if 'Price' in self.df.columns:
            # Create price categories based on quantiles
            self.df['Price_Category'] = pd.qcut(
                self.df['Price'], 
                q=4, 
                labels=['Budget', 'Standard', 'Premium', 'Luxury']
            )
            logger.info("Created Price_Category feature")
    
    def add_coordinate_features(self):
        """Add features based on coordinates"""
        if 'Lattitude' in self.df.columns and 'Longtitude' in self.df.columns:
            # Melbourne CBD coordinates (approximate)
            cbd_coords = (-37.8136, 144.9631)
            
            # Calculate distance to CBD
            distances = []
            for idx, row in self.df.iterrows():
                if pd.notna(row['Lattitude']) and pd.notna(row['Longtitude']):
                    property_coords = (row['Lattitude'], row['Longtitude'])
                    distance = geodesic(property_coords, cbd_coords).kilometers
                    distances.append(distance)
                else:
                    distances.append(np.nan)
            
            self.df['Distance_to_CBD'] = distances
            self.df['Distance_to_CBD'].fillna(self.df['Distance_to_CBD'].median(), inplace=True)
            logger.info("Added Distance_to_CBD feature")
    
    def create_interaction_features(self):
        """Create interaction features between important variables"""
        if 'Rooms' in self.df.columns and 'Bathroom' in self.df.columns:
            self.df['Rooms_Bathroom_Ratio'] = self.df['Rooms'] / self.df['Bathroom'].replace(0, 1)
        
        if 'Landsize' in self.df.columns and 'Rooms' in self.df.columns:
            self.df['Land_per_Room'] = self.df['Landsize'] / self.df['Rooms'].replace(0, 1)
        
        logger.info("Created interaction features")
    
    def run_feature_engineering(self):
        """Run all feature engineering steps"""
        logger.info("Starting feature engineering...")
        
        self.create_price_per_room()
        self.create_age_feature()
        self.create_land_efficiency()
        self.create_property_density()
        self.create_price_category()
        self.add_coordinate_features()
        self.create_interaction_features()
        
        logger.info(f"Feature engineering completed. New shape: {self.df.shape}")
        return self.df