from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer

app = Flask(__name__)

# Load the trained model
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'model.pkl')
model_info = joblib.load(model_path)
model = model_info[0]


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.form
        
        # Create feature dictionary
        features = {
            'Rooms': float(data['rooms']),
            'Type': data['type'],
            'Postcode': float(data['postcode']),  # Fixed: 'Postcard' to 'Postcode'
            'Bedroom2': float(data.get('bedrooms', data['rooms'])),
            'Bathroom': float(data.get('bathrooms', 1)),
            'Car': float(data.get('car', 1)),
            'Landsize': float(data['landsize']),
            'BuildingArea': float(data.get('building_area', data['landsize'])),
            'YearBuilt': float(data.get('year_built', 2000)),   
            'Propertycount': float(data.get('property_count', 1000))
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([features])
        
        # Apply feature engineering
        engineer = FeatureEngineer(input_df)
        input_df = engineer.run_feature_engineering()
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        return render_template('index.html', 
                             prediction_text=f'Estimated Price: ${prediction:,.2f}')
        
    except Exception as e: 
        return render_template('index.html', 
                             prediction_text=f'Error: {str(e)}')

# Optional: API endpoint for JSON responses
@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        
        features = {
            'Rooms': float(data['rooms']),
            'Type': data['type'],
            'Postcode': float(data['postcode']),
            'Bedroom2': float(data.get('bedrooms', data['rooms'])),
            'Bathroom': float(data.get('bathrooms', 1)),
            'Car': float(data.get('car', 1)),
            'Landsize': float(data['landsize']),
            'BuildingArea': float(data.get('building_area', data['landsize'])),
            'YearBuilt': float(data.get('year_built', 2000)),   
            'Propertycount': float(data.get('property_count', 1000))
        }

        input_df = pd.DataFrame([features])
        engineer = FeatureEngineer(input_df)
        input_df = engineer.run_feature_engineering()
        prediction = model.predict(input_df)[0]
        
        return jsonify({
            'success': True,
            'prediction': float(prediction),
            'formatted_price': f"${prediction:,.2f}"
        })
        
    except Exception as e: 
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)