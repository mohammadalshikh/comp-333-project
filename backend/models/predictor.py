import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

class MovieRatingPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = os.path.join(os.path.dirname(__file__), 'trained_model.joblib')
        
    def preprocess_data(self):
        # Load and merge datasets
        imdb_data = pd.read_csv('../data/imdb-movies.csv')
        online_data = pd.read_csv('../data/online-movies-and-show.csv')
        
        # Basic preprocessing (you'll need to expand this based on your data)
        # Clean and integrate the datasets
        # Handle missing values, encode categorical variables, etc.
        
        return X, y  # Return features and target (rating)
        
    def train(self):
        # Preprocess data
        X, y = self.preprocess_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Save model
        joblib.dump(self.model, self.model_path)
        
    def predict(self, features):
        if self.model is None and os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            
        if self.model is None:
            raise Exception("Model not trained yet")
            
        # Preprocess input features
        features_scaled = self.scaler.transform([features])
        
        # Make prediction
        prediction = self.model.predict(features_scaled)
        return float(prediction[0])