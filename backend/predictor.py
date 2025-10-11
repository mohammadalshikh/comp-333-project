import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
import traceback


class MovieRatingPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.model_path = os.path.join(
            os.path.dirname(__file__), "trained_model.joblib"
        )

    def describe(self):
        """Describe the model and its features."""
        pass
    
    def integrate(self):
        """Load and combine the datasets for training."""
        imdb_data = pd.read_csv(
            os.path.join(os.path.dirname(__file__), "../data/imdb-movies.csv")
        )
        online_data = pd.read_csv(
            os.path.join(
                os.path.dirname(__file__), "../data/online-movies-and-show.csv"
            )
        )

        # Filter out TV shows from online data
        online_data = online_data[online_data["Type"] == "Movie"]

    def prepare(self):
        """Prepare the model for training using the combined datasets."""
        pass

    def predict(self, features):
        """Predict rating for a movie based on its features."""
        try:
            # Check if model file exists
            if not os.path.exists(self.model_path) or self.model is None:
                if not os.path.exists(self.model_path):
                    raise Exception("Model file not found")
                    saved_data = joblib.load(self.model_path)
                    self.model = saved_data["model"]
                    self.scaler = saved_data["scaler"]
                    self.feature_names = saved_data["feature_names"]
                self.interaction_features = [
                    f for f in self.feature_names if f.endswith("_interaction")
                ]

        # Prediction logic

        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            print(f"Stack trace: {traceback.format_exc()}")
            return None
