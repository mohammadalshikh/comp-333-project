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
        self.feature_names = None
        self.model_path = os.path.join(
            os.path.dirname(__file__), "trained_model.joblib"
        )

    def preprocess_data(self):
        # Load and merge datasets
        try:
            imdb_data = pd.read_csv(
                os.path.join(os.path.dirname(__file__), "../data/imdb-movies.csv")
            )

            # Rename and select relevant columns from IMDb data
            imdb_data = imdb_data.rename(
                columns={
                    "title_year": "year",
                    "genres": "genre",
                    "language": "language",
                    "duration": "duration",
                    "imdb_score": "rating",
                }
            )

            # Clean up the data
            imdb_data["year"] = pd.to_numeric(imdb_data["year"], errors="coerce")
            imdb_data["duration"] = pd.to_numeric(
                imdb_data["duration"], errors="coerce"
            )

            # Take the first genre if multiple are present
            imdb_data["genre"] = imdb_data["genre"].str.split("|").str[0]

            # Select features and target
            features = ["year", "genre", "language", "duration"]
            X = imdb_data[features].copy()
            y = imdb_data["rating"]

            # Drop rows with missing values
            mask = X.notna().all(axis=1) & y.notna()
            X = X[mask]
            y = y[mask]

            # Convert year to int
            X["year"] = X["year"].astype("int")
            X["duration"] = X["duration"].astype("int")

            # One-hot encode categorical variables
            X = pd.get_dummies(X, columns=["genre", "language"])

            # Store feature names
            self.feature_names = X.columns.tolist()

            return X, y
        except Exception as e:
            raise Exception(f"Error preprocessing data: {str(e)}")

    def train(self):
        # Preprocess data
        X, y = self.preprocess_data()

        if len(X) == 0:
            raise Exception("No training data available")

        print(f"Training model with {len(X)} samples")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale numeric features
        numeric_features = ["year", "duration"]
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()

        X_train_scaled[numeric_features] = self.scaler.fit_transform(
            X_train[numeric_features]
        )
        X_test_scaled[numeric_features] = self.scaler.transform(
            X_test[numeric_features]
        )

        # Train model with more trees and better parameters
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
        )
        self.model.fit(X_train_scaled, y_train)

        # Evaluate model
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        print(f"Model R² score - Training: {train_score:.4f}, Test: {test_score:.4f}")

        # Get feature importance
        feature_importance = pd.DataFrame(
            {"feature": X_train.columns, "importance": self.model.feature_importances_}
        ).sort_values("importance", ascending=False)
        print("\nTop 10 most important features:\n")
        print(feature_importance.head(10))

        # Save model, scaler, and metadata
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "train_score": train_score,
            "test_score": test_score,
            "feature_importance": feature_importance.to_dict(),
        }
        print(f"\nSaving model with {len(self.feature_names)} features")
        joblib.dump(model_data, self.model_path)

    def predict(self, features):
        try:
            if self.model is None:
                if os.path.exists(self.model_path):
                    saved_data = joblib.load(self.model_path)
                    self.model = saved_data["model"]
                    self.scaler = saved_data["scaler"]
                    self.feature_names = saved_data["feature_names"]
                else:
                    raise Exception("Model not trained yet")

            # Input validation
            if (
                not isinstance(features["year"], (int, str))
                or not str(features["year"]).isdigit()
            ):
                raise ValueError("Year must be a valid number")
            if (
                not isinstance(features["duration"], (int, str))
                or not str(features["duration"]).isdigit()
            ):
                raise ValueError("Duration must be a valid number")
            if not isinstance(features["genre"], str) or not features["genre"].strip():
                raise ValueError("Genre must be a non-empty string")
            if (
                not isinstance(features["language"], str)
                or not features["language"].strip()
            ):
                raise ValueError("Language must be a non-empty string")

            # Create a DataFrame with the input features
            input_df = pd.DataFrame(
                [
                    {
                        "year": int(features["year"]),
                        "duration": int(features["duration"]),
                        "genre": features["genre"].strip(),
                        "language": features["language"].strip(),
                    }
                ]
            )

            print("\n\nProcessing input features:")
            print("{")
            for key, value in input_df.to_dict("records")[0].items():
                print(f"    {key}: {value}")
            print("}")

            # One-hot encode categorical variables
            input_df = pd.get_dummies(input_df, columns=["genre", "language"])

            # Ensure all features from training are present
            for feature in self.feature_names:
                if feature not in input_df.columns:
                    input_df[feature] = 0

            # Reorder columns to match training data
            input_df = input_df[self.feature_names]

            # Scale numeric features
            numeric_features = ["year", "duration"]
            input_df[numeric_features] = self.scaler.transform(
                input_df[numeric_features]
            )

            # Make prediction
            prediction = self.model.predict(input_df)

            print(f"Processed features shape: {input_df.shape}")
            print("Non-zero features used in prediction:")
            non_zero_features = input_df.loc[:, (input_df != 0).any()]
            features_dict = non_zero_features.to_dict("records")[0]
            print("{")
            for key, value in features_dict.items():
                print(f"    {key}: {value}")
            print("}")

            return float(prediction[0])
        except Exception as e:
            raise Exception(f"Error making prediction: {str(e)}")
