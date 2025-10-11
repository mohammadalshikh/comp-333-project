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
        self.data_path = os.path.dirname(os.path.dirname(__file__))

    def _load_datasets(self):
        """Helper function to load all datasets
        
        Returns:
            tuple: (imdb_data, tmdb_data) pandas DataFrames, or (None, None) if loading fails
        """
        try:
            imdb_path = os.path.join(self.data_path, "data/imdb.csv")
            tmdb_path = os.path.join(self.data_path, "data/tmdb.csv")

            if not os.path.exists(imdb_path) or not os.path.exists(tmdb_path):
                raise FileNotFoundError("One or both dataset files not found")

            imdb_data = pd.read_csv(imdb_path, low_memory=False)
            tmdb_data = pd.read_csv(tmdb_path, low_memory=False)

            return imdb_data, tmdb_data

        except Exception as e:
            print(f"Error loading datasets: {str(e)}")
            print(f"Stack trace: {traceback.format_exc()}")
            return None, None

    def _get_dataset_stats(self, df, name):
        """Helper function to get basic statistics for a dataset"""
        
        # Map dataset names to file names
        file_map = {
            "IMDb Movies": "imdb.csv",
            "TMDB Movies": "tmdb.csv"
        }
        
        # Get actual file size from the data file
        file_name = file_map.get(name)
        if file_name:
            file_path = os.path.join(self.data_path, "data", file_name)
            try:
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            except OSError:
                file_size_mb = 0
        else:
            file_size_mb = 0

        return {
            "name": name,
            "rows": len(df),
            "columns": len(df.columns),
            "size_mb": file_size_mb,
            "null_counts": df.isnull().sum().to_dict(),
            "columns_list": list(df.columns)
        }

    def _get_data_sources(self):
        """Helper function to document data sources"""
        return {
            "imdb": {
                "source": "IMDb Movies Dataset",
                "description": "Contains detailed movie information including ratings and metadata",
                "url": "https://www.kaggle.com/code/aditimulye/imdb-5000-movie-dataset-analysis",
            },
            "tmdb": {
                "source": "TMDB 45K Movies Dataset",
                "description": "Contains comprehensive movie information including budget, revenue, and detailed metadata",
                "url": "https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset/data?select=movies_metadata.csv",
            },
        }

    def _generate_erd(self):
        """Helper function to generate ERD information"""
        return {
            "entities": {
                "Movie": {
                    "attributes": [
                        "id (PK)",
                        "title",
                        "original_title",
                        "release_date",
                        "duration/runtime",
                        "budget",
                        "revenue/gross",
                        "language",
                        "country",
                        "content_rating",
                        "status",
                        "adult",
                        "video",
                        "homepage",
                        "tagline",
                        "overview",
                        "aspect_ratio",
                        "movie_facebook_likes"
                    ]
                },
                "MovieMetadata": {
                    "attributes": [
                        "metadata_id (PK)",
                        "movie_id (FK)",
                        "genres",
                        "plot_keywords",
                        "belongs_to_collection",
                        "popularity",
                        "poster_path",
                        "facenumber_in_poster"
                    ]
                },
                "Cast": {
                    "attributes": [
                        "cast_id (PK)",
                        "movie_id (FK)",
                        "actor_name",
                        "actor_position",
                        "facebook_likes",
                        "cast_total_facebook_likes"
                    ]
                },
                "Director": {
                    "attributes": [
                        "director_id (PK)",
                        "movie_id (FK)",
                        "director_name",
                        "director_facebook_likes"
                    ]
                },
                "Ratings": {
                    "attributes": [
                        "rating_id (PK)",
                        "movie_id (FK)",
                        "imdb_id",
                        "imdb_score",
                        "vote_average",
                        "vote_count",
                        "num_critic_reviews",
                        "num_user_reviews",
                        "num_voted_users"
                    ]
                },
                "Production": {
                    "attributes": [
                        "production_id (PK)",
                        "movie_id (FK)",
                        "production_companies",
                        "production_countries",
                        "spoken_languages"
                    ]
                }
            },
            "relationships": [
                "Movie (1) -- (1) MovieMetadata",
                "Movie (1) -- (N) Cast",
                "Movie (N) -- (1) Director",
                "Movie (1) -- (1) Ratings",
                "Movie (1) -- (1) Production"
            ]
        }

    def describe(self):
        """Generate comprehensive description of the datasets"""
        try:
            # Load datasets using helper function
            imdb_data, tmdb_data = self._load_datasets()
            if imdb_data is None or tmdb_data is None:
                return None

            description = {
                "dataset_statistics": {
                    "imdb": self._get_dataset_stats(imdb_data, "IMDb Movies"),
                    "tmdb": self._get_dataset_stats(tmdb_data, "TMDB Movies")
                },
                "data_sources": self._get_data_sources(),
                "erd": self._generate_erd()
            }

            return description

        except Exception as e:
            print(f"Error in describe(): {str(e)}")
            print(f"Stack trace: {traceback.format_exc()}")
            return None

    def integrate(self):
        """Load and combine the datasets for training."""
        # Load datasets using helper function
        imdb_data, tmdb_data = self._load_datasets()
        if imdb_data is None or tmdb_data is None:
            return None

        # TODO: Implement integration logic between IMDB and TMDB datasets
        # Key fields to match: imdb_id (from TMDB) with corresponding ID in IMDB dataset

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
