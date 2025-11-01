import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
import traceback
import ast
from transformers import pipeline
import torch

_P = {}  # cache for encoder embedding pipelines, don't delete or will load encoder for every string.


class MovieRatingPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.model_path = os.path.join(
            os.path.dirname(__file__), "trained_model.joblib"
        )
        self.data_path = os.path.dirname(os.path.dirname(__file__))
        self._text_embed_model = "sentence-transformers/all-MiniLM-L6-v2"
        self._text_embed_dim = 384

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
        file_map = {"IMDb Movies": "imdb.csv", "TMDB Movies": "tmdb.csv"}

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
            "columns_list": list(df.columns),
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
                        "movie_facebook_likes",
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
                        "facenumber_in_poster",
                    ]
                },
                "Cast": {
                    "attributes": [
                        "cast_id (PK)",
                        "movie_id (FK)",
                        "actor_name",
                        "actor_position",
                        "facebook_likes",
                        "cast_total_facebook_likes",
                    ]
                },
                "Director": {
                    "attributes": [
                        "director_id (PK)",
                        "movie_id (FK)",
                        "director_name",
                        "director_facebook_likes",
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
                        "num_voted_users",
                    ]
                },
                "Production": {
                    "attributes": [
                        "production_id (PK)",
                        "movie_id (FK)",
                        "production_companies",
                        "production_countries",
                        "spoken_languages",
                    ]
                },
            },
            "relationships": [
                "Movie (1) -- (1) MovieMetadata",
                "Movie (1) -- (N) Cast",
                "Movie (N) -- (1) Director",
                "Movie (1) -- (1) Ratings",
                "Movie (1) -- (1) Production",
            ],
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
                    "tmdb": self._get_dataset_stats(tmdb_data, "TMDB Movies"),
                },
                "data_sources": self._get_data_sources(),
                "erd": self._generate_erd(),
            }

            return description

        except Exception as e:
            print(f"Error in describe(): {str(e)}")
            print(f"Stack trace: {traceback.format_exc()}")
            return None

    def integrate(self):
        """Load and combine the datasets for training."""
        # 1. Load datasets using helper function
        imdb_data, tmdb_data = self._load_datasets()
        if imdb_data is None or tmdb_data is None:
            return None

        try:
            # 2. Drop unnecessary columns
            imdb_data = imdb_data.drop(
                columns=["movie_imdb_link", "num_user_for_reviews", "country"], axis=1
            )

            tmdb_data = tmdb_data.drop(
                columns=[
                    "id",
                    "imdb_id",
                    "original_language",
                    "belongs_to_collection",
                    "homepage",
                    "overview",
                    "popularity",
                    "poster_path",
                    "status",
                    "tagline",
                    "title",
                    "video",
                ],
                axis=1,
            )

            # 3. Standardize column names
            imdb_data = imdb_data.rename(
                columns={
                    "director_name": "director",
                    "director_facebook_likes": "director_likes",
                    "num_critic_for_reviews": "critic_votes",
                    "movie_title": "title",
                    "num_voted_users": "user_votes",
                    "title_year": "year",
                    "imdb_score": "rating",
                    "language": "languages",
                    "actor_1_name": "actor_1",
                    "actor_2_name": "actor_2",
                    "actor_3_name": "actor_3",
                    "actor_1_facebook_likes": "actor_1_likes",
                    "actor_2_facebook_likes": "actor_2_likes",
                    "actor_3_facebook_likes": "actor_3_likes",
                    "cast_total_facebook_likes": "cast_likes",
                    "movie_facebook_likes": "movie_likes",
                }
            )

            tmdb_data = tmdb_data.rename(
                columns={
                    "revenue": "gross",
                    "runtime": "duration",
                    "spoken_languages": "languages",
                    "release_date": "year",
                    "vote_average": "rating",
                    "vote_count": "user_votes",
                    "original_title": "title",
                }
            )

            # 4. Standardize genres
            imdb_data["genres"] = imdb_data["genres"].apply(
                lambda x: x.split("|") if isinstance(x, str) else []
            )
            tmdb_data["genres"] = tmdb_data["genres"].apply(
                lambda x: (
                    [d["name"] for d in ast.literal_eval(x)]
                    if isinstance(x, str)
                    else []
                )
            )

            # 5. Standardize languages
            imdb_data["languages"] = imdb_data["languages"].apply(
                lambda x: (
                    [lang.strip() for lang in x.split(",")]
                    if isinstance(x, str)
                    else []
                )
            )
            tmdb_data["languages"] = tmdb_data["languages"].apply(
                lambda x: (
                    [d["name"] for d in ast.literal_eval(x)]
                    if isinstance(x, str)
                    else []
                )
            )

            # 6. Standardize duration type
            imdb_data["duration"] = pd.to_numeric(
                imdb_data["duration"], errors="coerce"
            )
            tmdb_data["duration"] = pd.to_numeric(
                tmdb_data["duration"], errors="coerce"
            )

            # 7. Standardize budget type
            imdb_data["budget"] = pd.to_numeric(imdb_data["budget"], errors="coerce")
            tmdb_data["budget"] = pd.to_numeric(tmdb_data["budget"], errors="coerce")

            # 8. Standardize gross type
            imdb_data["gross"] = pd.to_numeric(imdb_data["gross"], errors="coerce")
            tmdb_data["gross"] = pd.to_numeric(tmdb_data["gross"], errors="coerce")

            # 9. Standardize production companies and countries
            def extract_names(x):
                if isinstance(x, str):
                    try:
                        data = ast.literal_eval(x)
                        if isinstance(data, list):
                            return [d.get("name") for d in data if isinstance(d, dict) and "name" in d]
                    except Exception:
                        pass
                return []

            tmdb_data["production_companies"] = tmdb_data["production_companies"].apply(extract_names)

            tmdb_data["production_countries"] = tmdb_data["production_countries"].apply(extract_names)

            # 10. Standardize year format
            tmdb_data["year"] = pd.to_datetime(
                tmdb_data["year"], errors="coerce"
            ).dt.year

            # 11. Standardize content rating
            # Convert string 'True'/'False' to boolean first
            tmdb_data["adult"] = tmdb_data["adult"].astype(str).str.lower() == 'true'
            tmdb_data["content_rating"] = tmdb_data["adult"].apply(
                lambda x: "NC-17" if x else None
            )
            tmdb_data.drop(columns=["adult"], axis=1, inplace=True)

            def normalize_title(title):
                return (
                    str(title)
                    .strip()
                    .lower()
                    .replace(":", "")
                    .replace("-", "")
                    .replace("(", "")
                    .replace(")", "")
                    .replace(",", "")
                )

            imdb_data["title"] = imdb_data["title"].apply(normalize_title)
            tmdb_data["title"] = tmdb_data["title"].apply(normalize_title)

            # 12. Merge on 'title' using OUTER join to keep all records
            merged = pd.merge(
                imdb_data,
                tmdb_data,
                on=["title"],
                how="outer",
                suffixes=("_x", "_y")
            )

            # 13. Combine shared string fields - prefer IMDB for content_rating (more comprehensive)
            merged["content_rating"] = merged["content_rating_x"].combine_first(
                merged["content_rating_y"]
            )

            # 14. Combine shared numeric fields - sum user_votes, average others
            merged["user_votes"] = merged["user_votes_x"].fillna(0) + merged[
                "user_votes_y"
            ].fillna(0)

            # For duration, gross, budget, rating: average when both exist, otherwise take non-null
            merged["duration"] = merged[["duration_x", "duration_y"]].mean(axis=1, skipna=True)
            merged["gross"] = merged[["gross_x", "gross_y"]].mean(axis=1, skipna=True)
            merged["budget"] = merged[["budget_x", "budget_y"]].mean(axis=1, skipna=True)
            merged["rating"] = merged[["rating_x", "rating_y"]].mean(axis=1, skipna=True)

            # 15. Combine shared list fields - merge unique values from both
            merged["genres"] = merged.apply(
                lambda row: list(
                    set(
                        (
                            row["genres_x"]
                            if isinstance(row["genres_x"], list)
                            else []
                        )
                        + (
                            row["genres_y"]
                            if isinstance(row["genres_y"], list)
                            else []
                        )
                    )
                ),
                axis=1,
            )
            merged["languages"] = merged.apply(
                lambda row: list(
                    set(
                        (
                            row["languages_x"]
                            if isinstance(row["languages_x"], list)
                            else []
                        )
                        + (
                            row["languages_y"]
                            if isinstance(row["languages_y"], list)
                            else []
                        )
                    )
                ),
                axis=1,
            )

            # 16. Year: prefer non-null value from either
            merged["year"] = merged["year_y"].combine_first(merged["year_x"])

            # 17. Drop the old duplicate columns
            merged.drop(
                columns=[
                    "user_votes_x",
                    "user_votes_y",
                    "duration_x",
                    "duration_y",
                    "gross_x",
                    "gross_y",
                    "budget_x",
                    "budget_y",
                    "rating_x",
                    "rating_y",
                    "year_x",
                    "year_y",
                    "genres_x",
                    "genres_y",
                    "languages_x",
                    "languages_y",
                    "content_rating_x",
                    "content_rating_y",
                ],
                inplace=True,
            )

            # NEW: run your prepare step (minimal call)
            merged = self.prepare(merged)
            return merged

        except Exception as e:
            print(f"Error integrating datasets: {str(e)}")
            print(f"Stack trace: {traceback.format_exc()}")
            return None

    def _get_embedding_pipeline(self):
        """Return a cached HuggingFace pipeline, moving to GPU if available."""
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
        else:
            device = -1  # CPU

        cache_key = (self._text_embed_model, device)
        p = _P.get(cache_key)
        if p is None:
            p = pipeline("feature-extraction", model=self._text_embed_model, device=device)
            hidden_size = getattr(getattr(p.model, "config", None), "hidden_size", None)
            if isinstance(hidden_size, int) and hidden_size > 0:
                self._text_embed_dim = hidden_size
            _P[cache_key] = p
        return p

    def _embed_string(self, text, normalize=True):
        """Return a mean-pooled, self-attention-based sentence vector (list[float])."""
        try:
            if isinstance(text, list):
                parts = []
                for item in text:
                    if item is None:
                        continue
                    parts.append(str(item))
                text = " ".join(parts)
            elif text is None:
                text = ""
            else:
                text = str(text)

            if not text.strip():
                text = "Unknown"

            p = self._get_embedding_pipeline()

            tok = p.tokenizer(text, return_tensors="pt", truncation=True)
            for k in list(tok.keys()):
                tok[k] = tok[k].to(p.model.device)

            with torch.no_grad():
                h = p.model(**tok).last_hidden_state  # [1, T, D]

            mask = tok["attention_mask"].unsqueeze(-1)  # [1, T, 1]
            s = (h * mask).sum(dim=1) / mask.sum(dim=1)  # [1, D]

            if normalize:
                s = s / (s.norm(dim=1, keepdim=True) + 1e-12)

            vec = s.squeeze(0).detach().cpu().tolist()
            return vec
        except Exception as e:
            print(f"Warning: embedding failed for text='{str(text)[:60]}...': {e}")
            return [0.0] * self._text_embed_dim

    def _embed_text_columns(self, df, cols):
        """Replace each text column with its embedding columns."""
        for col in cols:
            if col not in df.columns:
                continue
            try:
                vectors = []
                for i in range(len(df)):
                    v = self._embed_string(df.iloc[i][col], normalize=True)
                    vectors.append(v)

                if len(vectors) == 0:
                    continue

                dim = len(vectors[0])
                new_cols = []
                for j in range(dim):
                    new_cols.append(f"{col}_emb_{j}")

                emb_df = pd.DataFrame(vectors, index=df.index, columns=new_cols)
                df = pd.concat([df.drop(columns=[col]), emb_df], axis=1)

            except Exception as e:
                print(f"Warning: embedding column '{col}' failed: {e}")
        return df

    def prepare(self, df=None):
        """Your preparation logic; df is optional for back-compat (returns None if not provided)."""
        if df is None:
            return None

        # integers
        int_cols = ["critic_votes","duration","gross","user_votes","facenumber_in_poster","budget","year"]
        for col in int_cols:
            if col not in df.columns:
                continue
            s = pd.to_numeric(df[col], errors="coerce")
            mu = s.mean()
            df[col] = s.astype("Int64") if pd.isna(mu) else s.fillna(int(round(mu))).astype("Int64")

        # floats
        if "aspect_ratio" in df.columns:
            s = pd.to_numeric(df["aspect_ratio"], errors="coerce")
            mu = s.mean()
            df["aspect_ratio"] = (s if pd.isna(mu) else s.fillna(round(mu, 2))).astype("Float64")

        # keep list columns as lists
        for col in ["genres","languages"]:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda v: v if isinstance(v, list)
                    else ([] if (pd.isna(v) or v is None or v == "") else [str(v)])
                )

        # string columns
        for col in ["director","plot_keywords"]:
            if col in df.columns:
                s = df[col].astype("string").str.strip().replace(
                    ["", " ", "NA", "N/A","n/a","na","NaN","nan","NULL","null","None","?"], pd.NA
                )
                df[col] = s.fillna("Unknown")

        # ratings default
        if "content_rating" in df.columns:
            s = df["content_rating"].astype("string").str.strip().replace(
                ["", " ", "NA","N/A","n/a","na","NaN","nan","NULL","null","None","?"], pd.NA
            )
            df["content_rating"] = s.fillna("PG-13")

        # NEW: embed selected text features (minimal intrusion)
        text_cols = ["title", "director", "plot_keywords", "content_rating", "genres", "languages", "production_companies", "production_countries"]
        df = self._embed_text_columns(df, text_cols)

        return df

    def predict(self, features):
        """Predict rating for a movie based on its features."""
        try:
            # Check if model file exists / load only if needed (minimal fix)
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
