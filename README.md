# COMP 333 Project - Group #12

### How to run
From root folder:
```bash
chmod +x scripts/*.sh
```
```bash
./scripts/setup.sh
```
In one terminal run:
```bash
cd backend
python app.py
```
In another terminal run:
```bash
cd frontend
npm start
```

### Problem statement
Given the extensive amount of movie data available, it would be interesting to explore which factors contribute most to a movie’s success. Therefore, this project aims to identify the characteristics of movies that tend to achieve that, namely higher IMDb ratings. The analysis  will also aim to present visualized results using Python for better interpretation.

In this project, IMDb scores serve as the response variable, while the remaining attributes from the IMDb 5000 movie dataset as well as the online platforms dataset are used to make predictions. 

The insights gained can assist film studios in understanding what elements lead to a commercially successful movie.

### Goal
Predict a movie rating given its features such as year, crew, genres, number of votes, language etc.

### Prediction type
Regression, predicting a continuous numeric value.

### Datasets

- **[Kaggle: IMDb 5000 Movie Dataset](https://www.kaggle.com/code/aditimulye/imdb-5000-movie-dataset-analysis/input?select=movie_metadata.csv)**  
	Contains movies from IMDb (actors, directors, ratings, budget, income, etc.)

- **[Kaggle: TMDB 45000 Movie Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset/data?select=movies_metadata.csv)**  
	Contains movies from TMDB (budget, genres, languages, year, etc.)

### Contributors
- Mohammad Alshikh 
- Bishoy Abdelmalik
- Arina Sabzevari
- Lucas Miquet-Westphal
- Jalal Zakaria
