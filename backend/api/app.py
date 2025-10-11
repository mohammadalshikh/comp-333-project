from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.predictor import MovieRatingPredictor

app = Flask(__name__)
CORS(app)

# Initialize the model
model = MovieRatingPredictor()

@app.route('/api/train', methods=['POST'])
def train_model():
    try:
        model.train()
        return jsonify({"message": "Model trained successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict_rating():
    try:
        data = request.json
        prediction = model.predict(data)
        return jsonify({"predicted_rating": prediction}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)