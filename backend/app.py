from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys

# Import the predictor from the same directory
from predictor import MovieRatingPredictor

app = Flask(__name__)
CORS(app)

# Initialize the model
model = MovieRatingPredictor()


def check_model_status():
    """Check if the model is trained and return its status."""
    return os.path.exists(model.model_path)


@app.route("/api/train", methods=["POST"])
def train_model():
    """Endpoint to manually train the model."""
    try:
        print("Training model... This might take a few moments...")
        model.train()
        print("Model training completed successfully!")
        return jsonify({"message": "Model trained successfully"}), 200
    except Exception as e:
        print(f"Error training model: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/status", methods=["GET"])
def get_model_status():
    """Get the current training status of the model."""
    is_trained = check_model_status()
    return jsonify(
        {
            "trained": is_trained,
            "message": "Trained" if is_trained else "Untrained",
        }
    )


@app.route("/api/delete", methods=["POST"])
def delete_model():
    """Delete the trained model file."""
    try:
        if os.path.exists(model.model_path):
            os.remove(model.model_path)
            print("\nModel deleted successfully!\n")
            return jsonify({"message": "Model deleted successfully"}), 200
        else:
            return jsonify({"message": "No model file found"}), 404
    except Exception as e:
        print(f"Error deleting model: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def predict_rating():
    try:
        data = request.json
        # Validate input data silently
        required_fields = ["year", "genre", "language", "duration"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Ensure numeric fields are valid numbers
        try:
            data["year"] = int(data["year"])
            data["duration"] = int(data["duration"])
        except ValueError:
            return jsonify({"error": "Year and duration must be valid numbers"}), 400

        # Get prediction
        prediction = model.predict(data)
        return jsonify({"predicted_rating": prediction}), 200
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Disable all logging
    import logging

    log = logging.getLogger("werkzeug")
    log.disabled = True

    # Disable Flask logging
    cli = sys.modules["flask.cli"]
    cli.show_server_banner = lambda *x: None
    app.logger.disabled = True

    # Print initial status
    is_trained = check_model_status()
    print(f"\nModel status: {'Trained' if is_trained else 'Untrained'}")
    print("\nServer started\n")

    # Run without debug mode and logging
    app.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False)
