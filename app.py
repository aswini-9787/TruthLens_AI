
from flask import Flask, render_template, request, jsonify
import os
import joblib

app = Flask(__name__)

# Trained Machine Learning model
MODEL_PATH = os.path.join(
    "model",
    "fake_news_model.pkl"
)

# Load model once when Flask starts
model = None

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("TruthLens AI model loaded successfully!")
else:
    print("Trained model not found!")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():

    data = request.get_json(silent=True) or {}
    news = data.get("news", "").strip()

    # Check empty input
    if not news:
        return jsonify({
            "error": "Please enter a news article."
        }), 400

    # Check trained model
    if model is None:
        return jsonify({
            "error": "AI model is not available."
        }), 503

    try:
        # Predict news category
        prediction = model.predict([news])[0]

        # Calculate prediction probabilities
        probabilities = model.predict_proba([news])[0]

        # Get confidence of predicted category
        predicted_index = list(model.classes_).index(prediction)

        confidence = round(
            float(probabilities[predicted_index]) * 100,
            2
        )

        return jsonify({
            "prediction": str(prediction),
            "confidence": confidence,
            "message": "News analysis completed successfully."
        })

    except Exception as e:

        print("Analysis Error:", e)

        return jsonify({
            "error": "An error occurred during news analysis."
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
