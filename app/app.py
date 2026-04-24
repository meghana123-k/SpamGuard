import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from flask import Flask, request, jsonify
import joblib
from feedback import save_feedback
from preprocess import clean_text

# Fix import path


app = Flask(__name__)

# Load model
model = joblib.load("models/spam_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")


@app.route("/")
def home():
    return "Spam Detection API Running"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    text = data.get("text", "")

    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])

    prob = model.predict_proba(vec)[0][1]
    prediction = "spam" if prob >= 0.6 else "ham"

    return jsonify({
        "prediction": prediction,
        "confidence": float(prob)
    })
@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json()

    text = data.get("text", "")
    predicted = data.get("predicted")
    actual = data.get("actual")

    if predicted is None or actual is None:
        return jsonify({"error": "Missing predicted or actual label"}), 400

    save_feedback(text, predicted, actual)

    return jsonify({
        "message": "Feedback saved successfully"
    })

if __name__ == "__main__":
    app.run(debug=True)