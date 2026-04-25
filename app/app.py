import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from flask import Flask, request, jsonify, render_template
import joblib

from feedback import save_feedback, retrain_with_feedback

MODEL_PATH = "models/spam_model.pkl"

app = Flask(__name__)

# 🔹 Load model
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise Exception("Model not found. Train the model first.")
    return joblib.load(MODEL_PATH)

model = load_model()


# -------------------------------
# Home UI
# -------------------------------
@app.route("/")
def home():
    return render_template("index.html", confidence=0)


# -------------------------------
# API Prediction
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    global model

    data = request.get_json()
    text = data.get("text", "")

    prob = model.predict_proba([text])[0][1]
    prediction = "spam" if prob >= 0.6 else "ham"

    return jsonify({
        "prediction": prediction,
        "confidence": float(prob)
    })


# -------------------------------
# UI Prediction
# -------------------------------
@app.route("/predict_ui", methods=["POST"])
def predict_ui():
    global model

    text = request.form.get("text")

    prob = model.predict_proba([text])[0][1]
    prediction = "spam" if prob >= 0.6 else "ham"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=round(prob * 100, 2),
        text=text
    )


# -------------------------------
# Feedback
# -------------------------------
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


# -------------------------------
# Retrain
# -------------------------------
@app.route("/retrain", methods=["POST"])
def retrain():
    global model

    retrain_with_feedback()

    # 🔥 reload updated model
    model = load_model()

    return jsonify({
        "message": "Model retrained and reloaded successfully"
    })


# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)