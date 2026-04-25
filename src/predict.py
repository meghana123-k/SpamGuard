import joblib
import os

MODEL_PATH = "models/spam_model.pkl"


def predict_email(text, threshold=0.6):
    if not os.path.exists(MODEL_PATH):
        raise Exception("Model not found. Train the model first.")

    # Load full pipeline
    model = joblib.load(MODEL_PATH)

    # Model expects raw text (pipeline handles preprocessing)
    prob = model.predict_proba([text])[0][1]

    prediction = 1 if prob >= threshold else 0

    return prediction, prob