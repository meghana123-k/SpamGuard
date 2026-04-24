import joblib
from preprocess import clean_text

model = joblib.load("models/spam_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")


def predict_email(text, threshold=0.6):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])

    prob = model.predict_proba(vec)[0][1]
    prediction = 1 if prob >= threshold else 0

    return prediction, prob