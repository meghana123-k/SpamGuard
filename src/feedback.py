import pandas as pd
import os

FEEDBACK_FILE = "data/feedback.csv"


def init_feedback_file():
    if not os.path.exists(FEEDBACK_FILE):
        os.makedirs("data", exist_ok=True)
        df = pd.DataFrame(columns=["text", "predicted", "actual"])
        df.to_csv(FEEDBACK_FILE, index=False)


def save_feedback(text, predicted, actual):
    init_feedback_file()

    df = pd.read_csv(FEEDBACK_FILE)

    new_row = {
        "text": text,
        "predicted": int(predicted),
        "actual": int(actual)
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(FEEDBACK_FILE, index=False)

    print("✅ Feedback saved")


# 🔥 IMPORTANT: retraining just calls train.py
def retrain_with_feedback():
    print("🔄 Retraining model using full pipeline...")
    
    from src.train import train_model
    train_model()

    print("✅ Model retrained with feedback")