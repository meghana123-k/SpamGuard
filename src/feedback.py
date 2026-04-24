import pandas as pd

FEEDBACK_FILE = "data/feedback.csv"


def save_feedback(text, predicted, actual):
    df = pd.read_csv(FEEDBACK_FILE)

    new_row = {
        "text": text,
        "predicted": predicted,
        "actual": actual
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(FEEDBACK_FILE, index=False)

    print("Feedback saved.")