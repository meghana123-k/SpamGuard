import pandas as pd

FEEDBACK_FILE = "data/feedback.csv"

from preprocess import clean_text

def retrain_with_feedback():
    
    # Load original data
    enron = pd.read_csv("data/enron_spam_data.csv")
    sms = pd.read_csv("data/SMSSpamCollection", sep="\t", header=None, names=["label", "text"])
    sms['label'] = sms['label'].map({'ham': 0, 'spam': 1})

    enron['text'] = enron['Subject'].fillna('') + " " + enron['Message'].fillna('')
    enron = enron.rename(columns={'Spam/Ham': 'label'})
    enron['label'] = enron['label'].map({'ham': 0, 'spam': 1})

    # Load feedback
    feedback = pd.read_csv("data/feedback.csv")

    # Clean feedback
    feedback['clean_text'] = feedback['text'].apply(clean_text)
    feedback = feedback.rename(columns={'actual': 'label'})

    # Combine all
    df = pd.concat([
        enron[['text','label']].assign(clean_text=enron['text'].apply(clean_text)),
        sms[['text','label']].assign(clean_text=sms['text'].apply(clean_text)),
        feedback[['clean_text','label']]
    ])

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    import joblib
    import os

    vectorizer = TfidfVectorizer(max_features=7000, ngram_range=(1,2), min_df=2)
    X = vectorizer.fit_transform(df['clean_text'])
    y = df['label']

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/spam_model.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")

    print("Model retrained with feedback!")
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