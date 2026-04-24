import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import os
from preprocess import clean_text

os.makedirs("models", exist_ok=True)
def train_model():
    # Load datasets
    enron = pd.read_csv("data/enron_spam_data.csv")
    sms = pd.read_csv("data/SMSSpamCollection", sep="\t", header=None, names=["label", "text"])

    sms['label'] = sms['label'].map({'ham': 0, 'spam': 1})

    # Clean
    # Combine Subject + Message (same as notebook)
    enron['text'] = enron['Subject'].fillna('') + " " + enron['Message'].fillna('')

    # Clean
    enron['clean_text'] = enron['text'].apply(clean_text)

    # Fix label column
    enron = enron.rename(columns={'Spam/Ham': 'label'})
    enron['label'] = enron['label'].map({'ham': 0, 'spam': 1})
    sms['clean_text'] = sms['text'].apply(clean_text)

    # Combine
    df = pd.concat([
        enron[['clean_text', 'label']],
        sms[['clean_text', 'label']]
    ])

    # Vectorize
    vectorizer = TfidfVectorizer(max_features=7000, ngram_range=(1,2), min_df=5)
    X = vectorizer.fit_transform(df['clean_text'])
    y = df['label']

    # Train
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    # Save
    joblib.dump(model, "models/spam_model.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")

    print("Model trained and saved.")


if __name__ == "__main__":
    train_model()