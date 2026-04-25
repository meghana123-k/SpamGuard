import pandas as pd
import numpy as np
import os
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MinMaxScaler

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from preprocess import TextCleaner, MetadataExtractor


DATA_PATH = "data"
MODEL_PATH = "models"


# -------------------------------
# Load and prepare data
# -------------------------------
def load_data():

    # Load Enron
    enron = pd.read_csv(f"{DATA_PATH}/enron_spam_data.csv")

    enron['text'] = enron['Subject'].fillna('') + " " + enron['Message'].fillna('')
    enron = enron.rename(columns={'Spam/Ham': 'label'})
    enron['label'] = enron['label'].map({'ham': 0, 'spam': 1})
    enron = enron[['text', 'label']]

    # Load SMS
    sms = pd.read_csv(f"{DATA_PATH}/SMSSpamCollection", sep="\t", header=None, names=["label", "text"])
    sms['label'] = sms['label'].map({'ham': 0, 'spam': 1})
    sms = sms[['text', 'label']]

    # Load feedback
    feedback_path = f"{DATA_PATH}/feedback.csv"
    if os.path.exists(feedback_path):
        feedback = pd.read_csv(feedback_path)
        feedback = feedback.rename(columns={'actual': 'label'})
        feedback = feedback[['text', 'label']]

        # 🔥 Give more weight to feedback
        feedback = pd.concat([feedback] * 3, ignore_index=True)
    else:
        feedback = pd.DataFrame(columns=["text", "label"])

    # Combine all
    df = pd.concat([enron, sms, feedback], ignore_index=True)

    # Clean invalid rows
    df = df.dropna(subset=["text", "label"])

    return df


# -------------------------------
# Evaluation function
# -------------------------------
def evaluate(model, X_test, y_test, name):

    y_pred = model.predict(X_test)

    print(f"\n--- {name} ---")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1 Score :", f1_score(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return f1_score(y_test, y_pred)


# -------------------------------
# Training pipeline
# -------------------------------
def train_model():

    df = load_data()

    X = df['text']
    y = df['label'].astype(int)

    print(f"Dataset size: {len(df)}")
    print(f"Spam ratio : {sum(y)/len(y):.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -------------------------------
    # Feature Pipeline
    # -------------------------------
    feature_pipeline = FeatureUnion([
        ('text_pipeline', Pipeline([
            ('cleaner', TextCleaner()),
            ('tfidf', TfidfVectorizer(
                max_features=7000,
                ngram_range=(1, 3),
                min_df=1
            ))
        ])),
        ('metadata_pipeline', Pipeline([
            ('extractor', MetadataExtractor()),
            ('scaler', MinMaxScaler())
        ]))
    ])

    # -------------------------------
    # Naive Bayes Model (text only)
    # -------------------------------
    nb_pipeline = Pipeline([
        ('cleaner', TextCleaner()),
        ('tfidf', TfidfVectorizer(
            max_features=7000,
            ngram_range=(1, 2),
            min_df=1
        )),
        ('clf', MultinomialNB())
    ])

    print("\nTraining Naive Bayes...")
    nb_pipeline.fit(X_train, y_train)
    f1_nb = evaluate(nb_pipeline, X_test, y_test, "Naive Bayes")


    # -------------------------------
    # Random Forest Model
    # -------------------------------
    rf_pipeline = Pipeline([
        ('features', feature_pipeline),
        ('clf', RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        ))
    ])

    print("\nTraining Random Forest...")
    rf_pipeline.fit(X_train, y_train)
    f1_rf = evaluate(rf_pipeline, X_test, y_test, "Random Forest")


    # -------------------------------
    # Select Best Model
    # -------------------------------
    os.makedirs(MODEL_PATH, exist_ok=True)

    if f1_nb >= f1_rf:
        print("\n✅ Saving Naive Bayes as best model")
        joblib.dump(nb_pipeline, f"{MODEL_PATH}/spam_model.pkl")
    else:
        print("\n✅ Saving Random Forest as best model")
        joblib.dump(rf_pipeline, f"{MODEL_PATH}/spam_model.pkl")

    print("\n🎯 Training complete.")


# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    train_model()