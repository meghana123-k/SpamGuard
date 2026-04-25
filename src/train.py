import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import os
from src.preprocess import clean_text, extract_metadata, TextCleaner, MetadataExtractor

def ensure_data_exists():
    """Create dummy data if real datasets are missing."""
    enron_path = "data/enron_spam_data.csv"
    sms_path = "data/SMSSpamCollection"
    os.makedirs("data", exist_ok=True)
    
    if not os.path.exists(enron_path):
        print(f"Creating dummy Enron data at {enron_path}")
        df = pd.DataFrame([
            {"Subject": "Meeting", "Message": "Hello, let's meet at 5", "Spam/Ham": "ham"},
            {"Subject": "Congratulations", "Message": "You won the lottery!!! Click here http://scam.me", "Spam/Ham": "spam"},
            {"Subject": "Invoice", "Message": "Please check the attached invoice.", "Spam/Ham": "ham"},
            {"Subject": "Urgent", "Message": "URGENT ACTION REQUIRED. CLAIM YOUR PRIZE NOW!", "Spam/Ham": "spam"},
        ] * 20) # Multiply to have enough for grid search
        df.to_csv(enron_path, index=False)
        
    if not os.path.exists(sms_path):
        print(f"Creating dummy SMS data at {sms_path}")
        with open(sms_path, "w") as f:
            for _ in range(20):
                f.write("ham\tOk lor. Sony ericsson salesman...\n")
                f.write("spam\tFree entry in 2 a wkly comp... text WIN to 80082\n")

def train_model():
    ensure_data_exists()
    
    # 1. Load datasets
    enron = pd.read_csv("data/enron_spam_data.csv")
    sms = pd.read_csv("data/SMSSpamCollection", sep="\t", header=None, names=["label", "text"])
    sms['label'] = sms['label'].map({'ham': 0, 'spam': 1})

    enron['text'] = enron['Subject'].fillna('') + " " + enron['Message'].fillna('')
    if 'Spam/Ham' in enron.columns:
        enron = enron.rename(columns={'Spam/Ham': 'label'})
    enron['label'] = enron['label'].map({'ham': 0, 'spam': 1})

    # Load feedback if exists
    feedback_path = "data/feedback.csv"
    if os.path.exists(feedback_path):
        feedback = pd.read_csv(feedback_path)
        feedback_data = feedback[['text', 'actual']].rename(columns={'actual': 'label'})
    else:
        feedback_data = pd.DataFrame(columns=['text', 'label'])

    # Combine
    df = pd.concat([
        enron[['text', 'label']],
        sms[['text', 'label']],
        feedback_data
    ]).dropna().reset_index(drop=True)

    X = df['text']
    y = df['label'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Dataset Size: {len(df)}")
    print(f"Spam ratio: {sum(y)/len(y):.2%}")

    # 2. Build Pipeline
    # We use FeatureUnion to combine TF-IDF and Metadata
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('cleaner', TextCleaner()),
                ('tfidf', TfidfVectorizer(min_df=2))
            ])),
            ('metadata_pipeline', Pipeline([
                ('extractor', MetadataExtractor()),
                ('scaler', StandardScaler())
            ]))
        ])),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000))
    ])

    # 3. Hyperparameter Tuning
    param_grid = {
        'features__text_pipeline__tfidf__max_features': [5000, 7000],
        'features__text_pipeline__tfidf__ngram_range': [(1, 2), (1, 3)],
        'clf__C': [0.1, 1, 10]
    }

    print("\nStarting GridSearchCV for Logistic Regression...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_lr = grid_search.best_estimator_
    print(f"Best Params: {grid_search.best_params_}")

    # 4. Compare with MultinomialNB (using text only for NB since it doesn't handle negative scaled metadata well)
    nb_pipeline = Pipeline([
        ('cleaner', TextCleaner()),
        ('tfidf', TfidfVectorizer(max_features=7000, ngram_range=(1, 2), min_df=2)),
        ('clf', MultinomialNB())
    ])
    
    print("\nEvaluating MultinomialNB...")
    nb_pipeline.fit(X_train, y_train)

    # 5. Evaluation
    def evaluate(model, name):
        y_pred = model.predict(X_test)
        print(f"\n--- {name} Performance ---")
        print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
        print(f"Precision: {precision_score(y_test, y_pred):.4f}")
        print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
        print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        return f1_score(y_test, y_pred)

    f1_lr = evaluate(best_lr, "Logistic Regression (Tuned + Features)")
    f1_nb = evaluate(nb_pipeline, "MultinomialNB (Text Default)")

    # 6. Save Best Model
    os.makedirs("models", exist_ok=True)
    if f1_lr >= f1_nb:
        print("\nSaving Logistic Regression as the best model.")
        joblib.dump(best_lr, "models/spam_model.pkl")
    else:
        print("\nSaving MultinomialNB as the best model.")
        joblib.dump(nb_pipeline, "models/spam_model.pkl")

    print("\nTraining complete.")

if __name__ == "__main__":
    train_model()
