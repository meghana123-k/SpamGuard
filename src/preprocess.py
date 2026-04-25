import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin

# Ensure NLTK data is available
try:
    stop_words = set(stopwords.words('english'))
except Exception:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

try:
    lemmatizer = WordNetLemmatizer()
except Exception:
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()

custom_noise = {
    'subject', 'fw', 'fwd', 're', 'cc',
    'ect', 'hou', 'corp', 'forwarded'
}

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove numbers and special characters
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    
    # 3. Tokenize and clean
    words = text.split()
    cleaned_words = []
    
    for word in words:
        if word not in stop_words and word not in custom_noise:
            # 4. Lemmatize
            cleaned_words.append(lemmatizer.lemmatize(word))
            
    return " ".join(cleaned_words)

def extract_metadata(text):
    if not isinstance(text, str) or len(text) == 0:
        return [0.0, 0.0, 0.0]
    
    link_count = len(re.findall(r'https?://\S+|www\.\S+', text))
    upper_count = sum(1 for c in text if c.isupper())
    uppercase_ratio = upper_count / len(text)
    exclamation_count = text.count('!')
    
    return [float(link_count), float(uppercase_ratio), float(exclamation_count)]

class MetadataExtractor(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, posts):
        return np.array([extract_metadata(text) for text in posts])

class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, posts):
        return [clean_text(text) for text in posts]
