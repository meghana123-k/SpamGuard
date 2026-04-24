import re
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

custom_noise = {
    'subject', 'fw', 'fwd', 're', 'cc',
    'ect', 'hou', 'corp', 'forwarded'
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()
    words = [
        word for word in words
        if word not in stop_words and word not in custom_noise
    ]

    return " ".join(words)