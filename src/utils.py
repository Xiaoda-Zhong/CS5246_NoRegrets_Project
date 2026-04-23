import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

def get_text_to_analyze(text):
    return text[:2000]

def clean_text(text, remove_stopwords=False):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().strip()
    if remove_stopwords:
        stop = set(stopwords.words('english'))
        text = ' '.join([w for w in text.split() if w not in stop])
    return text