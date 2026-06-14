import re
import string
 
import joblib
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
 
 
# ---------------------------------------------------------------------------
# NLTK setup (downloads quietly if not already present)
# ---------------------------------------------------------------------------
for _pkg in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    nltk.download(_pkg, quiet=True)
 
_stop_words = set(stopwords.words("english"))
 
# Keep negation words — they flip sentiment and shouldn't be removed
_negations = {"not", "no", "nor", "don't", "isn't", "wasn't", "didn't"}
_stop_words = _stop_words - _negations
 
_lemmatizer = WordNetLemmatizer()
 
 
# ---------------------------------------------------------------------------
# Text cleaning / preprocessing
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Lowercase and strip URLs, HTML tags, digits, punctuation, extra spaces."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text
 
 
def preprocess(text: str) -> str:
    """Full pipeline: clean -> tokenize -> remove stopwords -> lemmatize."""
    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned)
    tokens = [t for t in tokens if t not in _stop_words and len(t) > 1]
    tokens = [_lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)
 
 
# ---------------------------------------------------------------------------
# Model artifact loading
# ---------------------------------------------------------------------------
def load_artifacts(
    model_path: str = "sentiment_model.pkl",
    vectorizer_path: str = "vectorizer.pkl",
    encoder_path: str = "label_encoder.pkl",
):
    """Load the trained model, TF-IDF vectorizer, and label encoder."""
    model = joblib.load(model_path)
    tfidf = joblib.load(vectorizer_path)
    le = joblib.load(encoder_path)
    return model, tfidf, le
 
 
# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def get_prediction(text: str, model, tfidf, le):
    """
    Returns:
        sentiment (str): predicted label, e.g. 'Positive'
        probs (np.ndarray): confidence scores aligned with le.classes_
        processed (str): the preprocessed text (useful for debugging)
    """
    processed = preprocess(text)
    vec = tfidf.transform([processed])
    pred = model.predict(vec)[0]
    sentiment = le.inverse_transform([pred])[0]
 
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec)[0]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(vec)[0]
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()
    else:
        probs = np.zeros(len(le.classes_))
 
    return sentiment, probs, processed
 