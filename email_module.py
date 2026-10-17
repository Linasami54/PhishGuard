# PhishGuard - Task 1: Phishing Email Classifier
# this file loads our trained model and checks if an email is phishing or safe

import re
from pathlib import Path

import joblib
import numpy as np

# settings
MODEL_PATH = Path(__file__).resolve().parent / "models" / "email_model.joblib"
MAX_CHARS = 20_000   # we only read the first 20,000 characters of an email
THRESHOLD = 0.5      # if probability is 0.5 or more -> phishing

# patterns we use to find links, emails, numbers and html tags in the text
_URL_RE = re.compile(r"(?:https?://|www\.)[^\s\"'<>]+")
# some links in the dataset are written with spaces like "http : / / www . site . com"
_HOST = r"[\w%~-]+(?:\s*[.-]\s*[\w%~-]+)*"
_PATH = r"(?:\s*/\s*[\w%~-]+(?:\s*[.?=&-]\s*[\w%~-]+)*)*(?:\s*/)?"
_SPACED_URL_RE = re.compile(
    r"\bhttps?\s*:\s*/\s*/\s*" + _HOST + _PATH + r"|\bwww\d*\s*\.\s*" + _HOST + _PATH)
_EMAIL_RE = re.compile(r"[\w+-]+(?:\s*\.\s*[\w+-]+)*\s*@\s*[\w-]+(?:\s*\.\s*[\w-]+)+")
_HTML_TAG_RE = re.compile(r"<[^>]*>")
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")
_SPACE_RE = re.compile(r"\s+")

# we keep the model here after loading it the first time
_model = None
_weights = None


# cleans the email text before giving it to the model
# (we use the same cleaning in training and in prediction)
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text[:MAX_CHARS].lower()
    text = _URL_RE.sub(" URL ", text)                    # links -> URL
    text = _HTML_TAG_RE.sub(lambda m: " URL " if "URL" in m.group(0) else " ", text)  # remove html tags
    text = _SPACED_URL_RE.sub(" URL ", text)             # links with spaces -> URL
    text = _EMAIL_RE.sub(" EMAIL ", text)                # email addresses -> EMAIL
    text = _NUM_RE.sub(" NUM ", text)                    # numbers -> NUM
    return _SPACE_RE.sub(" ", text).strip()              # remove extra spaces


# gives every word a weight from the model
# positive weight = the word points to phishing, negative = points to safe
def get_word_weights(pipeline):
    vectorizer = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "coef_"):                        # Logistic Regression
        weights = clf.coef_[0]
    elif hasattr(clf, "feature_log_prob_"):          # Naive Bayes
        weights = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    elif hasattr(clf, "calibrated_classifiers_"):    # Linear SVM
        weights = np.mean([c.estimator.coef_[0] for c in clf.calibrated_classifiers_], axis=0)
    else:
        raise TypeError(f"Unsupported classifier: {type(clf).__name__}")
    return vectorizer.get_feature_names_out(), np.asarray(weights).ravel()


# finds the words in this email that made the model think it is phishing
# score of each word = tf-idf value x word weight
def explain(pipeline, cleaned_text, top_n=5, weights=None):
    words, w = weights if weights is not None else get_word_weights(pipeline)
    row = pipeline.named_steps["tfidf"].transform([cleaned_text]).tocoo()
    contrib = row.data * w[row.col]
    order = np.argsort(contrib)[::-1]                # highest score first
    return [(str(words[row.col[i]]), round(float(contrib[i]), 3))
            for i in order[:top_n] if contrib[i] > 0]


# loads the saved model only one time so the app stays fast
def load_model(path=MODEL_PATH):
    global _model, _weights
    if _model is None:
        _model = joblib.load(path)
        _weights = get_word_weights(_model)
    return _model


# main function used by the app
# input: email text -> output: label (phishing/safe), probability, top 5 suspicious words
def predict_email(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Email text is empty.")
    model = load_model()
    cleaned = clean_text(text)
    probability = float(model.predict_proba([cleaned])[0, 1])
    return {
        "label": "phishing" if probability >= THRESHOLD else "safe",
        "probability": round(probability, 4),
        "top_words": explain(model, cleaned, top_n=5, weights=_weights),
    }


# quick test when we run this file directly
if __name__ == "__main__":
    examples = [
        "Your account will be suspended. Verify your password immediately.",
        "Hi team, the meeting is moved to 3pm tomorrow. See you there.",
    ]
    for e in examples:
        print(e, "\n ->", predict_email(e), "\n")
