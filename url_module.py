# PhishGuard - Task 2: Malicious URL Classifier
# this file turns a link into 18 numbers (features) and uses our trained model
# to say if the link is phishing or legitimate. the website is never opened.

import ipaddress
import math
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit, parse_qsl

import joblib
import pandas as pd
import tldextract
import xgboost

MODEL_PATH = Path(__file__).resolve().parent / "models" / "url_model.joblib"

# offline mode: tldextract uses its built-in list and never downloads anything
_extract = tldextract.TLDExtract(suffix_list_urls=())

SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd",
              "cutt.ly", "buff.ly", "rebrand.ly", "rb.gy"}
SUSPICIOUS_TLDS = {"xyz", "top", "tk", "ml", "ga", "cf", "gq", "click", "work", "zip"}
SENSITIVE_WORDS = ("login", "verify", "secure", "account", "update", "bank",
                   "password", "signin", "confirm")

FEATURE_NAMES = [
    "url_length", "domain_length", "num_dots", "num_hyphens", "num_digits",
    "num_special_chars", "has_at_symbol", "has_ip_address", "uses_https",
    "num_subdomains", "path_length", "num_query_params", "has_double_slash",
    "digit_letter_ratio", "domain_entropy", "is_shortened", "suspicious_tld",
    "has_sensitive_word",
]

# the model does not use num_query_params: in our datasets the legitimate links never have a
# "?" part (PhreshPhish removed it for privacy), so the model would learn "? = phishing", which is wrong
MODEL_FEATURES = [f for f in FEATURE_NAMES if f != "num_query_params"]

_model = None


# cleans the link before we read it
# if no http/https was typed we assume https, so a missing protocol is not counted against the link
# "site.com/" and "site.com" are treated the same
def prepare_url(url):
    if not isinstance(url, str) or not url.strip():
        raise ValueError("URL is empty.")
    url = url.strip()
    if re.search(r"\s", url):
        raise ValueError("URL cannot contain spaces.")
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "https://" + url.lstrip("/")
    parts = urlsplit(url)
    path = "" if parts.path == "/" else parts.path
    url = f"{parts.scheme.lower()}://{parts.netloc}{path}"
    if parts.query:
        url += "?" + parts.query
    if parts.fragment:
        url += "#" + parts.fragment
    return url


# entropy = how random the letters of the domain look (random names are common in phishing)
def _entropy(text):
    if not text:
        return 0.0
    counts = Counter(text)
    return -sum((n / len(text)) * math.log2(n / len(text)) for n in counts.values())


# turns one link into 18 features
# all features except num_query_params are measured on the link without its "?" part
def extract_url_features(url):
    full = prepare_url(url)
    query = urlsplit(full).query
    url = full.split("?", 1)[0].split("#", 1)[0]
    parts = urlsplit(url)
    try:
        host = (parts.hostname or "").rstrip(".")
    except ValueError:
        host = ""

    try:
        ipaddress.ip_address(host)
        is_ip = 1
    except ValueError:
        is_ip = 0

    ext = _extract(host) if host and not is_ip else None
    subdomain = ext.subdomain if ext else ""
    tld = host.rsplit(".", 1)[-1] if "." in host else ""
    after_scheme = url.split("://", 1)[1]
    digits = sum(c.isdigit() for c in url)
    letters = sum(c.isalpha() for c in url)

    return {
        "url_length": len(url),
        "domain_length": len(host),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_digits": digits,
        "num_special_chars": sum(not c.isalnum() for c in url),
        "has_at_symbol": int("@" in url),
        "has_ip_address": is_ip,
        "uses_https": int(parts.scheme == "https"),
        "num_subdomains": len(subdomain.split(".")) if subdomain else 0,
        "path_length": len(parts.path),
        "num_query_params": len(parse_qsl(query, keep_blank_values=True)),
        "has_double_slash": int("//" in after_scheme),
        "digit_letter_ratio": round(digits / max(letters, 1), 4),
        "domain_entropy": round(_entropy(host), 4),
        "is_shortened": int(host.lower() in SHORTENERS),
        "suspicious_tld": int(tld.lower() in SUSPICIOUS_TLDS),
        "has_sensitive_word": int(any(w in url.lower() for w in SENSITIVE_WORDS)),
    }


# loads the saved model only one time so the app stays fast
def load_model(path=MODEL_PATH):
    global _model
    if _model is None:
        _model = joblib.load(path)
    return _model


# finds which features made this link look like phishing
# xgboost can tell us how much each feature added to the phishing score
# for other models we put the normal value (average of legitimate links) in each feature
# and see how much the phishing probability goes down
def explain(model, features, legit_mean, top_n=5):
    row = pd.DataFrame([features])[MODEL_FEATURES].astype(float)
    if hasattr(model, "get_booster"):
        scores = model.get_booster().predict(xgboost.DMatrix(row), pred_contribs=True)[0][:-1]
    else:
        probability = model.predict_proba(row)[0, 1]
        changed = pd.concat([row] * len(MODEL_FEATURES), ignore_index=True)
        for i, name in enumerate(MODEL_FEATURES):
            changed.loc[i, name] = legit_mean[name]
        scores = probability - model.predict_proba(changed)[:, 1]
    ranked = sorted(zip(MODEL_FEATURES, scores), key=lambda x: x[1], reverse=True)
    return [(name, round(float(s), 3)) for name, s in ranked[:top_n] if s > 0]


# main function used by the app
# input: link -> output: label (phishing/legitimate), probability of phishing, features, top features
def predict_url(url):
    saved = load_model()
    features = extract_url_features(url)
    row = pd.DataFrame([features])[MODEL_FEATURES].astype(float)
    probability = float(saved["model"].predict_proba(row)[0, 1])
    return {
        "label": "phishing" if probability >= saved["threshold"] else "legitimate",
        "probability": round(probability, 4),
        "features": features,
        "top_features": explain(saved["model"], features, saved["legit_mean"]),
    }


# quick test when we run this file directly
if __name__ == "__main__":
    for link in ["https://www.google.com/search?q=test", "http://paypa1-secure-login.xyz/verify"]:
        print(link, "\n ->", predict_url(link), "\n")
