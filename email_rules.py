# PhishGuard - Task 3: Email Rules Engine
# finds phishing signs in an email (urgency, threats, password requests...) and explains them

import re
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
KEYWORDS_FILE = BASE_DIR / "data" / "keywords.json"

# points for each category
CATEGORY_POINTS = {
    "credentials": 30,
    "threat": 20,
    "urgency": 15,
    "reward": 15,
    "links": 10,
    "generic_greeting": 5,
    "formatting": 5,
}

# explanation shown to the user for each category
CATEGORY_MESSAGES = {
    "credentials": "طلب معلومات حساسة مثل كلمة المرور أو رمز التحقق (Credentials)",
    "threat": "تهديد بإيقاف الحساب أو إغلاقه (Threat)",
    "urgency": "لغة استعجال للضغط عليك حتى تتصرف بسرعة (Urgency)",
    "reward": "عرض جائزة أو استرداد أموال غير متوقع (Reward)",
    "links": "وجود رابط داخل الرسالة قد يوصلك لصفحة مزيفة (Links)",
    "generic_greeting": "تحية عامة بدل اسمك الحقيقي (Generic Greeting)",
    "formatting": "استخدام مبالغ فيه للحروف الكبيرة أو علامات التعجب (Formatting)",
}


# reads the keyword lists from data/keywords.json
def load_keywords():
    if not KEYWORDS_FILE.exists():
        return {}
    with open(KEYWORDS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# normalize text to catch common tricks like p@ssw0rd or p a s s w o r d
def normalize_text(text):
    if not text:
        return ""

    text = text.lower()

    replacements = {"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "@": "a", "$": "s", "!": "i"}
    for old, new in replacements.items():
        text = text.replace(old, new)

    # remove hidden zero-width characters
    text = re.sub(r"[​-‍﻿]", "", text)

    # remove spaces between single letters
    text = re.sub(r"(?<=\b\w)\s+(?=\w\b)", "", text)

    return text


# true if the keyword appears in the text
# english words must match as a whole word, so "pin" does not match "opinion"
# arabic words are checked as part of the text because arabic adds letters to the start of words
def contains_keyword(text, keyword):
    if re.search(r"[a-z]", keyword):
        return re.search(rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])", text) is not None
    return keyword in text


# checks the email and returns the phishing signs we found
def analyze_email_rules(text):
    reasons = []

    if not isinstance(text, str) or not text.strip():
        return reasons

    lower_text = text.lower()
    normalized_text = normalize_text(text)
    keywords = load_keywords()

    # keyword categories (one reason per category)
    for category, words in keywords.items():
        for word in words:
            word_lower = word.lower()
            if contains_keyword(lower_text, word_lower) or contains_keyword(normalized_text, word_lower):
                match = re.search(re.escape(word_lower), text, re.IGNORECASE)
                reasons.append({
                    "category": category,
                    "reason": CATEGORY_MESSAGES.get(category, "نشاط مشبوه"),
                    "points": CATEGORY_POINTS.get(category, 5),
                    "evidence": match.group(0) if match else word,
                })
                break

    # links inside the email
    url_match = re.search(r"(https?://[^\s]+|www\.[^\s]+)", text, re.IGNORECASE)
    if url_match:
        reasons.append({
            "category": "links",
            "reason": CATEGORY_MESSAGES["links"],
            "points": CATEGORY_POINTS["links"],
            "evidence": url_match.group(0),
        })

    # too many words in capital letters (3 or more)
    uppercase_words = [w for w in text.split() if w.isupper() and len(w) > 1]
    # repeated exclamation marks
    has_exclamations = "!!!" in text

    if len(uppercase_words) >= 3 or has_exclamations:
        reasons.append({
            "category": "formatting",
            "reason": CATEGORY_MESSAGES["formatting"],
            "points": CATEGORY_POINTS["formatting"],
            "evidence": "!!!" if has_exclamations else uppercase_words[0],
        })

    return reasons
