# PhishGuard - Task 5: fake versions of the six functions
# they return the same output format as the real ones, so the app could be built
# before the other tasks were ready. the app uses them only if a real file is missing.

from __future__ import annotations

import math
import re
from urllib.parse import urlparse

# shared constants

# Points per rule category (from the Task 3 / Task 4 specification).
CATEGORY_POINTS: dict[str, int] = {
    # email
    "credentials": 30, "threat": 20, "urgency": 15, "reward": 15,
    "links": 10, "generic_greeting": 5, "formatting": 5,
    # url
    "ip_address": 30, "at_symbol": 20, "lookalike_domain": 35,
    "brand_in_subdomain": 35, "punycode": 25, "shortener": 15,
    "suspicious_tld": 15, "many_subdomains": 10, "long_url": 10, "no_https": 5,
}

# email stubs

# category -> (human reason, regex alternatives; English + Arabic)
_EMAIL_PATTERNS: dict[str, tuple[str, str]] = {
    "credentials": (
        "The message asks for sensitive credentials",
        r"password|passcode|\botp\b|\bpin\b|credit card|verify your account|"
        r"confirm your (?:identity|account)|كلمة المرور|رمز التحقق",
    ),
    "threat": (
        "The message threatens negative consequences",
        r"suspended|locked|legal action|terminated|will be closed|"
        r"سيتم إيقاف حسابك|تم تعليق",
    ),
    "urgency": (
        "The message pressures you to act now",
        r"urgent|immediately|within 24 hours|act now|right away|عاجل|فورا|فوراً",
    ),
    "reward": (
        "The message promises a reward or refund",
        r"winner|prize|gift card|refund|congratulations|ربحت|جائزة",
    ),
    "generic_greeting": (
        "The greeting is generic instead of using your name",
        r"dear customer|dear user|dear client|عزيزي العميل|عزيزي المستخدم",
    ),
}

# word -> weight toward "phishing" (stub "model"). Negative = safe-leaning.
_PHISH_WORDS: dict[str, float] = {
    "password": 0.41, "suspended": 0.33, "verify": 0.30, "immediately": 0.27,
    "urgent": 0.25, "account": 0.18, "click": 0.17, "login": 0.16,
    "winner": 0.22, "prize": 0.21, "refund": 0.19, "bank": 0.14,
    "confirm": 0.13, "limited": 0.10, "locked": 0.28, "otp": 0.30,
}
_SAFE_WORDS: dict[str, float] = {
    "meeting": -0.22, "agenda": -0.20, "attached": -0.12, "schedule": -0.15,
    "thanks": -0.10, "regards": -0.09, "project": -0.14, "tomorrow": -0.08,
}


def _tokens(text: str) -> list[str]:
    # lowercase words of the text
    return re.findall(r"\w+", text.lower())


def predict_email(text: str) -> dict:
    # fake email model: scores the text using a small list of words
    tokens = set(_tokens(text or ""))
    hits = {w: v for w, v in {**_PHISH_WORDS, **_SAFE_WORDS}.items() if w in tokens}
    # logistic squashing of the summed weights, with a safe-leaning bias
    z = -2.4 + 4.0 * sum(hits.values())
    probability = round(1 / (1 + math.exp(-z)), 2)
    top_words = sorted(hits.items(), key=lambda kv: abs(kv[1]), reverse=True)[:8]
    return {
        "label": "phishing" if probability >= 0.5 else "safe",
        "probability": probability,
        "top_words": [(w, round(v, 2)) for w, v in top_words],
    }


def analyze_email_rules(text: str) -> list[dict]:
    # fake email rules: simple keyword checks
    text = text or ""
    reasons: list[dict] = []

    for category, (reason, pattern) in _EMAIL_PATTERNS.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            reasons.append({
                "category": category,
                "reason": reason,
                "points": CATEGORY_POINTS[category],
                "evidence": match.group(0),
            })

    link = re.search(r"https?://\S+|www\.\S+", text, flags=re.IGNORECASE)
    if link:
        reasons.append({
            "category": "links",
            "reason": "The message contains a link - check where it really goes",
            "points": CATEGORY_POINTS["links"],
            "evidence": link.group(0).rstrip(".,;)"),
        })

    letters = [c for c in text if c.isalpha()]
    caps_ratio = sum(c.isupper() for c in letters) / len(letters) if letters else 0
    exclaim = re.search(r"!{3,}", text)
    if exclaim or (len(letters) >= 20 and caps_ratio > 0.5):
        reasons.append({
            "category": "formatting",
            "reason": "Shouting capitals or repeated exclamation marks",
            "points": CATEGORY_POINTS["formatting"],
            "evidence": exclaim.group(0) if exclaim else "MANY CAPITAL LETTERS",
        })
    return reasons


# url stubs

_SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "cutt.ly"}
_SUSPICIOUS_TLDS = {"xyz", "top", "tk", "ml", "ga", "cf", "gq", "click", "work", "zip"}
_SENSITIVE_WORDS = ("login", "verify", "secure", "account", "update", "bank", "password")
_BRANDS = {  # tiny list for the stub; Task 4 owns the real data/brands.json
    "paypal": "paypal.com", "google": "google.com", "microsoft": "microsoft.com",
    "apple": "apple.com", "amazon": "amazon.com", "netflix": "netflix.com",
    "facebook": "facebook.com", "stc": "stc.com.sa", "absher": "absher.sa",
}
_IP_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")


def _parse(url: str):
    # reads a link without crashing on bad input
    try:
        return urlparse(url)
    except ValueError:
        return urlparse("")


def _host(url: str) -> str:
    # domain of the link in lowercase ('' if the link is broken)
    try:
        return (_parse(url).hostname or "").lower()
    except ValueError:
        return ""


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    return -sum((s.count(c) / len(s)) * math.log2(s.count(c) / len(s)) for c in set(s))


def _levenshtein(a: str, b: str) -> int:
    # small edit distance (the real one is in url_rules.py)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _registered_domain(host: str) -> tuple[str, str]:
    # simple split of the domain name and its ending
    parts = host.split(".")
    if len(parts) < 2:
        return host, host
    return parts[-2], ".".join(parts[-2:])


def _extract_features(url: str) -> dict:
    # the 18 URL features, computed in a simple way
    parsed = _parse(url)
    host = _host(url)
    parts = host.split(".") if host else []
    letters = sum(c.isalpha() for c in url)
    digits = sum(c.isdigit() for c in url)
    return {
        "url_length": len(url),
        "domain_length": len(host),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_digits": digits,
        "num_special_chars": sum(not c.isalnum() for c in url),
        "has_at_symbol": int("@" in url),
        "has_ip_address": int(bool(_IP_RE.match(host))),
        "uses_https": int(parsed.scheme == "https"),
        "num_subdomains": max(0, len(parts) - 2) if not _IP_RE.match(host) else 0,
        "path_length": len(parsed.path),
        "num_query_params": len([p for p in parsed.query.split("&") if p]),
        "has_double_slash": int("//" in url.split("://", 1)[-1]),
        "digit_letter_ratio": round(digits / letters, 3) if letters else 0.0,
        "domain_entropy": round(_entropy(host), 3),
        "is_shortened": int(host in _SHORTENERS),
        "suspicious_tld": int(bool(parts) and parts[-1] in _SUSPICIOUS_TLDS),
        "has_sensitive_word": int(any(w in url.lower() for w in _SENSITIVE_WORDS)),
    }


# feature -> weight toward phishing (stub "feature importance")
_URL_WEIGHTS = {
    "has_ip_address": 0.30, "has_at_symbol": 0.22, "suspicious_tld": 0.20,
    "has_sensitive_word": 0.16, "is_shortened": 0.14, "num_subdomains": 0.06,
    "num_hyphens": 0.05, "has_double_slash": 0.08,
}


def predict_url(url: str) -> dict:
    # fake URL model: scores the link using a few features
    features = _extract_features(url or "")
    contributions: dict[str, float] = {}
    for name, weight in _URL_WEIGHTS.items():
        value = features[name]
        if value:
            contributions[name] = round(weight * min(value, 3), 2)
    if not features["uses_https"]:
        contributions["uses_https"] = 0.05
    if features["url_length"] > 75:
        contributions["url_length"] = 0.10
    z = -2.0 + 7.0 * sum(contributions.values())
    probability = round(1 / (1 + math.exp(-z)), 2)
    top = sorted(contributions.items(), key=lambda kv: kv[1], reverse=True)[:5]
    return {
        "label": "phishing" if probability >= 0.5 else "legitimate",
        "probability": probability,
        "features": features,
        "top_features": top,
    }


def analyze_url_rules(url: str) -> list[dict]:
    # fake URL rules: a few of the real checks
    url = url or ""
    host = _host(url)
    parsed_scheme = _parse(url).scheme if host else ""
    reasons: list[dict] = []

    def add(category: str, reason: str, evidence: str) -> None:
        reasons.append({"category": category, "reason": reason,
                        "points": CATEGORY_POINTS[category], "evidence": evidence})

    if _IP_RE.match(host):
        add("ip_address", "The address is a raw IP, not a real domain name", host)
    if "@" in url.split("://", 1)[-1].split("/", 1)[0]:
        add("at_symbol", "An '@' can hide the real destination of the link", "@")
    if "xn--" in host:
        add("punycode", "Punycode can disguise look-alike characters", host)
    if host in _SHORTENERS:
        add("shortener", "A shortened link hides the real destination", host)
    if host and host.rsplit(".", 1)[-1] in _SUSPICIOUS_TLDS:
        add("suspicious_tld", "The domain ends with a TLD often abused by scammers",
            "." + host.rsplit(".", 1)[-1])
    if host and not _IP_RE.match(host) and len(host.split(".")) - 2 > 3:  # more than 3 subdomains
        add("many_subdomains", "The address has an unusually long chain of subdomains", host)
    if len(url) > 75:
        add("long_url", "Very long URLs are often used to hide the real domain",
            f"{len(url)} characters")
    if parsed_scheme == "http":
        add("no_https", "The connection is not encrypted (note: HTTPS alone does not mean safe)",
            "http://")

    label, registered = _registered_domain(host)
    if host and not _IP_RE.match(host):
        normalized = label.replace("0", "o").replace("1", "l").replace("rn", "m").replace("vv", "w")
        for brand, official in _BRANDS.items():
            # brand in subdomain/path, but the real registered domain differs
            elsewhere = brand in host.replace(registered, "") or brand in _parse(url).path.lower()
            if registered != official and elsewhere and brand != label:
                add("brand_in_subdomain",
                    f"The brand name '{brand}' appears, but the real domain is {registered}", host)
                break
            if registered != official and label != brand and \
                    _levenshtein(normalized, brand) in (0, 1, 2) and abs(len(label) - len(brand)) <= 2:
                add("lookalike_domain", f"The domain imitates {official}", registered)
                break
    return reasons


# risk + advice stubs (implement the real task 3 formulas, so they are accurate)

def combine_risk(probability: float | None, reasons: list[dict]) -> dict:
    # same formula as risk.py: 60% ML probability + 40% rule points
    rule_points = min(100, sum(int(r.get("points", 0)) for r in (reasons or [])))
    if probability is None:
        score = rule_points
    else:
        prob = min(1.0, max(0.0, float(probability)))
        score = 0.6 * (prob * 100) + 0.4 * rule_points
    score = int(round(min(100, max(0, score))))
    level = "Low" if score < 35 else "Medium" if score < 70 else "High"
    return {"score": score, "level": level}


_ADVICE_BY_CATEGORY = {
    "credentials": "Never enter your password, PIN or OTP through a link in a message - type the website address yourself.",
    "threat": "Threats of suspension are a pressure tactic. Contact the company through its official app or website.",
    "urgency": "Real organisations rarely demand action 'immediately'. Pause and verify before acting.",
    "reward": "Unexpected prizes and refunds are a classic lure - if you did not enter a contest, ignore it.",
    "links": "Hover over links (or long-press on mobile) to see the real destination before clicking.",
    "generic_greeting": "Your bank or provider normally uses your real name, not 'Dear customer'.",
    "formatting": "Shouting capitals and '!!!' are common in scam messages.",
    "ip_address": "Legitimate sites use domain names, not raw IP addresses. Do not enter any data here.",
    "at_symbol": "Everything before '@' in a link is ignored by the browser - the real site is after it.",
    "long_url": "Very long links can hide the true domain. Check the part just before the first '/'.",
    "many_subdomains": "Check the LAST part of the domain (before the first '/') - that is the real owner.",
    "shortener": "Expand shortened links with a preview tool before opening them.",
    "suspicious_tld": "Be extra careful with unusual domain endings such as .xyz, .top or .tk.",
    "punycode": "The domain may use look-alike foreign characters to imitate a real brand.",
    "no_https": "The connection is not encrypted - never submit passwords or card details here.",
    "lookalike_domain": "The domain imitates a known brand with small spelling changes. Go to the official site directly.",
    "brand_in_subdomain": "A brand name in the subdomain or path does not make a site official - read the real domain.",
}

_ADVICE_BY_LEVEL = {
    "Low": "No strong warning signs were found, but stay alert - no tool is perfect.",
    "Medium": "Some suspicious signs were found. Do not click or reply until you verify the sender through an official channel.",
    "High": "High risk: do not click links, open attachments or reply. Report it as phishing and delete it.",
}


def get_advice(level: str, reasons: list[dict]) -> list[str]:
    # one tip for the level and one tip for each category found
    advice = [_ADVICE_BY_LEVEL.get(level, _ADVICE_BY_LEVEL["Medium"])]
    seen: set[str] = set()
    for reason in reasons or []:
        category = reason.get("category", "")
        if category in _ADVICE_BY_CATEGORY and category not in seen:
            seen.add(category)
            advice.append(_ADVICE_BY_CATEGORY[category])
    return advice
