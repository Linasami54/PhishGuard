# PhishGuard - Task 4: URL Rules Engine
# checks a link for phishing signs (IP address, @, fake domain names...) and explains them

import ipaddress
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

import tldextract

# offline mode: uses the built-in list of domain endings, no download
extractor = tldextract.TLDExtract(suffix_list_urls=())

BRANDS_FILE = Path(__file__).resolve().parent / "data" / "brands.json"
with open(BRANDS_FILE, "r", encoding="utf-8") as file:
    BRANDS = json.load(file)

SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "is.gd", "ow.ly", "cutt.ly", "rb.gy"}
SUSPICIOUS_TLDS = {"xyz", "top", "tk", "gq", "ml", "cf", "ga"}

# words that phishing pages use to look like a login or payment page
PHISHING_WORDS = {"login", "verify", "secure", "account", "password", "signin", "update",
                  "confirm", "security", "billing", "payment", "delivery", "support"}


# returns the evidence exactly as it was written in the original link
def original_evidence(original_url, evidence):
    if not evidence:
        return ""
    position = original_url.lower().find(evidence.lower())
    if position == -1:
        return ""
    return original_url[position:position + len(evidence)]


# replaces characters that attackers use to copy brand names (paypa1 -> paypal)
def normalize_domain(name):
    name = name.lower()
    name = name.replace("rn", "m")
    name = name.replace("vv", "w")
    name = name.replace("0", "o")
    name = name.replace("1", "l")
    return name


# minimum number of letter changes to turn one word into another
def levenshtein(a, b):
    if len(a) < len(b):
        return levenshtein(b, a)

    previous_row = list(range(len(b) + 1))
    for i, char_a in enumerate(a):
        current_row = [i + 1]
        for j, char_b in enumerate(b):
            insertion = previous_row[j + 1] + 1
            deletion = current_row[j] + 1
            substitution = previous_row[j] + (char_a != char_b)
            current_row.append(min(insertion, deletion, substitution))
        previous_row = current_row

    return previous_row[-1]


# splits a host into (subdomain, domain name, ending)
# if the ending is unknown (like .example) we take the last part as the ending
def split_host(host):
    ext = extractor(host)
    if ext.suffix:
        return ext.subdomain, ext.domain, ext.suffix
    parts = host.split(".")
    if len(parts) < 2:
        return "", host, ""
    return ".".join(parts[:-2]), parts[-2], parts[-1]


# the real registered domain of a link, for example "secure-login.xyz" in paypal.com.secure-login.xyz
def registered_domain(host):
    _, domain, suffix = split_host(host)
    return f"{domain}.{suffix}" if suffix else domain


OFFICIAL_DOMAINS = {registered_domain(d).lower() for d in BRANDS.values()}


# main function: returns the list of phishing signs found in the link
def analyze_url_rules(url):
    reasons = []

    if not isinstance(url, str) or not url.strip():
        return reasons

    url = url.strip()
    original_url = url
    typed_scheme = "://" in url

    # add a scheme if the user did not type one, so we can read the link
    if not typed_scheme:
        url = "http://" + url

    # skip links that are not valid
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        if not hostname or parsed.scheme.lower() not in ("http", "https"):
            return reasons
        _ = parsed.port
        if "." not in hostname and not hostname.lower().startswith("localhost"):
            return reasons
    except ValueError:
        return reasons

    hostname = hostname.lower().rstrip(".")

    # rule 1: no https (only when the user really typed http)
    if typed_scheme and parsed.scheme.lower() != "https":
        reasons.append({
            "category": "no_https",
            "reason": "The URL does not use HTTPS",
            "points": 5,
            "evidence": original_evidence(original_url, parsed.scheme),
        })

    # rule 2: IP address instead of a domain name
    is_ip = False
    try:
        ipaddress.ip_address(hostname)
        is_ip = True
        reasons.append({
            "category": "ip_address",
            "reason": "The URL uses an IP address instead of a domain name",
            "points": 30,
            "evidence": original_evidence(original_url, hostname),
        })
    except ValueError:
        pass

    # rule 3: @ symbol (everything before it is ignored by the browser)
    if "@" in parsed.netloc:
        reasons.append({
            "category": "at_symbol",
            "reason": "The URL contains an @ symbol that may hide the real domain",
            "points": 20,
            "evidence": "@",
        })

    # rule 4: link shortener
    if hostname in SHORTENERS:
        reasons.append({
            "category": "shortener",
            "reason": "The URL uses a shortening service that hides its destination",
            "points": 15,
            "evidence": original_evidence(original_url, hostname),
        })

    # rule 5: suspicious domain ending
    tld = hostname.split(".")[-1]
    if not is_ip and tld in SUSPICIOUS_TLDS:
        reasons.append({
            "category": "suspicious_tld",
            "reason": "The URL uses a domain ending that is common in phishing",
            "points": 15,
            "evidence": original_evidence(original_url, "." + tld),
        })

    # rule 6: very long link
    if len(original_url) > 75:
        reasons.append({
            "category": "long_url",
            "reason": "The URL is unusually long and may hide suspicious content",
            "points": 10,
            "evidence": original_url[:76],
        })

    if is_ip:
        return reasons

    subdomain, domain_name, suffix = split_host(hostname)
    real_domain = registered_domain(hostname)

    # rule 7: more than 3 subdomains
    if len([part for part in subdomain.split(".") if part]) > 3:
        reasons.append({
            "category": "many_subdomains",
            "reason": "The URL contains more than 3 subdomains",
            "points": 10,
            "evidence": original_evidence(original_url, subdomain),
        })

    # rule 8: punycode (xn--) can hide look-alike letters
    puny = next((label for label in hostname.split(".") if label.startswith("xn--")), None)
    if puny:
        reasons.append({
            "category": "punycode",
            "reason": "The domain uses Punycode, which may disguise look-alike characters",
            "points": 25,
            "evidence": original_evidence(original_url, puny),
        })

    # official brand domains are never flagged by the brand rules
    if real_domain in OFFICIAL_DOMAINS:
        return reasons

    # rule 9: look-alike domain, like paypa1.com or rnicrosoft.com
    # same name with a different ending (google.com.sa) is not flagged, many brands have local sites
    for brand, official_domain in BRANDS.items():
        _, official_name, official_suffix = split_host(official_domain)
        distance = levenshtein(normalize_domain(domain_name), normalize_domain(official_name))
        uses_tricks = distance == 0 and domain_name != official_name
        close_name = 1 <= distance <= 2 and len(official_name) >= 5 and suffix == official_suffix
        if uses_tricks or close_name:
            reasons.append({
                "category": "lookalike_domain",
                "reason": f"The domain resembles {official_domain}",
                "points": 35,
                "evidence": original_evidence(original_url, real_domain),
            })
            break

    # rule 10: brand name used on a different domain
    # in a subdomain (paypal.com.secure-login.xyz), in the domain with a phishing word
    # (paypal-login.com), or in the path with a phishing word (site.com/paypal/verify)
    domain_words = set(domain_name.split("-"))
    path = parsed.path.lower()
    for brand in BRANDS:
        pattern = rf"(?<![a-z0-9]){re.escape(brand)}(?![a-z0-9])"
        in_subdomain = re.search(pattern, subdomain) is not None
        in_domain = brand in domain_words and len(domain_words) > 1 and bool(domain_words & PHISHING_WORDS)
        in_path = re.search(pattern, path) is not None and any(w in path for w in PHISHING_WORDS)
        if in_subdomain or in_domain or in_path:
            evidence = subdomain if in_subdomain else real_domain if in_domain else parsed.path
            reasons.append({
                "category": "brand_in_subdomain",
                "reason": f"The URL mentions {brand} but uses a different domain",
                "points": 35,
                "evidence": original_evidence(original_url, evidence),
            })
            break

    return reasons
