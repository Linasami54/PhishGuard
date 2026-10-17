# PhishGuard - Task 5: Streamlit app
# run with: streamlit run app.py
# the app calls the six functions from the other tasks. if a file is missing or broken,
# it uses the fake version in stubs.py so the app still works.

from __future__ import annotations

import html
import importlib
import os
import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import stubs

# settings

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

MAX_EMAIL_CHARS = 5000
ARABIC_NOTICE = "The ML model is trained on English; rules engine provides Arabic support."

# colors for each risk level (green, orange, red)
LEVEL_COLORS = {"Low": "#2e9e5b", "Medium": "#f39c12", "High": "#d93636"}
LEVEL_BG = {"Low": "#d4f5dd", "Medium": "#ffe8c2", "High": "#fbd0d0"}
LEVEL_ICONS = {"Low": "🟢", "Medium": "🟠", "High": "🔴"}

# which file has each function
FUNCTION_SOURCES = {
    "predict_email": "email_module",
    "predict_url": "url_module",
    "analyze_email_rules": "email_rules",
    "analyze_url_rules": "url_rules",
    "combine_risk": "risk",
    "get_advice": "risk",
}

# demo examples used if data/test_cases.csv is missing
FALLBACK_EMAILS = [
    ("Phishing (English)",
     "Dear customer,\n\nYour account will be suspended within 24 hours. "
     "Verify your password immediately: http://secure-paypa1-login.xyz/verify\n\n"
     "URGENT!!! Failure to act will result in legal action."),
    ("Safe (work email)",
     "Hi team,\n\nAttached is the agenda for tomorrow's project meeting. "
     "Please review it beforehand and send me any questions.\n\nThanks and regards,\nSara"),
    ("Phishing (Arabic)",
     "عزيزي العميل، سيتم إيقاف حسابك خلال 24 ساعة. "
     "أرسل كلمة المرور فورا عبر الرابط: http://absher-verify.top/login عاجل"),
]
FALLBACK_URLS = [
    ("Phishing (IP address)", "http://192.168.45.7/secure/login.php?verify=account"),
    ("Phishing (look-alike)", "http://paypa1-secure-login.xyz/verify"),
    ("Legitimate", "https://www.apple.com"),
]

TIPS = {
    "Social engineering": [
        ("Urgency and fear", "Scammers want you to act before you think. Messages like "
         "'your account will be closed in 24 hours' are designed to trigger panic. "
         "Stop, breathe, and verify through an official channel."),
        ("Too good to be true", "Surprise prizes, refunds and gift cards are lures. "
         "If you did not enter a contest, you did not win one."),
        ("Impersonation of authority", "Attackers pose as banks, government services "
         "(for example Absher), delivery companies or your own boss. Call the organisation "
         "using the number on its official website - never the number in the message."),
        ("Requests for secrets", "No legitimate service will ask for your password, PIN or "
         "one-time code (OTP) by email, SMS or chat. Treat any such request as an attack."),
        ("Generic greetings and odd wording", "'Dear customer' and awkward phrasing are weak "
         "signals on their own - but combined with urgency or links they matter."),
    ],
    "URL spoofing": [
        ("Read the domain from right to left", "The real owner is the part just before the "
         "first single '/'. In `paypal.com.secure-login.xyz` the owner is `secure-login.xyz`, "
         "not paypal.com."),
        ("Look-alike characters", "`paypa1.com` (digit one) and `rnicrosoft.com` ('rn' instead "
         "of 'm') are designed to fool a quick glance. Zoom in on every character."),
        ("The @ trick", "In `http://google.com@evil.example`, the browser ignores everything "
         "before the '@' and opens evil.example."),
        ("IP addresses and odd endings", "Raw IPs (`http://203.0.113.9/...`) and cheap endings "
         "such as .xyz, .top or .tk are common in phishing."),
        ("HTTPS is not a safety badge", "The padlock only means the connection is encrypted. "
         "Phishing sites can have it too. Judge the domain, not the padlock."),
        ("Shortened links", "bit.ly-style links hide the destination. Use a link-expander "
         "before opening one you did not expect."),
    ],
    "What to do": [
        ("Do not click or reply", "Type the website address yourself or use the official app."),
        ("Report it", "Report phishing to your email provider, your bank, or the relevant "
         "national cyber-security authority."),
        ("Already entered your password?", "Change it immediately everywhere you reuse it, "
         "enable two-factor authentication and contact your bank if payment data was shared."),
        ("Use a password manager", "It will not auto-fill credentials on a look-alike domain - "
         "a useful built-in warning."),
    ],
}

# loading the functions (runs only once)


@st.cache_resource(show_spinner="Loading PhishGuard models...")
def load_backend() -> dict:
    # loads the six functions once (real ones if they exist, else stubs)
    force_stubs = os.environ.get("PHISHGUARD_FORCE_STUBS") == "1"
    funcs, source, notes = {}, {}, {}
    for name, module_name in FUNCTION_SOURCES.items():
        if not force_stubs:
            try:
                funcs[name] = getattr(importlib.import_module(module_name), name)
                source[name] = "real"
                continue
            except AttributeError:
                notes[name] = f"{module_name}.py does not define {name}() yet"
            except Exception as exc:  # import error, missing joblib file, version clash...
                notes[name] = f"{module_name}: {type(exc).__name__}: {exc}"
        else:
            notes[name] = "forced via PHISHGUARD_FORCE_STUBS=1"
        funcs[name] = getattr(stubs, name)
        source[name] = "stub"

    # load the models now so the first click is fast
    for module_name in {FUNCTION_SOURCES[n] for n, src in source.items() if src == "real"}:
        loader = getattr(importlib.import_module(module_name), "load_model", None)
        if callable(loader):
            try:
                loader()
            except Exception as exc:
                notes[module_name] = f"{module_name}.load_model(): {type(exc).__name__}: {exc}"

    model_files = {
        "email_model.joblib": (MODELS_DIR / "email_model.joblib").exists(),
        "url_model.joblib": (MODELS_DIR / "url_model.joblib").exists(),
    }
    return {"funcs": funcs, "source": source, "notes": notes, "model_files": model_files}


@st.cache_data(show_spinner=False)
def load_demo_cases(csv_mtime: float | None) -> tuple[list, list]:
    # demo examples from data/test_cases.csv (3 emails and 3 links)
    path = DATA_DIR / "test_cases.csv"
    if csv_mtime is None or not path.exists():
        return FALLBACK_EMAILS, FALLBACK_URLS
    try:
        df = pd.read_csv(path).dropna(subset=["type", "content"])
    except Exception:
        return FALLBACK_EMAILS, FALLBACK_URLS

    def pick(kind: str, fallback: list) -> list:
        sub = df[df["type"].astype(str).str.lower() == kind]
        if sub.empty:
            return fallback
        # one phishing and one safe example first, then fill up to 3
        first_per_label = sub.drop_duplicates("label") if "label" in sub else sub
        chosen = pd.concat([first_per_label, sub]).drop_duplicates("id").head(3) \
            if "id" in sub else pd.concat([first_per_label, sub]).drop_duplicates().head(3)
        out = []
        for i, row in enumerate(chosen.itertuples(index=False), 1):
            note = str(getattr(row, "note", "") or "").strip()
            label = note if note and note.lower() != "nan" else f"{kind.title()} sample {i}"
            out.append((label if len(label) <= 32 else label[:31].rstrip() + "…", str(row.content)))
        return out or fallback

    return pick("email", FALLBACK_EMAILS), pick("url", FALLBACK_URLS)


def get_demo_cases() -> tuple[list, list]:
    path = DATA_DIR / "test_cases.csv"
    return load_demo_cases(path.stat().st_mtime if path.exists() else None)


# input helpers

_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")
_ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")


def normalize_url(raw: str) -> str:
    # adds http:// if the user did not type a protocol
    url = raw.strip()
    if url and not _SCHEME_RE.match(url):
        url = "http://" + url.lstrip("/")
    return url


def contains_arabic(text: str, min_ratio: float = 0.05) -> bool:
    # true if the text has Arabic letters
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    return sum(1 for c in letters if _ARABIC_RE.match(c)) / len(letters) >= min_ratio


def safe_call(label: str, func, *args):
    # runs a function and shows an error message instead of crashing
    try:
        return func(*args)
    except Exception as exc:
        st.error(f"⚠️ {label} failed: {type(exc).__name__}: {exc}")
        return None


# display helpers


def build_highlight_terms(top_words: list, reasons: list) -> list[str]:
    # words to highlight: ML top words and the evidence of each rule
    terms: set[str] = set()
    placeholders = {"NUM", "URL", "EMAIL"}  # tokens the email model substitutes for numbers/links/addresses
    for item in top_words or []:
        try:
            word, weight = item
            word = str(word).strip()
            if float(weight) > 0 and word and not (set(word.split()) & placeholders):
                terms.add(word)
        except (TypeError, ValueError):
            continue
    for reason in reasons or []:
        evidence = str(reason.get("evidence", "")).strip()
        if 2 <= len(evidence) <= 120:
            terms.add(evidence)
    return sorted(terms, key=len, reverse=True)  # longest first: prefer phrases over words


def highlight_text(text: str, terms: list[str]) -> str:
    # puts <mark> around the dangerous words (the text is escaped first, so it is safe)
    def esc(s: str) -> str:
        return html.escape(s).replace("$", "&#36;")

    if not text:
        return ""
    spans: list[tuple[int, int]] = []
    if terms:
        # match whole words only (works for Arabic too)
        pattern = re.compile(
            "|".join(f"(?<!\\w){re.escape(t)}(?!\\w)" for t in terms), flags=re.IGNORECASE
        )
        for m in pattern.finditer(text):
            if m.end() > m.start():
                spans.append((m.start(), m.end()))

    out, cursor = [], 0
    for start, end in spans:  # finditer yields non-overlapping, ordered matches
        out.append(esc(text[cursor:start]))
        out.append(f"<mark class='pg-mark'>{esc(text[start:end])}</mark>")
        cursor = end
    out.append(esc(text[cursor:]))
    return "".join(out).replace("\n", "<br>")


def make_gauge(score: int, level: str, title: str = "Risk score") -> go.Figure:
    # risk gauge from 0 to 100 (green, orange, red)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "/100", "font": {"size": 38}},
            title={"text": title, "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100], "tickvals": [0, 35, 70, 100]},
                "bar": {"color": LEVEL_COLORS.get(level, "#555"), "thickness": 0.3},
                "steps": [
                    {"range": [0, 35], "color": LEVEL_BG["Low"]},
                    {"range": [35, 70], "color": LEVEL_BG["Medium"]},
                    {"range": [70, 100], "color": LEVEL_BG["High"]},
                ],
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=60, b=10),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def level_badge(level: str, label: str | None = None) -> str:
    # colored label for the risk level
    text = html.escape(label or f"{level} risk")
    return (f"<span class='pg-badge' style='background:{LEVEL_COLORS.get(level, '#555')}'>"
            f"{LEVEL_ICONS.get(level, '')} {text}</span>")


def reasons_to_frame(reasons: list[dict]) -> pd.DataFrame:
    # table of the rules that were triggered
    return pd.DataFrame(
        [{"Category": r.get("category", ""), "Why it matters": r.get("reason", ""),
          "Points": int(r.get("points", 0)), "Evidence": str(r.get("evidence", ""))}
         for r in reasons]
    )


def render_rules_breakdown(reasons: list[dict]) -> None:
    total = sum(int(r.get("points", 0)) for r in reasons)
    with st.expander(f"🔎 Triggered rules ({len(reasons)}) - {total} points", expanded=bool(reasons)):
        if not reasons:
            st.write("No rule was triggered.")
            return
        st.dataframe(reasons_to_frame(reasons), hide_index=True, width="stretch",
                     column_config={"Points": st.column_config.NumberColumn(format="+%d")})
        st.caption("Risk score = 60% from the ML probability + 40% from the rule points (max 100).")


def render_advice(advice: list[str]) -> None:
    st.subheader("🛡️ What you should do")
    # dir='auto' shows Arabic tips right to left and English tips left to right
    items = "".join(f"<p dir='auto' style='margin:6px 0; text-align:start'>• {html.escape(tip)}</p>" for tip in advice)
    st.markdown(items, unsafe_allow_html=True)


def render_result_header(level: str, score: int, verdict: str, key: str) -> None:
    # gauge on the left, result on the right
    left, right = st.columns([1.1, 1])
    with left:
        st.plotly_chart(make_gauge(score, level), width="stretch", key=f"gauge_{key}")
    with right:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        st.markdown(level_badge(level), unsafe_allow_html=True)
        st.markdown(f"### {verdict}")
        st.caption("Risk levels: 🟢 Low < 35 · 🟠 Medium 35-69 · 🔴 High ≥ 70")


# button actions (fill or clear the input box)


def set_email(text: str) -> None:
    st.session_state["email_input"] = text
    st.session_state.pop("email_result", None)


def set_url(url: str) -> None:
    st.session_state["url_input"] = url
    st.session_state.pop("url_result", None)


def clear_email() -> None:
    st.session_state["email_input"] = ""
    st.session_state.pop("email_result", None)


def clear_url() -> None:
    st.session_state["url_input"] = ""
    st.session_state.pop("url_result", None)


# analysis: ML model + rules, then risk score and advice


def run_email_analysis(backend: dict, raw_text: str) -> dict | None:
    funcs = backend["funcs"]
    text = raw_text[:MAX_EMAIL_CHARS]  # long input is truncated (warning shown by caller)
    ml = safe_call("Email model", funcs["predict_email"], text)
    reasons = safe_call("Email rules engine", funcs["analyze_email_rules"], text)
    if ml is None or reasons is None:
        return None
    prob = ml.get("probability")
    risk = safe_call("Risk scoring", funcs["combine_risk"], prob, reasons)
    if risk is None:
        return None
    advice = safe_call("Advice", funcs["get_advice"], risk["level"], reasons) or []
    return {"text": text, "ml": ml, "reasons": reasons, "risk": risk, "advice": advice}


def run_url_analysis(backend: dict, url: str, raw_url: str | None = None) -> dict | None:
    funcs = backend["funcs"]
    ml = safe_call("URL model", funcs["predict_url"], raw_url if raw_url is not None else url)
    reasons = safe_call("URL rules engine", funcs["analyze_url_rules"], url)
    if ml is None or reasons is None:
        return None
    # if the user did not type http, we do not count "no https" against the link
    if raw_url is not None and not _SCHEME_RE.match(raw_url):
        reasons = [r for r in reasons if r.get("category") != "no_https"]
    risk = safe_call("Risk scoring", funcs["combine_risk"], ml.get("probability"), reasons)
    if risk is None:
        return None
    advice = safe_call("Advice", funcs["get_advice"], risk["level"], reasons) or []
    return {"url": url, "ml": ml, "reasons": reasons, "risk": risk, "advice": advice}


# page 1: email check


def page_email(backend: dict) -> None:
    st.header("📧 Email Check")
    st.write("Paste the text of a suspicious email. PhishGuard combines an ML model with "
             "explainable rules into one risk score.")

    emails, _ = get_demo_cases()
    st.caption("Try a demo example:")
    cols = st.columns(len(emails) + 1)
    for col, (label, content) in zip(cols, emails):
        col.button(f"📨 {label}", key=f"demo_email_{label}", on_click=set_email,
                   args=(content,), width="stretch")
    cols[-1].button("🧹 Clear", key="clear_email", on_click=clear_email, width="stretch")

    st.text_area("Email text", key="email_input", height=220,
                 placeholder="Paste the email subject and body here...")
    analyze = st.button("🔍 Analyze email", type="primary", key="analyze_email")

    if analyze:
        raw = st.session_state.get("email_input", "")
        if not raw.strip():
            st.session_state.pop("email_result", None)
            st.warning("Please paste some email text first - the box is empty.")
            st.toast("Empty input: nothing to analyze.", icon="⚠️")
        else:
            with st.spinner("Analyzing..."):
                st.session_state["email_result"] = run_email_analysis(backend, raw)
            st.session_state["email_truncated"] = len(raw) > MAX_EMAIL_CHARS
            st.session_state["email_arabic"] = contains_arabic(raw)

    result = st.session_state.get("email_result")
    if not result:
        return

    st.divider()
    if st.session_state.get("email_truncated"):
        st.warning(f"The email is longer than {MAX_EMAIL_CHARS:,} characters. Only the first "
                   f"{MAX_EMAIL_CHARS:,} were analyzed - dangerous content later in the text "
                   "may have been missed.")
    if st.session_state.get("email_arabic"):
        st.info(ARABIC_NOTICE, icon="🌐")

    ml, risk, reasons = result["ml"], result["risk"], result["reasons"]
    verdict = "Looks like PHISHING" if ml.get("label") == "phishing" else "Looks SAFE (per the ML model)"
    render_result_header(risk["level"], risk["score"], verdict, key="email")

    m1, m2, m3 = st.columns(3)
    m1.metric("ML verdict", str(ml.get("label", "?")).title())
    m2.metric("ML phishing probability", f"{float(ml.get('probability', 0)) * 100:.0f}%")
    m3.metric("Rule points", sum(int(r.get("points", 0)) for r in reasons))

    st.subheader("Dangerous words highlighted")
    # show the ML words only when the model says phishing
    ml_words = ml.get("top_words", []) if ml.get("label") == "phishing" else []
    terms = build_highlight_terms(ml_words, reasons)
    st.markdown(f"<div class='pg-text' dir='auto'>{highlight_text(result['text'], terms)}</div>",
                unsafe_allow_html=True)
    top_words = [(w, v) for w, v in ml_words if float(v) > 0 and w not in {"NUM", "URL", "EMAIL"}]
    if top_words:
        st.caption("Most influential words for the ML model: "
                   + ", ".join(f"`{w}` ({v:+.2f})" for w, v in top_words))

    render_rules_breakdown(reasons)
    render_advice(result["advice"])


# page 2: url check


def features_to_frame(features: dict, top_features: list) -> pd.DataFrame:
    # table of the URL features with their effect on the model
    influence = {}
    for item in top_features or []:
        try:
            influence[str(item[0])] = float(item[1])
        except (TypeError, ValueError, IndexError):
            continue
    rows = []
    for name, value in (features or {}).items():
        rows.append({
            "Feature": name,
            "Value": f"{value:g}" if isinstance(value, (int, float)) else str(value),
            "Model influence": influence.get(name),
        })
    df = pd.DataFrame(rows, columns=["Feature", "Value", "Model influence"])
    return df.sort_values("Model influence", ascending=False, na_position="last",
                          ignore_index=True)


def page_url(backend: dict) -> None:
    st.header("🔗 URL Check")
    st.write("Enter a link. Only the URL text is analyzed - **the website is never opened.**")

    _, urls = get_demo_cases()
    st.caption("Try a demo example:")
    cols = st.columns(len(urls) + 1)
    for col, (label, content) in zip(cols, urls):
        col.button(f"🌐 {label}", key=f"demo_url_{label}", on_click=set_url,
                   args=(content,), width="stretch")
    cols[-1].button("🧹 Clear", key="clear_url", on_click=clear_url, width="stretch")

    st.text_input("URL", key="url_input", placeholder="example.com/login or https://...")
    analyze = st.button("🔍 Analyze URL", type="primary", key="analyze_url")

    if analyze:
        raw = st.session_state.get("url_input", "")
        if not raw.strip():
            st.session_state.pop("url_result", None)
            st.warning("Please enter a URL first - the box is empty.")
            st.toast("Empty input: nothing to analyze.", icon="⚠️")
        elif re.search(r"\s", raw.strip()):
            st.session_state.pop("url_result", None)
            st.warning("A URL cannot contain spaces. Please check what you pasted.")
        else:
            url = normalize_url(raw)
            with st.spinner("Analyzing..."):
                st.session_state["url_result"] = run_url_analysis(backend, url, raw_url=raw.strip())
            st.session_state["url_added_scheme"] = url != raw.strip()

    result = st.session_state.get("url_result")
    if not result:
        return

    st.divider()
    if st.session_state.get("url_added_scheme"):
        st.info("You did not type http or https, so the protocol was not counted against this link.")

    ml, risk, reasons = result["ml"], result["risk"], result["reasons"]
    is_phish = ml.get("label") == "phishing"
    verdict = "Verdict: PHISHING" if is_phish else "Verdict: LEGITIMATE (per the ML model)"
    render_result_header(risk["level"], risk["score"], verdict, key="url")

    m1, m2, m3 = st.columns(3)
    m1.metric("ML verdict", str(ml.get("label", "?")).title())
    m2.metric("ML phishing probability", f"{float(ml.get('probability', 0)) * 100:.0f}%")
    m3.metric("Rule points", sum(int(r.get("points", 0)) for r in reasons))

    st.subheader("Extracted URL features")
    frame = features_to_frame(ml.get("features", {}), ml.get("top_features", []))
    st.dataframe(frame, hide_index=True, width="stretch", height=320,
                 column_config={"Model influence": st.column_config.NumberColumn(
                     format="%.2f", help="Weight of the feature in the ML decision "
                                         "(empty = not among the top features).")})

    render_rules_breakdown(reasons)
    render_advice(result["advice"])


# page 3: model results


def show_metrics_table(csv_name: str, owner: str) -> None:
    path = RESULTS_DIR / csv_name
    if not path.exists():
        st.info(f"📄 `results/{csv_name}` not found yet - it will appear here when {owner} delivers it.")
        return
    try:
        df = pd.read_csv(path)
        df = df.rename(columns={c: "" for c in df.columns if str(c).startswith("Unnamed")})
        numeric = df.select_dtypes("number").columns
        st.dataframe(df.style.format({c: "{:.3f}" for c in numeric}), hide_index=True,
                     width="stretch")
    except Exception as exc:
        st.error(f"Could not read results/{csv_name}: {exc}")


def show_images(patterns: list[str], owner: str, per_row: int = 3) -> None:
    # shows all images in results/ that match the names
    files: list[Path] = []
    for pattern in patterns:
        files += sorted(RESULTS_DIR.glob(pattern))
    if not files:
        st.info(f"🖼️ No images matching `results/{'`, `results/'.join(patterns)}` yet - "
                f"{owner} will add them.")
        return
    for i in range(0, len(files), per_row):
        cols = st.columns(per_row)
        for col, f in zip(cols, files[i:i + per_row]):
            col.image(str(f), caption=f.stem.replace("_", " "), width="stretch")


def page_results() -> None:
    st.header("📊 Model Results")
    st.write("Evaluation of the models and the full system. Tables and figures are loaded "
             "from the `results/` folder.")
    email_tab, url_tab, system_tab = st.tabs(["📧 Email model", "🔗 URL model", "🧪 Full system"])

    with email_tab:
        st.subheader("Model comparison")
        show_metrics_table("email_metrics.csv", "Task 1")
        st.subheader("Confusion matrices")
        show_images(["email_cm_*.png"], "Task 1")
        st.subheader("Cross-validation (5 folds)")
        show_metrics_table("email_cv_results.csv", "Task 1")
        st.subheader("Dataset and model comparison")
        show_images(["email_class_distribution.png", "email_model_comparison.png"], "Task 1", per_row=2)
        st.subheader("Top phishing / safe words")
        show_images(["email_top_words.png"], "Task 1", per_row=1)

    with url_tab:
        st.subheader("Model comparison")
        show_metrics_table("url_metrics.csv", "Task 2")
        st.subheader("Test on new links (never used in training)")
        show_metrics_table("url_new_links_metrics.csv", "Task 2")
        st.subheader("Cross-validation (5 folds)")
        show_metrics_table("url_cv_results.csv", "Task 2")
        st.subheader("Confusion matrices")
        show_images(["url_cm_*.png"], "Task 2")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("ROC curves")
            show_images(["url_roc.png"], "Task 2", per_row=1)
        with c2:
            st.subheader("Feature importance")
            show_images(["url_feature_importance.png"], "Task 2", per_row=1)

    with system_tab:
        st.subheader("ML only vs ML + rules (60 independent test cases)")
        show_metrics_table("system_evaluation.csv", "Task 4")
        st.subheader("Confusion matrices")
        show_images(["system_cm_*.png", "*system*confusion*.png", "cm_*_ml_*.png"], "Task 4")


# page 4: awareness tips


def page_tips() -> None:
    st.header("🎓 Awareness Tips")
    st.write("Technology helps, but the best defence is a pause before you click.")
    tabs = st.tabs([f"{icon} {name}" for icon, name in
                    zip(["🧠", "🔗", "✅"], TIPS.keys())])
    for tab, (_, tips) in zip(tabs, TIPS.items()):
        with tab:
            for title, body in tips:
                with st.expander(title, expanded=False):
                    st.markdown(body)
    st.success("**Golden rule:** if a message makes you feel rushed, scared or lucky - "
               "slow down and verify through an official channel.")


# main layout

CSS = """
<style>
.pg-badge {color:#fff; padding:6px 16px; border-radius:999px; font-weight:700;
           font-size:1.05rem; display:inline-block;}
.pg-text {border:1px solid rgba(128,128,128,.35); border-radius:8px; padding:12px 14px;
          line-height:1.7; word-break:break-word;}
.pg-mark {background:#ffd54a; color:#000; padding:1px 3px; border-radius:3px; font-weight:600;}
</style>
"""


def sidebar(backend: dict) -> None:
    with st.sidebar:
        st.title("🛡️ PhishGuard")
        st.caption("Explainable phishing detection for emails and URLs.")
        st.markdown("**Module status**")
        for name, src in backend["source"].items():
            st.markdown(f"{'✅' if src == 'real' else '🧪'} `{name}` - {src}")
        if "stub" in backend["source"].values():
            with st.expander("Why stubs?"):
                for name, why in backend["notes"].items():
                    st.caption(f"`{name}`: {why}")
        missing = [n for n, ok in backend["model_files"].items() if not ok]
        if missing:
            st.caption("Model files not found yet: " + ", ".join(f"`{m}`" for m in missing))
        if st.button("♻️ Reload modules", help="Re-import modules after a teammate delivers"):
            load_backend.clear()
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="PhishGuard", page_icon="🛡️", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    backend = load_backend()
    sidebar(backend)

    st.title("🛡️ PhishGuard")
    st.caption("Checks emails and links using a machine learning model and simple rules.")

    tab_email, tab_url, tab_results, tab_tips = st.tabs(
        ["📧 Email Check", "🔗 URL Check", "📊 Model Results", "🎓 Awareness Tips"]
    )
    with tab_email:
        page_email(backend)
    with tab_url:
        page_url(backend)
    with tab_results:
        page_results()
    with tab_tips:
        page_tips()


main()
