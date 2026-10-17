# PhishGuard - Task 3: Risk score and advice
# combines the ML probability with the rule points, then gives advice to the user


# final score from 0 to 100
# 60% from the ML model and 40% from the rules (rule points are capped at 100)
# if there is no ML probability we use the rules only
def combine_risk(probability=None, reasons=None):
    if reasons is None:
        reasons = []

    total_points = sum(r.get("points", 0) for r in reasons)

    if probability is not None:
        score_val = 0.6 * (probability * 100) + 0.4 * min(100, total_points)
    else:
        score_val = float(min(100, total_points))

    score = int(round(score_val))

    if score < 35:
        level = "Low"
    elif score <= 69:
        level = "Medium"
    else:
        level = "High"

    return {"score": score, "level": level}


# one tip for every rule category (email rules and URL rules)
ADVICE = {
    # email categories
    "credentials": "لا تشارك كلمة المرور أو رمز التحقق (OTP) مع أي جهة عبر البريد الإلكتروني.",
    "urgency": "لا تتسرع، فمعظم هجمات التصيد تستغل عامل الخوف والتسرع.",
    "threat": "لا تتسرع، فمعظم هجمات التصيد تستغل عامل الخوف والتسرع.",
    "reward": "إذا كان العرض مغرياً جداً وغير متوقع، فغالباً ما يكون احتيالاً.",
    "links": "لا تضغط على الروابط داخل الرسائل المشبوهة، وتحقق من عنوان الموقع عبر القنوات الرسمية.",
    "generic_greeting": "الجهات الرسمية عادة تخاطبك باسمك وليس بتحية عامة.",
    "formatting": "الحروف الكبيرة وعلامات التعجب الكثيرة أسلوب شائع في رسائل الاحتيال.",
    # URL categories
    "ip_address": "الرابط يستخدم رقم IP بدل اسم موقع، والمواقع الرسمية لا تفعل ذلك عادة.",
    "at_symbol": "علامة @ في الرابط قد تخفي الموقع الحقيقي، انتبه للجزء الذي بعدها.",
    "long_url": "الرابط طويل بشكل غير طبيعي، تأكد من اسم الموقع قبل فتحه.",
    "many_subdomains": "الرابط فيه أجزاء كثيرة قبل اسم الموقع، اسم الموقع الحقيقي هو الجزء الأخير قبل أول /.",
    "shortener": "هذا رابط مختصر يخفي الموقع الحقيقي، لا تفتحه إلا إذا كنت تعرف المرسل.",
    "suspicious_tld": "نهاية الموقع (مثل .xyz أو .top) تستخدم كثيراً في مواقع التصيد.",
    "punycode": "اسم الموقع فيه حروف شبيهة بحروف أخرى ليقلد موقعاً معروفاً.",
    "no_https": "الموقع لا يستخدم اتصالاً آمناً (HTTPS)، لا تدخل فيه أي بيانات.",
    "lookalike_domain": "اسم الموقع يشبه موقعاً معروفاً لكنه ليس هو، اكتب عنوان الموقع الرسمي بنفسك.",
    "brand_in_subdomain": "الرابط يذكر اسم شركة معروفة لكنه يوصلك إلى موقع آخر.",
}


# advice list: tips for what we found, then one general tip for the risk level
def get_advice(level, reasons=None):
    if reasons is None:
        reasons = []

    advice_list = []
    for r in reasons:
        tip = ADVICE.get(r.get("category"))
        if tip and tip not in advice_list:
            advice_list.append(tip)

    if level == "High":
        advice_list.append("تحذير: هذا المحتوى عالي الخطورة! لا تضغط على أي رابط ولا تدخل أي بيانات.")
    elif level == "Medium":
        advice_list.append("تنبيه: تأكد من المرسل أو الموقع عن طريق القنوات الرسمية قبل أي خطوة.")
    else:
        advice_list.append("يبدو آمناً بشكل عام، لكن انتبه دائماً قبل فتح الروابط أو المرفقات.")

    return advice_list
