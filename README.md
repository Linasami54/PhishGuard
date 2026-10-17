# PhishGuard

Explainable AI system that checks if an email or a link is phishing.
It uses two machine learning models (one for emails, one for links) together with simple rules, then gives a risk score from 0 to 100, the reasons, and advice.

FC112/313 Introduction to Cyber Security, group project (AI in Cyber Security).

## How to run

Use Python 3.11.

```
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501

To run the tests:

```
pytest -q
```

To evaluate the full system on the 60 test cases:

```
python evaluate_system.py
```

## Project parts

| Task | What it does | Files |
|---|---|---|
| 1 | Email model (TF-IDF + Linear SVM) | email_module.py, notebooks/email_model.ipynb, models/email_model.joblib |
| 2 | URL model (18 features + XGBoost) | url_module.py, notebooks/url_model.ipynb, models/url_model.joblib |
| 3 | Email rules, risk score and advice | email_rules.py, risk.py, data/keywords.json |
| 4 | URL rules, look-alike domains, test cases and system evaluation | url_rules.py, data/brands.json, data/test_cases.csv, evaluate_system.py |
| 5 | Streamlit app and integration | app.py, stubs.py, docs/architecture.png |

## Datasets

| Dataset | Used for | Link |
|---|---|---|
| Phishing Email Detection (18,650 emails) | Email model | https://www.kaggle.com/datasets/subhajournal/phishingemails |
| PhiUSIIL Phishing URL Dataset (235,795 links) | URL model | https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset |
| PhreshPhish (url and label columns only) | URL model | https://huggingface.co/datasets/phreshphish/phreshphish |

All datasets are in the data folder. The PhreshPhish files were downloaded with download_phreshphish.py (only the link and label, no web pages).

We added PhreshPhish because in PhiUSIIL every legitimate link is a homepage (https://www.site.com). A model trained only on it flags any normal link with a path as phishing.

## Results

Email model (test set, 3,417 emails)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.973 | 0.974 | 0.953 | 0.963 |
| Naive Bayes | 0.967 | 0.980 | 0.932 | 0.955 |
| Linear SVM (chosen) | 0.983 | 0.980 | 0.975 | 0.977 |

URL model (test set, 14,080 links)

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Random Forest | 0.928 | 0.946 | 0.904 | 0.924 | 0.972 |
| XGBoost (chosen) | 0.929 | 0.948 | 0.902 | 0.924 | 0.975 |
| Logistic Regression | 0.840 | 0.893 | 0.760 | 0.821 | 0.912 |

On 24,733 new links that were never used in training, XGBoost gets accuracy 0.801 and F1 0.759.

Full system (60 test cases we wrote ourselves, results/system_evaluation.csv)

| Input | Method | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Email | ML only | 0.80 | 0.74 | 0.93 | 0.82 |
| Email | ML + rules | 0.90 | 1.00 | 0.80 | 0.89 |
| URL | ML only | 0.93 | 1.00 | 0.87 | 0.93 |
| URL | ML + rules | 0.93 | 1.00 | 0.87 | 0.93 |

## Known limitations

- The email model was trained on old English emails (2000 to 2002). It does not understand Arabic, so Arabic emails depend on the rules.
- A short normal email that starts with "Hello" can be flagged by the email model, because many spam emails in the dataset start with Hello.
- The URL model reads only the text of the link. A phishing homepage like https://www.random-shop.com looks normal to it.
- Normal websites that still use http can get a high score from the URL model.
- The rules can be bypassed by a well written email with no warning words, or by text inside an image.

## Folders

```
phishguard/
  app.py, stubs.py              Streamlit app (Task 5)
  email_module.py               Task 1
  url_module.py                 Task 2
  email_rules.py, risk.py       Task 3
  url_rules.py                  Task 4
  evaluate_system.py            Task 4
  download_phreshphish.py       downloads the PhreshPhish links
  data/                         datasets, keywords, brands, test cases
  models/                       saved models
  notebooks/                    email_model.ipynb, url_model.ipynb
  results/                      tables and charts
  docs/                         architecture, rules table, evasion test, functional tests
  tests/                        pytest tests
```
