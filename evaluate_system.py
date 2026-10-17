
# PhishGuard - Task 4: System evaluation
# runs the 60 test cases and compares the ML model alone with ML model + rules

import csv
import importlib
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from url_rules import analyze_url_rules

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "test_cases.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_function(module_name, function_name):
    # use the real function if it exists, otherwise the stub
    module = importlib.import_module(module_name)

    if hasattr(module, function_name):
        return getattr(module, function_name), False

    stubs = importlib.import_module("stubs")
    return getattr(stubs, function_name), True


predict_email, email_stub = load_function(
    "email_module", "predict_email"
)
predict_url, url_stub = load_function(
    "url_module", "predict_url"
)
analyze_email_rules, rules_stub = load_function(
    "email_rules", "analyze_email_rules"
)
combine_risk, risk_stub = load_function(
    "risk", "combine_risk"
)

stub_mode = any([
    email_stub, url_stub, rules_stub, risk_stub
])

print("Using placeholder components:", stub_mode)


def evaluate():
    with open(DATA_FILE, encoding="utf-8-sig", newline="") as f:
        cases = list(csv.DictReader(f))

    predictions = []

    for case in cases:
        content = case["content"]
        actual = int(case["label"])

        if case["type"] == "url":
            ml_result = predict_url(content)
            reasons = analyze_url_rules(content)
        else:
            ml_result = predict_email(content)
            reasons = analyze_email_rules(content)

        probability = float(ml_result["probability"])

        # Contract: probability means P(phishing).
        ml_prediction = int(probability >= 0.5)

        risk = combine_risk(probability, reasons)

        # Medium and High are classified as phishing.
        combined_prediction = int(
            risk["level"] in ("Medium", "High")
        )

        predictions.append({
            "id": case["id"],
            "type": case["type"],
            "actual": actual,
            "ml_only": ml_prediction,
            "ml_and_rules": combined_prediction,
        })

    df = pd.DataFrame(predictions)

    df.to_csv(
        RESULTS_DIR / "system_predictions.csv",
        index=False
    )

    metrics_rows = []

    for input_type in ("email", "url"):
        subset = df[df["type"] == input_type]
        y_true = subset["actual"]

        for method in ("ml_only", "ml_and_rules"):
            y_pred = subset[method]

            metrics_rows.append({
                "type": input_type,
                "method": method,
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(
                    y_true, y_pred, zero_division=0
                ),
                "recall": recall_score(
                    y_true, y_pred, zero_division=0
                ),
                "f1": f1_score(
                    y_true, y_pred, zero_division=0
                ),
                "evaluation_mode": (
                    "STUB_DEMO" if stub_mode else "REAL"
                )
            })

            cm = confusion_matrix(
                y_true, y_pred, labels=[0, 1]
            )

            display = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=["Safe", "Phishing"]
            )

            display.plot(cmap="Blues", values_format="d")
            plt.title(f"{input_type}: {method}")
            plt.tight_layout()

            plt.savefig(
                RESULTS_DIR / f"cm_{input_type}_{method}.png"
            )
            plt.close()

    metrics = pd.DataFrame(metrics_rows)

    metrics.to_csv(
        RESULTS_DIR / "system_evaluation.csv",
        index=False
    )

    print("\nEvaluation results:")
    print(metrics.to_string(index=False))
    print("\nResults saved in:", RESULTS_DIR)


if __name__ == "__main__":
    evaluate()
