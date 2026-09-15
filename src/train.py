
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "customer_queries_dataset.csv"
MODELS_DIR = ROOT / "models"
RANDOM_STATE = 42


def build_feature_union() -> FeatureUnion:
    """Word n-grams capture vocabulary/meaning; char n-grams add typo
    robustness. Both are fit jointly via FeatureUnion so a single
    vectorizer object can be saved and reused at inference time."""
    word_vec = TfidfVectorizer(
        ngram_range=(1, 2), min_df=1, sublinear_tf=True, analyzer="word"
    )
    char_vec = TfidfVectorizer(
        ngram_range=(3, 5), min_df=1, sublinear_tf=True, analyzer="char_wb"
    )
    return FeatureUnion([("word", word_vec), ("char", char_vec)])


def evaluate(name, y_true, y_pred, labels):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.3f} | Macro-F1: {macro_f1:.3f}")
    print(report)
    return {"accuracy": acc, "macro_f1": macro_f1}


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    hard_df = df[df["is_hard_case"] == True].reset_index(drop=True)  # noqa: E712
    main_df = df[df["is_hard_case"] == False].reset_index(drop=True)  # noqa: E712

    X = main_df["query_text"].astype(str)
    y_cat = main_df["category"]
    y_pri = main_df["priority"]

    X_train, X_test, ycat_train, ycat_test, ypri_train, ypri_test = train_test_split(
        X, y_cat, y_pri, test_size=0.2, random_state=RANDOM_STATE, stratify=y_cat
    )

    # --- Feature extraction (fit ONLY on train, transform test) ---
    features = build_feature_union()
    Xtr = features.fit_transform(X_train)
    Xte = features.transform(X_test)

    cat_labels = sorted(y_cat.unique())
    pri_labels = sorted(y_pri.unique())

    # --- Category model: Logistic Regression ---
    # Chosen over SVM because it provides native predict_proba (no calibration
    # wrapper needed), trains faster, and is fully interpretable -- easy to
    # inspect which words drive each category via the learned coefficients.
    category_model = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
    )
    category_model.fit(Xtr, ycat_train)
    cat_metrics = evaluate("Category: Logistic Regression", ycat_test, category_model.predict(Xte), cat_labels)

    # --- Priority model (Logistic Regression on the same features) ---
    priority_model = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
    )
    priority_model.fit(Xtr, ypri_train)
    pri_metrics = evaluate("Priority: Logistic Regression", ypri_test, priority_model.predict(Xte), pri_labels)

    # --- Stress test on held-out hard cases ---
    hard_metrics = {}
    if len(hard_df) > 0:
        Xhard = features.transform(hard_df["query_text"].astype(str))
        hard_cat_pred = category_model.predict(Xhard)
        hard_acc = accuracy_score(hard_df["category"], hard_cat_pred)
        hard_metrics["hard_case_category_accuracy"] = hard_acc
        print(f"\n--- Hard case stress test ---\nCategory accuracy on {len(hard_df)} hard cases: {hard_acc:.3f}")
        for text, true_cat, pred_cat in zip(hard_df["query_text"], hard_df["category"], hard_cat_pred):
            flag = "OK" if true_cat == pred_cat else "MISS"
            print(f"  [{flag}] true={true_cat:<16} pred={pred_cat:<16} | {text[:70]}")

    # --- Save artifacts ---
    joblib.dump(features, MODELS_DIR / "feature_union.joblib")
    joblib.dump(category_model, MODELS_DIR / "category_classifier.joblib")
    joblib.dump(priority_model, MODELS_DIR / "priority_classifier.joblib")
    joblib.dump(cat_labels, MODELS_DIR / "category_labels.joblib")
    joblib.dump(pri_labels, MODELS_DIR / "priority_labels.joblib")

    metrics = {
        "category": cat_metrics,
        "priority": pri_metrics,
        **hard_metrics,
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nArtifacts saved to {MODELS_DIR}")
    return metrics


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
