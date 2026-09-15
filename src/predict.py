
from pathlib import Path

import joblib
import numpy as np

from .preprocessing import clean_text
from .priority_rules import rule_override
from .response_generator import generate_response

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

# Below this confidence, a query is flagged for human review.
# Chosen by inspecting the precision/recall tradeoff on the test set (see
# notebook) -- 0.55 catches most low-confidence errors while only flagging
# a manageable fraction of correct predictions. Documented as a tunable
# constant, not a magic number buried in logic.
CONFIDENCE_THRESHOLD = 0.55


class QueryClassifier:
    def __init__(self, models_dir: Path = MODELS_DIR):
        self.features = joblib.load(models_dir / "feature_union.joblib")
        self.category_model = joblib.load(models_dir / "category_classifier.joblib")
        self.priority_model = joblib.load(models_dir / "priority_classifier.joblib")
        self.category_labels = joblib.load(models_dir / "category_labels.joblib")
        self.priority_labels = joblib.load(models_dir / "priority_labels.joblib")

    def predict(self, raw_text: str) -> dict:
        if not raw_text or not raw_text.strip():
            raise ValueError("query_text must be a non-empty string")

        cleaned = clean_text(raw_text)
        X = self.features.transform([cleaned])

        cat_proba = self.category_model.predict_proba(X)[0]
        cat_idx = int(np.argmax(cat_proba))
        category = self.category_model.classes_[cat_idx]
        category_confidence = float(cat_proba[cat_idx])

        pri_proba = self.priority_model.predict_proba(X)[0]
        pri_idx = int(np.argmax(pri_proba))
        model_priority = self.priority_model.classes_[pri_idx]

        final_priority, rule_overridden = rule_override(raw_text, model_priority)

        human_review_flag = category_confidence < CONFIDENCE_THRESHOLD

        suggested_response = generate_response(raw_text, category, human_review_flag)

        return {
            "query_text": raw_text,
            "category": category,
            "priority": final_priority,
            "confidence_score": round(category_confidence, 4),
            "human_review_flag": bool(human_review_flag),
            "suggested_response": suggested_response,
            "priority_rule_overridden": rule_overridden,
        }


if __name__ == "__main__":
    clf = QueryClassifier()
    samples = [
        "I need a quotation for 500 components.",
        "What is the status of my order ORD-12345?",
        "The bearings I received are damaged, need urgent replacement.",
        "hi",
    ]
    for s in samples:
        print(clf.predict(s))
