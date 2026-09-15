# Intelligent Customer Query Classification & Response System

A small AI/ML system that reads a customer query and returns:
- **Category** — Quotation, Order Status, Complaint, or General Enquiry
- **Priority** — High, Medium, or Low
- **Confidence Score** — how sure the model is about its category prediction
- **Suggested Response** — an auto-generated reply
- **Human Review Flag** — whether a human should check this query before it goes out

Built as an assignment for an AI/ML Engineering assessment.

---

## 1. Problem Statement

Customer support teams receive a high volume of free-text queries that currently need to be read, categorized, prioritized, and responded to manually. This is slow, inconsistent across agents, and doesn't scale.

The goal of this system is to automate the first pass of that workflow: given a raw customer message, automatically determine what kind of request it is, how urgent it is, draft a first-response, and — critically — recognize when it *isn't* confident enough to make that call alone, so a human can step in instead of a wrong answer going out silently.

The domain used for this implementation is a B2B industrial components supplier (matching the assignment's own example query, *"I need a quotation for 500 components"*), so the dataset and response templates are written in that context. The same architecture generalizes to other support domains by swapping the training data and templates.

---

## 2. Setup Instructions

```bash
# 1. Extract and enter the project
unzip customer_query_classifier.zip && cd project

# 2. Create an isolated Python environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

Requires **Python 3.10+**. Tested on Python 3.12.

---

## 3. Dependencies

| Package | Purpose |
|---|---|
| pandas, numpy | Data loading and manipulation |
| scikit-learn | TF-IDF feature extraction, Logistic Regression, evaluation metrics |
| joblib | Saving/loading trained model artifacts |
| fastapi, uvicorn | REST API layer |
| pydantic | Request/response validation for the API |

Full pinned list in [`requirements.txt`](./requirements.txt).

---

## 4. How to Run the Model (Training)

Model artifacts are **not** shipped pre-trained in this repo — they're generated locally so training is fully reproducible from the included dataset.

```bash
python -m src.train
```

This will:
1. Load `data/customer_queries_dataset.csv`
2. Hold out the 10 rows flagged `is_hard_case=True` as a separate stress-test set (never trained on)
3. Split the remaining rows 80/20 (stratified by category)
4. Extract TF-IDF features (word n-grams + character n-grams via FeatureUnion)
5. Train a Logistic Regression model for category classification
6. Train a Logistic Regression model for priority classification
7. Print accuracy / macro-F1 / per-class precision-recall, plus stress-test results on the hard cases
8. Save everything to `models/`: `feature_union.joblib`, `category_classifier.joblib`, `priority_classifier.joblib`, label files, and `metrics.json`

Run this once before starting the API.

---

## 5. How to Run the Application (API)

```bash
uvicorn src.api:app --reload
```

- API root: `http://127.0.0.1:8000`
- Interactive docs (Swagger UI): `http://127.0.0.1:8000/docs`

**Example request:**
```bash
curl -X POST http://127.0.0.1:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"query_text": "I need a quotation for 500 components."}'
```

**Example response:**
```json
{
  "query_text": "I need a quotation for 500 components.",
  "category": "Quotation",
  "priority": "Medium",
  "confidence_score": 0.8852,
  "human_review_flag": false,
  "suggested_response": "Thank you for your interest in 500 units. Our sales team will share a detailed quotation shortly. If you need this urgently, please let us know your deadline.",
  "priority_rule_overridden": false
}
```

`GET /health` is available for basic liveness checks (e.g. container orchestration probes).

You can also use the classifier directly in Python without the API:
```python
from src.predict import QueryClassifier

clf = QueryClassifier()
result = clf.predict("The bearings I received are damaged, need urgent replacement.")
print(result)
```

---

## 6. Key Design Decisions

**Logistic Regression over deep learning and other classifiers.**
With ~550 training rows across 4 classes, a fine-tuned transformer is unlikely to outperform a well-tuned TF-IDF + Logistic Regression model, and is far harder to explain and debug. Logistic Regression was chosen because it provides native probability outputs (needed for the confidence score), trains in seconds, and its learned weights are directly interpretable — you can inspect exactly which words drive each category prediction. See *Improvements* for where a transformer would earn its place.

**Word + character TF-IDF combined (`FeatureUnion`).**
Word n-grams capture vocabulary and short phrases ("need a quotation"). Character n-grams (3–5 chars) add typo robustness — "qutation" still shares most of its character chunks with "quotation" even though it's a different word. This matters for real customer messages, which are rarely typo-free.

**Confidence score is the model's own predicted probability**, not a separately engineered metric — the probability of the winning class from `predict_proba()`.

**Priority uses a hybrid model + rule-override approach**, not a pure ML prediction. A Logistic Regression model makes an initial guess, but a small keyword list (e.g., "urgent", "damaged", "refund", "escalate") can force-escalate priority upward — never downward. This is a deliberate, conservative safety net: it's cheaper to over-flag a query than to silently under-prioritize a real escalation, and a statistical model trained on a modest dataset can miss rare-but-critical phrasing.

**Human Review Flag is a single confidence threshold** (`0.55`, tunable in `src/predict.py`), chosen by inspecting where prediction errors clustered on the confidence axis rather than picked arbitrarily.

**Suggested Response is template-based, not generative (LLM-written).**
This keeps output deterministic and fully testable, avoids external API cost/latency/failure modes, and keeps the whole system explainable — you can always point to the exact template and exact keyword/entity match that produced a given reply. The tradeoff (rigidity) is accepted for this scope and addressed under *Improvements*.

---

## 7. Assumptions

- The four categories given in the assignment brief (Quotation, Order Status, Complaint, General Enquiry) are exhaustive for this scope; queries outside these are expected to fall into "General Enquiry" or trigger human review via low confidence.
- Priority is treated as its own learnable label (present in the training data) rather than purely derived from category, since real support priority depends on more than just category (e.g. a Complaint about a minor invoice typo vs. a safety issue are both "Complaint" but not equally urgent).
- One query = one category. Multi-intent messages (see hard cases below) are treated as an edge case to be caught by low confidence / human review, not natively supported with multi-label output.
- English-language input only; no language detection or translation layer.
- The dataset provided with this submission is synthetic (see *Limitations*), built to be realistic for a B2B industrial components support context, since no sample dataset was provided at the start of this assignment.

---

## 8. Limitations

**Test accuracy is inflated by template structure in the training data.**
The synthetic dataset was built from a fixed set of sentence templates with slot-filled variables. On the standard train/test split, the category classifier reaches ~100% accuracy — this is partly because train and test rows share underlying template structure (a form of data leakage), not because the task is genuinely that easy. A held-out set of 10 deliberately hand-written "hard case" queries (ambiguous, multi-intent, or near-empty messages) tells a more honest story: category accuracy drops to roughly 30% on these. This gap is the single most important thing to understand about this system's real-world readiness — production performance on genuinely novel phrasing will look much closer to the hard-case number than the clean-split number. A template-aware (grouped) train/test split would give a more trustworthy accuracy figure and is a recommended next step before deployment.

**No handling of multi-intent messages.**
A query that's part Complaint, part Quotation follow-up gets forced into a single category — usually whichever has the strongest keyword signal, which isn't always the "correct" primary intent a human would pick.

**Template responses lack nuance.**
The suggested response doesn't adapt to customer tone (frustration, urgency phrasing beyond keyword matching) or answer specific factual questions embedded in the query.

**Small, synthetic training set.**
Real customer language is messier and more varied than what 550 synthetically generated examples can capture. Class imbalance is also synthetic rather than measured from a real queue.

**No authentication, rate limiting, or persistence layer** on the API — out of scope for this assignment, called out explicitly under *Production Considerations* as needed before real deployment.

**Confidence threshold is a single tuned constant**, not recalibrated per class — some categories may warrant a different threshold than others in a production setting.

---

## 9. Project Structure

```
project/
├── data/
│   └── customer_queries_dataset.csv
├── src/
│   ├── preprocessing.py        # text cleaning, order ID / quantity extraction
│   ├── priority_rules.py       # keyword-based priority escalation
│   ├── response_generator.py   # template + entity slot-fill responses
│   ├── train.py                # trains & evaluates category and priority models
│   ├── predict.py              # inference class, confidence & review-flag logic
│   └── api.py                  # FastAPI app (/classify, /health)
├── tests/
│   ├── test_pipeline.py
│   ├── test_api.py
│   └── conftest.py
├── models/                     # generated by `python -m src.train`
├── requirements.txt
└── README.md
```
