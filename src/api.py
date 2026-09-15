
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .predict import QueryClassifier

app = FastAPI(
    title="Customer Query Classification API",
    description="Classifies customer queries by category and priority, "
                "returns a confidence score, suggested response, and a "
                "human-review flag for low-confidence predictions.",
    version="1.0.0",
)

_classifier: QueryClassifier | None = None


def get_classifier() -> QueryClassifier:
    global _classifier
    if _classifier is None:
        _classifier = QueryClassifier()  #object
    return _classifier


class QueryRequest(BaseModel):
    query_text: str = Field(..., min_length=1, examples=["I need a quotation for 500 components."])


class QueryResponse(BaseModel):
    query_text: str
    category: str
    priority: str
    confidence_score: float
    human_review_flag: bool
    suggested_response: str
    priority_rule_overridden: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/classify", response_model=QueryResponse)
def classify(request: QueryRequest):
    if not request.query_text.strip():
        raise HTTPException(status_code=422, detail="query_text cannot be empty")
    try:
        result = get_classifier().predict(request.query_text)
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not found. Run `python -m src.train` first.",
        )
    return result
