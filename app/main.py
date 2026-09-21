import os
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from preprocess import clean_text

_DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "ml" / "sentiment_model.pkl"
)
MODEL_PATH = Path(os.getenv("SENTIMENT_MODEL_PATH", str(_DEFAULT_MODEL_PATH)))

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Sentiment model not found at {MODEL_PATH}. "
        "Set SENTIMENT_MODEL_PATH to the correct location."
    )

model = joblib.load(MODEL_PATH)

ID_TO_LABEL = {0: "negative", 1: "positive"}

app = FastAPI(
    title="IMDb Movie Review Sentiment API",
    description=(
        "Classifies a movie review as positive or negative using a "
        "CountVectorizer and LogisticRegression pipeline trained on the "
        "Stanford IMDb dataset."
    ),
    version="1.0.0",
)


class MovieReview(BaseModel):
    review: str = Field(..., description="The movie review text to classify.")


class PredictionResponse(BaseModel):
    review: str
    label: str
    label_id: int
    probability_positive: float


@app.get("/")
async def root():
    return {
        "message": "IMDb Movie Review Sentiment API",
        "endpoints": {
            "predict": "POST /predict/",
            "health": "GET /health",
            "docs": "GET /docs",
        },
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict/", response_model=PredictionResponse)
async def predict(movie_review: MovieReview):
    """Predict the sentiment of a single movie review."""
    raw = movie_review.review
    cleaned = clean_text(raw)

    if not cleaned:
        raise HTTPException(
            status_code=422,
            detail="Review contains no usable text after cleaning.",
        )

    probability_positive = float(model.predict_proba([cleaned])[0, 1])
    label_id = int(probability_positive >= 0.5)

    return PredictionResponse(
        review=raw,
        label=ID_TO_LABEL[label_id],
        label_id=label_id,
        probability_positive=round(probability_positive, 6),
    )
