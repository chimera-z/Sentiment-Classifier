import os
from pathlib import Path
import redis
from dotenv import load_dotenv

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

load_dotenv()
r = redis.Redis(
    host=os.getenv("REDIS_HOST"),
    port=int(os.getenv("REDIS_PORT")),
    username=os.getenv("REDIS_USERNAME"),
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=True,
)

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
    raw = movie_review.review
    cleaned = clean_text(raw)

    if not cleaned:
        raise HTTPException(
            status_code=422,
            detail="Review contains no usable text after cleaning.",
        )

    cache_key = f"sentiment:{cleaned}"

    result = r.get(cache_key)
    if result is not None:
        print("Cache HIT")
        probability_positive = float(result)
    else:
        print("Cache MISS")
        probability_positive = float(model.predict_proba([cleaned])[0, 1])
        r.set(cache_key, probability_positive)
    
    label_id = int(probability_positive >= 0.5)

    return PredictionResponse(
        review=raw,
        label=ID_TO_LABEL[label_id],
        label_id=label_id,
        probability_positive=round(probability_positive, 6),
    )
