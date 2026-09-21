"""Retrain the final model: bigram CountVectorizer + LogisticRegression.

Replicates the last cell of ``train_eval.ipynb`` and saves the result to
``sentiment_model.pkl``.
"""

import sys
from pathlib import Path

# Make `preprocessor` importable when this script is run from another directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

from preprocessor import clean_text

MODEL_PATH = Path(__file__).resolve().parent / "sentiment_model.pkl"


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "vectorizer",
                CountVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.95),
            ),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )


def main() -> None:
    print("Loading stanfordnlp/imdb dataset ...")
    dataset = load_dataset("stanfordnlp/imdb")

    train = dataset["train"]
    test = dataset["test"]

    X_train = [clean_text(text) for text in train["text"]]
    y_train = train["label"]
    X_test = [clean_text(text) for text in test["text"]]
    y_test = test["label"]

    clf = build_pipeline()
    print(f"Fitting pipeline on {len(X_train):,} reviews ...")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    vocab = len(clf.named_steps["vectorizer"].vocabulary_)
    print(f"Test accuracy: {acc:.4f}")
    print(f"Vocabulary size: {vocab:,}")

    joblib.dump(clf, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
