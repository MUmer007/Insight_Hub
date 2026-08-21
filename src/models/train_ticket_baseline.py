"""
train_ticket_baseline.py

Phase 3 - Classical NLP baseline
Predicts ticket_type from ticket_description text using TF-IDF + Logistic Regression.

Why TF-IDF first, before a transformer?
Same reason as Phase 2: always establish a simple, fast baseline first.
TF-IDF turns text into numbers based on how important/rare each word is.
If a transformer can't beat this, the extra complexity isn't worth it.

Run it with:  uv run python src/models/train_ticket_baseline.py
"""

from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib
import mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "processed" / "insighthub.db"
MLFLOW_DB_PATH = PROJECT_ROOT / "data" / "processed" / "mlflow.db"
MODEL_OUT_PATH = PROJECT_ROOT / "data" / "processed" / "ticket_classifier_baseline.joblib"

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
mlflow.set_experiment("ticket_classification")


def load_data() -> pd.DataFrame:
    engine = create_engine(f"sqlite:///{DB_PATH}")
    return pd.read_sql("SELECT * FROM support_tickets", engine)


def combine_text(df: pd.DataFrame) -> pd.Series:
    """Combine subject + description into one text field — more signal for the model."""
    subject = df.get("ticket_subject", pd.Series([""] * len(df))).fillna("")
    description = df["ticket_description"].fillna("")
    return (subject + " " + description).str.strip()


def main():
    print("Loading data...")
    df = load_data()

    X_text = combine_text(df)
    y = df["ticket_type"]

    print(f"Total tickets: {len(df)}")
    print(f"Ticket types: {y.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # TfidfVectorizer converts raw text into numeric features.
    # max_features caps vocabulary size (keeps only the most informative words)
    # stop_words="english" removes common filler words like "the", "is", "and"
    pipeline = Pipeline(steps=[
        ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    with mlflow.start_run(run_name="tfidf_logistic_regression"):
        mlflow.log_params({
            "model_type": "TF-IDF + LogisticRegression",
            "max_features": 5000,
            "ngram_range": "(1, 2)",
        })

        print("Training model...")
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1_macro)

        print("\n--- Classification Report ---")
        print(classification_report(y_test, y_pred))
        print(f"Accuracy: {acc:.3f} | Macro F1: {f1_macro:.3f}")

    joblib.dump(pipeline, MODEL_OUT_PATH)
    print(f"\nModel saved to: {MODEL_OUT_PATH}")


if __name__ == "__main__":
    main()
