"""
train_ticket_transformer_fast.py

Phase 3 - Deep Learning (CPU-friendly version)
Same idea as train_ticket_transformer.py, but trimmed down to actually
finish in a reasonable time on a CPU-only laptop:
  - 1 epoch instead of 3
  - shorter max token length (64 instead of 128)
  - trained on a subset of the data instead of all of it

This is a legitimate way to work: get the full pipeline correct and
fast first, THEN scale up data/epochs later if you have more time or
access to a GPU (e.g. a free Google Colab GPU runtime).

Run it with:  uv run python src/models/train_ticket_transformer_fast.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "processed" / "insighthub.db"
MLFLOW_DB_PATH = PROJECT_ROOT / "data" / "processed" / "mlflow.db"
MODEL_OUT_DIR = PROJECT_ROOT / "data" / "processed" / "ticket_classifier_transformer"

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 64          # shorter than before (was 128) — faster tokenization + training
TRAIN_SUBSET_SIZE = 2000  # only train on 2000 tickets instead of ~6700 — much faster
TEST_SUBSET_SIZE = 500

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
mlflow.set_experiment("ticket_classification")


def load_data() -> pd.DataFrame:
    engine = create_engine(f"sqlite:///{DB_PATH}")
    return pd.read_sql("SELECT * FROM support_tickets", engine)


def combine_text(df: pd.DataFrame) -> pd.Series:
    subject = df.get("ticket_subject", pd.Series([""] * len(df))).fillna("")
    description = df["ticket_description"].fillna("")
    return (subject + " " + description).str.strip()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
    }


def main():
    print("Loading data...")
    df = load_data()
    X_text = combine_text(df)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["ticket_type"])
    num_labels = len(label_encoder.classes_)
    print(f"Categories: {list(label_encoder.classes_)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    # Shrink to a manageable subset for CPU training
    X_train = X_train[:TRAIN_SUBSET_SIZE]
    y_train = y_train[:TRAIN_SUBSET_SIZE]
    X_test = X_test[:TEST_SUBSET_SIZE]
    y_test = y_test[:TEST_SUBSET_SIZE]
    print(f"Using {len(X_train)} training examples, {len(X_test)} test examples (subset for speed).")

    train_dataset = Dataset.from_dict({"text": X_train.tolist(), "label": y_train.tolist()})
    test_dataset = Dataset.from_dict({"text": X_test.tolist(), "label": y_test.tolist()})

    print(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)

    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels
    )

    training_args = TrainingArguments(
        output_dir=str(PROJECT_ROOT / "data" / "processed" / "transformer_checkpoints"),
        num_train_epochs=1,               # was 3 — 1 epoch is enough to prove the pipeline works
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    with mlflow.start_run(run_name="distilbert_finetuned_fast"):
        mlflow.log_params({
            "model_type": "DistilBERT (fine-tuned, CPU-fast config)",
            "base_model": MODEL_NAME,
            "epochs": 1,
            "max_length": MAX_LENGTH,
            "batch_size": 16,
            "train_subset_size": TRAIN_SUBSET_SIZE,
        })

        print("Fine-tuning DistilBERT (fast config)... should take a few minutes now.")
        trainer.train()

        print("Evaluating...")
        eval_results = trainer.evaluate()
        print(eval_results)

        mlflow.log_metric("accuracy", eval_results["eval_accuracy"])
        mlflow.log_metric("f1_macro", eval_results["eval_f1_macro"])

    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_OUT_DIR))
    tokenizer.save_pretrained(str(MODEL_OUT_DIR))
    print(f"\nModel saved to: {MODEL_OUT_DIR}")
    print(f"Label classes (in order): {list(label_encoder.classes_)}")


if __name__ == "__main__":
    main()
