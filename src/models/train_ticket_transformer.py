"""
train_ticket_transformer.py

Phase 3 - Deep Learning
Fine-tunes a pretrained DistilBERT model to classify support tickets.

This is a genuinely different approach from the TF-IDF baseline:
instead of counting words, DistilBERT already "understands" language
from being pretrained on huge amounts of text, and we're just teaching
it our specific 5 categories on top of that existing knowledge.
This is called "transfer learning" — a core concept in modern deep learning.

Run it with:  uv run python src/models/train_ticket_transformer.py

Note: this will take real minutes to run on CPU (no GPU needed, just patience).
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
MAX_LENGTH = 128  # cap how many tokens (word pieces) we look at per ticket

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

    # Neural networks need numeric labels, not strings — encode them
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["ticket_type"])
    num_labels = len(label_encoder.classes_)
    print(f"Categories: {list(label_encoder.classes_)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Convert to HuggingFace Dataset format (what Trainer expects)
    train_dataset = Dataset.from_dict({"text": X_train.tolist(), "label": y_train.tolist()})
    test_dataset = Dataset.from_dict({"text": X_test.tolist(), "label": y_test.tolist()})

    print(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        # Turns raw text into token IDs the model understands.
        # truncation=True cuts off text longer than MAX_LENGTH
        # padding="max_length" pads shorter text so all inputs are the same size
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)

    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels
    )

    training_args = TrainingArguments(
        output_dir=str(PROJECT_ROOT / "data" / "processed" / "transformer_checkpoints"),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="no",  # don't save checkpoints after every epoch, saves disk space
        logging_steps=20,
        report_to=[],  # we're logging to MLflow manually instead
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    with mlflow.start_run(run_name="distilbert_finetuned"):
        mlflow.log_params({
            "model_type": "DistilBERT (fine-tuned)",
            "base_model": MODEL_NAME,
            "epochs": 3,
            "max_length": MAX_LENGTH,
            "batch_size": 16,
        })

        print("Fine-tuning DistilBERT... this will take a few minutes.")
        trainer.train()

        print("Evaluating...")
        eval_results = trainer.evaluate()
        print(eval_results)

        mlflow.log_metric("accuracy", eval_results["eval_accuracy"])
        mlflow.log_metric("f1_macro", eval_results["eval_f1_macro"])

    # Save the fine-tuned model + tokenizer so we can load it later in the API
    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_OUT_DIR))
    tokenizer.save_pretrained(str(MODEL_OUT_DIR))
    print(f"\nModel saved to: {MODEL_OUT_DIR}")
    print(f"Label classes (in order): {list(label_encoder.classes_)}")


if __name__ == "__main__":
    main()
