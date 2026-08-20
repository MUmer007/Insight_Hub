"""
train_baseline.py

Phase 2 - Classical ML
Trains a baseline Logistic Regression model to predict customer churn.

Why start with Logistic Regression instead of something fancier?
It's fast, interpretable, and gives you a "floor" to beat with better models later.
Always start simple — a complex model that beats nothing isn't proof it's good.

Run it with:  uv run python src/models/train_baseline.py
"""

from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
)
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "processed" / "insighthub.db"
MODEL_OUT_PATH = PROJECT_ROOT / "data" / "processed" / "churn_baseline_model.joblib"


def load_data() -> pd.DataFrame:
    engine = create_engine(f"sqlite:///{DB_PATH}")
    df = pd.read_sql("SELECT * FROM customers", engine)
    return df


def prepare_features(df: pd.DataFrame):
    # Target: convert Yes/No into 1/0
    y = (df["churn"] == "Yes").astype(int)

    # Drop columns that aren't useful features
    # customerid = just an ID, not a pattern the model should learn from
    # churn = this is our target, can't be a feature too
    X = df.drop(columns=["churn", "customerid"], errors="ignore")

    numeric_features = ["tenure", "monthlycharges", "totalcharges"]
    categorical_features = [c for c in X.columns if c not in numeric_features]

    return X, y, numeric_features, categorical_features


def build_pipeline(numeric_features, categorical_features) -> Pipeline:
    # Numeric pipeline: fill missing values with median, then scale
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Categorical pipeline: fill missing with "missing", then one-hot encode
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])

    # class_weight="balanced" tells the model to pay more attention to the
    # minority class (churned customers), since they're outnumbered
    model = LogisticRegression(max_iter=1000, class_weight="balanced")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])
    return pipeline


def main():
    print("Loading data...")
    df = load_data()

    X, y, numeric_features, categorical_features = prepare_features(df)
    print(f"Features: {X.shape[1]} columns, Target churn rate: {y.mean():.1%}")

    # stratify=y keeps the same churn ratio in both train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    pipeline = build_pipeline(numeric_features, categorical_features)

    print("Training model...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC Score: {auc:.3f}")

    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred)
    print(pd.DataFrame(
        cm,
        index=["Actual: No Churn", "Actual: Churn"],
        columns=["Predicted: No Churn", "Predicted: Churn"],
    ))

    # Save the trained pipeline (preprocessing + model together) to disk
    joblib.dump(pipeline, MODEL_OUT_PATH)
    print(f"\nModel saved to: {MODEL_OUT_PATH}")


if __name__ == "__main__":
    main()
