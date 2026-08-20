"""
train_and_compare.py

Phase 2 - Model comparison with MLflow tracking
Trains two models (Logistic Regression baseline, XGBoost) and logs
both runs to MLflow so you can compare them side by side.

Run it with:  uv run python src/models/train_and_compare.py

Then view your results with:  uv run mlflow ui
(opens a dashboard at http://localhost:5000)
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
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from xgboost import XGBClassifier
import joblib
import mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "processed" / "insighthub.db"
MLFLOW_DB_PATH = PROJECT_ROOT / "data" / "processed" / "mlflow.db"
MODEL_OUT_PATH = PROJECT_ROOT / "data" / "processed" / "churn_best_model.joblib"
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
mlflow.set_experiment("churn_prediction")


def load_data() -> pd.DataFrame:
    engine = create_engine(f"sqlite:///{DB_PATH}")
    return pd.read_sql("SELECT * FROM customers", engine)


def prepare_features(df: pd.DataFrame):
    y = (df["churn"] == "Yes").astype(int)
    X = df.drop(columns=["churn", "customerid"], errors="ignore")
    numeric_features = ["tenure", "monthlycharges", "totalcharges"]
    categorical_features = [c for c in X.columns if c not in numeric_features]
    return X, y, numeric_features, categorical_features


def build_preprocessor(numeric_features, categorical_features) -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])


def train_and_log(name, model, preprocessor, X_train, X_test, y_train, y_test, params):
    """Train one model, evaluate it, and log everything to MLflow."""
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])

    with mlflow.start_run(run_name=name):
        # Log the hyperparameters we're using
        mlflow.log_params(params)

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)

        # Log the metrics so we can compare runs later in the MLflow UI
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("f1_score", f1)

        print(f"\n--- {name} ---")
        print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
        print(f"ROC-AUC: {auc:.3f} | F1: {f1:.3f}")

        return pipeline, auc


def main():
    print("Loading data...")
    df = load_data()
    X, y, numeric_features, categorical_features = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = {}

    # ── Model 1: Logistic Regression (baseline) ─────────────────
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    lr_model = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr_params = {"model_type": "LogisticRegression", "class_weight": "balanced"}
    lr_pipeline, lr_auc = train_and_log(
        "logistic_regression", lr_model, preprocessor,
        X_train, X_test, y_train, y_test, lr_params
    )
    results["logistic_regression"] = (lr_pipeline, lr_auc)

    # ── Model 2: XGBoost ─────────────────────────────────────────
    # scale_pos_weight helps XGBoost handle class imbalance, similar
    # to class_weight="balanced" in sklearn
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    preprocessor2 = build_preprocessor(numeric_features, categorical_features)
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )
    xgb_params = {
        "model_type": "XGBoost",
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
    }
    xgb_pipeline, xgb_auc = train_and_log(
        "xgboost", xgb_model, preprocessor2,
        X_train, X_test, y_train, y_test, xgb_params
    )
    results["xgboost"] = (xgb_pipeline, xgb_auc)

    # ── Pick the winner and save it ─────────────────────────────
    best_name = max(results, key=lambda k: results[k][1])
    best_pipeline, best_auc = results[best_name]

    print(f"\nBest model: {best_name} (ROC-AUC: {best_auc:.3f})")
    joblib.dump(best_pipeline, MODEL_OUT_PATH)
    print(f"Saved best model to: {MODEL_OUT_PATH}")


if __name__ == "__main__":
    main()
