import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import (
    create_features,
    build_preprocessor,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES
)


# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "train.csv"

MODEL_DIR = BASE_DIR / "artifacts" / "models"
MODEL_PATH = MODEL_DIR / "gradient_boosting_v1.joblib"

REPORT_DIR = BASE_DIR / "reports"
METRICS_PATH = REPORT_DIR / "training_metrics.json"


def main():

    print("=" * 50)
    print("TASK 5 - GRADIENT BOOSTING TRAINING")
    print("=" * 50)

    # Load dataset
    print("\nLoading training dataset...")

    train_df = pd.read_csv(DATA_PATH)

    # Feature engineering
    print("Performing feature engineering...")

    processed_df = create_features(train_df)

    feature_columns = (
        NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    )

    X = processed_df[feature_columns]
    y = train_df["Transported"].astype(int)

    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Build model pipeline
    print("Building Gradient Boosting pipeline...")

    model = Pipeline([
        ("preprocessor", build_preprocessor()),

        ("classifier", GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        ))
    ])

    # Train
    print("Training model...")

    model.fit(X_train, y_train)

    # Evaluate
    print("Evaluating model...")

    y_pred = model.predict(X_val)
    y_probability = model.predict_proba(X_val)[:, 1]

    accuracy = accuracy_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_probability)

    cm = confusion_matrix(y_val, y_pred)

    report = classification_report(
        y_val,
        y_pred,
        output_dict=True
    )

    # Create directories
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Save model
    joblib.dump(model, MODEL_PATH)

    # Save evaluation metrics
    metrics = {
        "model_name": "Gradient Boosting Classifier",
        "model_version": "1.0.0",
        "dataset": "Spaceship Titanic",
        "total_rows": len(train_df),
        "training_rows": len(X_train),
        "validation_rows": len(X_val),
        "validation_accuracy": accuracy,
        "validation_roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "parameters": {
            "n_estimators": 150,
            "learning_rate": 0.1,
            "max_depth": 3,
            "random_state": 42
        }
    }

    with open(METRICS_PATH, "w") as file:
        json.dump(metrics, file, indent=4)

    # Display results
    print("\n" + "=" * 50)
    print("TRAINING COMPLETED")
    print("=" * 50)

    print("Training Rows:", len(X_train))
    print("Validation Rows:", len(X_val))
    print("Validation Accuracy:", round(accuracy, 4))
    print("Validation ROC-AUC:", round(roc_auc, 4))

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))

    print("\nModel saved at:", MODEL_PATH)
    print("Metrics saved at:", METRICS_PATH)


if __name__ == "__main__":
    main()