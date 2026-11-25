"""Train candidates, record experiments, and export the selected model."""

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .features import FEATURES, NUMERIC_FEATURES, select_features

DEFAULT_DATA = Path("data/raw/ai4i2020.csv")
DEFAULT_MODEL = Path("artifacts/model.joblib")
DEFAULT_METRICS = Path("artifacts/metrics.json")
SEED = 42


def make_pipeline(estimator):
    preprocess = ColumnTransformer(
        [
            ("category", OneHotEncoder(handle_unknown="ignore"), ["Type"]),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ]
    )
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def candidates():
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
        "random_forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.9, eval_metric="logloss", random_state=SEED, n_jobs=-1,
        ),
    }


def choose_threshold(y_true, probabilities):
    thresholds = np.linspace(0.05, 0.95, 91)
    scores = [fbeta_score(y_true, probabilities >= threshold, beta=2, zero_division=0) for threshold in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def evaluate(y_true, probabilities, threshold):
    predicted = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "f2": float(fbeta_score(y_true, predicted, beta=2, zero_division=0)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def train(data_path=DEFAULT_DATA, model_path=DEFAULT_MODEL, metrics_path=DEFAULT_METRICS):
    frame = pd.read_csv(data_path)
    x, y = select_features(frame)
    if len(frame) < 100 or y.sum() < 10:
        raise ValueError("Dataset is too small or contains too few failures")

    x_develop, x_test, y_develop, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=SEED
    )
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_develop, y_develop, test_size=0.2, stratify=y_develop, random_state=SEED
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("predictive-maintenance-ai4i")
    best = None
    comparison = {}
    for name, estimator in candidates().items():
        model = make_pipeline(estimator)
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"model": name, "seed": SEED, "train_rows": len(x_train), "validation_rows": len(x_validation)})
            model.fit(x_train, y_train)
            probabilities = model.predict_proba(x_validation)[:, 1]
            threshold = choose_threshold(y_validation, probabilities)
            metrics = evaluate(y_validation, probabilities, threshold)
            mlflow.log_param("threshold", threshold)
            mlflow.log_metrics({f"validation_{key}": value for key, value in metrics.items()})
            comparison[name] = {"threshold": threshold, **metrics}
            if best is None or metrics["average_precision"] > best["validation_average_precision"]:
                best = {
                    "name": name,
                    "model": model,
                    "threshold": threshold,
                    "validation_average_precision": metrics["average_precision"],
                    "run_id": mlflow.active_run().info.run_id,
                }

    test_probabilities = best["model"].predict_proba(x_test)[:, 1]
    test_metrics = evaluate(y_test, test_probabilities, best["threshold"])
    result = {
        "dataset_rows": len(frame),
        "failure_rate": float(y.mean()),
        "split_rows": {"train": len(x_train), "validation": len(x_validation), "test": len(x_test)},
        "selected_model": best["name"],
        "threshold": best["threshold"],
        "validation": comparison,
        "test": test_metrics,
    }
    model_path = Path(model_path)
    metrics_path = Path(metrics_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best["model"], "threshold": best["threshold"], "features": FEATURES}, model_path)
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.log_metrics({f"test_{key}": value for key, value in test_metrics.items()})
        mlflow.log_artifact(str(model_path), artifact_path="selected_model")
        mlflow.log_artifact(str(metrics_path), artifact_path="selected_model")
    print(json.dumps({"selected_model": best["name"], "threshold": best["threshold"], "test": test_metrics}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    args = parser.parse_args()
    train(args.data, args.model, args.metrics)


if __name__ == "__main__":
    main()
