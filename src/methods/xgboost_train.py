"""Train XGBoost classifier for MUH hallucination detection."""

from __future__ import annotations

import bootstrap  # noqa: F401
import argparse
import json
from typing import Any

import joblib
import numpy as np
import shap
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from xgboost import XGBClassifier

from cli import add_model_arg
from paths import indicators_file, method_model_path, method_result_path

FEATURE_NAMES = ["U", "EAU", "S", "SAU"]


def parse_label(label: Any) -> int | None:
    if label is None:
        return None
    if isinstance(label, bool):
        return int(label)
    if isinstance(label, str):
        low = label.lower()
        if low in ("true", "1"):
            return 1
        if low in ("false", "0"):
            return 0
        return None
    try:
        return int(label)
    except (TypeError, ValueError):
        return None


def load_item(item: dict) -> tuple[list[float], int] | None:
    label = parse_label(item.get("hallucination"))
    if label is None:
        return None
    try:
        feat = [
            float(item.get("U", 0.0)),
            float(item.get("EAU", 0.0)),
            float(item.get("S", 0.0)),
            float(item.get("SAU", 0.0)),
        ]
    except (TypeError, ValueError):
        return None
    return feat, label


def train_and_evaluate(data_path, model_path, result_path) -> None:
    X_list: list[list[float]] = []
    y_list: list[int] = []
    bad_count = 0
    with data_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="MUH load"):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                bad_count += 1
                continue
            parsed = load_item(item)
            if parsed is None:
                bad_count += 1
                continue
            feat, label = parsed
            X_list.append(feat)
            y_list.append(label)

    X = np.array(X_list)
    y = np.array(y_list)
    print(f"Samples: {X.shape}, bad: {bad_count}, labels: {np.unique(y, return_counts=True)}")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=2 / 3, random_state=42, stratify=y_temp
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    val_prob = model.predict_proba(X_val)[:, 1]
    precision_val, recall_val, thresholds_val = precision_recall_curve(y_val, val_prob)
    f1_scores_val = 2 * precision_val * recall_val / (precision_val + recall_val + 1e-9)
    if len(thresholds_val) > 0:
        best_idx = np.argmax(f1_scores_val[:-1])
        best_threshold = thresholds_val[best_idx]
        best_val_f1 = f1_scores_val[best_idx]
    else:
        best_threshold = 0.5
        best_val_f1 = 0.0

    test_prob = model.predict_proba(X_test)[:, 1]
    pred = (test_prob > best_threshold).astype(int)
    acc = accuracy_score(y_test, pred)
    p = precision_score(y_test, pred, zero_division=0)
    r = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    auc_roc = roc_auc_score(y_test, test_prob)
    precision_test, recall_test, _ = precision_recall_curve(y_test, test_prob)
    auc_pr = auc(recall_test, precision_test)
    fpr, tpr, _ = roc_curve(y_test, test_prob)
    tpr_at_5fpr = float(np.interp(0.05, fpr, tpr))

    print("\n========== Test ==========")
    print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC-ROC: {auc_roc:.4f}, AUC-PR: {auc_pr:.4f}")
    print(f"TPR@5%FPR: {tpr_at_5fpr:.4f}, threshold: {best_threshold:.4f}")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap_mean = np.mean(np.abs(shap_values), axis=0)
    shap_importance = {name: float(val) for name, val in zip(FEATURE_NAMES, shap_mean)}

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "threshold": best_threshold}, model_path)

    result = {
        "total_samples": int(len(X)),
        "train_samples": int(len(X_train)),
        "validation_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "train_ratio": float(len(X_train) / len(X)),
        "validation_ratio": float(len(X_val) / len(X)),
        "test_ratio": float(len(X_test) / len(X)),
        "bad_count": int(bad_count),
        "validation_best_f1": float(best_val_f1),
        "best_threshold": float(best_threshold),
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "auc_roc": float(auc_roc),
        "auc_pr": float(auc_pr),
        "tpr_at_5fpr": float(tpr_at_5fpr),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "shap_feature_importance": shap_importance,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"Model: {model_path}")
    print(f"Results: {result_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MUH XGBoost detector.")
    add_model_arg(parser)
    args = parser.parse_args()
    train_and_evaluate(
        indicators_file(args.model),
        method_model_path(args.model),
        method_result_path(args.model),
    )


if __name__ == "__main__":
    main()
