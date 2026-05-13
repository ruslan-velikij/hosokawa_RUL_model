#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "models" else SCRIPT_DIR

PROCESSED_DIR = PROJECT_ROOT / "processed"
TOIR_DIR = PROJECT_ROOT / "toir_hosokawa"
RESULTS_DIR = PROJECT_ROOT / "results"

OUT_DIR = PROCESSED_DIR / "baseline_event_classifier"

NODES = {
    "compactor": {
        "train_path": PROCESSED_DIR / "compactor_train_prepared.parquet",
        "test_path": PROCESSED_DIR / "compactor_test_prepared.parquet",
    },
    "mill": {
        "train_path": PROCESSED_DIR / "mill_train_prepared.parquet",
        "test_path": PROCESSED_DIR / "mill_test_prepared.parquet",
    },
}

TARGET = "event_in_72h"

MAX_TRAIN_ROWS = 1_000_000

NEGATIVE_TO_POSITIVE_RATIO = 2

MAX_TEST_ROWS: int | None = None

RANDOM_STATE = 42

TRAIN_HIST_GRADIENT_BOOSTING = True

TRAIN_RANDOM_FOREST = False

CALCULATE_PERMUTATION_IMPORTANCE = True
PERMUTATION_IMPORTANCE_SAMPLE_ROWS = 30_000
PERMUTATION_IMPORTANCE_REPEATS = 3

THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50]


def log(msg: str = "") -> None:
    print(msg, flush=True)


def read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    return pd.read_parquet(path)


def get_feature_columns(df: pd.DataFrame, target: str) -> list[str]:
    excluded = {
        "DT",
        "split",
        "time_to_next_event_hours",
        "event_in_24h",
        "event_in_48h",
        "event_in_72h",
        "pre_event_window",
        target,
    }

    feature_cols = []
    for col in df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)

    return feature_cols


def downcast_numeric(df: pd.DataFrame, feature_cols: list[str], target: str) -> pd.DataFrame:
    for col in feature_cols:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="float")
        elif pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")

    if target in df.columns:
        df[target] = df[target].astype("int8")

    return df


def make_balanced_train_sample(
    train_df: pd.DataFrame,
    target: str,
    max_rows: int,
    negative_to_positive_ratio: int,
    random_state: int,
) -> pd.DataFrame:
    pos = train_df[train_df[target] == 1]
    neg = train_df[train_df[target] == 0]

    if len(pos) == 0:
        raise ValueError(f"В train нет положительных примеров для цели {target}.")
    if len(neg) == 0:
        raise ValueError(f"В train нет отрицательных примеров для цели {target}.")

    desired_pos = len(pos)
    desired_neg = min(len(neg), desired_pos * negative_to_positive_ratio)
    desired_total = desired_pos + desired_neg

    if desired_total > max_rows:
        desired_pos = max(1, max_rows // (negative_to_positive_ratio + 1))
        desired_pos = min(desired_pos, len(pos))
        desired_neg = min(max_rows - desired_pos, len(neg), desired_pos * negative_to_positive_ratio)

    pos_sample = pos.sample(n=desired_pos, random_state=random_state) if desired_pos < len(pos) else pos
    neg_sample = neg.sample(n=desired_neg, random_state=random_state) if desired_neg < len(neg) else neg

    sample = pd.concat([pos_sample, neg_sample], axis=0)
    sample = sample.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    return sample


def maybe_sample_test(test_df: pd.DataFrame, max_rows: int | None, random_state: int) -> pd.DataFrame:
    if max_rows is None or len(test_df) <= max_rows:
        return test_df.reset_index(drop=True)

    return test_df.sample(n=max_rows, random_state=random_state).reset_index(drop=True)


def class_balance(y: pd.Series) -> dict:
    counts = y.value_counts(dropna=False).sort_index()
    total = len(y)
    return {
        str(int(k)): {
            "count": int(v),
            "share": float(v / total) if total else None,
        }
        for k, v in counts.items()
    }


def metrics_at_threshold(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> dict:
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def safe_roc_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_proba))


def safe_average_precision(y_true: np.ndarray, y_proba: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, y_proba))


def save_json(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def fit_dummy_model(X_train: pd.DataFrame, y_train: pd.Series):
    model = DummyClassifier(strategy="most_frequent")
    model.fit(X_train, y_train)
    return model


def fit_hist_gradient_boosting(X_train: pd.DataFrame, y_train: pd.Series):
    model = HistGradientBoostingClassifier(
        max_iter=120,
        learning_rate=0.08,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=0.05,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def fit_random_forest(X_train: pd.DataFrame, y_train: pd.Series):
    model = RandomForestClassifier(
        n_estimators=80,
        max_depth=14,
        min_samples_leaf=50,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=0,
    )
    model.fit(X_train, y_train)
    return model


def predict_positive_proba(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.shape[1] == 1:
            return np.zeros(len(X), dtype=float)
        return proba[:, 1]

    pred = model.predict(X)
    return pred.astype(float)


def make_permutation_importance(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_cols: list[str],
    node: str,
    model_name: str,
    out_path: Path,
) -> None:
    if not CALCULATE_PERMUTATION_IMPORTANCE:
        return

    if len(X_test) == 0:
        return

    n = min(PERMUTATION_IMPORTANCE_SAMPLE_ROWS, len(X_test))
    sample_idx = np.random.default_rng(RANDOM_STATE).choice(len(X_test), size=n, replace=False)

    X_small = X_test.iloc[sample_idx]
    y_small = y_test.iloc[sample_idx]

    try:
        result = permutation_importance(
            model,
            X_small,
            y_small,
            scoring="average_precision",
            n_repeats=PERMUTATION_IMPORTANCE_REPEATS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        imp_df = pd.DataFrame({
            "node": node,
            "model": model_name,
            "feature": feature_cols,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }).sort_values("importance_mean", ascending=False)

        imp_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    except Exception as exc:
        warnings.warn(f"Не удалось посчитать permutation importance для {node}/{model_name}: {exc}")


def evaluate_model(
    node: str,
    model_name: str,
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_cols: list[str],
    out_prefix: Path,
) -> dict:
    y_true = y_test.to_numpy().astype(int)
    y_proba = predict_positive_proba(model, X_test)

    threshold_rows = []
    for thr in THRESHOLDS:
        threshold_rows.append(metrics_at_threshold(y_true, y_proba, thr))

    threshold_df = pd.DataFrame(threshold_rows)
    threshold_path = out_prefix.with_name(out_prefix.name + f"_{model_name}_threshold_metrics.csv")
    threshold_df.to_csv(threshold_path, index=False, encoding="utf-8-sig")

    main_metrics = {
        "node": node,
        "model": model_name,
        "target": TARGET,
        "roc_auc": safe_roc_auc(y_true, y_proba),
        "average_precision": safe_average_precision(y_true, y_proba),
        "threshold_metrics_path": str(threshold_path),
        "metrics_by_threshold": threshold_rows,
    }

    importance_path = out_prefix.with_name(out_prefix.name + f"_{model_name}_feature_importance.csv")
    make_permutation_importance(
        model=model,
        X_test=X_test,
        y_test=y_test,
        feature_cols=feature_cols,
        node=node,
        model_name=model_name,
        out_path=importance_path,
    )
    if importance_path.exists():
        main_metrics["feature_importance_path"] = str(importance_path)

    return main_metrics


def process_node(node: str, paths: dict) -> dict:
    log("=" * 80)
    log(f"Обучение baseline-классификатора для узла: {node}")
    log(f"Train: {paths['train_path']}")
    log(f"Test:  {paths['test_path']}")

    train_df = read_parquet(paths["train_path"])
    test_df = read_parquet(paths["test_path"])

    if TARGET not in train_df.columns:
        raise ValueError(f"В train-файле нет целевой колонки {TARGET}")
    if TARGET not in test_df.columns:
        raise ValueError(f"В test-файле нет целевой колонки {TARGET}")

    feature_cols = get_feature_columns(train_df, TARGET)
    if not feature_cols:
        raise ValueError(f"Не найдено числовых признаков для обучения узла {node}")

    log(f"Исходный train размер: {train_df.shape}")
    log(f"Исходный test размер:  {test_df.shape}")
    log(f"Количество признаков X: {len(feature_cols)}")
    log(f"Целевая переменная: {TARGET}")

    train_df = train_df[feature_cols + [TARGET]].copy()
    test_df = test_df[feature_cols + [TARGET]].copy()

    train_df = train_df[train_df[TARGET].notna()].copy()
    test_df = test_df[test_df[TARGET].notna()].copy()

    train_df[TARGET] = train_df[TARGET].astype("int8")
    test_df[TARGET] = test_df[TARGET].astype("int8")

    log("Баланс классов train до сэмплирования:")
    log(json.dumps(class_balance(train_df[TARGET]), ensure_ascii=False, indent=2))

    log("Баланс классов test:")
    log(json.dumps(class_balance(test_df[TARGET]), ensure_ascii=False, indent=2))

    train_sample = make_balanced_train_sample(
        train_df=train_df,
        target=TARGET,
        max_rows=MAX_TRAIN_ROWS,
        negative_to_positive_ratio=NEGATIVE_TO_POSITIVE_RATIO,
        random_state=RANDOM_STATE,
    )

    test_eval = maybe_sample_test(test_df, MAX_TEST_ROWS, RANDOM_STATE)

    log(f"Train после сэмплирования: {train_sample.shape}")
    log("Баланс классов train после сэмплирования:")
    log(json.dumps(class_balance(train_sample[TARGET]), ensure_ascii=False, indent=2))

    log(f"Test для оценки: {test_eval.shape}")

    train_sample = downcast_numeric(train_sample, feature_cols, TARGET)
    test_eval = downcast_numeric(test_eval, feature_cols, TARGET)

    X_train = train_sample[feature_cols]
    y_train = train_sample[TARGET]
    X_test = test_eval[feature_cols]
    y_test = test_eval[TARGET]

    out_prefix = OUT_DIR / f"{node}_{TARGET}"

    result = {
        "node": node,
        "target": TARGET,
        "feature_columns": feature_cols,
        "n_features": len(feature_cols),
        "train_rows_original": int(len(train_df)),
        "test_rows_original": int(len(test_df)),
        "train_rows_used": int(len(train_sample)),
        "test_rows_used": int(len(test_eval)),
        "class_balance_train_original": class_balance(train_df[TARGET]),
        "class_balance_test": class_balance(test_df[TARGET]),
        "class_balance_train_used": class_balance(train_sample[TARGET]),
        "settings": {
            "max_train_rows": MAX_TRAIN_ROWS,
            "negative_to_positive_ratio": NEGATIVE_TO_POSITIVE_RATIO,
            "max_test_rows": MAX_TEST_ROWS,
            "thresholds": THRESHOLDS,
            "train_hist_gradient_boosting": TRAIN_HIST_GRADIENT_BOOSTING,
            "train_random_forest": TRAIN_RANDOM_FOREST,
            "calculate_permutation_importance": CALCULATE_PERMUTATION_IMPORTANCE,
        },
        "models": {},
    }

    log("Обучение DummyClassifier...")
    dummy = fit_dummy_model(X_train, y_train)
    result["models"]["dummy_most_frequent"] = evaluate_model(
        node=node,
        model_name="dummy_most_frequent",
        model=dummy,
        X_test=X_test,
        y_test=y_test,
        feature_cols=feature_cols,
        out_prefix=out_prefix,
    )

    if TRAIN_HIST_GRADIENT_BOOSTING:
        log("Обучение HistGradientBoostingClassifier...")
        hgb = fit_hist_gradient_boosting(X_train, y_train)
        result["models"]["hist_gradient_boosting"] = evaluate_model(
            node=node,
            model_name="hist_gradient_boosting",
            model=hgb,
            X_test=X_test,
            y_test=y_test,
            feature_cols=feature_cols,
            out_prefix=out_prefix,
        )

    if TRAIN_RANDOM_FOREST:
        log("Обучение RandomForestClassifier...")
        rf = fit_random_forest(X_train, y_train)
        result["models"]["random_forest"] = evaluate_model(
            node=node,
            model_name="random_forest",
            model=rf,
            X_test=X_test,
            y_test=y_test,
            feature_cols=feature_cols,
            out_prefix=out_prefix,
        )

        rf_importance = pd.DataFrame({
            "node": node,
            "model": "random_forest_builtin",
            "feature": feature_cols,
            "importance": rf.feature_importances_,
        }).sort_values("importance", ascending=False)

        rf_importance_path = out_prefix.with_name(out_prefix.name + "_random_forest_builtin_feature_importance.csv")
        rf_importance.to_csv(rf_importance_path, index=False, encoding="utf-8-sig")
        result["models"]["random_forest"]["builtin_feature_importance_path"] = str(rf_importance_path)

    metrics_path = out_prefix.with_name(out_prefix.name + "_metrics.json")
    save_json(metrics_path, result)
    log(f"Метрики сохранены: {metrics_path}")

    return result


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log("Этап 4_4: обучение базового классификатора событий")
    log(f"Цель: {TARGET}")
    log(f"Выходная папка: {OUT_DIR.resolve()}")
    log("")

    all_results = {}

    for node, paths in NODES.items():
        all_results[node] = process_node(node, paths)

    summary_path = OUT_DIR / f"summary_{TARGET}.json"
    save_json(summary_path, all_results)

    log("")
    log("=" * 80)
    log("Готово.")
    log(f"Сводный отчет: {summary_path}")


if __name__ == "__main__":
    main()
