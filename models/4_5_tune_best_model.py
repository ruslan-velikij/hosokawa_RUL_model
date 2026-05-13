from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterSampler


RANDOM_STATE = 42
NODE_NAME = "mill"
TARGET_COLUMN = "event_in_72h"

PROJECT_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_DIR / "../processed"
OUTPUT_DIR = PROCESSED_DIR / "tuned_best_model_mill_event_in_72h"

TRAIN_PATH = PROCESSED_DIR / "mill_train_prepared.parquet"
TEST_PATH = PROCESSED_DIR / "mill_test_prepared.parquet"

VALIDATION_START = "2024-01-01"
TEST_START = "2025-01-01"

MAX_TRAIN_ROWS_FOR_SEARCH = 1_000_000
MAX_VALID_ROWS_FOR_SEARCH = 500_000
MAX_FINAL_TRAIN_ROWS = 1_500_000
NEGATIVE_TO_POSITIVE_RATIO = 2

N_ITER = 60
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.05), 2)

BASE_REAL_SIGNALS = [
    "MelnicaSpeed",
    "Tok_melnici",
    "Temp_korpusa_melnici",
    "Temp_perednego_podshipnika",
    "Temp_zadnego_podshipnika",
    "Skorost_shluza_melnici",
]

SERVICE_COLUMNS = {
    "N_Hosokawa",
    "Regim",
    "dt_hour",
    "dt_dayofweek",
    "dt_month",
    "dt_is_weekend",
    "hours_from_unit_start",
    "hours_from_prev_row",
}

TARGET_COLUMNS = {
    "event_in_24h",
    "event_in_48h",
    "event_in_72h",
    "pre_event_window",
    "time_to_next_event_hours",
}

PARAM_DISTRIBUTIONS: dict[str, list[Any]] = {
    "learning_rate": [0.02, 0.03, 0.05, 0.07, 0.10],
    "max_iter": [100, 150, 200, 300, 400],
    "max_leaf_nodes": [15, 31, 63],
    "min_samples_leaf": [50, 100, 200, 500, 1000],
    "l2_regularization": [0.0, 0.01, 0.1, 1.0, 5.0],
    "max_bins": [64, 128, 255],
}


def read_prepared_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл: {path}")

    df = pd.read_parquet(path)

    if "DT" in df.columns:
        df["DT"] = pd.to_datetime(df["DT"])
        df = df.sort_values("DT")
    elif isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()
        df["DT"] = df.index
    else:
        raise ValueError("Не найдено время DT: нет ни столбца DT, ни DatetimeIndex")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"В датасете нет целевой переменной {TARGET_COLUMN}")

    return df


def select_real_signal_features(df: pd.DataFrame) -> list[str]:
    expected_features: list[str] = []

    for signal in BASE_REAL_SIGNALS:
        expected_features.extend([
            signal,
            f"{signal}_lag1",
            f"{signal}_diff1",
        ])

    feature_columns = []
    for col in expected_features:
        if col not in df.columns:
            continue
        if col in SERVICE_COLUMNS:
            continue
        if col in TARGET_COLUMNS:
            continue
        if col.endswith("_was_missing"):
            continue
        if "_roll_" in col:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_columns.append(col)

    if not feature_columns:
        raise ValueError("После фильтрации не осталось признаков для обучения")

    return feature_columns


def split_train_valid(df_train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_start = pd.Timestamp(VALIDATION_START)

    inner_train = df_train[df_train["DT"] < valid_start].copy()
    valid = df_train[df_train["DT"] >= valid_start].copy()

    if inner_train.empty:
        raise ValueError(f"Пустая обучающая часть до {VALIDATION_START}")
    if valid.empty:
        raise ValueError(f"Пустая validation-часть с {VALIDATION_START} до {TEST_START}")

    return inner_train, valid


def sample_for_training(
    df: pd.DataFrame,
    target_col: str,
    max_rows: int,
    negative_to_positive_ratio: int,
    random_state: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    positives = df[df[target_col] == 1]
    negatives = df[df[target_col] == 0]

    if positives.empty:
        raise ValueError("В обучающей выборке нет положительного класса")
    if negatives.empty:
        raise ValueError("В обучающей выборке нет отрицательного класса")

    max_pos_by_total = max_rows // (negative_to_positive_ratio + 1)
    n_pos = min(len(positives), max_pos_by_total)
    n_neg = min(len(negatives), n_pos * negative_to_positive_ratio)

    pos_idx = rng.choice(positives.index.to_numpy(), size=n_pos, replace=False)
    neg_idx = rng.choice(negatives.index.to_numpy(), size=n_neg, replace=False)

    sampled = df.loc[np.concatenate([pos_idx, neg_idx])]
    sampled = sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return sampled


def sample_for_validation(
    df: pd.DataFrame,
    target_col: str,
    max_rows: int,
    random_state: int,
) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df.reset_index(drop=True)

    frac = max_rows / len(df)
    sampled_parts = []
    for _, part in df.groupby(target_col):
        n = max(1, int(len(part) * frac))
        sampled_parts.append(part.sample(n=n, random_state=random_state))

    sampled = pd.concat(sampled_parts, axis=0)
    sampled = sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return sampled


def make_xy(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    X = df[features].astype("float32")
    y = df[TARGET_COLUMN].astype("int8")
    return X, y


def class_balance(y: pd.Series) -> dict[str, dict[str, float | int]]:
    counts = y.value_counts(dropna=False).sort_index()
    total = len(y)
    return {
        str(int(k)): {
            "count": int(v),
            "share": float(v / total),
        }
        for k, v in counts.items()
    }


def evaluate_probabilities(y_true: pd.Series, y_prob: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "positive_share": float(np.mean(y_true)),
        "ap_lift": float(average_precision_score(y_true, y_prob) / max(np.mean(y_true), 1e-12)),
    }


def threshold_table(y_true: pd.Series, y_prob: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in THRESHOLDS:
        y_pred = (y_prob >= threshold).astype("int8")
        rows.append({
            "threshold": float(threshold),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "predicted_positive_share": float(np.mean(y_pred)),
        })
    return pd.DataFrame(rows)


def build_model(params: dict[str, Any]) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        **params,
        loss="log_loss",
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=RANDOM_STATE,
        verbose=0,
    )


def run_parameter_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    sampler = list(ParameterSampler(
        PARAM_DISTRIBUTIONS,
        n_iter=N_ITER,
        random_state=RANDOM_STATE,
    ))

    rows = []
    best_params = None
    best_score = -np.inf

    for idx, params in enumerate(sampler, start=1):
        print(f"[{idx:02d}/{len(sampler)}] params={params}")
        model = build_model(params)
        model.fit(X_train, y_train)

        valid_prob = model.predict_proba(X_valid)[:, 1]
        prob_metrics = evaluate_probabilities(y_valid, valid_prob)
        thr = threshold_table(y_valid, valid_prob)
        best_thr_row = thr.loc[thr["f1"].idxmax()].to_dict()

        row = {
            "candidate": idx,
            **params,
            **{f"valid_{k}": v for k, v in prob_metrics.items()},
            "valid_best_threshold": float(best_thr_row["threshold"]),
            "valid_best_f1": float(best_thr_row["f1"]),
            "valid_best_precision": float(best_thr_row["precision"]),
            "valid_best_recall": float(best_thr_row["recall"]),
        }
        rows.append(row)

        score = prob_metrics["average_precision"]
        if score > best_score:
            best_score = score
            best_params = params

        print(
            "    valid ROC-AUC={:.5f}, AP={:.5f}, AP lift={:.3f}, best F1={:.5f}".format(
                prob_metrics["roc_auc"],
                prob_metrics["average_precision"],
                prob_metrics["ap_lift"],
                best_thr_row["f1"],
            )
        )

    if best_params is None:
        raise RuntimeError("Не удалось выбрать лучшие параметры")

    results = pd.DataFrame(rows).sort_values(
        ["valid_average_precision", "valid_roc_auc", "valid_best_f1"],
        ascending=False,
    )
    return results, best_params


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Этап 4_5: подбор гиперпараметров лучшей модели")
    print(f"Узел: {NODE_NAME}")
    print(f"Целевая переменная: {TARGET_COLUMN}")
    print(f"Выходная папка: {OUTPUT_DIR}")

    df_train_full = read_prepared_dataset(TRAIN_PATH)
    df_test = read_prepared_dataset(TEST_PATH)

    features = select_real_signal_features(df_train_full)
    print(f"Количество признаков: {len(features)}")
    for feature in features:
        print(f"  - {feature}")

    inner_train, valid = split_train_valid(df_train_full)

    print("\nРазмеры до сэмплирования:")
    print(f"inner_train: {inner_train.shape}")
    print(f"valid:       {valid.shape}")
    print(f"test:        {df_test.shape}")

    train_search = sample_for_training(
        inner_train,
        TARGET_COLUMN,
        MAX_TRAIN_ROWS_FOR_SEARCH,
        NEGATIVE_TO_POSITIVE_RATIO,
        RANDOM_STATE,
    )
    valid_search = sample_for_validation(
        valid,
        TARGET_COLUMN,
        MAX_VALID_ROWS_FOR_SEARCH,
        RANDOM_STATE,
    )

    X_train_search, y_train_search = make_xy(train_search, features)
    X_valid_search, y_valid_search = make_xy(valid_search, features)

    print("\nРазмеры для подбора:")
    print(f"train_search: {X_train_search.shape}")
    print(f"valid_search: {X_valid_search.shape}")
    print("Баланс train_search:")
    print(json.dumps(class_balance(y_train_search), ensure_ascii=False, indent=2))
    print("Баланс valid_search:")
    print(json.dumps(class_balance(y_valid_search), ensure_ascii=False, indent=2))

    search_results, best_params = run_parameter_search(
        X_train_search,
        y_train_search,
        X_valid_search,
        y_valid_search,
    )

    search_results_path = OUTPUT_DIR / "hyperparameter_search_results.csv"
    search_results.to_csv(search_results_path, index=False, encoding="utf-8-sig")
    save_json(OUTPUT_DIR / "best_params.json", best_params)

    print("\nЛучшие параметры по validation Average Precision:")
    print(json.dumps(best_params, ensure_ascii=False, indent=2))

    final_train = sample_for_training(
        df_train_full,
        TARGET_COLUMN,
        MAX_FINAL_TRAIN_ROWS,
        NEGATIVE_TO_POSITIVE_RATIO,
        RANDOM_STATE + 1,
    )
    X_final_train, y_final_train = make_xy(final_train, features)
    X_valid_full, y_valid_full = make_xy(valid, features)
    X_test, y_test = make_xy(df_test, features)

    print("\nФинальное обучение на train < 2025-01-01:")
    print(f"final_train: {X_final_train.shape}")
    print("Баланс final_train:")
    print(json.dumps(class_balance(y_final_train), ensure_ascii=False, indent=2))

    final_model = build_model(best_params)
    final_model.fit(X_final_train, y_final_train)

    valid_prob = final_model.predict_proba(X_valid_full)[:, 1]
    valid_thresholds = threshold_table(y_valid_full, valid_prob)
    best_threshold_row = valid_thresholds.loc[valid_thresholds["f1"].idxmax()].to_dict()
    best_threshold = float(best_threshold_row["threshold"])

    test_prob = final_model.predict_proba(X_test)[:, 1]
    test_prob_metrics = evaluate_probabilities(y_test, test_prob)
    test_thresholds = threshold_table(y_test, test_prob)
    test_at_valid_threshold = test_thresholds.loc[
        (test_thresholds["threshold"] - best_threshold).abs().idxmin()
    ].to_dict()
    test_best_threshold_row = test_thresholds.loc[test_thresholds["f1"].idxmax()].to_dict()

    valid_thresholds.to_csv(
        OUTPUT_DIR / "validation_threshold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    test_thresholds.to_csv(
        OUTPUT_DIR / "test_threshold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics = {
        "node": NODE_NAME,
        "target": TARGET_COLUMN,
        "model": "HistGradientBoostingClassifier",
        "features": features,
        "n_features": len(features),
        "best_params": best_params,
        "split": {
            "inner_train": f"DT < {VALIDATION_START}",
            "validation": f"{VALIDATION_START} <= DT < {TEST_START}",
            "test": f"DT >= {TEST_START}",
        },
        "rows": {
            "train_full": int(len(df_train_full)),
            "inner_train": int(len(inner_train)),
            "validation": int(len(valid)),
            "test": int(len(df_test)),
            "train_search": int(len(train_search)),
            "valid_search": int(len(valid_search)),
            "final_train": int(len(final_train)),
        },
        "balance": {
            "train_search": class_balance(y_train_search),
            "valid_search": class_balance(y_valid_search),
            "final_train": class_balance(y_final_train),
            "validation_full": class_balance(y_valid_full),
            "test": class_balance(y_test),
        },
        "validation_probability_metrics": evaluate_probabilities(y_valid_full, valid_prob),
        "validation_best_threshold_by_f1": best_threshold_row,
        "test_probability_metrics": test_prob_metrics,
        "test_metrics_at_validation_best_threshold": test_at_valid_threshold,
        "test_best_threshold_by_f1": test_best_threshold_row,
    }
    save_json(OUTPUT_DIR / "tuned_model_metrics.json", metrics)

    importance_sample = sample_for_validation(df_test, TARGET_COLUMN, 100_000, RANDOM_STATE)
    X_importance, y_importance = make_xy(importance_sample, features)
    print("\nРасчет permutation importance на тестовой подвыборке...")
    perm = permutation_importance(
        final_model,
        X_importance,
        y_importance,
        scoring="average_precision",
        n_repeats=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    importance_df = pd.DataFrame({
        "feature": features,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False)
    importance_df.to_csv(
        OUTPUT_DIR / "feature_importance_permutation_ap.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nИтог на test:")
    print(json.dumps(metrics["test_probability_metrics"], ensure_ascii=False, indent=2))
    print("Метрики на test при пороге, выбранном по validation F1:")
    print(json.dumps(test_at_valid_threshold, ensure_ascii=False, indent=2))
    print(f"\nФайлы сохранены в: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
