#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    mean_squared_error,
    r2_score,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "models" else SCRIPT_DIR

PROCESSED_DIR = PROJECT_ROOT / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "rul_regression_calendar"

REGRESSION_TARGET = "time_to_next_event_hours"
DEFAULT_HORIZONS = [72.0, 168.0, 336.0]
RANDOM_STATE = 42

# Ограничение обучающей выборки, если слишком долго.
DEFAULT_MAX_TRAIN_ROWS = 600_000

# None означает полный test. Если будет медленно, можно указать, например, 300000
DEFAULT_MAX_TEST_ROWS: int | None = None

NODES = {
    "compactor": {
        "name_ru": "Компактор",
        "train_path": PROCESSED_DIR / "compactor_train_prepared.parquet",
        "test_path": PROCESSED_DIR / "compactor_test_prepared.parquet",
    },
    "mill": {
        "name_ru": "Мельница",
        "train_path": PROCESSED_DIR / "mill_train_prepared.parquet",
        "test_path": PROCESSED_DIR / "mill_test_prepared.parquet",
    },
}

TARGET_COLUMNS = {
    "event_in_24h",
    "event_in_48h",
    "event_in_72h",
    "pre_event_window",
    REGRESSION_TARGET,
}

EXCLUDE_SERVICE_FEATURES = {
    "DT",
    "split",
    "N_Hosokawa",
    "Regim",
    "dt_hour",
    "dt_dayofweek",
    "dt_month",
    "dt_is_weekend",
    "hours_from_unit_start",
    "hours_from_prev_row",
}

EXCLUDE_FEATURE_SUFFIXES = (
    "_was_missing",
)


def log(message: str = "") -> None:
    print(message, flush=True)


def parse_horizons(value: str) -> list[float]:
    parts = [part.strip().replace(",", ".") for part in value.split(";")]
    if len(parts) == 1 and "," in value and ";" not in value:
        parts = [part.strip() for part in value.split(",")]

    horizons: list[float] = []
    for part in parts:
        if not part:
            continue
        try:
            horizon = float(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Некорректный горизонт {part!r}. Пример: 72,168,336"
            ) from exc
        if horizon <= 0:
            raise argparse.ArgumentTypeError("Горизонт должен быть положительным числом часов.")
        horizons.append(horizon)

    if not horizons:
        raise argparse.ArgumentTypeError("Не указан ни один горизонт.")

    return horizons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Обучение RUL-регрессора по календарному времени до отказа."
    )
    parser.add_argument(
        "--horizons",
        type=parse_horizons,
        default=DEFAULT_HORIZONS,
        help="Горизонты прогноза в часах. Пример: 72,168,336",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Папка для результатов. По умолчанию: results/rul_regression_calendar",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=DEFAULT_MAX_TRAIN_ROWS,
        help=f"Максимальный размер обучающей подвыборки. По умолчанию: {DEFAULT_MAX_TRAIN_ROWS}",
    )
    parser.add_argument(
        "--max-test-rows",
        type=int,
        default=DEFAULT_MAX_TEST_ROWS,
        help="Максимальный размер test-подвыборки. По умолчанию используется весь test.",
    )
    parser.add_argument(
        "--prediction-sample-rows",
        type=int,
        default=50_000,
        help="Сколько строк с прогнозами сохранять в sample CSV для каждого узла/горизонта.",
    )
    parser.add_argument(
        "--include-service-features",
        action="store_true",
        help="Не исключать служебные/календарные признаки. По умолчанию они исключаются.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help="Фиксатор случайности.",
    )
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Файл не найден: {path}\n"
            "Сначала выполните этапы 4_1 и 4_3 для нужного эксперимента."
        )
    return pd.read_parquet(path)


def get_feature_columns(
    df: pd.DataFrame,
    include_service_features: bool = False,
) -> list[str]:
    excluded = set(TARGET_COLUMNS)

    feature_cols: list[str] = []
    for col in df.columns:
        if col in excluded:
            continue
        if not include_service_features and col in EXCLUDE_SERVICE_FEATURES:
            continue
        if not include_service_features and col.endswith(EXCLUDE_FEATURE_SUFFIXES):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)

    return feature_cols


def downcast_features(df: pd.DataFrame, feature_cols: Iterable[str]) -> pd.DataFrame:
    for col in feature_cols:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="float")
        elif pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def filter_by_horizon(df: pd.DataFrame, horizon_hours: float) -> pd.DataFrame:
    if REGRESSION_TARGET not in df.columns:
        raise ValueError(f"В датасете отсутствует целевая переменная {REGRESSION_TARGET!r}.")

    target = pd.to_numeric(df[REGRESSION_TARGET], errors="coerce")
    mask = target.gt(0) & target.le(horizon_hours) & np.isfinite(target)
    return df.loc[mask].copy()


def make_train_sample(
    df: pd.DataFrame,
    max_rows: int,
    random_state: int,
) -> pd.DataFrame:
    if max_rows is None or max_rows <= 0 or len(df) <= max_rows:
        return df

    target = df[REGRESSION_TARGET]
    n_bins = min(10, max(2, int(math.sqrt(len(df) / 10_000))))

    try:
        bins = pd.qcut(target, q=n_bins, duplicates="drop")
        sampled = (
            df.assign(_rul_bin=bins)
            .groupby("_rul_bin", group_keys=False, observed=True)
            .apply(
                lambda part: part.sample(
                    n=max(1, int(round(max_rows * len(part) / len(df)))),
                    random_state=random_state,
                )
                if len(part) > max(1, int(round(max_rows * len(part) / len(df))))
                else part
            )
            .drop(columns=["_rul_bin"])
        )

        if len(sampled) > max_rows:
            sampled = sampled.sample(n=max_rows, random_state=random_state)
        return sampled
    except Exception:
        return df.sample(n=max_rows, random_state=random_state)


def make_test_sample(
    df: pd.DataFrame,
    max_rows: int | None,
    random_state: int,
) -> pd.DataFrame:
    if max_rows is None or max_rows <= 0 or len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=random_state)


def train_model(random_state: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=0.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=random_state,
    )


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae_hours": float(mean_absolute_error(y_true, y_pred)),
        "median_ae_hours": float(median_absolute_error(y_true, y_pred)),
        "rmse_hours": rmse,
        "r2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan"),
    }


def baseline_median_metrics(y_train: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    median_value = float(np.median(y_train))
    pred = np.full_like(y_test, fill_value=median_value, dtype="float64")
    metrics = calculate_metrics(y_test, pred)
    return {
        "baseline_median_prediction_hours": median_value,
        "baseline_mae_hours": metrics["mae_hours"],
        "baseline_median_ae_hours": metrics["median_ae_hours"],
        "baseline_rmse_hours": metrics["rmse_hours"],
        "baseline_r2": metrics["r2"],
    }


def format_horizon(horizon: float) -> str:
    if float(horizon).is_integer():
        return str(int(horizon))
    return str(horizon).replace(".", "_")


def process_node_horizon(
    node_key: str,
    node_cfg: dict[str, object],
    horizon: float,
    output_dir: Path,
    max_train_rows: int,
    max_test_rows: int | None,
    prediction_sample_rows: int,
    include_service_features: bool,
    random_state: int,
) -> dict[str, object]:
    node_name_ru = str(node_cfg["name_ru"])
    train_path = Path(node_cfg["train_path"])
    test_path = Path(node_cfg["test_path"])

    log("=" * 80)
    log(f"Узел: {node_name_ru} ({node_key}), горизонт: {horizon:g} ч")
    log(f"Train: {train_path}")
    log(f"Test:  {test_path}")

    train_df = read_parquet(train_path)
    test_df = read_parquet(test_path)

    train_df = filter_by_horizon(train_df, horizon)
    test_df = filter_by_horizon(test_df, horizon)
    train_rows_before_sampling = len(train_df)
    test_rows_before_sampling = len(test_df)

    log(f"Train после фильтра 0 < RUL <= {horizon:g} ч: {train_df.shape}")
    log(f"Test после фильтра 0 < RUL <= {horizon:g} ч:  {test_df.shape}")

    if len(train_df) < 100:
        raise ValueError(
            f"{node_key}, {horizon:g} ч: слишком мало строк train для RUL-регрессии: {len(train_df)}"
        )
    if len(test_df) < 20:
        raise ValueError(
            f"{node_key}, {horizon:g} ч: слишком мало строк test для оценки: {len(test_df)}"
        )

    feature_cols = get_feature_columns(train_df, include_service_features=include_service_features)
    if not feature_cols:
        raise ValueError(f"{node_key}: не найдено числовых признаков для обучения.")

    feature_cols = [col for col in feature_cols if col in test_df.columns]
    if not feature_cols:
        raise ValueError(f"{node_key}: после пересечения train/test не осталось признаков.")

    train_df = make_train_sample(train_df, max_train_rows, random_state)
    test_df = make_test_sample(test_df, max_test_rows, random_state)

    train_df = downcast_features(train_df, feature_cols)
    test_df = downcast_features(test_df, feature_cols)

    X_train = train_df[feature_cols]
    y_train = train_df[REGRESSION_TARGET].astype("float64").to_numpy()
    X_test = test_df[feature_cols]
    y_test = test_df[REGRESSION_TARGET].astype("float64").to_numpy()

    log(f"Используемых признаков: {len(feature_cols)}")
    log(f"Train для обучения: {X_train.shape}")
    log(f"Test для оценки:   {X_test.shape}")
    log(
        "RUL train, ч: "
        f"min={np.min(y_train):.3f}, median={np.median(y_train):.3f}, "
        f"mean={np.mean(y_train):.3f}, max={np.max(y_train):.3f}"
    )
    log(
        "RUL test, ч:  "
        f"min={np.min(y_test):.3f}, median={np.median(y_test):.3f}, "
        f"mean={np.mean(y_test):.3f}, max={np.max(y_test):.3f}"
    )

    model = train_model(random_state)
    log("Обучение HistGradientBoostingRegressor...")
    model.fit(X_train, y_train)

    pred_raw = model.predict(X_test)
    pred = np.clip(pred_raw, 0.0, horizon)

    model_metrics = calculate_metrics(y_test, pred)
    base_metrics = baseline_median_metrics(y_train, y_test)

    mae_improvement = None
    if base_metrics["baseline_mae_hours"] > 0:
        mae_improvement = 1.0 - model_metrics["mae_hours"] / base_metrics["baseline_mae_hours"]

    result: dict[str, object] = {
        "node": node_key,
        "node_ru": node_name_ru,
        "horizon_hours": float(horizon),
        "train_rows_before_sampling": int(train_rows_before_sampling),
        "test_rows_before_sampling": int(test_rows_before_sampling),
        "train_rows_used": int(len(X_train)),
        "test_rows_used": int(len(X_test)),
        "n_features": int(len(feature_cols)),
        "target": REGRESSION_TARGET,
        "model": "HistGradientBoostingRegressor",
        "prediction_clipped_to_horizon": True,
        **model_metrics,
        **base_metrics,
        "mae_improvement_vs_median_baseline": float(mae_improvement) if mae_improvement is not None else None,
        "features": feature_cols,
    }

    sample_size = min(prediction_sample_rows, len(test_df))
    pred_df = pd.DataFrame(
        {
            "DT": test_df["DT"].values if "DT" in test_df.columns else np.arange(len(test_df)),
            "N_Hosokawa": test_df["N_Hosokawa"].values if "N_Hosokawa" in test_df.columns else np.nan,
            "y_true_hours": y_test,
            "y_pred_hours": pred,
            "abs_error_hours": np.abs(y_test - pred),
            "horizon_hours": horizon,
            "node": node_key,
        }
    )
    if sample_size < len(pred_df):
        pred_df = pred_df.sample(n=sample_size, random_state=random_state).sort_values(
            ["N_Hosokawa", "DT"], kind="stable"
        )

    pred_path = output_dir / f"{node_key}_rul_{format_horizon(horizon)}h_predictions_sample.csv"
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    log(
        f"Итог: MAE={model_metrics['mae_hours']:.3f} ч, "
        f"MedianAE={model_metrics['median_ae_hours']:.3f} ч, "
        f"RMSE={model_metrics['rmse_hours']:.3f} ч, "
        f"R²={model_metrics['r2']:.4f}"
    )
    log(f"Сэмпл прогнозов: {pred_path}")

    return result


def make_summary_text(results: list[dict[str, object]]) -> str:
    lines: list[str] = []
    lines.append("RUL-регрессия по календарному времени до ближайшего события")
    lines.append("=" * 72)
    lines.append("")
    lines.append(
        "Цель: предсказать time_to_next_event_hours для строк, где "
        "0 < time_to_next_event_hours <= выбранный горизонт."
    )
    lines.append("Модель: HistGradientBoostingRegressor.")
    lines.append("")

    for horizon in sorted({float(item["horizon_hours"]) for item in results}):
        lines.append(f"Горизонт {horizon:g} часов")
        lines.append("-" * 72)
        for item in [r for r in results if float(r["horizon_hours"]) == horizon]:
            lines.append(
                f"{item['node_ru']}: MAE = {float(item['mae_hours']):.3f} ч, "
                f"MedianAE = {float(item['median_ae_hours']):.3f} ч, "
                f"RMSE = {float(item['rmse_hours']):.3f} ч, "
                f"R² = {float(item['r2']):.4f}, "
                f"test = {int(item['test_rows_used'])} строк."
            )
        lines.append("")

    lines.append("Интерпретация:")
    lines.append(
        "MAE показывает, на сколько календарных часов в среднем ошибается модель "
        "при оценке остаточного времени до события внутри выбранного горизонта."
    )
    lines.append(
        "MedianAE показывает типичную ошибку без сильного влияния редких крупных промахов."
    )
    lines.append(
        "RMSE сильнее штрафует крупные ошибки, а R² показывает, насколько модель "
        "лучше объясняет разброс RUL по сравнению с простым прогнозом средним значением."
    )

    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log("Этап 5_2: RUL-регрессия по календарному времени")
    log(f"Корень проекта: {PROJECT_ROOT}")
    log(f"Папка processed: {PROCESSED_DIR}")
    log(f"Выходная папка: {output_dir}")
    log(f"Горизонты: {', '.join(f'{h:g}' for h in args.horizons)} ч")
    log(
        "Признаки: "
        + ("все числовые, включая служебные" if args.include_service_features else "реальные технологические и производные")
    )
    log("")

    results: list[dict[str, object]] = []

    for horizon in args.horizons:
        for node_key, node_cfg in NODES.items():
            result = process_node_horizon(
                node_key=node_key,
                node_cfg=node_cfg,
                horizon=float(horizon),
                output_dir=output_dir,
                max_train_rows=args.max_train_rows,
                max_test_rows=args.max_test_rows,
                prediction_sample_rows=args.prediction_sample_rows,
                include_service_features=args.include_service_features,
                random_state=args.random_state,
            )
            results.append(result)

    metrics_rows = []
    for item in results:
        row = {k: v for k, v in item.items() if k != "features"}
        metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = output_dir / "rul_regression_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    json_path = output_dir / "rul_regression_metrics.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    summary_text = make_summary_text(results)
    summary_path = output_dir / "rul_regression_summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")

    log("")
    log("=" * 80)
    log("Готово")
    log(f"Метрики CSV:  {metrics_path}")
    log(f"Метрики JSON: {json_path}")
    log(f"Краткий отчет: {summary_path}")
    log("")
    log(summary_text)


if __name__ == "__main__":
    main()
