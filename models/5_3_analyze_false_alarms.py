#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "models" else SCRIPT_DIR

PROCESSED_DIR = PROJECT_ROOT / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "error_analysis"

THRESHOLD_FILE_PATTERNS = [
    "**/*threshold_metrics*.csv",
    "**/*threshold*.csv",
]

REQUIRED_COLUMNS = {"threshold", "tp", "fp", "tn", "fn"}
NUMERIC_COLUMNS = [
    "threshold",
    "precision",
    "recall",
    "f1",
    "accuracy",
    "tp",
    "fp",
    "tn",
    "fn",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Анализ ложных срабатываний по threshold_metrics.csv."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Папка, где искать *_threshold_metrics.csv. По умолчанию: results.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Папка для сохранения результатов. По умолчанию: results/error_analysis.",
    )
    parser.add_argument(
        "--include-processed",
        action="store_true",
        help="Дополнительно искать threshold-файлы в processed.",
    )
    parser.add_argument(
        "--pattern",
        default=None,
        help="Свой glob-шаблон поиска, например '**/*threshold_metrics.csv'.",
    )
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def safe_div(numerator: float | int, denominator: float | int) -> float:
    if denominator == 0 or pd.isna(denominator):
        return np.nan
    return float(numerator) / float(denominator)


def read_csv_smart(path: Path) -> pd.DataFrame:
    """Читает CSV с запятой или точкой с запятой."""
    df = pd.read_csv(path)
    if len(df.columns) == 1:
        df = pd.read_csv(path, sep=";")
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {col: str(col).strip().lower() for col in df.columns}
    df = df.rename(columns=rename_map)
    return df


def infer_node_from_name(path: Path) -> str:
    name = path.name.lower()
    if "compactor" in name:
        return "compactor"
    if "mill" in name:
        return "mill"
    return "unknown"


def node_ru(node: str) -> str:
    return {
        "compactor": "Компактор",
        "mill": "Мельница",
        "unknown": "Не определено",
    }.get(node, node)


def infer_target_from_name(path: Path) -> str:
    name = path.name.lower()
    match = re.search(r"event_in_([0-9]+h)", name)
    if match:
        return f"event_in_{match.group(1)}"
    match = re.search(r"event_in_([0-9]+)", name)
    if match:
        return f"event_in_{match.group(1)}h"
    return "unknown"


def infer_experiment_name(path: Path, input_root: Path) -> str:
    try:
        rel = path.relative_to(input_root)
        if len(rel.parts) >= 2:
            return rel.parts[0]
        return path.parent.name
    except ValueError:
        return path.parent.name


def find_threshold_files(input_dirs: Iterable[Path], pattern: str | None = None) -> list[Path]:
    found: list[Path] = []
    patterns = [pattern] if pattern else THRESHOLD_FILE_PATTERNS

    for input_dir in input_dirs:
        if not input_dir.exists():
            continue
        for pat in patterns:
            for path in input_dir.glob(pat):
                if path.is_file() and path.suffix.lower() == ".csv":
                    if "error_analysis" in path.parts:
                        continue
                    found.append(path)

    # Убираем дубли, сохраняя порядок.
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in sorted(found):
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def validate_threshold_df(df: pd.DataFrame, path: Path) -> bool:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        print(f"Пропуск файла {path}: нет обязательных колонок {sorted(missing)}")
        return False
    return True


def enrich_threshold_metrics(df: pd.DataFrame, path: Path, input_root: Path) -> pd.DataFrame:
    df = normalize_columns(df.copy())

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    tp = df["tp"]
    fp = df["fp"]
    tn = df["tn"]
    fn = df["fn"]

    total = tp + fp + tn + fn
    actual_positive = tp + fn
    actual_negative = tn + fp
    predicted_positive = tp + fp
    predicted_negative = tn + fn

    df["total_rows"] = total
    df["actual_positive"] = actual_positive
    df["actual_negative"] = actual_negative
    df["predicted_positive"] = predicted_positive
    df["predicted_negative"] = predicted_negative

    # Метрики по отрицательному классу и ошибкам.
    df["specificity"] = [safe_div(tn_i, tn_i + fp_i) for tn_i, fp_i in zip(tn, fp)]
    df["false_positive_rate"] = [safe_div(fp_i, fp_i + tn_i) for fp_i, tn_i in zip(fp, tn)]
    df["false_negative_rate"] = [safe_div(fn_i, fn_i + tp_i) for fn_i, tp_i in zip(fn, tp)]
    df["false_discovery_rate"] = [safe_div(fp_i, fp_i + tp_i) for fp_i, tp_i in zip(fp, tp)]
    df["negative_predictive_value"] = [safe_div(tn_i, tn_i + fn_i) for tn_i, fn_i in zip(tn, fn)]
    df["balanced_accuracy"] = (df.get("recall", np.nan) + df["specificity"]) / 2

    # Удобные прикладные показатели.
    df["false_alarms_share_all_rows"] = [safe_div(fp_i, total_i) for fp_i, total_i in zip(fp, total)]
    df["missed_events_share_all_rows"] = [safe_div(fn_i, total_i) for fn_i, total_i in zip(fn, total)]
    df["false_alarms_per_true_alarm"] = [safe_div(fp_i, tp_i) for fp_i, tp_i in zip(fp, tp)]
    df["missed_events_per_detected_event"] = [safe_div(fn_i, tp_i) for fn_i, tp_i in zip(fn, tp)]

    node = infer_node_from_name(path)
    df.insert(0, "experiment", infer_experiment_name(path, input_root))
    df.insert(1, "node", node)
    df.insert(2, "node_ru", node_ru(node))
    df.insert(3, "target", infer_target_from_name(path))
    df.insert(4, "source_file", str(path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path))

    return df


def build_summary(full_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["experiment", "node", "node_ru", "target", "source_file"]

    for keys, part in full_df.groupby(group_cols, dropna=False):
        part = part.copy()
        part["f1_for_sort"] = pd.to_numeric(part.get("f1"), errors="coerce")

        if part["f1_for_sort"].notna().any():
            selected = part.loc[part["f1_for_sort"].idxmax()]
            selection_rule = "max_f1"
        else:
            selected = part.iloc[0]
            selection_rule = "first_row"

        row = dict(zip(group_cols, keys))
        row.update(
            {
                "selection_rule": selection_rule,
                "selected_threshold": selected.get("threshold"),
                "precision": selected.get("precision", np.nan),
                "recall": selected.get("recall", np.nan),
                "f1": selected.get("f1", np.nan),
                "accuracy": selected.get("accuracy", np.nan),
                "specificity": selected.get("specificity", np.nan),
                "false_positive_rate": selected.get("false_positive_rate", np.nan),
                "false_negative_rate": selected.get("false_negative_rate", np.nan),
                "false_discovery_rate": selected.get("false_discovery_rate", np.nan),
                "negative_predictive_value": selected.get("negative_predictive_value", np.nan),
                "balanced_accuracy": selected.get("balanced_accuracy", np.nan),
                "tp": selected.get("tp", np.nan),
                "fp": selected.get("fp", np.nan),
                "tn": selected.get("tn", np.nan),
                "fn": selected.get("fn", np.nan),
                "total_rows": selected.get("total_rows", np.nan),
                "predicted_positive": selected.get("predicted_positive", np.nan),
                "actual_positive": selected.get("actual_positive", np.nan),
                "false_alarms_share_all_rows": selected.get("false_alarms_share_all_rows", np.nan),
                "missed_events_share_all_rows": selected.get("missed_events_share_all_rows", np.nan),
                "false_alarms_per_true_alarm": selected.get("false_alarms_per_true_alarm", np.nan),
                "missed_events_per_detected_event": selected.get("missed_events_per_detected_event", np.nan),
            }
        )
        rows.append(row)

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(["experiment", "node", "target", "source_file"]).reset_index(drop=True)
    return summary


def add_metric_descriptions(output_dir: Path) -> None:
    text = """Описание дополнительных метрик анализа ложных срабатываний
================================================================

TP (true positive): модель выдала предупреждение, и событие действительно было.
FP (false positive): модель выдала предупреждение, но события не было. Это ложное срабатывание.
TN (true negative): модель не выдала предупреждение, и события действительно не было.
FN (false negative): модель не выдала предупреждение, но событие было. Это пропуск события.

specificity = TN / (TN + FP)
Специфичность. Доля обычных состояний, которые модель правильно оставила обычными.

false_positive_rate = FP / (FP + TN)
Доля ложных срабатываний среди всех обычных состояний. Чем ниже, тем меньше ложных тревог.

false_negative_rate = FN / (FN + TP)
Доля пропущенных предсобытийных состояний среди всех реальных предсобытийных состояний.

false_discovery_rate = FP / (FP + TP)
Доля ложных тревог среди всех предупреждений модели. Это удобно для ответа на вопрос: если модель сработала, как часто она ошибается?

negative_predictive_value = TN / (TN + FN)
Достоверность отрицательного прогноза. Показывает, насколько можно доверять ответу модели «события не будет».

balanced_accuracy = (recall + specificity) / 2
Сбалансированная точность. Учитывает качество как по положительному, так и по отрицательному классу.
"""
    (output_dir / "metric_definitions.txt").write_text(text, encoding="utf-8")


def save_outputs(full_df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    by_threshold_csv = output_dir / "false_alarm_analysis_by_threshold.csv"
    summary_csv = output_dir / "false_alarm_summary_by_experiment.csv"
    xlsx_path = output_dir / "false_alarm_analysis.xlsx"

    full_df.to_csv(by_threshold_csv, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        full_df.to_excel(writer, sheet_name="by_threshold", index=False)
        summary_df.to_excel(writer, sheet_name="summary", index=False)

    add_metric_descriptions(output_dir)

    print()
    print("Сохранены файлы:")
    print(f"  {by_threshold_csv}")
    print(f"  {summary_csv}")
    print(f"  {xlsx_path}")
    print(f"  {output_dir / 'metric_definitions.txt'}")


def print_short_report(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        return

    print()
    print("Краткая сводка по лучшему F1-порогу в каждом файле:")
    print("-" * 90)

    show_cols = [
        "experiment",
        "node_ru",
        "target",
        "selected_threshold",
        "precision",
        "recall",
        "f1",
        "specificity",
        "false_positive_rate",
        "false_discovery_rate",
        "negative_predictive_value",
        "tp",
        "fp",
        "tn",
        "fn",
    ]

    existing_cols = [col for col in show_cols if col in summary_df.columns]
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(summary_df[existing_cols].to_string(index=False))


def main() -> None:
    args = parse_args()

    input_dir = resolve_project_path(args.input_dir)
    output_dir = resolve_project_path(args.output_dir)

    input_dirs = [input_dir]
    if args.include_processed:
        input_dirs.append(PROCESSED_DIR)

    print("Анализ ложных срабатываний классификационных моделей")
    print(f"Корень проекта: {PROJECT_ROOT}")
    print("Папки поиска:")
    for directory in input_dirs:
        print(f"  - {directory}")
    print(f"Папка результатов: {output_dir}")

    files = find_threshold_files(input_dirs, args.pattern)

    if not files and input_dir == RESULTS_DIR and not args.include_processed:
        print()
        print("В results threshold-файлы не найдены. Пробую fallback-поиск в processed...")
        files = find_threshold_files([PROCESSED_DIR], args.pattern)

    if not files:
        raise FileNotFoundError(
            "Не найдены CSV-файлы с пороговыми метриками. "
            "Ожидались файлы вида *_threshold_metrics.csv в results/ или processed/."
        )

    print()
    print(f"Найдено threshold-файлов: {len(files)}")
    for path in files:
        print(f"  - {path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path}")

    enriched_parts: list[pd.DataFrame] = []
    for path in files:
        df = read_csv_smart(path)
        df = normalize_columns(df)
        if not validate_threshold_df(df, path):
            continue
        input_root_for_meta = input_dir if path.resolve().is_relative_to(input_dir.resolve()) else path.parent.parent
        enriched_parts.append(enrich_threshold_metrics(df, path, input_root_for_meta))

    if not enriched_parts:
        raise RuntimeError("Не осталось подходящих threshold-файлов после проверки колонок.")

    full_df = pd.concat(enriched_parts, ignore_index=True)
    full_df = full_df.sort_values(["experiment", "node", "target", "source_file", "threshold"]).reset_index(drop=True)
    summary_df = build_summary(full_df)

    save_outputs(full_df, summary_df, output_dir)
    print_short_report(summary_df)

    print()
    print("Готово.")


if __name__ == "__main__":
    main()
