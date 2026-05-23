#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
5_1_analyze_failure_intervals.py

Анализ интервалов между отказами/техническими событиями Hosokawa.

Скрипт читает файл:
    toir_hosokawa/jtiny_hosokawa_events_by_node.xlsx

Берёт листы:
    - mill
    - compactor

Фильтрует события:
    event_class == "Неисправность/отказ"

Для каждой пары:
    узел + N_Hosokawa

сортирует события по времени и считает:
    - интервал от предыдущего события;
    - интервал до следующего события;
    - сводные статистики по интервалам.

Выходные файлы:
    results/failure_intervals/failure_intervals_by_node.csv
    results/failure_intervals/failure_intervals_summary.csv
    results/failure_intervals/failure_intervals.xlsx

Запуск из корня проекта:
    python models/5_1_analyze_failure_intervals.py
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Iterable

import pandas as pd

warnings.filterwarnings("ignore", message="Could not infer format")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "models" else SCRIPT_DIR

TOIR_DIR = PROJECT_ROOT / "toir_hosokawa"
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_INPUT = TOIR_DIR / "jtiny_hosokawa_events_by_node.xlsx"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "failure_intervals"

DEFAULT_SHEETS = ["mill", "compactor"]
DEFAULT_EVENT_CLASS = "Неисправность/отказ"
DEFAULT_DATE_COLUMN = "Дата/время начала простоя"

DETAIL_COLUMNS_CANDIDATES = [
    "source_sheet",
    "node",
    "node_assignment_type",
    "№ обр",
    "N_Hosokawa",
    "event_class",
    "event_time",
    "previous_event_time",
    "next_event_time",
    "interval_since_previous_hours",
    "interval_since_previous_days",
    "time_to_next_event_hours",
    "time_to_next_event_days",
    "interval_category_since_previous",
    "event_order_in_unit_node",
    "Дата/время начала простоя",
    "Дата/время окончания простоя",
    "Дата/время начала ремонта",
    "Дата/время окончания ремонта",
    "Вид меропр",
    "Наименование оборудования",
    "Краткое описание отказа (причины простоя)",
    "Краткое описание выполненных работ",
    "Краткое описание  выполненных работ",
    "Комментарий эксперта",
    "Примечание",
]


def log(message: str = "") -> None:
    print(message, flush=True)


def normalize_col_name(value: object) -> str:
    text = str(value).strip().lower().replace("ё", "е")
    text = text.replace("\u00a0", " ")
    return "".join(ch for ch in text if ch.isalnum())


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def find_column(df: pd.DataFrame, candidates: Iterable[str], required: bool = True) -> str | None:
    normalized_map = {normalize_col_name(col): col for col in df.columns}
    for candidate in candidates:
        key = normalize_col_name(candidate)
        if key in normalized_map:
            return normalized_map[key]
    if required:
        raise ValueError(f"Не найден обязательный столбец. Проверялись варианты: {list(candidates)}")
    return None


def find_event_time_column(df: pd.DataFrame, preferred: str) -> str:
    if preferred in df.columns:
        return preferred

    candidates = [
        preferred,
        "Дата/время начала простоя",
        "Дата время начала простоя",
        "Дата/время начала",
        "Дата начала простоя",
        "Дата начала",
        "Начало простоя",
        "Дата события",
        "Дата",
    ]
    found = find_column(df, candidates, required=False)
    if found is not None:
        return found

    best_col = None
    best_score = -1.0
    for col in df.columns:
        norm = normalize_col_name(col)
        if not any(token in norm for token in ["дата", "время", "начало", "прост"]):
            continue
        if any(token in norm for token in ["оконч", "конец", "заверш"]):
            continue
        converted = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
        valid_share = converted.notna().mean()
        if valid_share <= 0:
            continue
        score = valid_share
        if "начал" in norm:
            score += 1.0
        if "прост" in norm:
            score += 1.0
        if score > best_score:
            best_col = col
            best_score = score

    if best_col is None:
        raise ValueError("Не удалось определить колонку времени события.")
    return best_col


def normalize_node_name(sheet_name: str) -> str:
    sheet = sheet_name.strip().lower()
    if sheet in {"mill", "melnica", "мельница"}:
        return "mill"
    if sheet in {"compactor", "компактор", "shnek", "шнек"}:
        return "compactor"
    return sheet_name


def interval_category(days: float | None) -> str:
    if days is None or pd.isna(days):
        return "first_event"
    if days < 0:
        return "bad_negative_interval"
    if days == 0:
        return "same_timestamp"
    if days <= 0.5:
        return "within_12h"
    if days <= 1:
        return "within_1d"
    if days <= 7:
        return "1_7d"
    if days <= 30:
        return "1_4w"
    if days <= 90:
        return "1_3m"
    if days <= 365:
        return "3_12m"
    return "more_1y"


def safe_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def describe_intervals(values: pd.Series) -> dict[str, float | int | None]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return {
            "interval_count": 0,
            "interval_min_days": None,
            "interval_q25_days": None,
            "interval_median_days": None,
            "interval_mean_days": None,
            "interval_q75_days": None,
            "interval_max_days": None,
            "interval_std_days": None,
        }
    return {
        "interval_count": int(len(values)),
        "interval_min_days": safe_float(values.min()),
        "interval_q25_days": safe_float(values.quantile(0.25)),
        "interval_median_days": safe_float(values.median()),
        "interval_mean_days": safe_float(values.mean()),
        "interval_q75_days": safe_float(values.quantile(0.75)),
        "interval_max_days": safe_float(values.max()),
        "interval_std_days": safe_float(values.std(ddof=1)) if len(values) > 1 else 0.0,
    }


def autosize_excel_columns(writer: pd.ExcelWriter) -> None:
    for ws in writer.book.worksheets:
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for column_cells in ws.columns:
            col_letter = column_cells[0].column_letter
            max_len = 0
            for cell in column_cells[:300]:
                if cell.value is not None:
                    max_len = max(max_len, len(str(cell.value)))
            ws.column_dimensions[col_letter].width = min(max(max_len + 2, 10), 55)


def load_failure_events(input_path: Path, sheets: list[str], event_class: str, date_column: str) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Не найден входной Excel-файл: {input_path}")

    xls = pd.ExcelFile(input_path)
    missing_sheets = [sheet for sheet in sheets if sheet not in xls.sheet_names]
    if missing_sheets:
        raise ValueError(
            f"Во входном файле нет листов: {missing_sheets}. "
            f"Доступные листы: {xls.sheet_names}"
        )

    parts: list[pd.DataFrame] = []
    for sheet in sheets:
        df = pd.read_excel(input_path, sheet_name=sheet)
        df = df.copy()
        df["source_sheet"] = sheet
        df["node"] = normalize_node_name(sheet)

        event_class_col = find_column(df, ["event_class"], required=True)
        unit_col = find_column(df, ["N_Hosokawa"], required=True)
        time_col = find_event_time_column(df, date_column)

        df[event_class_col] = df[event_class_col].astype(str).str.strip()
        df = df[df[event_class_col] == event_class].copy()

        df["event_class"] = df[event_class_col]
        df["N_Hosokawa"] = pd.to_numeric(df[unit_col], errors="coerce")
        df["event_time"] = pd.to_datetime(df[time_col], errors="coerce", dayfirst=True)

        before = len(df)
        df = df.dropna(subset=["N_Hosokawa", "event_time"]).copy()
        dropped = before - len(df)
        df["N_Hosokawa"] = df["N_Hosokawa"].astype("int64")

        duplicate_subset = ["node", "N_Hosokawa", "event_time"]
        event_id_col = find_column(df, ["№ обр", "номер обр", "номер обращения"], required=False)
        if event_id_col is not None:
            duplicate_subset.append(event_id_col)

        before_dedup = len(df)
        df = df.drop_duplicates(subset=duplicate_subset, keep="first").copy()
        duplicates_removed = before_dedup - len(df)

        log(
            f"Лист {sheet}: событий класса {event_class!r}: {len(df)} "
            f"(удалено без времени/установки: {dropped}, точных дублей: {duplicates_removed})"
        )
        parts.append(df)

    if not parts:
        return pd.DataFrame()
    events = pd.concat(parts, ignore_index=True)
    return events.sort_values(["node", "N_Hosokawa", "event_time"], kind="mergesort").reset_index(drop=True)


def add_interval_columns(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events
    df = events.copy()
    group_cols = ["node", "N_Hosokawa"]
    df = df.sort_values([*group_cols, "event_time"], kind="mergesort").reset_index(drop=True)
    grouped = df.groupby(group_cols, observed=True, sort=False)

    df["event_order_in_unit_node"] = grouped.cumcount() + 1
    df["previous_event_time"] = grouped["event_time"].shift(1)
    df["next_event_time"] = grouped["event_time"].shift(-1)

    since_previous = df["event_time"] - df["previous_event_time"]
    to_next = df["next_event_time"] - df["event_time"]

    df["interval_since_previous_hours"] = since_previous.dt.total_seconds() / 3600.0
    df["interval_since_previous_days"] = df["interval_since_previous_hours"] / 24.0
    df["time_to_next_event_hours"] = to_next.dt.total_seconds() / 3600.0
    df["time_to_next_event_days"] = df["time_to_next_event_hours"] / 24.0
    df["interval_category_since_previous"] = df["interval_since_previous_days"].map(interval_category)
    return df


def make_summary(intervals: pd.DataFrame) -> pd.DataFrame:
    if intervals.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []

    for (node, unit), group in intervals.groupby(["node", "N_Hosokawa"], observed=True, sort=True):
        first_event = group["event_time"].min()
        last_event = group["event_time"].max()
        span_days = (last_event - first_event).total_seconds() / 86400 if pd.notna(first_event) and pd.notna(last_event) else None
        row: dict[str, object] = {
            "summary_level": "node_unit",
            "node": node,
            "N_Hosokawa": unit,
            "event_count": int(len(group)),
            "first_event_time": first_event,
            "last_event_time": last_event,
            "observation_span_days": span_days,
            "events_per_year_by_span": float(len(group) / (span_days / 365.25)) if span_days is not None and span_days > 0 else None,
        }
        row.update(describe_intervals(group["interval_since_previous_days"]))
        rows.append(row)

    for node, group in intervals.groupby("node", observed=True, sort=True):
        first_event = group["event_time"].min()
        last_event = group["event_time"].max()
        span_days = (last_event - first_event).total_seconds() / 86400 if pd.notna(first_event) and pd.notna(last_event) else None
        row = {
            "summary_level": "node_total",
            "node": node,
            "N_Hosokawa": "all",
            "event_count": int(len(group)),
            "first_event_time": first_event,
            "last_event_time": last_event,
            "observation_span_days": span_days,
            "events_per_year_by_span": float(len(group) / (span_days / 365.25)) if span_days is not None and span_days > 0 else None,
        }
        row.update(describe_intervals(group["interval_since_previous_days"]))
        rows.append(row)

    first_event = intervals["event_time"].min()
    last_event = intervals["event_time"].max()
    span_days = (last_event - first_event).total_seconds() / 86400 if pd.notna(first_event) and pd.notna(last_event) else None
    row = {
        "summary_level": "overall",
        "node": "all",
        "N_Hosokawa": "all",
        "event_count": int(len(intervals)),
        "first_event_time": first_event,
        "last_event_time": last_event,
        "observation_span_days": span_days,
        "events_per_year_by_span": float(len(intervals) / (span_days / 365.25)) if span_days is not None and span_days > 0 else None,
    }
    row.update(describe_intervals(intervals["interval_since_previous_days"]))
    rows.append(row)

    summary = pd.DataFrame(rows)
    for col in [
        "interval_min_days",
        "interval_q25_days",
        "interval_median_days",
        "interval_mean_days",
        "interval_q75_days",
        "interval_max_days",
        "interval_std_days",
    ]:
        if col in summary.columns:
            summary[col.replace("_days", "_hours")] = pd.to_numeric(summary[col], errors="coerce") * 24.0

    preferred_order = [
        "summary_level", "node", "N_Hosokawa", "event_count", "interval_count",
        "first_event_time", "last_event_time", "observation_span_days", "events_per_year_by_span",
        "interval_min_hours", "interval_q25_hours", "interval_median_hours", "interval_mean_hours",
        "interval_q75_hours", "interval_max_hours", "interval_std_hours",
        "interval_min_days", "interval_q25_days", "interval_median_days", "interval_mean_days",
        "interval_q75_days", "interval_max_days", "interval_std_days",
    ]
    existing = [col for col in preferred_order if col in summary.columns]
    rest = [col for col in summary.columns if col not in existing]
    return summary[existing + rest]


def select_detail_columns(intervals: pd.DataFrame) -> pd.DataFrame:
    if intervals.empty:
        return intervals
    existing = [col for col in DETAIL_COLUMNS_CANDIDATES if col in intervals.columns]
    rest = [col for col in intervals.columns if col not in existing]
    return intervals[existing + rest]


def save_outputs(intervals: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    intervals_path = output_dir / "failure_intervals_by_node.csv"
    summary_path = output_dir / "failure_intervals_summary.csv"
    excel_path = output_dir / "failure_intervals.xlsx"

    intervals.to_csv(intervals_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        intervals.to_excel(writer, sheet_name="failure_intervals_by_node", index=False)
        summary.to_excel(writer, sheet_name="failure_intervals_summary", index=False)
        autosize_excel_columns(writer)

    log()
    log("Сохранены файлы:")
    log(f"  {intervals_path}")
    log(f"  {summary_path}")
    log(f"  {excel_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Анализ интервалов между отказами Hosokawa по узлам и установкам.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Путь к jtiny_hosokawa_events_by_node.xlsx.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Папка для сохранения результатов.")
    parser.add_argument("--sheets", nargs="+", default=DEFAULT_SHEETS, help="Листы для анализа. По умолчанию: mill compactor.")
    parser.add_argument("--event-class", default=DEFAULT_EVENT_CLASS, help='Класс событий для анализа. По умолчанию: "Неисправность/отказ".')
    parser.add_argument("--date-column", default=DEFAULT_DATE_COLUMN, help="Колонка времени события. По умолчанию: 'Дата/время начала простоя'.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = resolve_project_path(args.input)
    output_dir = resolve_project_path(args.output_dir)

    log("=" * 90)
    log("Анализ интервалов между отказами Hosokawa")
    log("=" * 90)
    log(f"Корень проекта: {PROJECT_ROOT}")
    log(f"Входной файл: {input_path}")
    log(f"Выходная папка: {output_dir}")
    log(f"Листы: {args.sheets}")
    log(f"Класс событий: {args.event_class!r}")
    log(f"Колонка времени: {args.date_column!r}")
    log()

    events = load_failure_events(
        input_path=input_path,
        sheets=args.sheets,
        event_class=args.event_class,
        date_column=args.date_column,
    )
    if events.empty:
        raise RuntimeError("После фильтрации не осталось событий для анализа.")

    intervals = add_interval_columns(events)
    intervals = select_detail_columns(intervals)
    summary = make_summary(intervals)

    log()
    log("Краткая сводка по узлам:")
    if not summary.empty:
        node_total = summary[summary["summary_level"] == "node_total"].copy()
        cols = ["node", "event_count", "interval_count", "interval_median_days", "interval_mean_days", "interval_min_days", "interval_max_days"]
        existing_cols = [col for col in cols if col in node_total.columns]
        log(node_total[existing_cols].to_string(index=False))

    save_outputs(intervals, summary, output_dir)
    log()
    log("Готово.")


if __name__ == "__main__":
    main()
