#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="Could not infer format")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "models" else SCRIPT_DIR
TOIR_DIR = PROJECT_ROOT / "toir_hosokawa"

DEFAULT_INPUT_CANDIDATES = [
    TOIR_DIR / "jtiny_hosokawa_events_by_node.xlsx",
    PROJECT_ROOT / "jtiny_hosokawa_events_by_node.xlsx",
]
DEFAULT_OUTPUT = TOIR_DIR / "jtiny_hosokawa_events_quality_labeled.xlsx"
SHEETS_TO_PROCESS = ["mill", "compactor"]
INCIDENT_WINDOW_HOURS = 12

MIN_OUTPUT_COLUMNS = [
    "source_sheet",
    "№ обр",
    "N_Hosokawa",
    "node",
    "recommended_node",
    "strict_event_class",
    "label_quality",
    "use_for_strict_model",
    "_detected_event_time",
    "Дата/время начала простоя",
    "Вид меропр",
    "Наименование оборудования",
    "Краткое описание отказа (причины простоя)",
    "Краткое описание выполненных работ",
    "incident_group_note",
    "suggested_incident_time",
]

NODE_KEYWORDS = {
    "mill": [
        r"\bmill\b",
        r"melnica",
        r"мельниц",
        r"мелниц",
        r"мельн",
        r"мелн",
        r"подшипник[^\n\r]{0,40}мельниц",
        r"корпус[^\n\r]{0,40}мельниц",
        r"шлюз[^\n\r]{0,40}мельниц",
        r"ток[^\n\r]{0,40}мельниц",
        r"двигател[^\n\r]{0,40}мельниц",
        r"скорост[^\n\r]{0,40}мельниц",
    ],
    "compactor": [
        r"\bcompactor\b",
        r"компактор",
        r"компакт",
        r"шнек",
        r"валк",
        r"валок",
        r"валки",
        r"давлен[^\n\r]{0,40}валк",
        r"ток[^\n\r]{0,40}шнек",
        r"ток[^\n\r]{0,40}компакт",
        r"сил[аы][^\n\r]{0,40}сжат",
        r"прессован",
        r"гранулятор",
    ],
}

STRICT_FAILURE_KEYWORDS = [
    r"неисправ",
    r"отказ",
    r"авари",
    r"не\s+запуска",
    r"не\s+старту",
    r"не\s+работа",
    r"не\s+вращ",
    r"останов",
    r"остановк",
    r"перегрев",
    r"повышенн[^\n\r]{0,30}температур",
    r"высок[^\n\r]{0,30}температур",
    r"заклин",
    r"клинит",
    r"закусыв",
    r"заедан",
    r"перегруз",
    r"повышенн[^\n\r]{0,30}ток",
    r"ток[^\n\r]{0,30}высок",
    r"вибрац",
    r"шум",
    r"разруш",
    r"обрыв",
    r"сгорел",
    r"сгорев",
    r"выбива",
    r"течь",
    r"утеч",
    r"засор",
    r"забив",
    r"замен[^\n\r]{0,40}подшип",
    r"замен[^\n\r]{0,40}двигател",
    r"замен[^\n\r]{0,40}валк",
    r"замен[^\n\r]{0,40}шнек",
]

PLANNED_WORK_KEYWORDS = [
    r"планов",
    r"ппр",
    r"\bто\b",
    r"техобслуж",
    r"регламент",
    r"профилактик",
    r"по\s+график",
    r"периодическ",
]

CLEANING_CHANGEOVER_KEYWORDS = [
    r"зачист",
    r"мойк",
    r"мыть",
    r"уборк",
    r"переход",
    r"смен[аы][^\n\r]{0,30}продукт",
    r"смен[аы][^\n\r]{0,30}парт",
    r"переналад",
    r"подготовк[^\n\r]{0,30}лини",
    r"промыв",
    r"санитар",
]

TIME_COLUMN_PRIORITY = [
    "дата/время начала простоя",
    "дата время начала простоя",
    "начало простоя",
    "дата начала простоя",
    "дата и время начала простоя",
    "дата/время начала",
    "дата начала",
    "начало",
    "дата события",
    "дата",
]

TIME_COLUMN_BAD_WORDS = ["оконч", "конец", "заверш", "закрыт", "выполн"]


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower().replace("ё", "е")
    text = re.sub(r"[\t\r\n]+", " ", text)
    text = re.sub(r"[;,.!?()\[\]{}:|/\\\"'`]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def has_any(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def resolve_project_path(path: Path) -> Path:
    """Возвращает абсолютный путь. Относительные пути считаются от корня проекта."""
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def choose_input_path() -> Path:
    for path in DEFAULT_INPUT_CANDIDATES:
        if path.exists():
            return path

    checked = "\n".join(f"  - {path}" for path in DEFAULT_INPUT_CANDIDATES)
    raise FileNotFoundError(
        "Не найден входной файл jtiny_hosokawa_events_by_node.xlsx.\n"
        "Проверьте, что сначала выполнен скрипт 3_4_classify_hosokawa_nodes.py, "
        "или укажите путь явно через --input.\n"
        f"Проверенные пути:\n{checked}"
    )


def build_quality_text(df: pd.DataFrame) -> pd.Series:
    text_cols = []
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            text_cols.append(col)
    if not text_cols:
        return pd.Series(["" for _ in range(len(df))], index=df.index)
    return df[text_cols].fillna("").astype(str).agg(" | ".join, axis=1).map(normalize_text)


def choose_event_time_column(df: pd.DataFrame) -> str | None:
    candidates = []
    for col in df.columns:
        norm_col = normalize_text(col)
        if any(bad in norm_col for bad in TIME_COLUMN_BAD_WORDS):
            continue
        if any(key in norm_col for key in ["дата", "время", "начало", "прост", "событ"]):
            candidates.append(col)

    best_col = None
    best_score = -1.0

    for col in candidates:
        norm_col = normalize_text(col)
        converted = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
        valid_share = converted.notna().mean()
        if valid_share < 0.05:
            continue

        score = valid_share * 20
        for idx, priority in enumerate(TIME_COLUMN_PRIORITY):
            if priority in norm_col:
                score += 100 - idx * 5
        if "нач" in norm_col:
            score += 25
        if "прост" in norm_col:
            score += 15

        if score > best_score:
            best_score = score
            best_col = col

    return best_col


def detect_event_time(df: pd.DataFrame) -> tuple[pd.Series, str | None]:
    col = choose_event_time_column(df)
    if col is None:
        return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]"), None
    return pd.to_datetime(df[col], errors="coerce", dayfirst=True), col


def extract_n_hosokawa(row: pd.Series, text: str) -> object:
    for col in row.index:
        if normalize_text(col) == "n_hosokawa":
            value = row[col]
            if pd.notna(value):
                try:
                    return int(float(value))
                except (TypeError, ValueError):
                    return value

    for pattern in [
        r"хосокава\s*[-№#]?\s*([1-4])",
        r"хос\s*[-№#]?\s*([1-4])",
        r"hosokawa\s*[-№#]?\s*([1-4])",
        r"hos\s*[-№#]?\s*([1-4])",
    ]:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return np.nan


def existing_node_hint(row: pd.Series) -> str | None:
    for col in row.index:
        if normalize_text(col) == "node":
            value = normalize_text(row[col])
            if "мельниц" in value or "mill" in value:
                return "mill"
            if "комп" in value or "шнек" in value or "валк" in value or "compactor" in value:
                return "compactor"
    return None


def determine_recommended_node(row: pd.Series, text: str, sheet_node: str) -> str:
    has_mill = has_any(text, NODE_KEYWORDS["mill"])
    has_compactor = has_any(text, NODE_KEYWORDS["compactor"])

    if has_mill and has_compactor:
        return "both"
    if has_mill:
        return "mill"
    if has_compactor:
        return "compactor"

    hint = existing_node_hint(row)
    if hint in {"mill", "compactor"}:
        return hint

    return sheet_node


def classify_row(row: pd.Series, sheet_node: str) -> dict[str, object]:
    text = row["_quality_text"]
    event_time = row["_detected_event_time"]
    recommended_node = determine_recommended_node(row, text, sheet_node)

    has_time = pd.notna(event_time)
    node_is_usable = recommended_node == sheet_node
    is_both = recommended_node == "both"
    is_cleaning = has_any(text, CLEANING_CHANGEOVER_KEYWORDS)
    is_nonplanned = has_any(text, [r"неплан"])
    is_planned = has_any(text, PLANNED_WORK_KEYWORDS) and not is_nonplanned
    is_strict_failure = has_any(text, STRICT_FAILURE_KEYWORDS)

    if has_time and node_is_usable and is_strict_failure and not is_cleaning and not is_planned:
        return {
            "strict_event_class": "strict_failure",
            "label_quality": "good",
            "recommended_node": recommended_node,
            "use_for_strict_model": 1,
        }

    return {
        "strict_event_class": "uncertain" if not is_both else "strict_failure",
        "label_quality": "uncertain",
        "recommended_node": recommended_node,
        "use_for_strict_model": 0,
    }


def add_incident_notes(strict_events: pd.DataFrame) -> pd.DataFrame:
    df = strict_events.copy()
    df["incident_group_note"] = ""
    df["suggested_incident_time"] = pd.NaT

    if df.empty:
        return df

    df = df.sort_values(["N_Hosokawa", "recommended_node", "_detected_event_time"])
    incident_counter = 0

    for (_, _), group in df.groupby(["N_Hosokawa", "recommended_node"], dropna=False):
        current_indices = []
        current_start = None
        previous_time = None

        for idx, row in group.iterrows():
            current_time = row["_detected_event_time"]
            if previous_time is None:
                current_indices = [idx]
                current_start = current_time
                previous_time = current_time
                continue

            hours_delta = (current_time - previous_time).total_seconds() / 3600
            if 0 <= hours_delta <= INCIDENT_WINDOW_HOURS:
                current_indices.append(idx)
                previous_time = current_time
            else:
                if len(current_indices) > 1:
                    incident_counter += 1
                    mark_incident(df, current_indices, incident_counter, current_start)
                current_indices = [idx]
                current_start = current_time
                previous_time = current_time

        if len(current_indices) > 1:
            incident_counter += 1
            mark_incident(df, current_indices, incident_counter, current_start)

    return df.sort_index()


def mark_incident(df: pd.DataFrame, indices: list[int], incident_counter: int, start_time: object) -> None:
    incident_id = f"incident_candidate_{incident_counter:04d}"
    for idx in indices:
        df.at[idx, "incident_group_note"] = (
            f"возможное объединение в один инцидент: {incident_id}; "
            f"окно {INCIDENT_WINDOW_HOURS} ч"
        )
        df.at[idx, "suggested_incident_time"] = start_time


def process_sheet(df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    df = df.copy()
    df["source_sheet"] = sheet_name
    df["_quality_text"] = build_quality_text(df)

    detected_time, time_col = detect_event_time(df)
    df["_detected_event_time"] = detected_time
    df["N_Hosokawa"] = [extract_n_hosokawa(row, row["_quality_text"]) for _, row in df.iterrows()]

    classifications = df.apply(lambda row: classify_row(row, sheet_name), axis=1, result_type="expand")
    for col in classifications.columns:
        df[col] = classifications[col]

    mask = (
        (df["label_quality"] == "good")
        & (df["strict_event_class"] == "strict_failure")
        & (df["use_for_strict_model"] == 1)
        & (df["recommended_node"] == sheet_name)
    )
    strict_events = df.loc[mask].copy()
    strict_events = add_incident_notes(strict_events)

    existing_cols = [col for col in MIN_OUTPUT_COLUMNS if col in strict_events.columns]
    strict_events = strict_events[existing_cols]

    print(f"Лист {sheet_name}: всего {len(df)}, для строгой модели {len(strict_events)}, колонка времени: {time_col}")
    return strict_events


def save_strict_events(strict_events: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        strict_events.to_excel(writer, sheet_name="strict_model_events", index=False)

        ws = writer.book["strict_model_events"]
        ws.freeze_panes = "A2"
        for col_cells in ws.columns:
            col_letter = col_cells[0].column_letter
            max_len = 0
            for cell in col_cells[:200]:
                if cell.value is not None:
                    max_len = max(max_len, len(str(cell.value)))
            ws.column_dimensions[col_letter].width = min(max(max_len + 2, 10), 45)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Качественная разметка Hosokawa: только strict_model_events.")
    parser.add_argument("--input", type=Path, default=None, help="Путь к jtiny_hosokawa_events_by_node.xlsx")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Путь к итоговому xlsx")
    parser.add_argument("--sheets", nargs="+", default=SHEETS_TO_PROCESS, help="Листы для обработки")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = resolve_project_path(args.input) if args.input is not None else choose_input_path()
    output_path = resolve_project_path(args.output)

    print("Качественная разметка событий Hosokawa")
    print(f"Входной файл: {input_path}")
    print(f"Выходной файл: {output_path}")
    print(f"Листы: {args.sheets}")
    print("=" * 80)

    xls = pd.ExcelFile(input_path)
    missing_sheets = [sheet for sheet in args.sheets if sheet not in xls.sheet_names]
    if missing_sheets:
        raise ValueError(f"Во входном файле нет листов: {missing_sheets}. Доступные листы: {xls.sheet_names}")

    strict_parts = []
    for sheet_name in args.sheets:
        df = pd.read_excel(input_path, sheet_name=sheet_name)
        strict_parts.append(process_sheet(df, sheet_name))

    strict_events = pd.concat(strict_parts, ignore_index=True) if strict_parts else pd.DataFrame(columns=MIN_OUTPUT_COLUMNS)
    strict_events = strict_events.sort_values(["source_sheet", "N_Hosokawa", "_detected_event_time"], na_position="last")

    save_strict_events(strict_events, output_path)

    print("=" * 80)
    print("Готово.")
    print(f"Итоговый файл: {output_path}")
    print(f"Событий для строгой модели всего: {len(strict_events)}")
    if not strict_events.empty:
        print("Распределение по узлам:")
        print(strict_events["source_sheet"].value_counts().to_string())


if __name__ == "__main__":
    main()
