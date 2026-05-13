from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "models" else SCRIPT_DIR

MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "processed"
TOIR_DIR = PROJECT_ROOT / "toir_hosokawa"
RESULTS_DIR = PROJECT_ROOT / "results"

DEFAULT_EVENTS_XLSX_PATH = TOIR_DIR / "jtiny_hosokawa_events_by_node.xlsx"

UNIT_COL = "N_Hosokawa"
DATE_START_COL = "Дата/время начала простоя"
EVENT_ID_COL = "№ обр"
EVENT_CLASS_COL = "event_class"
EVENT_CLASS_FILTER: Optional[list[str]] = None

LABEL_COLUMNS = [
    "event_in_24h",
    "event_in_48h",
    "event_in_72h",
    "time_to_next_event_hours",
    "pre_event_window",
]

OUTPUTS = {
    "compactor": {
        "sheet": "compactor",
        "path": PROCESSED_DIR / "compactor_dataset_labeled.parquet",
        "features": [
            "DT",
            "N_Hosokawa",
            "Regim",
            "ShneckSpeed",
            "Tok_shneka",
            "CompactorSpeed",
            "Tok_kompaktora",
            "ValkiPressure",
            "ValkiSpeed",
            "GranulatorSpeed",
            "Regulator_sili_sjatia_input_znach",
        ],
    },
    "mill": {
        "sheet": "mill",
        "path": PROCESSED_DIR / "mill_dataset_labeled.parquet",
        "features": [
            "DT",
            "N_Hosokawa",
            "Regim",
            "MelnicaSpeed",
            "Tok_melnici",
            "Temp_korpusa_melnici",
            "Temp_perednego_podshipnika",
            "Temp_zadnego_podshipnika",
            "Skorost_shluza_melnici",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Увязка событий ТОИР с временными рядами parquet. "
            "Входной parquet обязательно задаётся через --input."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Путь к входному parquet, например processed/all_data_2020_2025_with_ssd_no_duplicates.parquet",
    )
    parser.add_argument(
        "--events",
        type=Path,
        default=DEFAULT_EVENTS_XLSX_PATH,
        help="Путь к Excel-файлу с событиями. По умолчанию: toir_hosokawa/jtiny_hosokawa_events_by_node.xlsx",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Суффикс для выходных parquet-файлов, например _ssd.",
    )
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def output_path_with_suffix(path: Path, suffix: str) -> Path:
    if not suffix:
        return path
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def normalize_parquet_dataframe(table: pa.Table) -> pd.DataFrame:
    df = table.to_pandas()
    if "DT" not in df.columns and getattr(df.index, "name", None) == "DT":
        df = df.reset_index()
    if "DT" not in df.columns:
        raise ValueError("В parquet не найден столбец или индекс DT")
    return df


def load_events(events_xlsx_path: Path, sheet_name: str) -> pd.DataFrame:
    events = pd.read_excel(events_xlsx_path, sheet_name=sheet_name)

    required = [DATE_START_COL, UNIT_COL]
    missing = [col for col in required if col not in events.columns]
    if missing:
        raise ValueError(f"На листе {sheet_name!r} отсутствуют обязательные столбцы: {missing}")

    events = events.copy()
    events[DATE_START_COL] = pd.to_datetime(events[DATE_START_COL], errors="coerce")
    events[UNIT_COL] = pd.to_numeric(events[UNIT_COL], errors="coerce").astype("Int64")
    events = events.dropna(subset=[DATE_START_COL, UNIT_COL])

    if EVENT_CLASS_FILTER is not None:
        if EVENT_CLASS_COL not in events.columns:
            raise ValueError(f"Для фильтра по классу событий отсутствует столбец {EVENT_CLASS_COL!r}")
        events = events[events[EVENT_CLASS_COL].isin(EVENT_CLASS_FILTER)]

    if EVENT_ID_COL not in events.columns:
        events[EVENT_ID_COL] = np.arange(1, len(events) + 1)

    events = events.sort_values([UNIT_COL, DATE_START_COL, EVENT_ID_COL]).reset_index(drop=True)
    return events


def make_event_lookup(events: pd.DataFrame) -> dict[int, np.ndarray]:
    lookup: dict[int, np.ndarray] = {}
    for unit, part in events.groupby(UNIT_COL, dropna=True):
        times = part[DATE_START_COL].to_numpy(dtype="datetime64[ns]")
        lookup[int(unit)] = np.sort(times)
    return lookup


def add_time_labels(df: pd.DataFrame, lookup: dict[int, np.ndarray]) -> pd.DataFrame:
    n = len(df)
    time_to_event = np.full(n, np.nan, dtype="float64")

    if n == 0:
        for col in LABEL_COLUMNS:
            df[col] = []
        return df

    units = pd.to_numeric(df[UNIT_COL], errors="coerce")
    dt_values = pd.to_datetime(df["DT"], errors="coerce").to_numpy(dtype="datetime64[ns]")

    for unit in sorted(pd.Series(units.dropna().unique()).astype(int).tolist()):
        event_times = lookup.get(unit)
        if event_times is None or len(event_times) == 0:
            continue

        mask = (units.to_numpy() == unit) & ~pd.isna(dt_values)
        if not mask.any():
            continue

        row_times = dt_values[mask]
        idx = np.searchsorted(event_times, row_times, side="left")
        valid = idx < len(event_times)

        local_delta = np.full(len(row_times), np.nan, dtype="float64")
        local_delta[valid] = (event_times[idx[valid]] - row_times[valid]) / np.timedelta64(1, "h")
        time_to_event[mask] = local_delta

    df["time_to_next_event_hours"] = time_to_event
    df["event_in_24h"] = ((time_to_event >= 0) & (time_to_event <= 24)).astype("int8")
    df["event_in_48h"] = ((time_to_event >= 0) & (time_to_event <= 48)).astype("int8")
    df["event_in_72h"] = ((time_to_event >= 0) & (time_to_event <= 72)).astype("int8")
    df["pre_event_window"] = df["event_in_72h"].astype("int8")
    return df


def safe_read_row_group(pf: pq.ParquetFile, row_group: int, columns: list[str]) -> pd.DataFrame:
    table = pf.read_row_group(row_group, columns=columns)
    return normalize_parquet_dataframe(table)


def write_chunk(writer: Optional[pq.ParquetWriter], path: Path, df: pd.DataFrame) -> pq.ParquetWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema, compression="zstd")
    writer.write_table(table)
    return writer


def collect_needed_columns() -> list[str]:
    needed_columns: list[str] = []
    for cfg in OUTPUTS.values():
        for col in cfg["features"]:
            if col not in needed_columns:
                needed_columns.append(col)
    return needed_columns


def main() -> None:
    args = parse_args()
    parquet_path = resolve_project_path(args.input)
    events_xlsx_path = resolve_project_path(args.events)
    output_suffix = args.output_suffix

    if not parquet_path.exists():
        raise FileNotFoundError(f"Не найден входной parquet: {parquet_path}")
    if not events_xlsx_path.exists():
        raise FileNotFoundError(f"Не найден Excel с событиями: {events_xlsx_path}")

    pf = pq.ParquetFile(parquet_path)
    parquet_columns = set(pf.schema_arrow.names)
    needed_columns = collect_needed_columns()

    missing_features = [col for col in needed_columns if col not in parquet_columns]
    if missing_features:
        raise ValueError(
            "Во входном parquet отсутствуют нужные признаки: "
            f"{missing_features}\n"
            "Для SSD-эксперимента проверь, что используется файл с GranulatorSpeed и ValkiSpeed."
        )

    print("Проверка parquet")
    print(f"Файл: {parquet_path}")
    print(f"Строк: {pf.metadata.num_rows:,}".replace(",", " "))
    print(f"Столбцов: {pf.metadata.num_columns}")
    print(f"Групп строк: {pf.num_row_groups}")

    n_check = normalize_parquet_dataframe(pq.read_table(parquet_path, columns=[UNIT_COL, "DT"]))
    print("\nN_Hosokawa в parquet:")
    print(n_check[UNIT_COL].value_counts(dropna=False).sort_index().to_string())
    print(f"DT min: {n_check['DT'].min()}")
    print(f"DT max: {n_check['DT'].max()}")
    del n_check

    lookups: dict[str, dict[int, np.ndarray]] = {}
    for node_name, cfg in OUTPUTS.items():
        events = load_events(events_xlsx_path, cfg["sheet"])
        lookups[node_name] = make_event_lookup(events)

        print(f"\nСобытия {node_name} / лист {cfg['sheet']!r}: {len(events)}")
        if EVENT_CLASS_COL in events.columns:
            print(events[EVENT_CLASS_COL].value_counts(dropna=False).to_string())
        print(events.groupby(UNIT_COL).size().sort_index().to_string())

    writers: dict[str, Optional[pq.ParquetWriter]] = {node_name: None for node_name in OUTPUTS}
    output_paths = {
        node_name: output_path_with_suffix(cfg["path"], output_suffix)
        for node_name, cfg in OUTPUTS.items()
    }
    totals = {
        node_name: {"rows": 0, "event_24": 0, "event_48": 0, "event_72": 0}
        for node_name in OUTPUTS
    }

    try:
        for rg in range(pf.num_row_groups):
            base = safe_read_row_group(pf, rg, needed_columns)
            base["DT"] = pd.to_datetime(base["DT"], errors="coerce")
            base[UNIT_COL] = pd.to_numeric(base[UNIT_COL], errors="coerce").astype("Int64")

            print(f"\nОбработка row group {rg + 1}/{pf.num_row_groups}: {len(base):,} строк".replace(",", " "))

            for node_name, cfg in OUTPUTS.items():
                out = base[cfg["features"]].copy()
                out = add_time_labels(out, lookups[node_name])
                writers[node_name] = write_chunk(writers[node_name], output_paths[node_name], out)

                totals[node_name]["rows"] += len(out)
                totals[node_name]["event_24"] += int(out["event_in_24h"].sum())
                totals[node_name]["event_48"] += int(out["event_in_48h"].sum())
                totals[node_name]["event_72"] += int(out["event_in_72h"].sum())

                del out
            del base
    finally:
        for writer in writers.values():
            if writer is not None:
                writer.close()

    print("\nГотово. Итоговые датасеты:")
    for node_name, path in output_paths.items():
        stat = totals[node_name]
        print(f"{node_name}: {path}")
        print(f"  rows: {stat['rows']:,}".replace(",", " "))
        print(f"  event_in_24h: {stat['event_24']:,}".replace(",", " "))
        print(f"  event_in_48h: {stat['event_48']:,}".replace(",", " "))
        print(f"  event_in_72h / pre_event_window: {stat['event_72']:,}".replace(",", " "))


if __name__ == "__main__":
    main()
