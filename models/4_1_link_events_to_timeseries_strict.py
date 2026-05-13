from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "../processed"
TOIR_DIR = BASE_DIR / "../toir_hosokawa"

DEFAULT_EVENTS_XLSX_PATH = TOIR_DIR / "jtiny_hosokawa_events_quality_labeled.xlsx"
EVENTS_SHEET_NAME = "strict_model_events"

UNIT_COL = "N_Hosokawa"
DT_COL = "DT"
EVENT_TIME_COL = "_detected_event_time"
EVENT_ID_COL = "№ обр"
SOURCE_SHEET_COL = "source_sheet"
RECOMMENDED_NODE_COL = "recommended_node"
STRICT_CLASS_COL = "strict_event_class"
LABEL_QUALITY_COL = "label_quality"
USE_FOR_STRICT_COL = "use_for_strict_model"

OPTIONAL_FEATURES = {
    "GranulatorSpeed",
    "ValkiSpeed",
}

BASE_OUTPUTS = {
    "compactor": {
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

LABEL_COLUMNS = [
    "event_in_24h",
    "event_in_48h",
    "event_in_72h",
    "time_to_next_event_hours",
    "pre_event_window",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Увязка строгих событий ТОИР из strict_model_events "
            "с временными рядами Hosokawa."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Путь к входному parquet без полных дублей.",
    )
    parser.add_argument(
        "--events",
        type=Path,
        default=DEFAULT_EVENTS_XLSX_PATH,
        help="Путь к Excel-файлу со строгой разметкой событий.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Папка для выходных labeled parquet-файлов.",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help=(
            "Суффикс для выходных файлов. Например, _ssd даст "
            "compactor_dataset_labeled_ssd.parquet."
        ),
    )
    return parser.parse_args()


def build_outputs(output_dir: Path, output_suffix: str) -> dict[str, dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, object]] = {}

    for node_name, cfg in BASE_OUTPUTS.items():
        outputs[node_name] = {
            "path": output_dir / f"{node_name}_dataset_labeled{output_suffix}.parquet",
            "features": list(cfg["features"]),
        }
    return outputs


def normalize_parquet_dataframe(table: pa.Table) -> pd.DataFrame:
    df = table.to_pandas()
    if DT_COL not in df.columns and getattr(df.index, "name", None) == DT_COL:
        df = df.reset_index()
    if DT_COL not in df.columns:
        raise ValueError("В parquet не найден столбец или индекс DT")
    return df


def normalize_node_value(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def load_strict_events(events_path: Path, node_name: str) -> pd.DataFrame:
    events = pd.read_excel(events_path, sheet_name=EVENTS_SHEET_NAME)

    required = [
        EVENT_TIME_COL,
        UNIT_COL,
        SOURCE_SHEET_COL,
        RECOMMENDED_NODE_COL,
        STRICT_CLASS_COL,
        LABEL_QUALITY_COL,
        USE_FOR_STRICT_COL,
    ]
    missing = [col for col in required if col not in events.columns]
    if missing:
        raise ValueError(
            f"На листе {EVENTS_SHEET_NAME!r} отсутствуют обязательные столбцы: {missing}"
        )

    events = events.copy()
    events[EVENT_TIME_COL] = pd.to_datetime(events[EVENT_TIME_COL], errors="coerce")
    events[UNIT_COL] = pd.to_numeric(events[UNIT_COL], errors="coerce").astype("Int64")
    events[USE_FOR_STRICT_COL] = (
        pd.to_numeric(events[USE_FOR_STRICT_COL], errors="coerce")
        .fillna(0)
        .astype("int8")
    )

    events[SOURCE_SHEET_COL] = events[SOURCE_SHEET_COL].map(normalize_node_value)
    events[RECOMMENDED_NODE_COL] = events[RECOMMENDED_NODE_COL].map(normalize_node_value)
    events[STRICT_CLASS_COL] = events[STRICT_CLASS_COL].map(normalize_node_value)
    events[LABEL_QUALITY_COL] = events[LABEL_QUALITY_COL].map(normalize_node_value)

    events = events.dropna(subset=[EVENT_TIME_COL, UNIT_COL])

    events = events[
        (events[SOURCE_SHEET_COL] == node_name)
        & (events[RECOMMENDED_NODE_COL] == node_name)
        & (events[STRICT_CLASS_COL] == "strict_failure")
        & (events[LABEL_QUALITY_COL] == "good")
        & (events[USE_FOR_STRICT_COL] == 1)
    ].copy()

    if EVENT_ID_COL not in events.columns:
        events[EVENT_ID_COL] = np.arange(1, len(events) + 1)

    events = events.sort_values([UNIT_COL, EVENT_TIME_COL, EVENT_ID_COL]).reset_index(drop=True)
    return events


def make_event_lookup(events: pd.DataFrame) -> dict[int, np.ndarray]:
    lookup: dict[int, np.ndarray] = {}
    for unit, part in events.groupby(UNIT_COL, dropna=True):
        times = part[EVENT_TIME_COL].to_numpy(dtype="datetime64[ns]")
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
    unit_values = units.to_numpy()
    dt_values = pd.to_datetime(df[DT_COL], errors="coerce").to_numpy(dtype="datetime64[ns]")
    valid_dt_mask = ~np.isnat(dt_values)

    unique_units = sorted(pd.Series(units.dropna().unique()).astype(int).tolist())
    for unit in unique_units:
        event_times = lookup.get(unit)
        if event_times is None or len(event_times) == 0:
            continue

        mask = (unit_values == unit) & valid_dt_mask
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
    table = pa.Table.from_pandas(df, preserve_index=False)
    if writer is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()
        writer = pq.ParquetWriter(path, table.schema, compression="zstd")
    writer.write_table(table)
    return writer


def collect_needed_columns(outputs: dict[str, dict[str, object]]) -> list[str]:
    needed_columns: list[str] = []
    for cfg in outputs.values():
        for col in cfg["features"]:
            if col not in needed_columns:
                needed_columns.append(col)
    return needed_columns


def remove_missing_optional_features(
    outputs: dict[str, dict[str, object]],
    parquet_columns: set[str],
) -> dict[str, dict[str, object]]:
    cleaned_outputs: dict[str, dict[str, object]] = {}
    missing_required: list[str] = []
    missing_optional: list[str] = []

    for node_name, cfg in outputs.items():
        cleaned_features: list[str] = []
        for feature in cfg["features"]:
            if feature in parquet_columns:
                cleaned_features.append(feature)
                continue

            if feature in OPTIONAL_FEATURES:
                missing_optional.append(feature)
                continue

            missing_required.append(feature)

        cleaned_outputs[node_name] = {
            "path": cfg["path"],
            "features": cleaned_features,
        }

    if missing_required:
        unique_missing = sorted(set(missing_required))
        raise ValueError(f"В parquet отсутствуют обязательные признаки: {unique_missing}")

    if missing_optional:
        unique_missing_optional = sorted(set(missing_optional))
        print(
            "\nПредупреждение: во входном parquet отсутствуют optional SSD-признаки, "
            f"они будут пропущены: {unique_missing_optional}"
        )

    return cleaned_outputs


def main() -> None:
    args = parse_args()
    parquet_path = args.input
    events_path = args.events
    outputs = build_outputs(args.output_dir, args.output_suffix)

    if not parquet_path.exists():
        raise FileNotFoundError(f"Не найден parquet: {parquet_path}")
    if not events_path.exists():
        raise FileNotFoundError(f"Не найден Excel с событиями: {events_path}")

    pf = pq.ParquetFile(parquet_path)
    parquet_columns = set(pf.schema_arrow.names)
    outputs = remove_missing_optional_features(outputs, parquet_columns)

    print("Увязка строгих событий ТОИР с временными рядами Hosokawa")
    print(f"Parquet: {parquet_path}")
    print(f"Excel событий: {events_path}")
    print(f"Лист событий: {EVENTS_SHEET_NAME}")
    print(f"Суффикс выходных файлов: {args.output_suffix!r}")
    print(f"Строк parquet: {pf.metadata.num_rows:,}".replace(",", " "))
    print(f"Столбцов parquet: {pf.metadata.num_columns}")
    print(f"Групп строк: {pf.num_row_groups}")

    needed_columns = collect_needed_columns(outputs)

    n_check = normalize_parquet_dataframe(pq.read_table(parquet_path, columns=[UNIT_COL, DT_COL]))
    n_check[DT_COL] = pd.to_datetime(n_check[DT_COL], errors="coerce")
    print("\nN_Hosokawa в parquet:")
    print(n_check[UNIT_COL].value_counts(dropna=False).sort_index().to_string())
    print(f"DT min: {n_check[DT_COL].min()}")
    print(f"DT max: {n_check[DT_COL].max()}")
    del n_check

    lookups: dict[str, dict[int, np.ndarray]] = {}
    for node_name in outputs:
        events = load_strict_events(events_path, node_name)
        lookups[node_name] = make_event_lookup(events)

        print(f"\nСтрогие события для узла {node_name}: {len(events)}")
        if len(events) > 0:
            print(events.groupby(UNIT_COL).size().sort_index().to_string())
        else:
            print("Нет событий после строгой фильтрации")

    writers: dict[str, Optional[pq.ParquetWriter]] = {node_name: None for node_name in outputs}
    totals = {
        node_name: {"rows": 0, "event_24": 0, "event_48": 0, "event_72": 0}
        for node_name in outputs
    }

    try:
        for rg in range(pf.num_row_groups):
            base = safe_read_row_group(pf, rg, needed_columns)
            base[DT_COL] = pd.to_datetime(base[DT_COL], errors="coerce")
            base[UNIT_COL] = pd.to_numeric(base[UNIT_COL], errors="coerce").astype("Int64")

            print(f"\nОбработка row group {rg + 1}/{pf.num_row_groups}: {len(base):,} строк".replace(",", " "))

            for node_name, cfg in outputs.items():
                features = cfg["features"]
                out = base[features].copy()
                out = add_time_labels(out, lookups[node_name])
                writers[node_name] = write_chunk(writers[node_name], cfg["path"], out)

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
    for node_name, cfg in outputs.items():
        stat = totals[node_name]
        print(f"{node_name}: {cfg['path']}")
        print(f"  rows: {stat['rows']:,}".replace(",", " "))
        print(f"  event_in_24h: {stat['event_24']:,}".replace(",", " "))
        print(f"  event_in_48h: {stat['event_48']:,}".replace(",", " "))
        print(f"  event_in_72h / pre_event_window: {stat['event_72']:,}".replace(",", " "))


if __name__ == "__main__":
    main()
