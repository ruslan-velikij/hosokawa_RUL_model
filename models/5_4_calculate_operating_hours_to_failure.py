#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "models" else SCRIPT_DIR

PROCESSED_DIR = PROJECT_ROOT / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

DEFAULT_OUTPUT_DIR = RESULTS_DIR / "operating_hours_to_failure"

UNIT_COL = "N_Hosokawa"
DT_COL = "DT"
TARGET_COL = "time_to_next_event_hours"

NODE_CONFIG = {
    "compactor": {
        "node_ru": "Компактор",
        "input": PROCESSED_DIR / "compactor_dataset_labeled.parquet",
        "speed_columns": [
            "ShneckSpeed",
            "CompactorSpeed",
            "ValkiSpeed",
            "GranulatorSpeed",
        ],
        "current_columns": [
            "Tok_shneka",
            "Tok_kompaktora",
        ],
        "extra_columns": [
            "ValkiPressure",
            "Regulator_sili_sjatia_input_znach",
        ],
    },
    "mill": {
        "node_ru": "Мельница",
        "input": PROCESSED_DIR / "mill_dataset_labeled.parquet",
        "speed_columns": [
            "MelnicaSpeed",
            "Skorost_shluza_melnici",
        ],
        "current_columns": [
            "Tok_melnici",
        ],
        "extra_columns": [
            "Temp_korpusa_melnici",
            "Temp_perednego_podshipnika",
            "Temp_zadnego_podshipnika",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Расчёт рабочих часов до ближайшего отказа по размеченным датасетам Hosokawa."
    )

    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Папка processed. По умолчанию: PROJECT_ROOT/processed.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Папка для сохранения результатов.",
    )

    parser.add_argument(
        "--nodes",
        type=str,
        default="compactor,mill",
        help="Список узлов через запятую: compactor,mill.",
    )

    parser.add_argument(
        "--horizons",
        type=str,
        default="72,168,336",
        help="Горизонты анализа в календарных часах через запятую.",
    )

    parser.add_argument(
        "--max-gap-hours",
        type=float,
        default=1.0,
        help=(
            "Максимальный интервал между соседними строками, который можно засчитывать "
            "как непрерывную работу. Большие разрывы обрезаются. По умолчанию 1 час."
        ),
    )

    parser.add_argument(
        "--speed-threshold",
        type=float,
        default=0.0,
        help="Порог скорости для определения работы оборудования. По умолчанию > 0.",
    )

    parser.add_argument(
        "--current-threshold",
        type=float,
        default=0.0,
        help="Порог тока для определения работы оборудования. По умолчанию > 0.",
    )

    parser.add_argument(
        "--sample-rows",
        type=int,
        default=10000,
        help="Сколько строк сохранить в общий CSV-пример. По умолчанию 10000.",
    )

    parser.add_argument(
        "--save-full",
        action="store_true",
        help="Если указан, сохраняет полные parquet-файлы с расчётом по каждому узлу.",
    )

    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_horizons(value: str) -> list[float]:
    horizons = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        horizons.append(float(part))
    if not horizons:
        raise ValueError("Список горизонтов пуст.")
    return horizons


def parquet_columns(path: Path) -> set[str]:
    return set(pq.ParquetFile(path).schema_arrow.names)


def read_node_dataset(node: str, processed_dir: Path) -> tuple[pd.DataFrame, dict]:
    cfg = NODE_CONFIG[node].copy()
    input_path = processed_dir / cfg["input"].name

    if not input_path.exists():
        raise FileNotFoundError(
            f"Не найден входной файл для узла {node}: {input_path}\n"
            "Сначала выполните 4_1_link_events_to_timeseries.py."
        )

    existing = parquet_columns(input_path)

    base_columns = [DT_COL, UNIT_COL, TARGET_COL, "event_in_24h", "event_in_48h", "event_in_72h"]
    signal_columns = cfg["speed_columns"] + cfg["current_columns"] + cfg["extra_columns"]

    needed_columns = []
    for col in base_columns + signal_columns:
        if col in existing and col not in needed_columns:
            needed_columns.append(col)

    missing_required = [col for col in [DT_COL, UNIT_COL, TARGET_COL] if col not in needed_columns]
    if missing_required:
        raise ValueError(f"В файле {input_path} отсутствуют обязательные столбцы: {missing_required}")

    missing_signals = [col for col in signal_columns if col not in existing]
    if missing_signals:
        print(f"{node}: предупреждение, отсутствуют сигналы: {missing_signals}")

    df = pd.read_parquet(input_path, columns=needed_columns)

    df[DT_COL] = pd.to_datetime(df[DT_COL], errors="coerce")
    df[UNIT_COL] = pd.to_numeric(df[UNIT_COL], errors="coerce")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

    return df, cfg


def make_is_running(
    df: pd.DataFrame,
    speed_columns: Iterable[str],
    current_columns: Iterable[str],
    speed_threshold: float,
    current_threshold: float,
) -> pd.Series:
    running = pd.Series(False, index=df.index)

    used_speed = []
    for col in speed_columns:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce").abs()
        running |= values > speed_threshold
        used_speed.append(col)

    used_current = []
    for col in current_columns:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce").abs()
        running |= values > current_threshold
        used_current.append(col)

    if not used_speed and not used_current:
        raise ValueError("Не найдено ни одного сигнала скорости/тока для определения is_running.")

    return running.astype("int8")


def calculate_operating_hours_for_group(
    group: pd.DataFrame,
    max_gap_hours: float,
) -> pd.DataFrame:
    group = group.sort_values(DT_COL).copy()

    n = len(group)
    if n == 0:
        return group

    dt = pd.to_datetime(group[DT_COL], errors="coerce")
    valid_dt = dt.notna().to_numpy()

    result = pd.DataFrame(index=group.index)
    result["delta_to_next_row_hours"] = np.nan
    result["running_interval_hours"] = np.nan
    result["cumulative_operating_hours"] = np.nan
    result["operating_hours_to_next_failure"] = np.nan
    result["operating_to_calendar_ratio"] = np.nan

    if valid_dt.sum() < 2:
        return pd.concat([group, result], axis=1)

    valid_group = group.loc[valid_dt].copy()
    valid_index = valid_group.index

    times = valid_group[DT_COL].to_numpy(dtype="datetime64[ns]").astype("int64")
    times_float = times.astype("float64")

    n_valid = len(valid_group)

    delta_ns = np.empty(n_valid, dtype="float64")
    delta_ns[:-1] = np.diff(times_float)
    delta_ns[-1] = 0.0

    delta_hours_raw = delta_ns / 3_600_000_000_000.0
    delta_hours = np.where(
        np.isfinite(delta_hours_raw) & (delta_hours_raw > 0),
        np.minimum(delta_hours_raw, max_gap_hours),
        0.0,
    )

    is_running = valid_group["is_running"].to_numpy(dtype=bool)
    running_interval = delta_hours * is_running.astype("float64")

    cumulative_start = np.empty(n_valid, dtype="float64")
    cumulative_start[0] = 0.0
    if n_valid > 1:
        cumulative_start[1:] = np.cumsum(running_interval[:-1])

    target_hours = valid_group[TARGET_COL].to_numpy(dtype="float64")
    valid_event = np.isfinite(target_hours) & (target_hours > 0)

    operating_to_event = np.full(n_valid, np.nan, dtype="float64")
    ratio = np.full(n_valid, np.nan, dtype="float64")

    if valid_event.any():
        row_positions = np.flatnonzero(valid_event)
        event_times_float = times_float[row_positions] + target_hours[row_positions] * 3_600_000_000_000.0

        event_idx = np.searchsorted(times_float, event_times_float, side="right") - 1
        event_idx = np.clip(event_idx, 0, n_valid - 1)

        cum_at_event = cumulative_start[event_idx].astype("float64")

        partial_hours = (event_times_float - times_float[event_idx]) / 3_600_000_000_000.0
        partial_hours = np.where(np.isfinite(partial_hours) & (partial_hours > 0), partial_hours, 0.0)

        can_add_partial = event_idx < (n_valid - 1)
        partial_clipped = np.minimum(partial_hours, delta_hours[event_idx])
        partial_add = np.where(
            can_add_partial,
            partial_clipped * is_running[event_idx].astype("float64"),
            0.0,
        )

        cum_at_event = cum_at_event + partial_add

        current_cum = cumulative_start[row_positions]
        local_operating = cum_at_event - current_cum
        local_operating = np.where(local_operating >= 0, local_operating, 0.0)

        operating_to_event[row_positions] = local_operating
        ratio[row_positions] = np.where(
            target_hours[row_positions] > 0,
            local_operating / target_hours[row_positions],
            np.nan,
        )

    result.loc[valid_index, "delta_to_next_row_hours"] = delta_hours
    result.loc[valid_index, "running_interval_hours"] = running_interval
    result.loc[valid_index, "cumulative_operating_hours"] = cumulative_start
    result.loc[valid_index, "operating_hours_to_next_failure"] = operating_to_event
    result.loc[valid_index, "operating_to_calendar_ratio"] = ratio

    return pd.concat([group, result], axis=1)


def calculate_operating_hours(
    df: pd.DataFrame,
    max_gap_hours: float,
) -> pd.DataFrame:
    parts = []

    for unit, group in df.groupby(UNIT_COL, dropna=True, sort=True):
        print(f"  расчёт рабочих часов для N_Hosokawa={unit}: {len(group):,} строк".replace(",", " "))
        part = calculate_operating_hours_for_group(group, max_gap_hours=max_gap_hours)
        parts.append(part)

    if not parts:
        raise ValueError("Не найдено ни одной группы N_Hosokawa для расчёта.")

    out = pd.concat(parts, axis=0).sort_values([UNIT_COL, DT_COL]).reset_index(drop=True)
    return out


def numeric_summary(series: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()

    if values.empty:
        return {
            "mean": np.nan,
            "median": np.nan,
            "min": np.nan,
            "max": np.nan,
            "p25": np.nan,
            "p75": np.nan,
        }

    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "min": float(values.min()),
        "max": float(values.max()),
        "p25": float(values.quantile(0.25)),
        "p75": float(values.quantile(0.75)),
    }


def build_summary(
    df: pd.DataFrame,
    node: str,
    node_ru: str,
    horizons: list[float],
    group_col: str | None = None,
) -> list[dict]:
    rows = []
    groups: list[tuple[str, pd.DataFrame]]

    if group_col is None:
        groups = [("all", df)]
    else:
        groups = [(str(k), v) for k, v in df.groupby(group_col, dropna=True, sort=True)]

    for group_value, part in groups:
        running_share = float(part["is_running"].mean()) if len(part) else np.nan

        for horizon in horizons:
            mask = (
                part[TARGET_COL].notna()
                & (part[TARGET_COL] > 0)
                & (part[TARGET_COL] <= horizon)
                & part["operating_hours_to_next_failure"].notna()
            )
            h = part.loc[mask].copy()

            calendar_stats = numeric_summary(h[TARGET_COL])
            operating_stats = numeric_summary(h["operating_hours_to_next_failure"])
            ratio_stats = numeric_summary(h["operating_to_calendar_ratio"])

            row = {
                "node": node,
                "node_ru": node_ru,
                "group_column": group_col if group_col is not None else "all",
                "group_value": group_value,
                "horizon_calendar_hours": horizon,
                "rows_total": int(len(part)),
                "rows_in_horizon": int(len(h)),
                "running_share_all_rows": running_share,
                "calendar_hours_mean": calendar_stats["mean"],
                "calendar_hours_median": calendar_stats["median"],
                "calendar_hours_p25": calendar_stats["p25"],
                "calendar_hours_p75": calendar_stats["p75"],
                "calendar_hours_min": calendar_stats["min"],
                "calendar_hours_max": calendar_stats["max"],
                "operating_hours_mean": operating_stats["mean"],
                "operating_hours_median": operating_stats["median"],
                "operating_hours_p25": operating_stats["p25"],
                "operating_hours_p75": operating_stats["p75"],
                "operating_hours_min": operating_stats["min"],
                "operating_hours_max": operating_stats["max"],
                "operating_to_calendar_ratio_mean": ratio_stats["mean"],
                "operating_to_calendar_ratio_median": ratio_stats["median"],
                "operating_to_calendar_ratio_p25": ratio_stats["p25"],
                "operating_to_calendar_ratio_p75": ratio_stats["p75"],
            }
            rows.append(row)

    return rows


def select_sample(df: pd.DataFrame, node: str, node_ru: str, sample_rows: int) -> pd.DataFrame:
    columns = [
        DT_COL,
        UNIT_COL,
        "is_running",
        "delta_to_next_row_hours",
        "running_interval_hours",
        "cumulative_operating_hours",
        TARGET_COL,
        "operating_hours_to_next_failure",
        "operating_to_calendar_ratio",
    ]

    optional = ["event_in_24h", "event_in_48h", "event_in_72h"]
    columns += [col for col in optional if col in df.columns]

    existing = [col for col in columns if col in df.columns]

    valid = df[df["operating_hours_to_next_failure"].notna()]
    if len(valid) > sample_rows:
        sample = valid.sample(sample_rows, random_state=42)
    else:
        sample = valid

    sample = sample[existing].copy()
    sample.insert(0, "node_ru", node_ru)
    sample.insert(0, "node", node)
    return sample.sort_values(["node", UNIT_COL, DT_COL]).reset_index(drop=True)


def save_outputs(
    output_dir: Path,
    summary_by_node: pd.DataFrame,
    summary_by_unit: pd.DataFrame,
    sample_rows: pd.DataFrame,
    full_paths: list[Path],
    params: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_by_node_path = output_dir / "operating_hours_summary_by_node.csv"
    summary_by_unit_path = output_dir / "operating_hours_summary_by_unit.csv"
    sample_path = output_dir / "operating_hours_to_failure_sample.csv"
    xlsx_path = output_dir / "operating_hours_to_failure.xlsx"
    params_path = output_dir / "operating_hours_params.json"
    txt_path = output_dir / "operating_hours_summary.txt"

    summary_by_node.to_csv(summary_by_node_path, index=False, encoding="utf-8-sig")
    summary_by_unit.to_csv(summary_by_unit_path, index=False, encoding="utf-8-sig")
    sample_rows.to_csv(sample_path, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        summary_by_node.to_excel(writer, sheet_name="summary_by_node", index=False)
        summary_by_unit.to_excel(writer, sheet_name="summary_by_unit", index=False)
        sample_rows.to_excel(writer, sheet_name="sample_rows", index=False)

    params["full_output_parquet"] = [str(path.relative_to(PROJECT_ROOT)) for path in full_paths]
    params_path.write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("Расчёт рабочих часов до ближайшего отказа")
    lines.append("=" * 72)
    lines.append("")
    lines.append("Рабочее время считается только на интервалах, где оборудование находится в работе.")
    lines.append("Признак is_running определяется по положительным значениям скоростей и токов.")
    lines.append("")
    lines.append("Сводка по узлам:")
    lines.append("-" * 72)

    for _, row in summary_by_node.iterrows():
        lines.append(
            f"{row['node_ru']}, горизонт {row['horizon_calendar_hours']:.0f} ч: "
            f"строк в горизонте = {int(row['rows_in_horizon'])}, "
            f"медиана календарного времени = {row['calendar_hours_median']:.2f} ч, "
            f"медиана рабочих часов = {row['operating_hours_median']:.2f} ч, "
            f"медианное отношение рабочее/календарное = {row['operating_to_calendar_ratio_median']:.3f}."
        )

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print("Результаты сохранены:")
    print(f"  {summary_by_node_path}")
    print(f"  {summary_by_unit_path}")
    print(f"  {sample_path}")
    print(f"  {xlsx_path}")
    print(f"  {params_path}")
    print(f"  {txt_path}")


def main() -> None:
    args = parse_args()

    processed_dir = resolve_project_path(args.processed_dir)
    output_dir = resolve_project_path(args.output_dir)

    nodes = parse_list(args.nodes)
    horizons = parse_horizons(args.horizons)

    unknown_nodes = [node for node in nodes if node not in NODE_CONFIG]
    if unknown_nodes:
        raise ValueError(f"Неизвестные узлы: {unknown_nodes}. Доступно: {list(NODE_CONFIG)}")

    print("Расчёт рабочих часов до ближайшего отказа")
    print(f"Папка processed: {processed_dir}")
    print(f"Папка результатов: {output_dir}")
    print(f"Узлы: {nodes}")
    print(f"Горизонты, ч: {horizons}")
    print(f"Максимальный учитываемый разрыв между строками: {args.max_gap_hours} ч")
    print()

    all_summary_by_node = []
    all_summary_by_unit = []
    all_samples = []
    full_paths = []

    for node in nodes:
        cfg = NODE_CONFIG[node]
        node_ru = cfg["node_ru"]

        print("=" * 80)
        print(f"Узел: {node_ru} ({node})")

        df, cfg = read_node_dataset(node, processed_dir)

        print(f"Исходный размер: {df.shape}")

        df = df.dropna(subset=[DT_COL, UNIT_COL]).copy()
        df[UNIT_COL] = df[UNIT_COL].astype("int64")

        df["is_running"] = make_is_running(
            df,
            speed_columns=cfg["speed_columns"],
            current_columns=cfg["current_columns"],
            speed_threshold=args.speed_threshold,
            current_threshold=args.current_threshold,
        )

        print(f"Доля строк, где оборудование работает: {df['is_running'].mean():.4f}")

        out = calculate_operating_hours(df, max_gap_hours=args.max_gap_hours)

        out["operating_to_calendar_ratio"] = out["operating_to_calendar_ratio"].clip(lower=0)

        node_summary = build_summary(
            out,
            node=node,
            node_ru=node_ru,
            horizons=horizons,
            group_col=None,
        )
        unit_summary = build_summary(
            out,
            node=node,
            node_ru=node_ru,
            horizons=horizons,
            group_col=UNIT_COL,
        )

        all_summary_by_node.extend(node_summary)
        all_summary_by_unit.extend(unit_summary)
        all_samples.append(select_sample(out, node=node, node_ru=node_ru, sample_rows=args.sample_rows))

        if args.save_full:
            node_dir = output_dir / "full"
            node_dir.mkdir(parents=True, exist_ok=True)
            full_path = node_dir / f"{node}_operating_hours_to_failure.parquet"

            keep_cols = [
                DT_COL,
                UNIT_COL,
                "is_running",
                "delta_to_next_row_hours",
                "running_interval_hours",
                "cumulative_operating_hours",
                TARGET_COL,
                "operating_hours_to_next_failure",
                "operating_to_calendar_ratio",
            ]
            keep_cols += [col for col in ["event_in_24h", "event_in_48h", "event_in_72h"] if col in out.columns]
            keep_cols = [col for col in keep_cols if col in out.columns]

            out[keep_cols].to_parquet(full_path, index=False)
            full_paths.append(full_path)
            print(f"Полный файл сохранён: {full_path}")

        del df
        del out

    summary_by_node = pd.DataFrame(all_summary_by_node)
    summary_by_unit = pd.DataFrame(all_summary_by_unit)
    sample_rows = pd.concat(all_samples, axis=0, ignore_index=True) if all_samples else pd.DataFrame()

    params = {
        "processed_dir": str(processed_dir),
        "output_dir": str(output_dir),
        "nodes": nodes,
        "horizons_hours": horizons,
        "max_gap_hours": args.max_gap_hours,
        "speed_threshold": args.speed_threshold,
        "current_threshold": args.current_threshold,
        "sample_rows": args.sample_rows,
        "save_full": bool(args.save_full),
        "definition": {
            "is_running": "Оборудование считается работающим, если хотя бы один сигнал скорости или тока выше заданного порога.",
            "running_interval_hours": "Длительность интервала до следующей строки, засчитанная как рабочее время, если is_running = 1.",
            "operating_hours_to_next_failure": "Сумма рабочих интервалов от текущей строки до ближайшего будущего события.",
            "operating_to_calendar_ratio": "Отношение рабочих часов до события к календарным часам до события.",
        },
    }

    save_outputs(
        output_dir=output_dir,
        summary_by_node=summary_by_node,
        summary_by_unit=summary_by_unit,
        sample_rows=sample_rows,
        full_paths=full_paths,
        params=params,
    )

    print()
    print("Готово.")


if __name__ == "__main__":
    main()
