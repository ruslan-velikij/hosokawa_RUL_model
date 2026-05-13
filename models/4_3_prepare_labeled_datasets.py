from __future__ import annotations

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)


BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "../processed"
PROCESSED_DIR.mkdir(exist_ok=True)

INPUTS = {
    "compactor": PROCESSED_DIR / "compactor_dataset_labeled.parquet",
    "mill": PROCESSED_DIR / "mill_dataset_labeled.parquet",
}

OUTPUTS = {
    "compactor": {
        "prepared": PROCESSED_DIR / "compactor_dataset_prepared.parquet",
        "train": PROCESSED_DIR / "compactor_train_prepared.parquet",
        "test": PROCESSED_DIR / "compactor_test_prepared.parquet",
        "report": PROCESSED_DIR / "compactor_prepare_report.json",
    },
    "mill": {
        "prepared": PROCESSED_DIR / "mill_dataset_prepared.parquet",
        "train": PROCESSED_DIR / "mill_train_prepared.parquet",
        "test": PROCESSED_DIR / "mill_test_prepared.parquet",
        "report": PROCESSED_DIR / "mill_prepare_report.json",
    },
}

TRAIN_END_DATE = pd.Timestamp("2025-01-01")

MAX_MISSING_SHARE = 0.95

FFILL_LIMIT: int | None = None

MAKE_ROLLING_FEATURES = False
ROLLING_WINDOWS_ROWS = [6, 30]

DT_COL = "DT"
UNIT_COL = "N_Hosokawa"
SPLIT_COL = "split"

TARGET_COLUMNS = [
    "event_in_24h",
    "event_in_48h",
    "event_in_72h",
    "pre_event_window",
]

REGRESSION_TARGET = "time_to_next_event_hours"

PROTECTED_COLUMNS = [
    DT_COL,
    UNIT_COL,
    SPLIT_COL,
    REGRESSION_TARGET,
    *TARGET_COLUMNS,
]


def log(message: str) -> None:
    print(message, flush=True)


def assert_required_columns(df: pd.DataFrame, node_name: str) -> None:
    required = [DT_COL, UNIT_COL, REGRESSION_TARGET, *TARGET_COLUMNS]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{node_name}: отсутствуют обязательные столбцы: {missing}")


def normalize_basic_columns(df: pd.DataFrame, node_name: str) -> pd.DataFrame:
    df = df.copy()

    df[DT_COL] = pd.to_datetime(df[DT_COL], errors="coerce")
    df[UNIT_COL] = pd.to_numeric(df[UNIT_COL], errors="coerce")

    before = len(df)
    df = df.dropna(subset=[DT_COL, UNIT_COL])
    df = df[df[UNIT_COL].isin([1, 2, 3, 4])]
    dropped = before - len(df)
    if dropped:
        log(f"{node_name}: удалено строк с некорректными DT/N_Hosokawa: {dropped:,}".replace(",", " "))

    df[UNIT_COL] = df[UNIT_COL].astype("int8")

    for col in TARGET_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int8")

    df[REGRESSION_TARGET] = pd.to_numeric(df[REGRESSION_TARGET], errors="coerce")

    df["__row_order"] = np.arange(len(df), dtype="int64")
    df = df.sort_values([UNIT_COL, DT_COL, "__row_order"], kind="mergesort").drop(columns="__row_order")
    df = df.reset_index(drop=True)

    return df


def add_split_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[SPLIT_COL] = np.where(df[DT_COL] < TRAIN_END_DATE, "train", "test")
    return df


def get_candidate_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = set(PROTECTED_COLUMNS)
    return [col for col in df.columns if col not in excluded]


def remove_bad_features(df: pd.DataFrame, node_name: str) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    df = df.copy()
    train_mask = df[SPLIT_COL] == "train"

    candidate_cols = get_candidate_feature_columns(df)
    numeric_candidate_cols = [col for col in candidate_cols if pd.api.types.is_numeric_dtype(df[col])]

    dropped_non_numeric = [col for col in candidate_cols if col not in numeric_candidate_cols]

    dropped_missing: list[str] = []
    dropped_constant: list[str] = []

    for col in numeric_candidate_cols:
        missing_share = float(df.loc[train_mask, col].isna().mean())
        if missing_share > MAX_MISSING_SHARE:
            dropped_missing.append(col)

    remaining = [col for col in numeric_candidate_cols if col not in dropped_missing]

    for col in remaining:
        nunique = int(df.loc[train_mask, col].nunique(dropna=True))
        if nunique <= 1:
            dropped_constant.append(col)

    to_drop = dropped_non_numeric + dropped_missing + dropped_constant
    if to_drop:
        df = df.drop(columns=to_drop)

    log(f"{node_name}: удалено нечисловых признаков: {len(dropped_non_numeric)}")
    log(f"{node_name}: удалено почти пустых признаков: {len(dropped_missing)}")
    log(f"{node_name}: удалено константных признаков: {len(dropped_constant)}")

    return df, {
        "dropped_non_numeric": dropped_non_numeric,
        "dropped_missing": dropped_missing,
        "dropped_constant": dropped_constant,
    }


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["dt_hour"] = df[DT_COL].dt.hour.astype("int8")
    df["dt_dayofweek"] = df[DT_COL].dt.dayofweek.astype("int8")
    df["dt_month"] = df[DT_COL].dt.month.astype("int8")
    df["dt_is_weekend"] = df["dt_dayofweek"].isin([5, 6]).astype("int8")

    first_dt_by_unit = df.groupby(UNIT_COL, observed=True)[DT_COL].transform("min")
    df["hours_from_unit_start"] = (df[DT_COL] - first_dt_by_unit).dt.total_seconds() / 3600.0

    prev_dt = df.groupby(UNIT_COL, observed=True)[DT_COL].shift(1)
    df["hours_from_prev_row"] = (df[DT_COL] - prev_dt).dt.total_seconds() / 3600.0
    df["hours_from_prev_row"] = df["hours_from_prev_row"].clip(lower=0)

    return df


def get_signal_columns(df: pd.DataFrame) -> list[str]:
    excluded = set(PROTECTED_COLUMNS)
    excluded.update(
        {
            "Regim",
            "dt_hour",
            "dt_dayofweek",
            "dt_month",
            "dt_is_weekend",
            "hours_from_unit_start",
            "hours_from_prev_row",
        }
    )

    signal_cols: list[str] = []
    for col in df.columns:
        if col in excluded:
            continue
        if col.endswith("_was_missing"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            signal_cols.append(col)
    return signal_cols


def add_missing_indicators(df: pd.DataFrame, signal_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in signal_cols:
        df[f"{col}_was_missing"] = df[col].isna().astype("int8")
    return df


def add_lag_diff_features(df: pd.DataFrame, signal_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    grouped = df.groupby(UNIT_COL, observed=True, sort=False)

    for col in signal_cols:
        lag_col = f"{col}_lag1"
        diff_col = f"{col}_diff1"
        df[lag_col] = grouped[col].shift(1)
        df[diff_col] = df[col] - df[lag_col]

    return df


def add_rolling_features(df: pd.DataFrame, signal_cols: list[str]) -> pd.DataFrame:
    if not MAKE_ROLLING_FEATURES:
        return df

    df = df.copy()
    grouped = df.groupby(UNIT_COL, observed=True, sort=False)

    for col in signal_cols:
        for window in ROLLING_WINDOWS_ROWS:
            mean_name = f"{col}_roll{window}_mean"
            std_name = f"{col}_roll{window}_std"
            df[mean_name] = grouped[col].transform(lambda s: s.rolling(window=window, min_periods=2).mean())
            df[std_name] = grouped[col].transform(lambda s: s.rolling(window=window, min_periods=2).std())

    return df


def fill_missing_values(df: pd.DataFrame, node_name: str) -> tuple[pd.DataFrame, dict[str, float]]:
    df = df.copy()
    train_mask = df[SPLIT_COL] == "train"

    fill_cols = [
        col
        for col in df.columns
        if col not in [DT_COL, SPLIT_COL, REGRESSION_TARGET]
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    missing_before = int(df[fill_cols].isna().sum().sum())

    df[fill_cols] = df.groupby(UNIT_COL, observed=True, sort=False)[fill_cols].ffill(limit=FFILL_LIMIT)

    medians = df.loc[train_mask, fill_cols].median(numeric_only=True)
    medians = medians.fillna(0)
    df[fill_cols] = df[fill_cols].fillna(medians)

    missing_after = int(df[fill_cols].isna().sum().sum())
    log(f"{node_name}: пропусков в признаках до заполнения: {missing_before:,}".replace(",", " "))
    log(f"{node_name}: пропусков в признаках после заполнения: {missing_after:,}".replace(",", " "))

    return df, {str(k): float(v) for k, v in medians.to_dict().items()}


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in df.columns:
        if col in [DT_COL, SPLIT_COL]:
            continue
        if col in TARGET_COLUMNS:
            df[col] = df[col].astype("int8")
            continue
        if col == UNIT_COL:
            df[col] = df[col].astype("int8")
            continue
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype("float32")

    return df


def build_summary(df: pd.DataFrame, node_name: str, kept_columns: list[str], dropped: dict[str, list[str]]) -> dict:
    train_df = df[df[SPLIT_COL] == "train"]
    test_df = df[df[SPLIT_COL] == "test"]

    summary = {
        "node": node_name,
        "train_end_date_exclusive": str(TRAIN_END_DATE.date()),
        "rows_total": int(len(df)),
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "dt_min": str(df[DT_COL].min()),
        "dt_max": str(df[DT_COL].max()),
        "columns_total": int(len(df.columns)),
        "kept_columns": kept_columns,
        "dropped_columns": dropped,
        "target_positive_counts_total": {col: int(df[col].sum()) for col in TARGET_COLUMNS},
        "target_positive_counts_train": {col: int(train_df[col].sum()) for col in TARGET_COLUMNS},
        "target_positive_counts_test": {col: int(test_df[col].sum()) for col in TARGET_COLUMNS},
        "time_to_next_event_hours_nan_total": int(df[REGRESSION_TARGET].isna().sum()),
        "time_to_next_event_hours_nan_train": int(train_df[REGRESSION_TARGET].isna().sum()),
        "time_to_next_event_hours_nan_test": int(test_df[REGRESSION_TARGET].isna().sum()),
    }
    return summary


def prepare_one_dataset(node_name: str, input_path: Path, output_cfg: dict[str, Path]) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Не найден входной файл: {input_path}")

    log("\n" + "=" * 80)
    log(f"Подготовка датасета: {node_name}")
    log(f"Входной файл: {input_path}")

    df = pd.read_parquet(input_path)
    log(f"Исходный размер: {df.shape}")

    assert_required_columns(df, node_name)
    df = normalize_basic_columns(df, node_name)
    df = add_split_column(df)

    split_counts = df[SPLIT_COL].value_counts(dropna=False).to_dict()
    log(f"Разбиение по времени: {split_counts}")

    df, dropped = remove_bad_features(df, node_name)

    df = add_time_features(df)
    signal_cols = get_signal_columns(df)
    log(f"{node_name}: сигналов для инженерных признаков: {len(signal_cols)}")

    df = add_missing_indicators(df, signal_cols)
    df = add_lag_diff_features(df, signal_cols)
    df = add_rolling_features(df, signal_cols)
    df, medians = fill_missing_values(df, node_name)
    df = optimize_dtypes(df)

    kept_columns = list(df.columns)
    summary = build_summary(df, node_name, kept_columns, dropped)
    summary["fill_medians_train"] = medians
    summary["make_rolling_features"] = MAKE_ROLLING_FEATURES
    summary["rolling_windows_rows"] = ROLLING_WINDOWS_ROWS if MAKE_ROLLING_FEATURES else []

    prepared_path = output_cfg["prepared"]
    train_path = output_cfg["train"]
    test_path = output_cfg["test"]
    report_path = output_cfg["report"]

    df.to_parquet(prepared_path, index=False, compression="zstd")
    df[df[SPLIT_COL] == "train"].to_parquet(train_path, index=False, compression="zstd")
    df[df[SPLIT_COL] == "test"].to_parquet(test_path, index=False, compression="zstd")

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log(f"Готовый общий файл: {prepared_path}")
    log(f"Train-файл: {train_path}")
    log(f"Test-файл: {test_path}")
    log(f"Отчет подготовки: {report_path}")
    log(f"Итоговый размер: {df.shape}")

    del df


def main() -> None:
    log("Этап 4_3: подготовка размеченных датасетов под моделирование")
    log(f"Папка processed: {PROCESSED_DIR}")
    log(f"Граница train/test: DT < {TRAIN_END_DATE.date()} -> train, иначе test")

    for node_name, input_path in INPUTS.items():
        prepare_one_dataset(node_name, input_path, OUTPUTS[node_name])

    log("\nВсе датасеты подготовлены.")


if __name__ == "__main__":
    main()
