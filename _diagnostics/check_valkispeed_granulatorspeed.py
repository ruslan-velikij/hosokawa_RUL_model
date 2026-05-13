#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import sys
import pandas as pd


INPUT_PARQUET = Path("../processed/all_data_2020_2025_no_duplicates.parquet")

SIGNALS_TO_CHECK = [
    "ValkiSpeed",
    "GranulatorSpeed",
]

UNIT_COL = "N_Hosokawa"
DT_COL = "DT"

OUT_DIR = Path("processed")
OUT_REPORT_TXT = OUT_DIR / "check_valkispeed_granulatorspeed_report.txt"
OUT_BY_YEAR_CSV = OUT_DIR / "check_valkispeed_granulatorspeed_by_year.csv"
OUT_BY_UNIT_CSV = OUT_DIR / "check_valkispeed_granulatorspeed_by_unit.csv"
OUT_BY_YEAR_UNIT_CSV = OUT_DIR / "check_valkispeed_granulatorspeed_by_year_unit.csv"


def print_and_store(lines: list[str], text: str = "") -> None:
    print(text)
    lines.append(text)


def ensure_dt_column(df: pd.DataFrame) -> pd.DataFrame:
    if DT_COL in df.columns:
        df[DT_COL] = pd.to_datetime(df[DT_COL], errors="coerce")
        return df

    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        # После reset_index имя столбца может быть DT или index
        if DT_COL not in df.columns:
            first_col = df.columns[0]
            df = df.rename(columns={first_col: DT_COL})
        df[DT_COL] = pd.to_datetime(df[DT_COL], errors="coerce")
        return df

    raise ValueError(
        f"Не найден столбец {DT_COL}, и индекс не является DatetimeIndex. "
        "Проверить структуру parquet-файла."
    )


def existing_columns(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        return pf.schema.names
    except Exception:
        # Запасной вариант: читаем 1 строку
        return list(pd.read_parquet(path).head(1).columns)


def add_existing_dt_name(cols: list[str]) -> str | None:
    candidates = [DT_COL, "index", "__index_level_0__"]
    for c in candidates:
        if c in cols:
            return c
    return None


def main() -> None:
    report_lines: list[str] = []

    print_and_store(report_lines, "Проверка признаков ValkiSpeed и GranulatorSpeed")
    print_and_store(report_lines, "=" * 80)
    print_and_store(report_lines, f"Входной файл: {INPUT_PARQUET.resolve()}")

    if not INPUT_PARQUET.exists():
        print_and_store(report_lines, "")
        print_and_store(report_lines, "ОШИБКА: входной parquet-файл не найден.")
        print_and_store(report_lines, "Проверить имя файла в переменной INPUT_PARQUET.")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        OUT_REPORT_TXT.write_text("\n".join(report_lines), encoding="utf-8")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cols = existing_columns(INPUT_PARQUET)
    dt_name_in_file = add_existing_dt_name(cols)

    needed_cols = []
    if dt_name_in_file is not None:
        needed_cols.append(dt_name_in_file)

    for col in [UNIT_COL] + SIGNALS_TO_CHECK:
        if col in cols:
            needed_cols.append(col)

    needed_cols = list(dict.fromkeys(needed_cols))

    print_and_store(report_lines, f"Найдено столбцов в parquet-схеме: {len(cols)}")
    print_and_store(report_lines, f"Будут загружены столбцы: {needed_cols}")

    missing_signals = [c for c in SIGNALS_TO_CHECK if c not in cols]
    if missing_signals:
        print_and_store(report_lines, "")
        print_and_store(report_lines, "ВНИМАНИЕ: в parquet отсутствуют признаки:")
        for c in missing_signals:
            print_and_store(report_lines, f"  - {c}")

    if UNIT_COL not in cols:
        print_and_store(report_lines, "")
        print_and_store(report_lines, f"ВНИМАНИЕ: столбец {UNIT_COL} не найден. Анализ по установкам будет пропущен.")

    if not any(c in cols for c in SIGNALS_TO_CHECK):
        print_and_store(report_lines, "")
        print_and_store(report_lines, "ОШИБКА: ни один из проверяемых сигналов не найден.")
        OUT_REPORT_TXT.write_text("\n".join(report_lines), encoding="utf-8")
        sys.exit(1)

    df = pd.read_parquet(INPUT_PARQUET, columns=needed_cols)
    df = ensure_dt_column(df)

    df["year"] = df[DT_COL].dt.year

    print_and_store(report_lines, "")
    print_and_store(report_lines, "Общая информация")
    print_and_store(report_lines, "-" * 80)
    print_and_store(report_lines, f"Загружено строк: {len(df):,}".replace(",", " "))
    print_and_store(report_lines, f"DT min: {df[DT_COL].min()}")
    print_and_store(report_lines, f"DT max: {df[DT_COL].max()}")
    print_and_store(report_lines, f"Пустых DT: {df[DT_COL].isna().sum():,}".replace(",", " "))

    if UNIT_COL in df.columns:
        print_and_store(report_lines, "")
        print_and_store(report_lines, f"Распределение {UNIT_COL}:")
        unit_counts = df[UNIT_COL].value_counts(dropna=False).sort_index()
        for idx, val in unit_counts.items():
            print_and_store(report_lines, f"  {idx}: {val:,}".replace(",", " "))

    print_and_store(report_lines, "")
    print_and_store(report_lines, "Наличие данных по сигналам")
    print_and_store(report_lines, "-" * 80)

    for sig in SIGNALS_TO_CHECK:
        if sig not in df.columns:
            continue

        non_null = int(df[sig].notna().sum())
        nulls = int(df[sig].isna().sum())
        share = non_null / len(df) * 100 if len(df) else 0

        print_and_store(report_lines, f"{sig}:")
        print_and_store(report_lines, f"  непустых строк: {non_null:,}".replace(",", " "))
        print_and_store(report_lines, f"  пустых строк:    {nulls:,}".replace(",", " "))
        print_and_store(report_lines, f"  доля непустых:   {share:.4f}%")

        if non_null > 0:
            s = df.loc[df[sig].notna(), sig]
            dt_non_null = df.loc[df[sig].notna(), DT_COL]
            print_and_store(report_lines, f"  первый DT с данными: {dt_non_null.min()}")
            print_and_store(report_lines, f"  последний DT с данными: {dt_non_null.max()}")
            print_and_store(report_lines, f"  min: {s.min()}")
            print_and_store(report_lines, f"  median: {s.median()}")
            print_and_store(report_lines, f"  max: {s.max()}")
            print_and_store(report_lines, f"  уникальных непустых значений: {s.nunique(dropna=True):,}".replace(",", " "))
        print_and_store(report_lines, "")

    year_rows = []
    for year, group in df.groupby("year", dropna=False):
        row = {
            "year": year,
            "rows_total": len(group),
        }
        for sig in SIGNALS_TO_CHECK:
            if sig in group.columns:
                row[f"{sig}_non_null"] = int(group[sig].notna().sum())
                row[f"{sig}_non_null_share_pct"] = group[sig].notna().mean() * 100
        year_rows.append(row)

    by_year = pd.DataFrame(year_rows).sort_values("year")
    by_year.to_csv(OUT_BY_YEAR_CSV, index=False, encoding="utf-8-sig")

    print_and_store(report_lines, "")
    print_and_store(report_lines, "Наличие данных по годам")
    print_and_store(report_lines, "-" * 80)
    print_and_store(report_lines, by_year.to_string(index=False))

    if UNIT_COL in df.columns:
        unit_rows = []
        for unit, group in df.groupby(UNIT_COL, dropna=False):
            row = {
                UNIT_COL: unit,
                "rows_total": len(group),
            }
            for sig in SIGNALS_TO_CHECK:
                if sig in group.columns:
                    row[f"{sig}_non_null"] = int(group[sig].notna().sum())
                    row[f"{sig}_non_null_share_pct"] = group[sig].notna().mean() * 100
            unit_rows.append(row)

        by_unit = pd.DataFrame(unit_rows).sort_values(UNIT_COL)
        by_unit.to_csv(OUT_BY_UNIT_CSV, index=False, encoding="utf-8-sig")

        print_and_store(report_lines, "")
        print_and_store(report_lines, "Наличие данных по установкам")
        print_and_store(report_lines, "-" * 80)
        print_and_store(report_lines, by_unit.to_string(index=False))

        year_unit_rows = []
        for (year, unit), group in df.groupby(["year", UNIT_COL], dropna=False):
            row = {
                "year": year,
                UNIT_COL: unit,
                "rows_total": len(group),
            }
            for sig in SIGNALS_TO_CHECK:
                if sig in group.columns:
                    row[f"{sig}_non_null"] = int(group[sig].notna().sum())
                    row[f"{sig}_non_null_share_pct"] = group[sig].notna().mean() * 100
            year_unit_rows.append(row)

        by_year_unit = pd.DataFrame(year_unit_rows).sort_values(["year", UNIT_COL])
        by_year_unit.to_csv(OUT_BY_YEAR_UNIT_CSV, index=False, encoding="utf-8-sig")

        print_and_store(report_lines, "")
        print_and_store(report_lines, "Наличие данных по годам и установкам")
        print_and_store(report_lines, "-" * 80)
        print_and_store(report_lines, by_year_unit.to_string(index=False))

    df_2020 = df[df["year"] == 2020]

    print_and_store(report_lines, "")
    print_and_store(report_lines, "Отдельная проверка 2020 года")
    print_and_store(report_lines, "-" * 80)
    print_and_store(report_lines, f"Строк за 2020 год: {len(df_2020):,}".replace(",", " "))

    for sig in SIGNALS_TO_CHECK:
        if sig not in df_2020.columns:
            continue

        non_null_2020 = int(df_2020[sig].notna().sum())
        share_2020 = non_null_2020 / len(df_2020) * 100 if len(df_2020) else 0
        print_and_store(report_lines, f"{sig} в 2020:")
        print_and_store(report_lines, f"  непустых строк: {non_null_2020:,}".replace(",", " "))
        print_and_store(report_lines, f"  доля непустых:   {share_2020:.4f}%")

        if non_null_2020 > 0:
            dt_2020 = df_2020.loc[df_2020[sig].notna(), DT_COL]
            print_and_store(report_lines, f"  первый DT с данными: {dt_2020.min()}")
            print_and_store(report_lines, f"  последний DT с данными: {dt_2020.max()}")

    OUT_REPORT_TXT.write_text("\n".join(report_lines), encoding="utf-8")

    print()
    print("Готово.")
    print(f"Текстовый отчет: {OUT_REPORT_TXT}")
    print(f"CSV по годам: {OUT_BY_YEAR_CSV}")
    print(f"CSV по установкам: {OUT_BY_UNIT_CSV}")
    print(f"CSV по годам и установкам: {OUT_BY_YEAR_UNIT_CSV}")


if __name__ == "__main__":
    main()
