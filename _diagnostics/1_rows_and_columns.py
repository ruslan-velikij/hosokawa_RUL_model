from pathlib import Path
from collections import Counter
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PARQUET = PROJECT_ROOT / "../processed" / "all_data_2020_2025.parquet"
BATCH_SIZE = 250_000


def normalize_name(name: str) -> str:
    return name.lower().replace(" ", "").replace("-", "_")


def group_columns(columns: list[str]) -> dict[str, list[str]]:
    groups = {
        "Компактор_шнек": [],
        "Мельница": [],
        "Дробилка_гранулятор_шлюзы_дозатор": [],
        "Режимы_и_статусы": [],
        "Наработка_и_время": [],
        "Общесистемные_или_спорные": [],
    }

    for col in columns:
        c = normalize_name(col)

        if col == "DT" or c.startswith("vrema_"):
            groups["Наработка_и_время"].append(col)

        elif (
            "shneck" in c
            or "compactor" in c
            or "kompakt" in c
            or "valki" in c
            or "pred_uplotn" in c
            or "pre_comp" in c
            or "sjatia" in c
        ):
            groups["Компактор_шнек"].append(col)

        elif (
            "melnic" in c
            or "mill" in c
            or "podshipnik" in c
            or "hammer_mill" in c
            or "shluza_melnici" in c
            or "ventilatora_melnici" in c
        ):
            groups["Мельница"].append(col)

        elif (
            "drobilk" in c
            or "granulator" in c
            or "shluz" in c
            or "dozator" in c
            or "clapan" in c
            or "fluodisation" in c
            or "filtr" in c
        ):
            groups["Дробилка_гранулятор_шлюзы_дозатор"].append(col)

        elif (
            "regim" in c
            or c.startswith("stat.")
            or col == "N_Hosokawa"
        ):
            groups["Режимы_и_статусы"].append(col)

        else:
            groups["Общесистемные_или_спорные"].append(col)

    return groups


def find_suspicious_columns(columns: list[str]) -> list[str]:
    suspicious = []

    for col in columns:
        c = normalize_name(col)

        if "unnamed" in c:
            suspicious.append(f"{col} -> похоже на случайно сохранённый индекс")

        if "\\" in col:
            suspicious.append(f"{col} -> техническое имя из SCADA/источника, смысл надо уточнять")

        if col == "N_Hosokawa":
            suspicious.append(f"{col} -> похоже на идентификатор/счётчик/номер, смысл надо проверить")

    if "Perepad_davleniya_filtra" in columns and "Perepad_davl_filtra" in columns:
        suspicious.append(
            "Perepad_davleniya_filtra и Perepad_davl_filtra -> возможно дублирующие или почти дублирующие сигналы"
        )

    col_set = set(columns)
    for col in columns:
        if col.endswith("_ssd"):
            base = col[:-4]
            if base in col_set:
                suspicious.append(f"{base} и {col} -> пара основной/ssd версии сигнала")

    return suspicious


def format_series(series: pd.Series, top_n: int | None = None) -> str:
    if top_n is not None:
        series = series.head(top_n)

    lines = []
    for idx, val in series.items():
        lines.append(f"  {idx}: {val}")
    return "\n".join(lines)


def iter_df_batches(parquet_file: pq.ParquetFile, columns=None, batch_size=BATCH_SIZE):
    for batch in parquet_file.iter_batches(columns=columns, batch_size=batch_size):
        yield batch.to_pandas(ignore_metadata=True)


def analyze_parquet(file_path: Path) -> None:
    if not file_path.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    print("=" * 80)
    print("БЫСТРЫЙ ТЕХОСМОТР PARQUET")
    print("=" * 80)
    print(f"Файл: {file_path}")

    parquet_file = pq.ParquetFile(file_path)
    total_rows = parquet_file.metadata.num_rows
    columns = parquet_file.schema.names
    total_cols = len(columns)

    print(f"\nФорма таблицы (по метаданным): ({total_rows}, {total_cols})")

    print("\nСтолбцы:")
    for col in columns:
        print(f"  - {col}")

    print("\n[1/4] Считаю пропуски по столбцам...")
    na_counts = pd.Series(0, index=columns, dtype="int64")

    for df_batch in iter_df_batches(parquet_file):
        batch_na = df_batch.isna().sum().reindex(columns, fill_value=0)
        na_counts = na_counts.add(batch_na, fill_value=0).astype("int64")

    na_counts = na_counts.sort_values(ascending=False)

    print("[2/4] Анализирую DT...")
    dt_min = None
    dt_max = None
    bad_dt_count = 0
    duplicate_dt_count = None

    if "DT" in columns:
        dt_chunks = []

        for df_batch in iter_df_batches(parquet_file, columns=["DT"]):
            if "DT" not in df_batch.columns:
                continue

            dt = pd.to_datetime(df_batch["DT"], errors="coerce")
            bad_dt_count += int(dt.isna().sum())

            dt_valid = dt.dropna().astype("int64").to_numpy()
            if len(dt_valid) > 0:
                dt_chunks.append(dt_valid)

        if dt_chunks:
            all_dt = np.concatenate(dt_chunks)
            dt_min = pd.to_datetime(all_dt.min())
            dt_max = pd.to_datetime(all_dt.max())
            unique_dt = np.unique(all_dt).size
            duplicate_dt_count = int(all_dt.size - unique_dt)
    else:
        print("  ВНИМАНИЕ: столбец DT отсутствует в parquet")

    print("[3/4] Считаю режимы Regim...")
    regim_counts = pd.Series(dtype="int64")

    if "Regim" in columns:
        regim_counter = Counter()

        for df_batch in iter_df_batches(parquet_file, columns=["Regim"]):
            if "Regim" not in df_batch.columns:
                continue

            regim_series = df_batch["Regim"].fillna("<<NA>>").astype(str)
            regim_counter.update(regim_series.value_counts(dropna=False).to_dict())

        regim_counts = pd.Series(regim_counter).sort_values(ascending=False)

    print("[4/4] Группирую столбцы по смыслу...")
    groups = group_columns(columns)
    suspicious = find_suspicious_columns(columns)

    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТ")
    print("=" * 80)

    print("\n1) БАЗОВАЯ ИНФОРМАЦИЯ")
    print(f"  Строк: {total_rows}")
    print(f"  Столбцов: {total_cols}")

    print("\n2) ВРЕМЕННОЙ ДИАПАЗОН")
    print(f"  DT min: {dt_min}")
    print(f"  DT max: {dt_max}")
    print(f"  Некорректных/пустых DT после преобразования: {bad_dt_count}")
    print(f"  Дубликатов по DT: {duplicate_dt_count}")

    print("\n3) ПРОПУСКИ ПО СТОЛБЦАМ (топ-20)")
    print(format_series(na_counts, top_n=20))

    print("\n4) РЕЖИМЫ Regim")
    if not regim_counts.empty:
        print(format_series(regim_counts))
    else:
        print("  Столбец Regim не найден или не прочитался")

    print("\n5) СМЫСЛОВЫЕ ГРУППЫ СТОЛБЦОВ")
    for group_name, group_cols in groups.items():
        print(f"\n  {group_name}:")
        if group_cols:
            for col in group_cols:
                print(f"    - {col}")
        else:
            print("    (нет столбцов)")

    print("\n6) МУСОРНЫЕ / СПОРНЫЕ / ТРЕБУЮЩИЕ ПРОВЕРКИ СТОЛБЦЫ")
    if suspicious:
        for item in suspicious:
            print(f"  - {item}")
    else:
        print("  Явно подозрительных столбцов не найдено")

    print("\n" + "=" * 80)
    print("КОРОТКИЙ ВЫВОД")
    print("=" * 80)
    print("  - Главные ответы для первого техосмотра получены.")
    print("  - Дальше можно переходить к Excel-журналам и простому EDA по ключевым сигналам.")
    print("  - Следующий логичный шаг: быстрая инвентаризация файлов из папки toir_hosokawa.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        parquet_path = Path(sys.argv[1]).resolve()
    else:
        parquet_path = DEFAULT_PARQUET

    analyze_parquet(parquet_path)
