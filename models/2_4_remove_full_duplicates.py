from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


OUTPUT_SUFFIX = "_no_duplicates"


def build_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}{OUTPUT_SUFFIX}{input_path.suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Удаление полных дублей из parquet-файла."
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="input_path",
        type=Path,
        default=None,
        help="Путь к исходному parquet-файлу.",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        type=Path,
        default=None,
        help="Путь к выходному parquet-файлу. Если не указан, имя формируется автоматически.",
    )
    parser.add_argument(
        "--ignore-index",
        action="store_true",
        help=(
            "Не учитывать индекс при поиске дублей. По умолчанию индекс учитывается, "
            "если он содержит DT или другой смысловой индекс."
        ),
    )
    return parser.parse_args()


def get_duplicate_mask(df: pd.DataFrame, ignore_index: bool) -> pd.Series:
    if ignore_index:
        return df.duplicated()

    has_meaningful_index = not isinstance(df.index, pd.RangeIndex) or df.index.name is not None
    if has_meaningful_index:
        return df.reset_index().duplicated()

    return df.duplicated()


def main() -> None:
    args = parse_args()

    if args.input_path is None:
        raise SystemExit(
            "Ошибка: файл не выбран. Укажите путь к parquet через -i или --input.\n"
            "Пример: python 2_4_remove_full_duplicates.py "
            "--input processed/all_data_2020_2025_with_ssd.parquet"
        )

    input_file = args.input_path.expanduser().resolve()

    if not input_file.exists():
        raise FileNotFoundError(f"Файл не найден: {input_file}")
    if input_file.suffix.lower() != ".parquet":
        raise ValueError(f"Ожидался parquet-файл, получено: {input_file}")

    output_file = args.output_path.expanduser().resolve() if args.output_path else build_output_path(input_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("УДАЛЕНИЕ ПОЛНЫХ ДУБЛЕЙ ИЗ PARQUET")
    print("=" * 80)
    print(f"Исходный файл: {input_file}")
    print(f"Выходной файл: {output_file}")

    print("\n[1/4] Читаю parquet...")
    df = pd.read_parquet(input_file)

    print(f"Форма до очистки: {df.shape}")
    print(f"Имя индекса: {df.index.name}")
    print(f"Тип индекса: {type(df.index).__name__}")

    print("\n[2/4] Считаю полные дубли...")
    duplicate_mask = get_duplicate_mask(df, ignore_index=args.ignore_index)
    full_duplicates = int(duplicate_mask.sum())
    print(f"Полных дублей найдено: {full_duplicates:,}".replace(",", " "))

    print("\n[3/4] Удаляю полные дубли...")
    df_clean = df.loc[~duplicate_mask.to_numpy()].copy()

    removed_rows = len(df) - len(df_clean)

    print(f"Удалено строк: {removed_rows:,}".replace(",", " "))
    print(f"Форма после очистки: {df_clean.shape}")

    print("\n[4/4] Сохраняю очищенный parquet...")
    df_clean.to_parquet(output_file, compression="zstd")

    print("\n" + "=" * 80)
    print("ГОТОВО")
    print("=" * 80)
    print(f"Сохранено в: {output_file}")
    print("\nКороткий вывод:")
    print("- Полные дубли удалены.")
    print("- Повторы по DT с разными значениями НЕ тронуты.")
    print("- Выходное имя сформировано автоматически, если --output не был указан.")


if __name__ == "__main__":
    main()
