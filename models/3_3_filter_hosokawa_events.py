from pathlib import Path
import re
import sys
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
TOIR_DIR = PROJECT_ROOT / "../toir_hosokawa"

DEFAULT_INPUT = TOIR_DIR / "jtiny_merged_clean_from_2020.xlsx"
DEFAULT_OUTPUT = TOIR_DIR / "jtiny_hosokawa_events.xlsx"


SEARCH_COLUMNS = [
    "Наименование оборудования",
    "Вид меропр",
    "Краткое описание отказа (причины простоя)",
    "Краткое описание  выполненных работ",
    "Затраченный ЗиП (наименование, тип/марка, кат. №, кол-во)",
    "Комментарий эксперта",
    "Примечание",
]


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip().lower()
    text = text.replace("ё", "е")
    text = re.sub(r"\s+", " ", text)
    return text


def make_search_text(row: pd.Series, columns: list[str]) -> str:
    parts = []
    for col in columns:
        if col in row.index:
            parts.append(normalize_text(row[col]))
    return " | ".join(parts)


def is_hosokawa_event(text: str) -> bool:
    patterns = [
        r"хосокава",
        r"hosokawa",
        r"\bhos\b",
        r"хос\s*[-№]?\s*[1-4]",
    ]

    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def extract_hosokawa_number(text: str):
    patterns = [
        r"хосокава\s*[-№]?\s*([1-4])",
        r"hosokawa\s*[-№]?\s*([1-4])",
        r"\bhos\s*[-№]?\s*([1-4])",
        r"хос\s*[-№]?\s*([1-4])",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))

    return pd.NA


def classify_event_class_by_vid(vid_value) -> str:
    vid = normalize_text(vid_value)

    if not vid:
        return "Прочее"

    if "предложение по улучшению" in vid:
        return "Прочее"

    if "переход" in vid and "зачист" in vid:
        return "Зачистка"

    if "дополнительные работы" in vid:
        return "Работы"

    if "замечание по работе" in vid:
        return "Неисправность/отказ"

    if "неплановое техобслуживание" in vid:
        return "Неисправность/отказ"

    if "неплановый" in vid and "аварийный" in vid and "ремонт" in vid:
        return "Неисправность/отказ"

    if "неплан" in vid and "ремонт" in vid:
        return "Неисправность/отказ"

    return "Прочее"


def move_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns.tolist()

    priority_cols = [
        "N_Hosokawa",
        "event_class",
    ]

    for col in priority_cols:
        if col in cols:
            cols.remove(col)

    if "Наименование оборудования" in cols:
        insert_pos = cols.index("Наименование оборудования") + 1
    else:
        insert_pos = 1

    for offset, col in enumerate(priority_cols):
        if col in df.columns:
            cols.insert(insert_pos + offset, col)

    return df[cols]


def autofit_worksheet(ws):
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions

    for column_cells in ws.columns:
        max_len = 0
        col_letter = column_cells[0].column_letter

        for cell in column_cells[:200]:
            if cell.value is not None:
                max_len = max(max_len, len(str(cell.value)))

        ws.column_dimensions[col_letter].width = min(max(max_len + 2, 10), 45)


def main():
    input_file = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_INPUT
    output_file = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else DEFAULT_OUTPUT

    if not input_file.exists():
        raise FileNotFoundError(f"Файл не найден: {input_file}")

    print("=" * 80)
    print("ФИЛЬТРАЦИЯ СОБЫТИЙ ПО HOSOKAWA + КЛАССИФИКАЦИЯ ПО 'ВИД МЕРОПР'")
    print("=" * 80)
    print(f"Входной файл: {input_file}")

    df = pd.read_excel(input_file)

    print(f"\nРазмер исходной таблицы: {df.shape}")

    existing_search_columns = [col for col in SEARCH_COLUMNS if col in df.columns]

    if not existing_search_columns:
        raise ValueError(
            "Не найдено ни одного текстового столбца для поиска событий Hosokawa."
        )

    if "Вид меропр" not in df.columns:
        raise ValueError("Не найден столбец 'Вид меропр'.")

    print("\nСтолбцы, по которым выполняется поиск Hosokawa:")
    for col in existing_search_columns:
        print(f"  - {col}")

    df["__search_text__"] = df.apply(
        lambda row: make_search_text(row, existing_search_columns),
        axis=1,
    )

    hos_mask = df["__search_text__"].apply(is_hosokawa_event)
    df_hos = df[hos_mask].copy()

    print(f"\nНайдено событий Hosokawa: {len(df_hos)}")

    df_hos["N_Hosokawa"] = df_hos["__search_text__"].apply(extract_hosokawa_number)
    df_hos["N_Hosokawa"] = df_hos["N_Hosokawa"].astype("Int64")

    df_hos["event_class"] = df_hos["Вид меропр"].apply(classify_event_class_by_vid)

    df_hos = move_columns(df_hos)

    print("\nРаспределение по установкам Hosokawa:")
    print(df_hos["N_Hosokawa"].value_counts(dropna=False).sort_index())

    print("\nРаспределение по типам событий:")
    print(df_hos["event_class"].value_counts(dropna=False))

    print("\nРаспределение по установкам и типам событий:")
    print(
        df_hos.pivot_table(
            index="N_Hosokawa",
            columns="event_class",
            values=df_hos.columns[0],
            aggfunc="count",
            fill_value=0,
        )
    )

    df_hos_result = df_hos.drop(columns=["__search_text__"])

    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nСохраняю результат: {output_file}")

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df_hos_result.to_excel(writer, index=False, sheet_name="hosokawa_events")

        summary_class = (
            df_hos_result["event_class"]
            .value_counts(dropna=False)
            .reset_index()
        )
        summary_class.columns = ["event_class", "count"]
        summary_class.to_excel(writer, index=False, sheet_name="summary_by_event_class")

        summary_unit_class = (
            df_hos_result.pivot_table(
                index="N_Hosokawa",
                columns="event_class",
                values=df_hos_result.columns[0],
                aggfunc="count",
                fill_value=0,
            )
            .reset_index()
        )
        summary_unit_class.to_excel(writer, index=False, sheet_name="summary_unit_event")

        for ws in writer.book.worksheets:
            autofit_worksheet(ws)

    print("\n" + "=" * 80)
    print("ГОТОВО")
    print("=" * 80)
    print(f"Итоговый файл: {output_file}")
    print("\nВ файле будут листы:")
    print("  - hosokawa_events")
    print("  - summary_by_event_class")
    print("  - summary_unit_event")


if __name__ == "__main__":
    main()
