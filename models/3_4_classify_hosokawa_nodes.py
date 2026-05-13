from pathlib import Path
import re
import sys
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
TOIR_DIR = PROJECT_ROOT / "../toir_hosokawa"

DEFAULT_INPUT = TOIR_DIR / "jtiny_hosokawa_events.xlsx"
DEFAULT_OUTPUT = TOIR_DIR / "jtiny_hosokawa_events_by_node.xlsx"


SEARCH_COLUMNS = [
    "Наименование оборудования",
    "Краткое описание отказа (причины простоя)",
    "Краткое описание  выполненных работ",
    "Затраченный ЗиП (наименование, тип/марка, кат. №, кол-во)",
    "Комментарий эксперта",
    "Примечание",
]


COMPACTOR_PATTERNS = {
    "компактор": r"\bкомпактор\w*",
    "compact": r"\bcompact\w*",
    "шнек": r"\bшнек\w*",
    "валки": r"\bвалк\w*",
    "ролик/ролики": r"\bролик\w*",
    "предуплотнитель": r"\bпред[\s-]?уплот\w*",
    "уплотнитель": r"\bуплотн\w*",
    "прессование": r"\bпрессован\w*",
    "сжатие": r"\bсжат\w*",
}


MILL_PATTERNS = {
    "мельница": r"\bмельниц\w*",
    "мелница": r"\bмелниц\w*",
    "mill": r"\bmill\b|\bhammer\s*mill\b",
    "молотковая": r"\bмолотк\w*",
    "ротор мельницы": r"\bротор\w*",
    "сито": r"\bсит\w*",
    "помол": r"\bпомол\w*",
    "измельчение": r"\bизмельч\w*",
}


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower()
    text = text.replace("ё", "е")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def make_search_text(row: pd.Series, columns: list[str]) -> str:
    parts = []
    for col in columns:
        if col in row.index:
            parts.append(normalize_text(row[col]))
    return " | ".join(parts)


def find_matches(text: str, patterns: dict[str, str]) -> list[str]:
    matches = []
    for label, pattern in patterns.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            matches.append(label)
    return matches


def classify_event(text: str) -> dict:
    compactor_matches = find_matches(text, COMPACTOR_PATTERNS)
    mill_matches = find_matches(text, MILL_PATTERNS)

    has_compactor = len(compactor_matches) > 0
    has_mill = len(mill_matches) > 0

    if has_compactor and has_mill:
        return {
            "nodes": ["Компактор (шнек)", "Мельница"],
            "assignment_type": "both",
        }

    if has_compactor:
        return {
            "nodes": ["Компактор (шнек)"],
            "assignment_type": "direct_compactor",
        }

    if has_mill:
        return {
            "nodes": ["Мельница"],
            "assignment_type": "direct_mill",
        }

    return {
        "nodes": ["Прочее"],
        "assignment_type": "other",
    }


def expand_events_by_node(df: pd.DataFrame, search_text: pd.Series) -> pd.DataFrame:
    rows = []

    for original_idx, row in df.iterrows():
        classification = classify_event(search_text.loc[original_idx])

        for node in classification["nodes"]:
            new_row = row.copy()
            new_row["node"] = node
            new_row["node_assignment_type"] = classification["assignment_type"]
            rows.append(new_row)

    return pd.DataFrame(rows)


def move_node_columns_to_front(df: pd.DataFrame) -> pd.DataFrame:
    node_cols = [
        "node",
        "node_assignment_type",
    ]

    cols = df.columns.tolist()

    for col in node_cols:
        if col in cols:
            cols.remove(col)

    if "N_Hosokawa" in cols:
        insert_pos = cols.index("N_Hosokawa") + 1
    elif "Наименование оборудования" in cols:
        insert_pos = cols.index("Наименование оборудования") + 1
    else:
        insert_pos = 1

    for offset, col in enumerate(node_cols):
        if col in df.columns:
            cols.insert(insert_pos + offset, col)

    return df[cols]


def build_summary_by_unit_node(df: pd.DataFrame) -> pd.DataFrame:
    if "N_Hosokawa" not in df.columns:
        summary = df["node"].value_counts(dropna=False).reset_index()
        summary.columns = ["node", "count"]
        return summary

    pivot = (
        df.pivot_table(
            index="N_Hosokawa",
            columns="node",
            values=df.columns[0],
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
    )

    node_columns = [col for col in pivot.columns if col != "N_Hosokawa"]

    total_row = {"N_Hosokawa": "Итого"}
    for col in node_columns:
        total_row[col] = pivot[col].sum()

    pivot = pd.concat([pivot, pd.DataFrame([total_row])], ignore_index=True)

    return pivot


def build_summary_unit_event(source_df: pd.DataFrame) -> pd.DataFrame:
    if "N_Hosokawa" not in source_df.columns or "event_class" not in source_df.columns:
        return pd.DataFrame()

    pivot = (
        source_df.pivot_table(
            index="N_Hosokawa",
            columns="event_class",
            values=source_df.columns[0],
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
    )

    event_columns = [col for col in pivot.columns if col != "N_Hosokawa"]

    pivot["Итого"] = pivot[event_columns].sum(axis=1)

    total_row = {"N_Hosokawa": "Итого"}
    for col in event_columns:
        total_row[col] = pivot[col].sum()

    total_row["Итого"] = pivot["Итого"].sum()

    pivot = pd.concat([pivot, pd.DataFrame([total_row])], ignore_index=True)

    return pivot


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


def save_excel(df_expanded: pd.DataFrame, source_df: pd.DataFrame, output_file: Path):
    output_file.parent.mkdir(parents=True, exist_ok=True)

    summary_by_unit_node = build_summary_by_unit_node(df_expanded)
    summary_unit_event = build_summary_unit_event(source_df)

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df_expanded.to_excel(writer, index=False, sheet_name="all_hosokawa_events")

        df_expanded[df_expanded["node"] == "Компактор (шнек)"].to_excel(
            writer, index=False, sheet_name="compactor"
        )

        df_expanded[df_expanded["node"] == "Мельница"].to_excel(
            writer, index=False, sheet_name="mill"
        )

        df_expanded[df_expanded["node"] == "Прочее"].to_excel(
            writer, index=False, sheet_name="other"
        )

        summary_by_unit_node.to_excel(
            writer, index=False, sheet_name="summary_by_unit_node"
        )

        if summary_unit_event is not None and not summary_unit_event.empty:
            summary_unit_event.to_excel(
                writer, index=False, sheet_name="summary_unit_event"
            )

        for ws in writer.book.worksheets:
            autofit_worksheet(ws)


def main():
    input_file = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_INPUT
    output_file = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else DEFAULT_OUTPUT

    if not input_file.exists():
        raise FileNotFoundError(f"Файл не найден: {input_file}")

    print("=" * 80)
    print("РАЗНЕСЕНИЕ СОБЫТИЙ HOSOKAWA ПО УЗЛАМ")
    print("=" * 80)
    print(f"Входной файл: {input_file}")

    df = pd.read_excel(input_file, sheet_name=0)

    print(f"\nРазмер исходной таблицы: {df.shape}")

    existing_search_columns = [col for col in SEARCH_COLUMNS if col in df.columns]

    if not existing_search_columns:
        raise ValueError("Не найдено текстовых столбцов для классификации событий.")

    print("\nСтолбцы для поиска ключевых слов:")
    for col in existing_search_columns:
        print(f"  - {col}")

    search_text = df.apply(
        lambda row: make_search_text(row, existing_search_columns),
        axis=1,
    )

    df_expanded = expand_events_by_node(df, search_text)
    df_expanded = move_node_columns_to_front(df_expanded)

    print("\nРазмер после разнесения по узлам:")
    print(df_expanded.shape)

    print("\nРаспределение событий по узлам:")
    print(df_expanded["node"].value_counts(dropna=False))

    print("\nРаспределение по типу назначения узла:")
    print(df_expanded["node_assignment_type"].value_counts(dropna=False))

    if "N_Hosokawa" in df_expanded.columns:
        print("\nРаспределение по установкам и узлам:")
        print(
            df_expanded.pivot_table(
                index="N_Hosokawa",
                columns="node",
                values=df_expanded.columns[0],
                aggfunc="count",
                fill_value=0,
            )
        )

    if "№ обр" in df_expanded.columns:
        both_original_count = (
            df_expanded[df_expanded["node_assignment_type"] == "both"]
            .drop_duplicates(subset=["№ обр"])
            .shape[0]
        )
    else:
        both_original_count = (
            df_expanded[df_expanded["node_assignment_type"] == "both"]
            .drop_duplicates()
            .shape[0]
        )

    print(f"\nИсходных событий, разнесенных в оба узла: {both_original_count}")
    print(
        "Важно: события с node_assignment_type='both' представлены двумя строками — "
        "одна для компактора, одна для мельницы."
    )

    print(f"\nСохраняю результат: {output_file}")
    save_excel(df_expanded, df, output_file)

    print("\n" + "=" * 80)
    print("ГОТОВО")
    print("=" * 80)
    print(f"Итоговый файл: {output_file}")
    print("\nЛисты в итоговом файле:")
    print("  - all_hosokawa_events")
    print("  - compactor")
    print("  - mill")
    print("  - other")
    print("  - summary_by_unit_node")
    print("  - summary_unit_event")


if __name__ == "__main__":
    main()
