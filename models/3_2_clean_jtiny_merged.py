from pathlib import Path
import sys
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
TOIR_DIR = PROJECT_ROOT / "../toir_hosokawa"

DEFAULT_INPUT = TOIR_DIR / "jtiny_merged.xlsx"
DEFAULT_OUTPUT = TOIR_DIR / "jtiny_merged_clean_from_2020.xlsx"

DATE_START = pd.Timestamp("2020-10-20")

EXPECTED_HEADERS = [
    "№ обр",
    "Дата/время начала простоя",
    "Наименование оборудования",
    "Корпус, участок",
    "Вид меропр",
    "Краткое описание отказа (причины простоя)",
    "Дата/время окончания простоя",
    "Ф.И.О. мастера",
    "Дата/время начала ремонта",
    "Краткое описание выполненных работ",
    "Затраченный ЗиП (наименование, тип/марка, кат. №, кол-во)",
    "Дата/время окончания ремонта",
    "Специальность ремонтного персонала",
    "Ф.И.О. ремонтного персонала",
    "Время простоя",
    "Время простоя (абс)",
    "Отв за простой",
    "Отметка УТОП",
    "Комментарий эксперта",
    "Простой рем.сл.",
    "Время рем.сл.",
    "Простой производство",
    "Время производство",
    "Невыполнение в кг",
    "Невыполнение в минутах",
    "Примечание",
]


def normalize_cell(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def make_unique_columns(columns):
    result = []
    seen = {}

    for i, col in enumerate(columns):
        col = normalize_cell(col)
        if not col:
            col = f"column_{i + 1}"

        if col not in seen:
            seen[col] = 0
            result.append(col)
        else:
            seen[col] += 1
            result.append(f"{col}_{seen[col]}")

    return result


def build_expected_headers(n_cols):
    headers = EXPECTED_HEADERS.copy()

    if n_cols > len(headers):
        headers.append("__source_file__")

    while len(headers) < n_cols:
        headers.append(f"extra_column_{len(headers) + 1}")

    return headers[:n_cols]


def find_real_header_row(raw):
    for idx in range(min(10, len(raw))):
        values = [normalize_cell(x).lower().replace(" ", "") for x in raw.iloc[idx].tolist()]

        has_id = any(x == "№обр" for x in values)
        has_start_date = any("дат" in x and "начал" in x and "просто" in x for x in values)

        if has_id and has_start_date:
            return idx

    return None


def read_jtiny_merged(input_file):
    raw = pd.read_excel(input_file, sheet_name=0, header=None, dtype=object)
    raw = raw.dropna(how="all")

    header_row_idx = find_real_header_row(raw)

    if header_row_idx is not None:
        headers = raw.iloc[header_row_idx].tolist()
        df = raw.iloc[header_row_idx + 1:].copy()
        print(f"Найдена строка с корректными заголовками: {header_row_idx + 1}")
    else:
        headers = build_expected_headers(raw.shape[1])
        df = raw.iloc[1:].copy()
        print("Корректная строка заголовков не найдена. Заголовки восстановлены по шаблону.")

    df.columns = make_unique_columns(headers)
    df = df.dropna(how="all").copy()

    return df


def find_column(df, possible_names):
    normalized_map = {
        str(col).strip().lower().replace(" ", ""): col
        for col in df.columns
    }

    for name in possible_names:
        key = name.strip().lower().replace(" ", "")
        if key in normalized_map:
            return normalized_map[key]

    raise ValueError(f"Не найден ни один из столбцов: {possible_names}")


def clean_data(df):
    print(f"Размер до очистки: {df.shape}")

    id_col = find_column(df, ["№ обр"])
    start_date_col = find_column(df, ["Дата/время начала простоя"])

    print(f"Столбец ID: {id_col}")
    print(f"Столбец даты начала: {start_date_col}")

    df[id_col] = pd.to_numeric(df[id_col], errors="coerce")
    before = len(df)
    df = df[df[id_col].notna()].copy()
    print(f"Удалено строк без корректного '№ обр': {before - len(df)}")

    df[id_col] = df[id_col].astype("Int64")

    date_cols = [col for col in df.columns if "Дата/время" in str(col)]

    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)

    before = len(df)
    df = df.drop_duplicates(subset=[id_col], keep="first").copy()
    print(f"Удалено дублей по '{id_col}': {before - len(df)}")

    before = len(df)
    df = df[df[start_date_col].notna()].copy()
    print(f"Удалено строк без даты начала простоя: {before - len(df)}")

    before = len(df)
    df = df[df[start_date_col] >= DATE_START].copy()
    print(f"Удалено строк раньше {DATE_START.date()}: {before - len(df)}")

    df = df.sort_values(by=[id_col], kind="stable").reset_index(drop=True)

    print(f"Размер после очистки: {df.shape}")

    return df, id_col, date_cols


def save_result(df, output_file, date_cols):
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="jtiny_clean")

        ws = writer.book["jtiny_clean"]
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions

        header_to_col = {
            cell.value: cell.column
            for cell in ws[1]
        }

        for col_name in date_cols:
            if col_name in header_to_col:
                col_idx = header_to_col[col_name]
                for row in range(2, ws.max_row + 1):
                    ws.cell(row=row, column=col_idx).number_format = "yyyy-mm-dd hh:mm:ss"

        for column_cells in ws.columns:
            max_len = 0
            col_letter = column_cells[0].column_letter

            for cell in column_cells[:200]:
                if cell.value is not None:
                    max_len = max(max_len, len(str(cell.value)))

            ws.column_dimensions[col_letter].width = min(max(max_len + 2, 10), 45)

    print(f"Сохранено: {output_file}")


def main():
    input_file = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_INPUT
    output_file = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else DEFAULT_OUTPUT

    if not input_file.exists():
        raise FileNotFoundError(f"Файл не найден: {input_file}")

    print("=" * 80)
    print("ОЧИСТКА jtiny_merged.xlsx")
    print("=" * 80)
    print(f"Входной файл: {input_file}")

    df = read_jtiny_merged(input_file)
    df_clean, id_col, date_cols = clean_data(df)
    save_result(df_clean, output_file, date_cols)

    print("=" * 80)
    print("ГОТОВО")
    print("=" * 80)
    print(f"Итоговый файл: {output_file}")


if __name__ == "__main__":
    main()
