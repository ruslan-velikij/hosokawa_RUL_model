from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
TOIR_DIR = PROJECT_ROOT / "../toir_hosokawa"

FILES = [
    TOIR_DIR / "Хос_ТОИР_all.xlsx",
    TOIR_DIR / "Hosokawa 2024.xlsx",
    TOIR_DIR / "toir_2025.xlsx",
]

SHEET_NAME = "jtiny"
OUTPUT_FILE = TOIR_DIR / "jtiny_merged.xlsx"


def read_jtiny_correct(file_path: Path, sheet_name: str = "jtiny") -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

    header_row = raw.iloc[1].tolist()

    df = raw.iloc[2:].copy()
    df.columns = [str(x).strip() if pd.notna(x) else "" for x in header_row]

    df = df.dropna(how="all").copy()

    df.columns = [col.strip() for col in df.columns]

    df["__source_file__"] = file_path.name

    return df


def normalize_id_column(df: pd.DataFrame) -> str:
    candidates = []
    for col in df.columns:
        norm = str(col).strip().lower().replace(" ", "")
        if norm in {"№обр", "номеробр", "номеробразца"}:
            candidates.append(col)

    if candidates:
        return candidates[0]

    raise ValueError("Не найден столбец '№ обр'")


def main():
    print("=" * 80)
    print("ОБЪЕДИНЕНИЕ ЛИСТОВ jtiny С ПРАВИЛЬНЫМИ ЗАГОЛОВКАМИ")
    print("=" * 80)

    frames = []

    for file_path in FILES:
        print(f"\nЧитаю: {file_path.name}")
        df = read_jtiny_correct(file_path, SHEET_NAME)
        print(f"  Размер после корректного чтения: {df.shape}")
        frames.append(df)

    print("\nСклеиваю таблицы...")
    merged = pd.concat(frames, ignore_index=True)

    print(f"Размер до удаления дублей: {merged.shape}")

    id_col = normalize_id_column(merged)
    print(f"Ключевой столбец: {id_col}")

    merged[id_col] = merged[id_col].astype(str).str.strip()

    merged = merged[merged[id_col].notna()]
    merged = merged[merged[id_col] != ""]
    merged = merged[merged[id_col].str.lower() != "nan"]

    before = len(merged)
    merged = merged.drop_duplicates(subset=[id_col], keep="first")
    removed = before - len(merged)

    print(f"Удалено дублей по '{id_col}': {removed}")
    print(f"Размер после удаления дублей: {merged.shape}")

    sort_key = pd.to_numeric(merged[id_col], errors="coerce")
    if sort_key.notna().any():
        merged = (
            merged.assign(__sort_key__=sort_key)
            .sort_values(by=["__sort_key__", id_col], kind="stable")
            .drop(columns="__sort_key__")
        )
    else:
        merged = merged.sort_values(by=id_col, kind="stable")

    print(f"\nСохраняю: {OUTPUT_FILE}")
    merged.to_excel(OUTPUT_FILE, index=False)

    print("\n" + "=" * 80)
    print("ГОТОВО")
    print("=" * 80)
    print(f"Итоговый файл: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
