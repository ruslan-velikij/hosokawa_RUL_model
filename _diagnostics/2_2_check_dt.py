from pathlib import Path
import pandas as pd

file_path = Path("../processed/all_data_2020_2025.parquet")

df = pd.read_parquet(file_path, columns=["DT"])

print("Форма:")
print(df.shape)

print("\nКолонки:")
print(df.columns.tolist())

print("\nТип индекса:")
print(type(df.index))
print("Имя индекса:", df.index.name)
print("dtype индекса:", df.index.dtype)

print("\nПервые 10 значений индекса:")
print(df.index[:10])

dt = pd.to_datetime(df.index, errors="coerce")

print("\nПосле pd.to_datetime(index):")
print("dtype:", dt.dtype)
print("min:", dt.min())
print("max:", dt.max())
print("isna:", dt.isna().sum())

dup_count = pd.Index(df.index).duplicated().sum()
print("\nДубликаты по DT:")
print(dup_count)
