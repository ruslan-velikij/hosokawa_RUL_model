from pathlib import Path
import pandas as pd

file_path = Path("../processed/all_data_2020_2025.parquet")

df = pd.read_parquet(file_path)

print("Форма:", df.shape)
print("Имя индекса:", df.index.name)

print("\nПолные дубли строк:")
print(df.duplicated().sum())

print("\nДубли только по индексу DT:")
print(pd.Index(df.index).duplicated().sum())

print("\nПример строк с одинаковым DT:")
dup_dt = df.index[pd.Index(df.index).duplicated(keep=False)]
sample_dt = dup_dt[0]
print("DT:", sample_dt)
print(df.loc[sample_dt].head(10))
