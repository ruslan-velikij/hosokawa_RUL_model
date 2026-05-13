import pandas as pd

df = pd.read_parquet("../processed/all_data_2020_2025.parquet")
print(df["N_Hosokawa"].value_counts(dropna=False).sort_index())
