import pandas as pd

compactor = pd.read_parquet("processed/compactor_dataset_labeled.parquet")
mill = pd.read_parquet("processed/mill_dataset_labeled.parquet")

print(compactor.shape)
print(mill.shape)

print(compactor[["event_in_24h", "event_in_48h", "event_in_72h", "pre_event_window"]].sum())
print(mill[["event_in_24h", "event_in_48h", "event_in_72h", "pre_event_window"]].sum())

print(compactor["time_to_next_event_hours"].describe())
print(mill["time_to_next_event_hours"].describe())
