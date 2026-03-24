#setup and imports
import os
import pandas as pd
import numpy as np

#paths
raw_data_path = "/Users/paigeblackstone/Library/Mobile Documents/com~apple~CloudDocs/Formula 1 Fun"
processed_path = "data_processed"

os.makedirs(processed_path, exist_ok=True)

#raw tables
tables = {}

for file in os.listdir(raw_data_path):
    if file.endswith(".csv"):
        name = file.replace(".csv", "")
        tables[name] = pd.read_csv(os.path.join(raw_data_path, file))

sorted(tables.keys())

#master table
results = tables["results"].copy()
races = tables["races"].copy()
drivers = tables["drivers"].copy()
constructors = tables["constructors"].copy()
circuits = tables["circuits"].copy()
status = tables["status"].copy()
qualifying = tables["qualifying"].copy()

for df in [races, drivers, constructors, circuits]:
    if "url" in df.columns:
        df.drop(columns=["url"], inplace=True)

races["date"] = pd.to_datetime(races["date"], errors="coerce")

master = (
    results
    .merge(races, on="raceId", how="left", validate="m:1")
    .merge(drivers, on="driverId", how="left", validate="m:1")
    .merge(constructors, on="constructorId", how="left", validate="m:1")
    .merge(circuits, on="circuitId", how="left", validate="m:1")
    .merge(status, on="statusId", how="left", validate="m:1")
)

#add qually
qualifying = qualifying.rename(columns={"position": "qualifying_position"})

master = master.merge(
    qualifying[["raceId", "driverId", "qualifying_position"]],
    on=["raceId", "driverId"],
    how="left"
)

#add weather
master = master.merge(
    weather_df.drop(columns=["race_name", "year"]),
    on="raceId",
    how="left"
)


#base cleaned and derived fields
master["target_points"] = (master["points"] > 0).astype(int)
master["finish_position"] = master["positionOrder"]

master["grid_clean"] = pd.to_numeric(master["grid"], errors="coerce")
master.loc[master["grid_clean"] == 0, "grid_clean"] = np.nan

master["month"] = pd.to_datetime(master["date"]).dt.month
master["abs_lat"] = master["lat"].abs()
master["temp_range"] = master["temp_max"] - master["temp_min"]
master["is_wet_race"] = (master["precipitation"] > 0).astype(int)
master["high_altitude_track"] = (master["alt"] >= 500).astype(int)

#create DNF field
finish_like = [
    "Finished",
    "+1 Lap", "+2 Laps", "+3 Laps", "+4 Laps",
    "+5 Laps", "+6 Laps", "+7 Laps", "+8 Laps", "+9 Laps"
]

master["is_dnf"] = (~master["status"].isin(finish_like)).astype(int)

#rolling driver features
driver_sorted = master.sort_values(["driverId", "date"]).copy()

master["driver_avg_finish_last5"] = (
    driver_sorted.groupby("driverId")["positionOrder"]
    .transform(lambda x: x.shift(1).rolling(5).mean())
)

master["driver_points_last5"] = (
    driver_sorted.groupby("driverId")["points"]
    .transform(lambda x: x.shift(1).rolling(5).sum())
)

master["driver_dnf_rate_last5"] = (
    driver_sorted.groupby("driverId")["is_dnf"]
    .transform(lambda x: x.shift(1).rolling(5).mean())
)


#rolling constructor features
constructor_sorted = master.sort_values(["constructorId", "date"]).copy()

master["constructor_points_last5"] = (
    constructor_sorted.groupby("constructorId")["points"]
    .transform(lambda x: x.shift(1).rolling(5).sum())
)

master["constructor_dnf_rate_last5"] = (
    constructor_sorted.groupby("constructorId")["is_dnf"]
    .transform(lambda x: x.shift(1).rolling(5).mean())
)

#save feature store
master.to_csv(os.path.join(processed_path, "f1_feature_store.csv"), index=False)
master.to_parquet(os.path.join(processed_path, "f1_feature_store.parquet"), index=False)

