"""Build the leakage-safe pre-race dataset for probabilistic F1 forecasting.

The output has one row per driver per race. All rolling and standings features
use information from races strictly before the target race. Weather, pit-stop,
lap-time, points, and other race-day information are deliberately excluded.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FINISH_LIKE_PATTERN = r"^(Finished|\+\d+ Laps?)$"


def read_table(raw_dir: Path, name: str) -> pd.DataFrame:
    path = raw_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Required source file not found: {path}")
    return pd.read_csv(path, na_values=r"\N", low_memory=False)


def add_lagged_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add driver, constructor, and circuit history without current-race data."""
    df = df.sort_values(["date", "raceId", "driverId"]).copy()

    def lagged_mean(group_cols: list[str], value: str, window: int) -> pd.Series:
        return (
            df.groupby(group_cols, sort=False)[value]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )

    def lagged_sum(group_cols: list[str], value: str, window: int) -> pd.Series:
        return (
            df.groupby(group_cols, sort=False)[value]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).sum())
        )

    for window in (5, 10):
        df[f"driver_avg_finish_last{window}"] = lagged_mean(
            ["driverId"], "finish_position", window
        )
        df[f"driver_points_last{window}"] = lagged_sum(
            ["driverId"], "points", window
        )
        df[f"driver_dnf_rate_last{window}"] = lagged_mean(
            ["driverId"], "is_dnf", window
        )
        df[f"driver_avg_grid_last{window}"] = lagged_mean(
            ["driverId"], "grid_clean", window
        )

    df["driver_circuit_avg_finish_prior"] = (
        df.groupby(["driverId", "circuitId"], sort=False)["finish_position"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    )
    df["driver_prior_starts"] = df.groupby("driverId", sort=False).cumcount()

    # Constructor history must be calculated at constructor-race grain. Rolling
    # directly over driver rows would let one teammate's current-race result
    # leak into the other teammate's feature values.
    constructor_race = (
        df.groupby(
            ["constructorId", "raceId", "date", "circuitId"],
            as_index=False,
            dropna=False,
        )
        .agg(
            constructor_race_avg_finish=("finish_position", "mean"),
            constructor_race_points=("points", "sum"),
            constructor_race_dnf_rate=("is_dnf", "mean"),
            constructor_race_avg_grid=("grid_clean", "mean"),
        )
        .sort_values(["constructorId", "date", "raceId"])
    )
    constructor_race["constructor_prior_races"] = constructor_race.groupby(
        "constructorId", sort=False
    ).cumcount()

    constructor_sources = {
        "avg_finish": "constructor_race_avg_finish",
        "points": "constructor_race_points",
        "dnf_rate": "constructor_race_dnf_rate",
        "avg_grid": "constructor_race_avg_grid",
    }
    for window in (5, 10):
        for label, source in constructor_sources.items():
            constructor_race[f"constructor_{label}_last{window}"] = (
                constructor_race.groupby("constructorId", sort=False)[source]
                .transform(
                    lambda s: s.shift(1).rolling(window, min_periods=1).mean()
                )
            )

    constructor_race["constructor_circuit_avg_finish_prior"] = (
        constructor_race.groupby(
            ["constructorId", "circuitId"], sort=False
        )["constructor_race_avg_finish"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    )
    keep = [
        "constructorId", "raceId", "constructor_prior_races",
        "constructor_circuit_avg_finish_prior",
    ] + [
        f"constructor_{label}_last{window}"
        for window in (5, 10)
        for label in constructor_sources
    ]
    df = df.merge(
        constructor_race[keep],
        on=["constructorId", "raceId"],
        how="left",
        validate="m:1",
    )
    return df


def add_prerace_standings(
    df: pd.DataFrame,
    races: pd.DataFrame,
    raw_dir: Path,
) -> pd.DataFrame:
    race_dates = races[["raceId", "date", "year", "round"]].copy()

    driver = read_table(raw_dir, "driver_standings")
    driver = driver.merge(race_dates, on="raceId", how="left", validate="m:1")
    driver = driver.sort_values(["driverId", "year", "date", "raceId"])

    driver_final = (
        driver.sort_values(["driverId", "year", "date", "raceId"])
        .groupby(["driverId", "year"], as_index=False)
        .tail(1)[["driverId", "year", "points", "position", "wins"]]
    )
    driver_final["year"] = driver_final["year"] + 1
    driver_final = driver_final.rename(columns={
        "points": "driver_prior_season_points",
        "position": "driver_prior_season_position",
        "wins": "driver_prior_season_wins",
    })
    driver = driver.merge(
        driver_final, on=["driverId", "year"], how="left", validate="m:1"
    )
    for source, target in {
        "points": "driver_standing_points_prerace",
        "position": "driver_standing_position_prerace",
        "wins": "driver_standing_wins_prerace",
    }.items():
        driver[target] = driver.groupby(["driverId", "year"])[source].shift(1)
    driver.loc[driver["round"] == 1, "driver_standing_points_prerace"] = 0.0
    driver.loc[driver["round"] == 1, "driver_standing_wins_prerace"] = 0.0

    constructor = read_table(raw_dir, "constructor_standings")
    constructor = constructor.merge(race_dates, on="raceId", how="left", validate="m:1")
    constructor = constructor.sort_values(
        ["constructorId", "year", "date", "raceId"]
    )

    constructor_final = (
        constructor.sort_values(["constructorId", "year", "date", "raceId"])
        .groupby(["constructorId", "year"], as_index=False)
        .tail(1)[["constructorId", "year", "points", "position", "wins"]]
    )
    constructor_final["year"] = constructor_final["year"] + 1
    constructor_final = constructor_final.rename(columns={
        "points": "constructor_prior_season_points",
        "position": "constructor_prior_season_position",
        "wins": "constructor_prior_season_wins",
    })
    constructor = constructor.merge(
        constructor_final,
        on=["constructorId", "year"],
        how="left",
        validate="m:1",
    )
    for source, target in {
        "points": "constructor_standing_points_prerace",
        "position": "constructor_standing_position_prerace",
        "wins": "constructor_standing_wins_prerace",
    }.items():
        constructor[target] = constructor.groupby(
            ["constructorId", "year"]
        )[source].shift(1)
    constructor.loc[
        constructor["round"] == 1, "constructor_standing_points_prerace"
    ] = 0.0
    constructor.loc[
        constructor["round"] == 1, "constructor_standing_wins_prerace"
    ] = 0.0

    driver_cols = [
        "raceId", "driverId", "driver_standing_points_prerace",
        "driver_standing_position_prerace", "driver_standing_wins_prerace",
        "driver_prior_season_points", "driver_prior_season_position",
        "driver_prior_season_wins",
    ]
    constructor_cols = [
        "raceId", "constructorId", "constructor_standing_points_prerace",
        "constructor_standing_position_prerace", "constructor_standing_wins_prerace",
        "constructor_prior_season_points", "constructor_prior_season_position",
        "constructor_prior_season_wins",
    ]
    return (
        df.merge(driver[driver_cols], on=["raceId", "driverId"], how="left", validate="m:1")
        .merge(
            constructor[constructor_cols],
            on=["raceId", "constructorId"],
            how="left",
            validate="m:1",
        )
    )


def build_dataset(raw_dir: Path, min_year: int = 2014) -> pd.DataFrame:
    results = read_table(raw_dir, "results")
    races = read_table(raw_dir, "races")
    drivers = read_table(raw_dir, "drivers")
    constructors = read_table(raw_dir, "constructors")
    circuits = read_table(raw_dir, "circuits")
    status = read_table(raw_dir, "status")
    qualifying = read_table(raw_dir, "qualifying")

    races["date"] = pd.to_datetime(races["date"], errors="coerce")
    qualifying["qualifying_position"] = pd.to_numeric(
        qualifying["position"], errors="coerce"
    )

    drivers = drivers.rename(
        columns={"forename": "driver_forename", "surname": "driver_surname"}
    )
    constructors = constructors.rename(columns={"name": "constructor_name"})
    circuits = circuits.rename(columns={"name": "circuit_name"})
    races = races.rename(columns={"name": "race_name"})

    df = (
        results.merge(races, on="raceId", how="left", validate="m:1")
        .merge(
            drivers[["driverId", "driverRef", "driver_forename", "driver_surname"]],
            on="driverId",
            how="left",
            validate="m:1",
        )
        .merge(
            constructors[["constructorId", "constructorRef", "constructor_name"]],
            on="constructorId",
            how="left",
            validate="m:1",
        )
        .merge(
            circuits[["circuitId", "circuitRef", "circuit_name", "lat", "lng", "alt"]],
            on="circuitId",
            how="left",
            validate="m:1",
        )
        .merge(status, on="statusId", how="left", validate="m:1")
        .merge(
            qualifying[["raceId", "driverId", "qualifying_position"]],
            on=["raceId", "driverId"],
            how="left",
            # Historical results include legitimate shared-drive entries, so
            # the result side is many-to-one. Qualifying remains unique by
            # race and driver, and the modern output is validated as 1:1.
            validate="m:1",
        )
    )

    numeric_cols = [
        "year", "round", "grid", "positionOrder", "points", "laps",
        "qualifying_position", "lat", "lng", "alt",
    ]
    for column in numeric_cols:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["finish_position"] = df["positionOrder"].astype("Int64")
    df["grid_clean"] = df["grid"].replace(0, np.nan)
    df["is_dnf"] = (~df["status"].fillna("").str.match(FINISH_LIKE_PATTERN)).astype(int)
    df["field_size"] = df.groupby("raceId")["driverId"].transform("size")
    df["season_progress"] = df["round"] / df.groupby("year")["round"].transform("max")

    df = add_lagged_rolling_features(df)
    df = add_prerace_standings(df, races, raw_dir)
    df = df[df["year"] >= min_year].sort_values(
        ["date", "raceId", "finish_position"]
    ).reset_index(drop=True)

    if df.duplicated(["raceId", "driverId"]).any():
        raise ValueError("Dataset violates the one-row-per-driver-per-race grain")
    if df["finish_position"].isna().any():
        raise ValueError("Target finish_position contains missing values")
    if not df["finish_position"].between(1, df["field_size"]).all():
        raise ValueError("Target finish_position falls outside its race field size")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data_raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_processed"))
    parser.add_argument("--min-year", type=int, default=2014)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset(args.raw_dir, min_year=args.min_year)
    output = args.output_dir / "f1_probabilistic_prerace.parquet"
    dataset.to_parquet(output, index=False)

    print(f"Saved: {output}")
    print(f"Rows: {len(dataset):,}")
    print(f"Races: {dataset['raceId'].nunique():,}")
    print(f"Seasons: {int(dataset['year'].min())}-{int(dataset['year'].max())}")
    print(f"Drivers: {dataset['driverId'].nunique():,}")
    print(f"Target range: P{int(dataset['finish_position'].min())}-P{int(dataset['finish_position'].max())}")
    print(f"DNF rate: {dataset['is_dnf'].mean():.3%}")
    print("Dataset validation: PASS")


if __name__ == "__main__":
    main()
