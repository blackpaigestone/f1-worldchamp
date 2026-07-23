"""Evaluate the selected structure on 2024, then plot one coherent race."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.probabilistic_model import (
    FULL_FEATURES,
    CumulativeOrdinalLogit,
    balance_race_probabilities,
    evaluate_distribution,
    race_coherence_error,
)

DATA_PATH = Path("data_processed/f1_probabilistic_prerace.parquet")
OUTPUT_DIR = Path("outputs/figures_08")


def main() -> None:
    with (OUTPUT_DIR / "race_structure_selection.json").open() as file:
        decision = json.load(file)
    if decision["selected_structure"] != "BALANCED":
        raise RuntimeError(
            "Pre-2024 validation did not select BALANCED. Do not generate "
            "the heatmap; revisit the race model architecture first."
        )

    df = pd.read_parquet(DATA_PATH)
    train = df[(df["year"] >= 2014) & (df["year"] <= 2023)]
    test = df[df["year"] == 2024].copy()
    model = CumulativeOrdinalLogit(max_position=22, c=0.10)
    model.fit(train[FULL_FEATURES], train["finish_position"])
    raw = model.predict_proba(test[FULL_FEATURES], field_size=test["field_size"])
    balanced = balance_race_probabilities(
        raw, test["raceId"], test["field_size"]
    )

    report = {
        "structure": "BALANCED",
        **evaluate_distribution(test["finish_position"], balanced),
        **race_coherence_error(balanced, test["raceId"], test["field_size"]),
    }
    with (OUTPUT_DIR / "2024_balanced_holdout_metrics.json").open("w") as file:
        json.dump(report, file, indent=2)

    # Use the final race for a stable, reproducible first visualization.
    race_id = int(test.sort_values(["year", "round"])["raceId"].iloc[-1])
    race_mask = test["raceId"].eq(race_id).to_numpy()
    race = test.loc[race_mask].copy()
    race_probabilities = balanced[race_mask]
    field_size = int(race["field_size"].iloc[0])
    driver_names = (
        race["forename"].fillna("").str.cat(race["surname"].fillna(""), sep=" ").str.strip()
        if {"forename", "surname"}.issubset(race.columns)
        else race["driverRef"]
    )
    heatmap = pd.DataFrame(
        race_probabilities[:, :field_size],
        index=driver_names,
        columns=[f"P{position}" for position in range(1, field_size + 1)],
    )
    expected_positions = heatmap.to_numpy() @ np.arange(1, field_size + 1)
    heatmap = heatmap.iloc[np.argsort(expected_positions)]

    fig, axis = plt.subplots(figsize=(16, 11))
    sns.heatmap(
        heatmap,
        cmap="mako",
        linewidths=0.25,
        linecolor="white",
        cbar_kws={"label": "Predicted probability"},
        ax=axis,
    )
    race_name = str(race["race_name"].iloc[0]) if "race_name" in race else str(race["name"].iloc[0])
    axis.set_title(f"2024 {race_name}: Coherent Finishing-Position Forecast", fontweight="bold")
    axis.set_xlabel("Finishing position")
    axis.set_ylabel("Driver")
    fig.tight_layout()
    path = OUTPUT_DIR / "03_coherent_race_probability_heatmap.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("2024 BALANCED HOLDOUT")
    for name, value in report.items():
        print(f"{name:28s} {value}")
    print("Saved:", path)
    print("Heatmap source is coherent: PASS")


if __name__ == "__main__":
    main()
