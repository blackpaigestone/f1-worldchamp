"""Diagnose threshold crossing and show representative driver distributions."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.probabilistic_model import (
    FULL_FEATURES,
    CumulativeOrdinalLogit,
    balance_race_probabilities,
    cumulative_crossing_diagnostics,
    distribution_shape_diagnostics,
)

DATA_PATH = Path("data_processed/f1_probabilistic_prerace.parquet")
OUTPUT_DIR = Path("outputs/figures_09")


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    # Keep the diagnostic inside the development window: train through 2022
    # and inspect 2023. The challenger comparison therefore remains independent
    # of revised 2024 performance.
    train = df[(df["year"] >= 2014) & (df["year"] <= 2022)]
    test = df[df["year"] == 2023].copy()
    model = CumulativeOrdinalLogit(max_position=22, c=0.10)
    model.fit(train[FULL_FEATURES], train["finish_position"])

    raw_cdf = model.predict_cumulative(test[FULL_FEATURES], repair=False)
    marginal = model.predict_proba(test[FULL_FEATURES], field_size=test["field_size"])
    coherent = balance_race_probabilities(
        marginal, test["raceId"], test["field_size"]
    )
    diagnostics = {
        "raw_cumulative_crossing": cumulative_crossing_diagnostics(raw_cdf),
        "unbalanced_shape": distribution_shape_diagnostics(marginal),
        "balanced_shape": distribution_shape_diagnostics(coherent),
    }

    # Diagnose the final race and select three examples reproducibly:
    # highest peak, greatest fragmentation, and broadest uncertainty.
    race_id = int(test.sort_values(["year", "round"])["raceId"].iloc[-1])
    mask = test["raceId"].eq(race_id).to_numpy()
    race = test.loc[mask].reset_index(drop=True)
    probs = coherent[mask]
    field_size = int(race["field_size"].iloc[0])
    active = probs[:, :field_size]
    peaks = np.zeros(len(race), dtype=int)
    peaks += (active[:, 0] > active[:, 1]).astype(int)
    peaks += (active[:, -1] > active[:, -2]).astype(int)
    peaks += ((active[:, 1:-1] > active[:, :-2]) & (active[:, 1:-1] > active[:, 2:])).sum(axis=1)
    entropy = -np.sum(active * np.log(np.clip(active, 1e-15, 1.0)), axis=1)
    chosen = []
    for index in [int(active.max(axis=1).argmax()), int(peaks.argmax()), int(entropy.argmax())]:
        if index not in chosen:
            chosen.append(index)
    for index in np.argsort(-entropy):
        if len(chosen) == 3:
            break
        if int(index) not in chosen:
            chosen.append(int(index))

    positions = np.arange(1, field_size + 1)
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True, sharey=True)
    for axis, index in zip(axes, chosen):
        label = str(race.loc[index, "driverRef"])
        actual = int(race.loc[index, "finish_position"])
        axis.bar(positions, active[index], color="#4472C4", alpha=0.88)
        axis.axvline(actual, color="#C00000", linestyle="--", linewidth=2, label=f"Actual: P{actual}")
        axis.set_title(f"{label}: {peaks[index]} local peaks; entropy={entropy[index]:.2f}", loc="left", fontweight="bold")
        axis.set_ylabel("Probability")
        axis.legend(frameon=False)
    axes[-1].set_xlabel("Finishing position")
    axes[-1].set_xticks(positions)
    fig.suptitle("2023 Development Diagnostic: Representative Driver Distributions", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = OUTPUT_DIR / "01_driver_distribution_diagnostics.png"
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    with (OUTPUT_DIR / "distribution_shape_diagnostics.json").open("w") as file:
        json.dump(diagnostics, file, indent=2)

    print(json.dumps(diagnostics, indent=2))
    print("Selected drivers:", [str(race.loc[index, "driverRef"]) for index in chosen])
    print("Saved:", figure_path)
    print("2024 examined by this script: False")


if __name__ == "__main__":
    main()
