"""Select race-matrix structure using rolling-origin data through 2023 only."""

from pathlib import Path
import json

import pandas as pd

from src.probabilistic_model import (
    FULL_FEATURES,
    CumulativeOrdinalLogit,
    balance_race_probabilities,
    evaluate_distribution,
    race_coherence_error,
)

DATA_PATH = Path("data_processed/f1_probabilistic_prerace.parquet")
OUTPUT_DIR = Path("outputs/figures_08")
VALIDATION_YEARS = range(2018, 2024)
MODEL_C = 0.10
MAX_RPS_DEGRADATION = 0.01
COHERENCE_TOLERANCE = 1e-8


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    if (df["year"] == 2024).sum() == 0:
        raise ValueError("Expected an untouched 2024 holdout in the dataset")

    records = []
    for validation_year in VALIDATION_YEARS:
        train = df[(df["year"] >= 2014) & (df["year"] < validation_year)]
        valid = df[df["year"] == validation_year].copy()

        model = CumulativeOrdinalLogit(max_position=22, c=MODEL_C)
        model.fit(train[FULL_FEATURES], train["finish_position"])
        raw = model.predict_proba(
            valid[FULL_FEATURES], field_size=valid["field_size"]
        )
        balanced = balance_race_probabilities(
            raw, valid["raceId"], valid["field_size"]
        )

        for structure, probabilities in (
            ("UNBALANCED", raw),
            ("BALANCED", balanced),
        ):
            records.append(
                {
                    "validation_year": validation_year,
                    "structure": structure,
                    **evaluate_distribution(valid["finish_position"], probabilities),
                    **race_coherence_error(
                        probabilities, valid["raceId"], valid["field_size"]
                    ),
                }
            )

    results = pd.DataFrame(records)
    metric_columns = [
        "rps",
        "log_loss",
        "expected_position_mae",
        "modal_position_accuracy",
        "win_brier",
        "podium_brier",
        "points_brier",
        "max_row_error",
        "mean_row_error",
        "max_column_error",
        "mean_column_error",
    ]
    summary = results.groupby("structure")[metric_columns].mean()

    raw_rps = float(summary.loc["UNBALANCED", "rps"])
    balanced_rps = float(summary.loc["BALANCED", "rps"])
    rps_degradation = balanced_rps / raw_rps - 1.0
    coherence_pass = bool(
        summary.loc["BALANCED", "max_row_error"] <= COHERENCE_TOLERANCE
        and summary.loc["BALANCED", "max_column_error"] <= COHERENCE_TOLERANCE
    )
    selected = (
        "BALANCED"
        if coherence_pass and rps_degradation <= MAX_RPS_DEGRADATION
        else "REVISIT_MODEL_ARCHITECTURE"
    )

    decision = {
        "data_used": "2014-2023 only; rolling validation 2018-2023",
        "model_c": MODEL_C,
        "selection_rule": (
            "Select BALANCED if max row and column error are <= 1e-8 and "
            "mean RPS degradation versus UNBALANCED is <= 1%; otherwise "
            "revisit the model architecture before generating a heatmap."
        ),
        "selected_structure": selected,
        "balanced_rps_degradation_pct": 100.0 * rps_degradation,
        "coherence_pass": coherence_pass,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_json(
        OUTPUT_DIR / "race_balancing_validation.json", orient="records", indent=2
    )
    with (OUTPUT_DIR / "race_structure_selection.json").open("w") as file:
        json.dump(decision, file, indent=2)

    print("BALANCED VERSUS UNBALANCED: 2018-2023 ROLLING ORIGIN")
    print(summary.to_string(float_format=lambda value: f"{value:.6f}"))
    print(f"\nBalanced RPS degradation: {100 * rps_degradation:.3f}%")
    print("Coherence gate:", "PASS" if coherence_pass else "FAIL")
    print("Selected structure:", selected)
    print("2024 examined by this script: False")


if __name__ == "__main__":
    main()
