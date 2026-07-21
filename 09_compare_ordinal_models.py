"""Compare independent thresholds with true ordinal regression before 2024."""

from pathlib import Path
import json

import pandas as pd

from src.probabilistic_model import (
    FULL_FEATURES,
    CumulativeOrdinalLogit,
    ProportionalOddsLogit,
    balance_race_probabilities,
    distribution_shape_diagnostics,
    evaluate_distribution,
    race_coherence_error,
)

DATA_PATH = Path("data_processed/f1_probabilistic_prerace.parquet")
OUTPUT_DIR = Path("outputs/figures_09")
VALIDATION_YEARS = range(2018, 2024)


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    records = []
    for year in VALIDATION_YEARS:
        print(f"\nValidation season {year}")
        train = df[(df["year"] >= 2014) & (df["year"] < year)]
        valid = df[df["year"] == year]
        challengers = {
            "INDEPENDENT_BALANCED": CumulativeOrdinalLogit(max_position=22, c=0.10),
            "PROPORTIONAL_ODDS_BALANCED": ProportionalOddsLogit(max_position=22),
        }
        for name, model in challengers.items():
            print("  fitting", name)
            model.fit(train[FULL_FEATURES], train["finish_position"])
            marginal = model.predict_proba(
                valid[FULL_FEATURES], field_size=valid["field_size"]
            )
            coherent = balance_race_probabilities(
                marginal, valid["raceId"], valid["field_size"]
            )
            record = {
                "validation_year": year,
                "model": name,
                **evaluate_distribution(valid["finish_position"], coherent),
                **distribution_shape_diagnostics(coherent),
                **race_coherence_error(coherent, valid["raceId"], valid["field_size"]),
            }
            records.append(record)
            print(
                f"    RPS={record['rps']:.5f} | log loss={record['log_loss']:.5f} | "
                f"multimodal={record['share_multimodal']:.1%}"
            )

    results = pd.DataFrame(records)
    numeric = results.select_dtypes("number").columns.drop("validation_year")
    summary = results.groupby("model")[numeric].mean()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_json(OUTPUT_DIR / "ordinal_model_comparison.json", orient="records", indent=2)
    summary.reset_index().to_json(
        OUTPUT_DIR / "ordinal_model_comparison_summary.json", orient="records", indent=2
    )
    print("\nMEAN 2018-2023 PERFORMANCE AND SHAPE")
    print(summary.to_string(float_format=lambda value: f"{value:.6f}"))
    print("\n2024 examined by this script: False")
    print("No final challenger selected automatically: inspect accuracy and shape together.")


if __name__ == "__main__":
    main()
