"""Freeze the selected probabilistic architecture using pre-2024 evidence."""

from pathlib import Path
import json

import pandas as pd

OUTPUT_DIR = Path("outputs/figures_09")
SUMMARY_PATH = OUTPUT_DIR / "ordinal_model_comparison_summary.json"
FREEZE_PATH = OUTPUT_DIR / "frozen_model_specification.json"


def main() -> None:
    summary = pd.read_json(SUMMARY_PATH).set_index("model")
    incumbent = summary.loc["INDEPENDENT_BALANCED"]
    challenger = summary.loc["PROPORTIONAL_ODDS_BALANCED"]

    rps_degradation = challenger["rps"] / incumbent["rps"] - 1
    log_loss_improvement = 1 - challenger["log_loss"] / incumbent["log_loss"]
    mae_improvement = 1 - challenger["expected_position_mae"] / incumbent["expected_position_mae"]
    shape_pass = challenger["share_multimodal"] <= 0.05
    coherence_pass = (
        challenger["max_row_error"] <= 1e-8
        and challenger["max_column_error"] <= 1e-8
    )
    selected = bool(
        rps_degradation <= 0.02
        and log_loss_improvement >= 0.05
        and shape_pass
        and coherence_pass
    )
    if not selected:
        raise RuntimeError("Proportional odds did not pass the predeclared freeze gate")

    specification = {
        "status": "FROZEN_BEFORE_2024_CHALLENGER_EVALUATION",
        "selected_model": "PROPORTIONAL_ODDS_BALANCED",
        "estimator": "ProportionalOddsLogit",
        "features": "FULL_FEATURES",
        "training_window_for_holdout": "2014-2023",
        "holdout": 2024,
        "race_coherence": "Sinkhorn balancing",
        "probability_floor": 1e-6,
        "selection_data": "Rolling-origin validation seasons 2018-2023 only",
        "selection_rule": {
            "maximum_rps_degradation": 0.02,
            "minimum_log_loss_improvement": 0.05,
            "maximum_multimodal_share": 0.05,
            "maximum_coherence_error": 1e-8,
        },
        "observed_tradeoff": {
            "rps_degradation_pct": 100 * float(rps_degradation),
            "log_loss_improvement_pct": 100 * float(log_loss_improvement),
            "expected_mae_improvement_pct": 100 * float(mae_improvement),
            "multimodal_share": float(challenger["share_multimodal"]),
        },
        "known_limitation": (
            "Correlated predictors prevent reliable Hessian inversion and "
            "coefficient standard errors; model selection is predictive, not inferential."
        ),
    }
    with FREEZE_PATH.open("w") as file:
        json.dump(specification, file, indent=2)

    print(json.dumps(specification, indent=2))
    print("Saved:", FREEZE_PATH)
    print("2024 examined by this script: False")


if __name__ == "__main__":
    main()
