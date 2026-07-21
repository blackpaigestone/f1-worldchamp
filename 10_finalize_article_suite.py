"""Evaluate the frozen 2024 challenger and create publication-ready figures."""

from pathlib import Path
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tools.sm_exceptions import HessianInversionWarning

from src.probabilistic_model import (
    BASELINE_FEATURES,
    FULL_FEATURES,
    CumulativeOrdinalLogit,
    ProportionalOddsLogit,
    balance_race_probabilities,
    distribution_frame,
    distribution_shape_diagnostics,
    evaluate_distribution,
    race_coherence_error,
)

DATA_PATH = Path("data_processed/f1_probabilistic_prerace.parquet")
FREEZE_PATH = Path("outputs/figures_09/frozen_model_specification.json")
VALIDATION_PATH = Path("outputs/figures_09/ordinal_model_comparison_summary.json")
OUTPUT_DIR = Path("outputs/article_probabilistic_forecasting")
PREDICTION_PATH = Path("data_processed/f1_2024_frozen_model_probabilities.parquet")

BLUE = "#234E70"
TEAL = "#2A9D8F"
GOLD = "#E9C46A"
RED = "#C8553D"
GRAY = "#667085"


def style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#FBFCFE",
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "DejaVu Sans",
    })


def driver_labels(frame: pd.DataFrame) -> pd.Series:
    if {"forename", "surname"}.issubset(frame.columns):
        labels = frame["forename"].fillna("").str.cat(
            frame["surname"].fillna(""), sep=" "
        ).str.strip()
        if labels.ne("").all():
            return labels
    return frame["driverRef"].str.replace("_", " ").str.title()


def quantile_position(probabilities: np.ndarray, q: float) -> np.ndarray:
    return (np.cumsum(probabilities, axis=1) < q).sum(axis=1) + 1


def fit_predict(model, features, train, test) -> np.ndarray:
    model.fit(train[features], train["finish_position"])
    marginal = model.predict_proba(test[features], field_size=test["field_size"])
    return balance_race_probabilities(
        marginal, test["raceId"], test["field_size"]
    )


def main() -> None:
    with FREEZE_PATH.open() as file:
        frozen = json.load(file)
    if frozen.get("selected_model") != "PROPORTIONAL_ODDS_BALANCED":
        raise RuntimeError("The required pre-2024 proportional-odds freeze is missing")

    style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA_PATH)
    train = df[(df["year"] >= 2014) & (df["year"] <= 2023)]
    test = df[df["year"] == 2024].copy().reset_index(drop=True)

    models = {
        "Qualifying baseline": (
            CumulativeOrdinalLogit(max_position=22, c=1.0), BASELINE_FEATURES
        ),
        "Previous independent model": (
            CumulativeOrdinalLogit(max_position=22, c=0.10), FULL_FEATURES
        ),
        "Frozen proportional odds": (
            ProportionalOddsLogit(max_position=22), FULL_FEATURES
        ),
    }
    probabilities = {}
    records = []
    for name, (model, features) in models.items():
        print("Fitting:", name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", HessianInversionWarning)
            predicted = fit_predict(model, features, train, test)
        probabilities[name] = predicted
        records.append({
            "model": name,
            **evaluate_distribution(test["finish_position"], predicted),
            **distribution_shape_diagnostics(predicted),
            **race_coherence_error(predicted, test["raceId"], test["field_size"]),
        })

    metrics = pd.DataFrame(records)
    metrics.to_json(OUTPUT_DIR / "2024_frozen_model_comparison.json", orient="records", indent=2)
    selected = probabilities["Frozen proportional odds"]

    metadata_columns = [
        column for column in [
            "raceId", "year", "round", "race_name", "name", "driverId",
            "driverRef", "constructorRef", "grid_clean", "qualifying_position",
            "finish_position", "field_size",
        ] if column in test.columns
    ]
    long_predictions = distribution_frame(test[metadata_columns], selected)
    long_predictions["model"] = "PROPORTIONAL_ODDS_BALANCED_FROZEN"
    long_predictions.to_parquet(PREDICTION_PATH, index=False)

    # Figure 1: development-window tradeoff.
    validation = pd.read_json(VALIDATION_PATH).set_index("model")
    tradeoff_metrics = ["rps", "log_loss", "expected_position_mae", "share_multimodal"]
    tradeoff_labels = ["RPS", "Log loss", "Expected-position MAE", "Multimodal share"]
    incumbent = validation.loc["INDEPENDENT_BALANCED", tradeoff_metrics]
    challenger = validation.loc["PROPORTIONAL_ODDS_BALANCED", tradeoff_metrics]
    change = 100 * (challenger / incumbent - 1)
    fig, axis = plt.subplots(figsize=(12, 7))
    colors = [RED if value > 0 else TEAL for value in change]
    axis.barh(tradeoff_labels, change, color=colors)
    axis.axvline(0, color="#222222", linewidth=1)
    for index, value in enumerate(change):
        axis.text(value + (0.8 if value >= 0 else -0.8), index, f"{value:+.1f}%", ha="left" if value >= 0 else "right", va="center", fontweight="bold")
    axis.set_title("Why the Ordinal Model Was Selected", loc="left")
    axis.set_xlabel("Change versus independent thresholds")
    axis.text(0, -0.18, "Lower is better for all four measures; selection used 2018–2023 only.", transform=axis.transAxes, color=GRAY)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_model_selection_tradeoff.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: honest 2024 model comparison.
    plot_metrics = ["rps", "log_loss", "expected_position_mae"]
    titles = ["Ranked Probability Score", "Exact-position log loss", "Expected-position MAE"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    palette = [GOLD, BLUE, TEAL]
    for axis, metric, title in zip(axes, plot_metrics, titles):
        ordered = metrics.sort_values(metric)
        axis.barh(ordered["model"], ordered[metric], color=[palette[list(metrics["model"]).index(name)] for name in ordered["model"]])
        axis.invert_yaxis()
        axis.set_title(title)
        axis.set_xlabel("Lower is better")
        for patch, value in zip(axis.patches, ordered[metric]):
            axis.text(value, patch.get_y() + patch.get_height()/2, f" {value:.3f}", va="center", fontsize=11)
    fig.suptitle("2024 Holdout: Frozen Model Comparison", fontsize=22, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUTPUT_DIR / "02_2024_holdout_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Work with the final 2024 race for the remaining figures.
    race_id = int(test.sort_values(["year", "round"])["raceId"].iloc[-1])
    mask = test["raceId"].eq(race_id).to_numpy()
    race = test.loc[mask].reset_index(drop=True)
    race_probs = selected[mask]
    field_size = int(race["field_size"].iloc[0])
    race_probs = race_probs[:, :field_size]
    labels = driver_labels(race).reset_index(drop=True)
    positions = np.arange(1, field_size + 1)
    expected = race_probs @ positions
    order = np.argsort(expected)
    race_name_column = "race_name" if "race_name" in race.columns else "name"
    race_name = str(race[race_name_column].iloc[0])

    # Figure 3: smooth coherent heatmap.
    heatmap = pd.DataFrame(race_probs[order], index=labels.iloc[order], columns=[f"P{i}" for i in positions])
    fig, axis = plt.subplots(figsize=(17, 11))
    sns.heatmap(heatmap, cmap="mako", linewidths=0.25, linecolor="white", cbar_kws={"label": "Predicted probability"}, ax=axis)
    axis.set_title(f"2024 {race_name}: Smooth, Coherent Race Forecast", loc="left", fontsize=20)
    axis.set_xlabel("Finishing position")
    axis.set_ylabel("Driver")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_smooth_coherent_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Figure 4: favorite, midfield, and long-shot distributions.
    representatives = [order[0], order[len(order)//2], order[-1]]
    roles = ["Favorite", "Midfield", "Long shot"]
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True, sharey=True)
    for axis, index, role in zip(axes, representatives, roles):
        axis.fill_between(positions, race_probs[index], color=TEAL, alpha=0.28)
        axis.plot(positions, race_probs[index], color=BLUE, linewidth=2.5, marker="o", markersize=4)
        axis.axvline(race.loc[index, "finish_position"], color=RED, linestyle="--", label=f"Actual P{int(race.loc[index, 'finish_position'])}")
        axis.set_title(f"{role}: {labels.iloc[index]} — expected P{expected[index]:.1f}", loc="left")
        axis.set_ylabel("Probability")
        axis.legend(frameon=False, loc="upper right")
    axes[-1].set_xlabel("Finishing position")
    axes[-1].set_xticks(positions)
    fig.suptitle("Different Drivers, Different Probability Distributions", fontsize=21, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUTPUT_DIR / "04_driver_distribution_examples.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Figure 5: win, podium, and points probabilities.
    events = pd.DataFrame({
        "Driver": labels,
        "Win": race_probs[:, :1].sum(axis=1),
        "Podium": race_probs[:, :3].sum(axis=1),
        "Points": race_probs[:, :10].sum(axis=1),
        "expected": expected,
    }).sort_values("expected")
    fig, axes = plt.subplots(1, 3, figsize=(19, 10), sharey=True)
    for axis, event, color in zip(axes, ["Win", "Podium", "Points"], [GOLD, TEAL, BLUE]):
        axis.barh(events["Driver"], events[event], color=color)
        axis.invert_yaxis()
        axis.set_title(f"P({event.lower()})")
        axis.set_xlabel("Probability")
        axis.set_xlim(0, 1)
        axis.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    fig.suptitle(f"2024 {race_name}: Event Probabilities", fontsize=22, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUTPUT_DIR / "05_win_podium_points_probabilities.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Figure 6: expected finish and central 80% interval.
    lower = quantile_position(race_probs, 0.10)
    upper = quantile_position(race_probs, 0.90)
    actual = race["finish_position"].to_numpy(dtype=int)
    fig, axis = plt.subplots(figsize=(14, 10))
    y = np.arange(field_size)
    axis.hlines(y, lower[order], upper[order], color="#9CB3C9", linewidth=7, alpha=0.8, label="Central 80% interval")
    axis.scatter(expected[order], y, color=BLUE, s=85, zorder=3, label="Expected finish")
    axis.scatter(actual[order], y, color=RED, marker="x", s=80, linewidth=2.5, zorder=4, label="Actual finish")
    axis.set_yticks(y, labels.iloc[order])
    axis.invert_yaxis()
    axis.invert_xaxis()
    axis.set_xticks(positions)
    axis.set_xlabel("Finishing position — better finishes are farther right")
    axis.set_title(f"2024 {race_name}: Forecast Range and Actual Result", loc="left")
    axis.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "06_expected_finish_uncertainty.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "frozen_model": frozen,
        "2024_metrics": records,
        "race_visualized": {"raceId": race_id, "race_name": race_name},
        "figures": [
            "01_model_selection_tradeoff.png",
            "02_2024_holdout_comparison.png",
            "03_smooth_coherent_heatmap.png",
            "04_driver_distribution_examples.png",
            "05_win_podium_points_probabilities.png",
            "06_expected_finish_uncertainty.png",
        ],
    }
    with (OUTPUT_DIR / "article_suite_manifest.json").open("w") as file:
        json.dump(manifest, file, indent=2)

    print("\n2024 HOLDOUT RESULTS — ALL MODELS RACE-BALANCED")
    print(metrics.set_index("model")[["rps", "log_loss", "expected_position_mae", "modal_position_accuracy", "win_brier", "podium_brier", "points_brier", "share_multimodal"]].to_string(float_format=lambda value: f"{value:.6f}"))
    print("\nSaved article suite:", OUTPUT_DIR)
    print("Saved frozen predictions:", PREDICTION_PATH)
    print("Article visualization suite: PASS")


if __name__ == "__main__":
    main()
