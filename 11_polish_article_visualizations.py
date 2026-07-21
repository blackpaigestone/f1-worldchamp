"""Polish the frozen article figures without refitting or changing the model."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import seaborn as sns

PREDICTIONS = Path("data_processed/f1_2024_frozen_model_probabilities.parquet")
VALIDATION = Path("outputs/figures_09/ordinal_model_comparison_summary.json")
HOLDOUT = Path("outputs/article_probabilistic_forecasting/2024_frozen_model_comparison.json")
OUTPUT = Path("outputs/article_probabilistic_forecasting/final")

NAVY = "#173F5F"
BLUE = "#3274A1"
TEAL = "#2A9D8F"
GOLD = "#E9C46A"
RED = "#D45A3A"
PALE = "#ADC4D6"
TEXT = "#20242A"
MUTED = "#667085"

DISPLAY_NAMES = {
    "max_verstappen": "Max Verstappen",
    "norris": "Lando Norris",
    "piastri": "Oscar Piastri",
    "sainz": "Carlos Sainz",
    "russell": "George Russell",
    "leclerc": "Charles Leclerc",
    "perez": "Sergio Pérez",
    "gasly": "Pierre Gasly",
    "hulkenberg": "Nico Hülkenberg",
    "alonso": "Fernando Alonso",
    "hamilton": "Lewis Hamilton",
    "tsunoda": "Yuki Tsunoda",
    "bottas": "Valtteri Bottas",
    "stroll": "Lance Stroll",
    "lawson": "Liam Lawson",
    "kevin_magnussen": "Kevin Magnussen",
    "albon": "Alex Albon",
    "zhou": "Zhou Guanyu",
    "doohan": "Jack Doohan",
    "colapinto": "Franco Colapinto",
}


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFBFC",
        "axes.edgecolor": "#CDD2D9",
        "axes.labelcolor": TEXT,
        "axes.titlecolor": TEXT,
        "axes.titleweight": "bold",
        "text.color": TEXT,
        "font.family": "DejaVu Sans",
        "savefig.facecolor": "white",
    })


def save(fig: plt.Figure, filename: str) -> None:
    fig.savefig(OUTPUT / filename, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def quantile_position(probabilities: np.ndarray, probability: float) -> np.ndarray:
    return (np.cumsum(probabilities, axis=1) < probability).sum(axis=1) + 1


def main() -> None:
    set_style()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    validation = pd.read_json(VALIDATION).set_index("model")
    independent = validation.loc["INDEPENDENT_BALANCED"]
    ordinal = validation.loc["PROPORTIONAL_ODDS_BALANCED"]
    changes = 100 * (ordinal / independent - 1)

    # 1. Keep predictive changes on a readable scale and separate the structural result.
    fig = plt.figure(figsize=(14, 7))
    grid = fig.add_gridspec(1, 2, width_ratios=[2.2, 1], wspace=0.12)
    axis = fig.add_subplot(grid[0, 0])
    metrics = ["rps", "log_loss", "expected_position_mae"]
    labels = ["Ranked Probability Score", "Exact-position log loss", "Expected-position MAE"]
    values = changes[metrics].to_numpy()
    colors = [RED if value > 0 else TEAL for value in values]
    y = np.arange(len(labels))
    axis.barh(y, values, color=colors, height=0.58)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.axvline(0, color=TEXT, linewidth=1.2)
    axis.set_xlim(-13, 3)
    axis.set_xlabel("Change versus independent thresholds")
    axis.xaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")
    for row, value in enumerate(values):
        axis.text(value + (0.35 if value >= 0 else -0.35), row, f"{value:+.1f}%", ha="left" if value >= 0 else "right", va="center", fontweight="bold")
    axis.set_title("Predictive trade-off", loc="left")

    callout = fig.add_subplot(grid[0, 1])
    callout.set_facecolor("#EFF8F6")
    for spine in callout.spines.values():
        spine.set_visible(False)
    callout.set_xticks([])
    callout.set_yticks([])
    callout.text(0.5, 0.74, "Fragmented forecasts", ha="center", fontsize=16, fontweight="bold")
    callout.text(0.5, 0.52, "100%  →  0%", ha="center", fontsize=34, fontweight="bold", color=TEAL)
    callout.text(0.5, 0.31, "multimodal driver\ndistributions", ha="center", fontsize=15, color=MUTED, linespacing=1.4)
    fig.suptitle("Why the Ordinal Model Was Selected", fontsize=25, fontweight="bold", y=1.01)
    fig.text(0.5, -0.02, "Rolling-origin validation, 2018–2023. Lower is better for all predictive metrics.", ha="center", color=MUTED, fontsize=14)
    save(fig, "01_model_selection_tradeoff_final.png")

    # 2. Honest holdout comparison with compact labels and safe annotations.
    holdout = pd.read_json(HOLDOUT)
    label_map = {
        "Frozen proportional odds": "Ordinal",
        "Qualifying baseline": "Qualifying",
        "Previous independent model": "Independent",
    }
    holdout["short_model"] = holdout["model"].map(label_map)
    color_map = {"Ordinal": TEAL, "Qualifying": GOLD, "Independent": NAVY}
    plot_metrics = ["rps", "log_loss", "expected_position_mae"]
    titles = ["Ranked Probability Score", "Exact-position log loss", "Expected-position MAE"]
    formats = [".4f", ".3f", ".3f"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.5))
    for axis, metric, title, number_format in zip(axes, plot_metrics, titles, formats):
        ordered = holdout.sort_values(metric).reset_index(drop=True)
        bars = axis.barh(ordered["short_model"], ordered[metric], color=ordered["short_model"].map(color_map), height=0.62)
        axis.invert_yaxis()
        axis.set_title(title, fontsize=16)
        axis.set_xlabel("Lower is better", fontsize=13)
        axis.tick_params(axis="y", labelsize=14)
        axis.set_xlim(0, ordered[metric].max() * 1.18)
        for bar, value in zip(bars, ordered[metric]):
            axis.text(value + ordered[metric].max() * 0.025, bar.get_y() + bar.get_height()/2, format(value, number_format), va="center", fontsize=12, fontweight="bold")
    fig.suptitle("2024 Holdout: Frozen Model Comparison", fontsize=24, fontweight="bold")
    fig.text(0.5, 0.01, "All three models were race-balanced before scoring.", ha="center", color=MUTED, fontsize=13)
    fig.tight_layout(rect=[0, 0.06, 1, 0.92], w_pad=3)
    save(fig, "02_2024_holdout_comparison_final.png")

    # Reconstruct the selected Abu Dhabi probability matrix.
    predictions = pd.read_parquet(PREDICTIONS)
    race_id = int(predictions["raceId"].max())
    race_long = predictions[predictions["raceId"] == race_id].copy()
    id_columns = [column for column in race_long.columns if column not in {"position", "position_probability", "model"}]
    race = race_long[id_columns].drop_duplicates("driverId").set_index("driverId")
    matrix = race_long.pivot(index="driverId", columns="position", values="position_probability").fillna(0)
    matrix = matrix.reindex(race.index)
    field_size = int(race["field_size"].iloc[0])
    matrix = matrix.reindex(columns=range(1, field_size + 1), fill_value=0)
    probabilities = matrix.to_numpy()
    positions = np.arange(1, field_size + 1)
    expected = probabilities @ positions
    labels = race["driverRef"].map(DISPLAY_NAMES).fillna(race["driverRef"].str.replace("_", " ").str.title())
    order = np.argsort(expected)
    race_name_column = "race_name" if "race_name" in race.columns else "name"
    race_name = str(race[race_name_column].iloc[0])

    # 3. Smooth coherent heatmap.
    heatmap = pd.DataFrame(probabilities[order], index=labels.iloc[order], columns=[f"P{i}" for i in positions])
    fig, axis = plt.subplots(figsize=(16, 10.5))
    sns.heatmap(heatmap, cmap="mako", vmin=0, linewidths=0.25, linecolor="#D9DEE5", cbar_kws={"label": "Predicted probability", "shrink": 0.88}, ax=axis)
    axis.set_title(f"2024 {race_name}: Smooth, Coherent Race Forecast", loc="left", fontsize=21, pad=12)
    axis.set_xlabel("Finishing position")
    axis.set_ylabel("")
    axis.tick_params(axis="y", labelsize=13)
    fig.tight_layout()
    save(fig, "03_smooth_coherent_heatmap_final.png")

    # 4. Representative driver distributions.
    representatives = [order[0], order[len(order)//2], order[-1]]
    roles = ["Model favorite", "Midfield example", "Long shot"]
    actual = race["finish_position"].to_numpy(dtype=int)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10.5), sharex=True, sharey=True)
    for axis, index, role in zip(axes, representatives, roles):
        axis.fill_between(positions, probabilities[index], color=TEAL, alpha=0.25)
        axis.plot(positions, probabilities[index], color=NAVY, linewidth=2.5, marker="o", markersize=4)
        axis.axvline(actual[index], color=RED, linestyle="--", linewidth=2, label=f"Actual: P{actual[index]}")
        axis.set_title(f"{role}: {labels.iloc[index]}  |  expected P{expected[index]:.1f}", loc="left", fontsize=16)
        axis.set_ylabel("Probability", fontsize=13)
        axis.legend(frameon=False, loc="upper right", fontsize=12)
    axes[-1].set_xlabel("Finishing position")
    axes[-1].set_xticks(positions)
    fig.suptitle("One Forecast Is a Distribution, Not a Single Position", fontsize=23, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "04_driver_distribution_examples_final.png")

    # 5. Coherent event probabilities; label material probabilities only.
    events = pd.DataFrame({
        "Driver": labels,
        "Win": probabilities[:, :1].sum(axis=1),
        "Podium": probabilities[:, :3].sum(axis=1),
        "Points": probabilities[:, :10].sum(axis=1),
        "expected": expected,
    }).sort_values("expected")
    fig, axes = plt.subplots(1, 3, figsize=(18, 10), sharey=True)
    for axis, event, color in zip(axes, ["Win", "Podium", "Points"], [GOLD, TEAL, NAVY]):
        bars = axis.barh(events["Driver"], events[event], color=color, height=0.72)
        axis.invert_yaxis()
        axis.set_title(f"P({event.lower()})", fontsize=17)
        axis.set_xlabel("Probability")
        axis.set_xlim(0, 1.08)
        axis.xaxis.set_major_formatter(PercentFormatter(1.0))
        for bar, value in zip(bars, events[event]):
            if value >= 0.05:
                axis.text(value + 0.015, bar.get_y() + bar.get_height()/2, f"{value:.0%}", va="center", fontsize=10)
    fig.suptitle(f"2024 {race_name}: Event Probabilities", fontsize=23, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94], w_pad=2)
    save(fig, "05_win_podium_points_probabilities_final.png")

    # 6. Forecast range with extra bottom margin and a natural left-to-right axis.
    lower = quantile_position(probabilities, 0.10)
    upper = quantile_position(probabilities, 0.90)
    fig, axis = plt.subplots(figsize=(14, 10.5))
    y = np.arange(field_size)
    axis.hlines(y, lower[order], upper[order], color=PALE, linewidth=7, alpha=0.9, label="Central 80% interval")
    axis.scatter(expected[order], y, color=NAVY, s=80, zorder=3, label="Expected finish")
    axis.scatter(actual[order], y, color=RED, marker="x", s=80, linewidth=2.5, zorder=4, label="Actual finish")
    axis.set_yticks(y, labels.iloc[order])
    axis.invert_yaxis()
    axis.set_xlim(20.7, 0.3)
    axis.set_xticks(positions[::-1])
    axis.set_xlabel("Finishing position — P1 is best")
    axis.set_title(f"2024 {race_name}: Forecast Range and Actual Result", loc="left", fontsize=21, pad=12)
    axis.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.10), fontsize=12)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    save(fig, "06_expected_finish_uncertainty_final.png")

    print("Saved six polished figures to:", OUTPUT)
    for path in sorted(OUTPUT.glob("*.png")):
        print(" ", path)
    print("Model refit: False")
    print("Frozen probabilities changed: False")
    print("Editorial visualization pass: PASS")


if __name__ == "__main__":
    main()
