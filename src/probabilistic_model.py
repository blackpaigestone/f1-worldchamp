"""Models and metrics for Formula 1 finishing-position distributions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


MAX_POSITION = 22

BASELINE_FEATURES = [
    "grid_clean",
    "qualifying_position",
]

FULL_FEATURES = [
    "grid_clean",
    "qualifying_position",
    "field_size",
    "season_progress",
    "driver_prior_starts",
    "constructor_prior_races",
    "driver_avg_finish_last5",
    "driver_points_last5",
    "driver_dnf_rate_last5",
    "driver_avg_grid_last5",
    "driver_avg_finish_last10",
    "driver_points_last10",
    "driver_dnf_rate_last10",
    "driver_avg_grid_last10",
    "constructor_avg_finish_last5",
    "constructor_points_last5",
    "constructor_dnf_rate_last5",
    "constructor_avg_grid_last5",
    "constructor_avg_finish_last10",
    "constructor_points_last10",
    "constructor_dnf_rate_last10",
    "constructor_avg_grid_last10",
    "driver_circuit_avg_finish_prior",
    "constructor_circuit_avg_finish_prior",
    "driver_standing_points_prerace",
    "driver_standing_position_prerace",
    "driver_standing_wins_prerace",
    "constructor_standing_points_prerace",
    "constructor_standing_position_prerace",
    "constructor_standing_wins_prerace",
    "driver_prior_season_points",
    "driver_prior_season_position",
    "driver_prior_season_wins",
    "constructor_prior_season_points",
    "constructor_prior_season_position",
    "constructor_prior_season_wins",
]


@dataclass
class ConstantBinaryModel:
    """Fallback for a cumulative threshold with only one training class."""

    probability: float

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        p = np.full(len(x), self.probability, dtype=float)
        return np.column_stack([1.0 - p, p])


class CumulativeOrdinalLogit:
    """Estimate P(Y <= k) at each ordered finishing-position threshold.

    Threshold probabilities are fitted independently, monotonically repaired,
    differenced into a probability mass function, masked to each race's field
    size, and normalized to sum to one.
    """

    def __init__(
        self,
        max_position: int = MAX_POSITION,
        c: float = 1.0,
        max_iter: int = 2_000,
        random_state: int = 42,
        clip_value: float = 10.0,
        min_probability: float = 1e-6,
    ) -> None:
        self.max_position = max_position
        self.c = c
        self.max_iter = max_iter
        self.random_state = random_state
        self.clip_value = clip_value
        self.min_probability = min_probability
        self.imputer = SimpleImputer(strategy="median", add_indicator=True)
        self.scaler = StandardScaler()
        self.models: list[LogisticRegression | ConstantBinaryModel] = []

    def fit(self, x: pd.DataFrame, y: pd.Series | np.ndarray) -> "CumulativeOrdinalLogit":
        x_array = self.imputer.fit_transform(x)
        x_array = self.scaler.fit_transform(x_array)
        x_array = np.clip(x_array, -self.clip_value, self.clip_value)
        y_array = np.asarray(y, dtype=int)
        self.models = []

        for threshold in range(1, self.max_position):
            binary_y = (y_array <= threshold).astype(int)
            classes = np.unique(binary_y)
            if len(classes) == 1:
                model: LogisticRegression | ConstantBinaryModel = ConstantBinaryModel(
                    float(classes[0])
                )
            else:
                model = LogisticRegression(
                    C=self.c,
                    max_iter=self.max_iter,
                    solver="liblinear",
                    random_state=self.random_state,
                )
                model.fit(x_array, binary_y)
            self.models.append(model)
        return self

    def predict_proba(
        self,
        x: pd.DataFrame,
        field_size: pd.Series | np.ndarray | None = None,
    ) -> np.ndarray:
        if not self.models:
            raise RuntimeError("Model must be fitted before prediction")

        x_array = self.imputer.transform(x)
        x_array = self.scaler.transform(x_array)
        x_array = np.clip(x_array, -self.clip_value, self.clip_value)
        cumulative = np.column_stack(
            [model.predict_proba(x_array)[:, 1] for model in self.models]
        )

        # Independent threshold models can cross. A cumulative maximum is a
        # deterministic monotonic repair before converting CDFs to masses.
        cumulative = np.maximum.accumulate(cumulative, axis=1)
        cumulative = np.clip(cumulative, 0.0, 1.0)
        bounded = np.column_stack(
            [np.zeros(len(x_array)), cumulative, np.ones(len(x_array))]
        )
        probabilities = np.diff(bounded, axis=1)
        probabilities = np.clip(probabilities, 0.0, None)

        positions = np.arange(1, self.max_position + 1)
        if field_size is not None:
            sizes = np.asarray(field_size, dtype=int)
            allowed = positions[None, :] <= sizes[:, None]
        else:
            allowed = np.ones_like(probabilities, dtype=bool)

        # Monotonic CDF repair can create flat sections and therefore exact
        # zero-mass positions. A very small floor prevents a single surprise
        # result from dominating log loss while leaving RPS effectively intact.
        probabilities = np.where(
            allowed,
            np.maximum(probabilities, self.min_probability),
            0.0,
        )

        row_sums = probabilities.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0):
            raise ValueError("At least one predicted distribution has no probability mass")
        return probabilities / row_sums


def ranked_probability_score(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
) -> float:
    y = np.asarray(y_true, dtype=int)
    predicted_cdf = np.cumsum(probabilities, axis=1)[:, :-1]
    thresholds = np.arange(1, probabilities.shape[1])
    observed_cdf = (y[:, None] <= thresholds[None, :]).astype(float)
    return float(np.mean(np.mean((predicted_cdf - observed_cdf) ** 2, axis=1)))


def distribution_log_loss(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    epsilon: float = 1e-15,
) -> float:
    y = np.asarray(y_true, dtype=int)
    actual_probability = probabilities[np.arange(len(y)), y - 1]
    return float(-np.mean(np.log(np.clip(actual_probability, epsilon, 1.0))))


def event_brier_score(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    cutoff: int,
) -> float:
    y = np.asarray(y_true, dtype=int)
    predicted = probabilities[:, :cutoff].sum(axis=1)
    observed = (y <= cutoff).astype(float)
    return float(np.mean((predicted - observed) ** 2))


def apply_temperature(
    probabilities: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Soften or sharpen distributions while preserving row normalization."""
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")
    adjusted = np.power(np.asarray(probabilities, dtype=float), 1.0 / temperature)
    row_sums = adjusted.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("At least one adjusted distribution has no probability mass")
    return adjusted / row_sums


def balance_race_matrix(
    probabilities: np.ndarray,
    field_size: int | None = None,
    tolerance: float = 1e-10,
    max_iter: int = 10_000,
    min_probability: float = 1e-12,
) -> np.ndarray:
    """Balance one race into a coherent driver-by-position probability matrix.

    The active portion of the matrix is square: one row per driver and one
    column per available finishing position. Iterative proportional fitting
    (Sinkhorn balancing) preserves the model's relative probability structure
    while making every row and every active position column sum to one.
    """
    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("probabilities must be a two-dimensional array")
    if not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise ValueError("probabilities must be finite and non-negative")

    n_drivers, n_positions = matrix.shape
    active_size = n_drivers if field_size is None else int(field_size)
    if active_size != n_drivers:
        raise ValueError(
            "A coherent race matrix requires field_size to equal the number "
            "of drivers in the race"
        )
    if active_size > n_positions:
        raise ValueError("field_size exceeds the available position columns")

    active = np.maximum(matrix[:, :active_size], min_probability).copy()
    converged = False
    for _ in range(max_iter):
        active /= active.sum(axis=1, keepdims=True)
        active /= active.sum(axis=0, keepdims=True)
        row_error = np.max(np.abs(active.sum(axis=1) - 1.0))
        column_error = np.max(np.abs(active.sum(axis=0) - 1.0))
        if max(row_error, column_error) <= tolerance:
            converged = True
            break

    if not converged:
        raise RuntimeError("Race-matrix balancing did not converge")

    balanced = np.zeros_like(matrix)
    balanced[:, :active_size] = active
    return balanced


def balance_race_probabilities(
    probabilities: np.ndarray,
    race_ids: pd.Series | np.ndarray,
    field_sizes: pd.Series | np.ndarray,
    **balance_kwargs: float | int,
) -> np.ndarray:
    """Balance a stacked prediction array independently within every race."""
    matrix = np.asarray(probabilities, dtype=float)
    races = np.asarray(race_ids)
    sizes = np.asarray(field_sizes, dtype=int)
    if len(matrix) != len(races) or len(matrix) != len(sizes):
        raise ValueError("probabilities, race_ids, and field_sizes must align")

    balanced = np.zeros_like(matrix)
    for race_id in pd.unique(races):
        indices = np.flatnonzero(races == race_id)
        race_sizes = np.unique(sizes[indices])
        if len(race_sizes) != 1:
            raise ValueError(f"Race {race_id!r} has inconsistent field sizes")
        balanced[indices] = balance_race_matrix(
            matrix[indices],
            field_size=int(race_sizes[0]),
            **balance_kwargs,
        )
    return balanced


def race_coherence_error(
    probabilities: np.ndarray,
    race_ids: pd.Series | np.ndarray,
    field_sizes: pd.Series | np.ndarray,
) -> dict[str, float]:
    """Summarize row/position deviations from a coherent race forecast."""
    matrix = np.asarray(probabilities, dtype=float)
    races = np.asarray(race_ids)
    sizes = np.asarray(field_sizes, dtype=int)
    row_errors: list[float] = []
    column_errors: list[float] = []
    for race_id in pd.unique(races):
        indices = np.flatnonzero(races == race_id)
        active_size = int(np.unique(sizes[indices]).item())
        active = matrix[indices, :active_size]
        row_errors.extend(np.abs(active.sum(axis=1) - 1.0))
        column_errors.extend(np.abs(active.sum(axis=0) - 1.0))
    return {
        "max_row_error": float(np.max(row_errors)),
        "mean_row_error": float(np.mean(row_errors)),
        "max_column_error": float(np.max(column_errors)),
        "mean_column_error": float(np.mean(column_errors)),
    }


def evaluate_distribution(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    positions = np.arange(1, probabilities.shape[1] + 1)
    expected_position = probabilities @ positions
    modal_position = probabilities.argmax(axis=1) + 1

    return {
        "rps": ranked_probability_score(y, probabilities),
        "log_loss": distribution_log_loss(y, probabilities),
        "expected_position_mae": float(np.mean(np.abs(expected_position - y))),
        "modal_position_accuracy": float(np.mean(modal_position == y)),
        "win_brier": event_brier_score(y, probabilities, cutoff=1),
        "podium_brier": event_brier_score(y, probabilities, cutoff=3),
        "points_brier": event_brier_score(y, probabilities, cutoff=10),
    }


def distribution_frame(
    metadata: pd.DataFrame,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """Return one row per driver-position probability for analysis and plots."""
    probability_columns = [
        f"p_position_{position}" for position in range(1, probabilities.shape[1] + 1)
    ]
    wide = pd.concat(
        [
            metadata.reset_index(drop=True),
            pd.DataFrame(probabilities, columns=probability_columns),
        ],
        axis=1,
    )
    long = wide.melt(
        id_vars=list(metadata.columns),
        value_vars=probability_columns,
        var_name="position_label",
        value_name="position_probability",
    )
    long["position"] = long["position_label"].str.extract(r"(\d+)$").astype(int)
    return long.drop(columns="position_label")
