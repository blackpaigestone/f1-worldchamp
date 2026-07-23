"""Models and metrics for Formula 1 finishing-position distributions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
        cumulative = self._predict_raw_cumulative_array(x_array)

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

    def _predict_raw_cumulative_array(self, x_array: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [model.predict_proba(x_array)[:, 1] for model in self.models]
        )

    def predict_cumulative(
        self,
        x: pd.DataFrame,
        repair: bool = False,
    ) -> np.ndarray:
        """Return threshold CDFs before or after monotonic repair."""
        if not self.models:
            raise RuntimeError("Model must be fitted before prediction")
        x_array = self.imputer.transform(x)
        x_array = self.scaler.transform(x_array)
        x_array = np.clip(x_array, -self.clip_value, self.clip_value)
        cumulative = self._predict_raw_cumulative_array(x_array)
        return np.maximum.accumulate(cumulative, axis=1) if repair else cumulative


class ProportionalOddsLogit:
    """True ordinal logistic model with shared coefficients and ordered cuts."""

    def __init__(
        self,
        max_position: int = MAX_POSITION,
        max_iter: int = 500,
        clip_value: float = 10.0,
        min_probability: float = 1e-6,
    ) -> None:
        self.max_position = max_position
        self.max_iter = max_iter
        self.clip_value = clip_value
        self.min_probability = min_probability
        self.imputer = SimpleImputer(strategy="median", add_indicator=True)
        self.scaler = StandardScaler()
        self.nonconstant_mask_: np.ndarray | None = None
        self.center_offset_: np.ndarray | None = None
        self.model: Any | None = None
        self.result = None
        self.classes_: np.ndarray | None = None

    def fit(self, x: pd.DataFrame, y: pd.Series | np.ndarray) -> "ProportionalOddsLogit":
        try:
            from statsmodels.miscmodels.ordinal_model import OrderedModel
        except ImportError as error:
            raise ImportError(
                "ProportionalOddsLogit requires statsmodels. Install it with "
                "`python -m pip install statsmodels`."
            ) from error
        x_array = self.imputer.fit_transform(x)
        feature_variance = np.var(x_array, axis=0)
        self.nonconstant_mask_ = np.isfinite(feature_variance) & (feature_variance > 1e-12)
        if not self.nonconstant_mask_.any():
            raise ValueError("No nonconstant predictors remain after preprocessing")
        x_array = x_array[:, self.nonconstant_mask_]
        x_array = self.scaler.fit_transform(x_array)
        # OrderedModel includes threshold intercepts and must not receive a
        # separate constant. Recenter explicitly, then bypass statsmodels'
        # rank-based detector, which can mistake correlated predictors for an
        # implicit intercept even after zero-variance columns are removed.
        self.center_offset_ = x_array.mean(axis=0, keepdims=True)
        x_array = x_array - self.center_offset_
        x_array = np.clip(x_array, -self.clip_value, self.clip_value)
        y_array = np.asarray(y, dtype=int)
        self.classes_ = np.sort(np.unique(y_array))
        self.model = OrderedModel(
            y_array,
            x_array,
            distr="logit",
            hasconst=False,
        )
        self.result = self.model.fit(
            method="lbfgs",
            maxiter=self.max_iter,
            disp=False,
        )
        return self

    def predict_proba(
        self,
        x: pd.DataFrame,
        field_size: pd.Series | np.ndarray | None = None,
    ) -> np.ndarray:
        if (
            self.result is None
            or self.classes_ is None
            or self.nonconstant_mask_ is None
            or self.center_offset_ is None
        ):
            raise RuntimeError("Model must be fitted before prediction")
        x_array = self.imputer.transform(x)
        x_array = x_array[:, self.nonconstant_mask_]
        x_array = self.scaler.transform(x_array)
        x_array = x_array - self.center_offset_
        x_array = np.clip(x_array, -self.clip_value, self.clip_value)
        predicted = np.asarray(self.result.model.predict(self.result.params, exog=x_array))
        probabilities = np.zeros((len(x_array), self.max_position), dtype=float)
        probabilities[:, self.classes_ - 1] = predicted

        positions = np.arange(1, self.max_position + 1)
        allowed = (
            positions[None, :] <= np.asarray(field_size, dtype=int)[:, None]
            if field_size is not None
            else np.ones_like(probabilities, dtype=bool)
        )
        probabilities = np.where(
            allowed, np.maximum(probabilities, self.min_probability), 0.0
        )
        return probabilities / probabilities.sum(axis=1, keepdims=True)


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


def distribution_shape_diagnostics(probabilities: np.ndarray) -> dict[str, float]:
    """Measure roughness, fragmentation, concentration, and effective width."""
    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("probabilities must be a two-dimensional array")
    local_peaks = (
        (matrix[:, 1:-1] > matrix[:, :-2])
        & (matrix[:, 1:-1] > matrix[:, 2:])
    ).sum(axis=1)
    local_peaks += (matrix[:, 0] > matrix[:, 1]).astype(int)
    local_peaks += (matrix[:, -1] > matrix[:, -2]).astype(int)
    total_variation = np.abs(np.diff(matrix, axis=1)).sum(axis=1)
    entropy = -np.sum(matrix * np.log(np.clip(matrix, 1e-15, 1.0)), axis=1)
    sorted_probability = np.sort(matrix, axis=1)[:, ::-1]
    effective_width_80 = (np.cumsum(sorted_probability, axis=1) < 0.8).sum(axis=1) + 1
    return {
        "mean_local_peaks": float(np.mean(local_peaks)),
        "share_multimodal": float(np.mean(local_peaks > 1)),
        "mean_total_variation": float(np.mean(total_variation)),
        "mean_entropy": float(np.mean(entropy)),
        "mean_effective_width_80": float(np.mean(effective_width_80)),
        "mean_max_probability": float(np.mean(matrix.max(axis=1))),
    }


def cumulative_crossing_diagnostics(cumulative: np.ndarray) -> dict[str, float]:
    """Measure violations of P(Y<=k) <= P(Y<=k+1)."""
    cdf = np.asarray(cumulative, dtype=float)
    violations = np.diff(cdf, axis=1) < 0
    magnitudes = np.maximum(-np.diff(cdf, axis=1), 0.0)
    return {
        "crossing_pair_count": int(violations.sum()),
        "rows_with_crossing": int(violations.any(axis=1).sum()),
        "share_rows_with_crossing": float(violations.any(axis=1).mean()),
        "mean_crossings_per_row": float(violations.sum(axis=1).mean()),
        "maximum_crossing_magnitude": float(magnitudes.max()),
        "mean_crossing_magnitude_when_present": float(
            magnitudes[violations].mean() if violations.any() else 0.0
        ),
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
