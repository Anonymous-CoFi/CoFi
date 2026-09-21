"""Reference implementation of the final CoFi C4+F28 selector."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .kernel import build_kernel_geometry, cross_rbf
from .panels import focus_unit, panel_unit
from .posterior import condition_on_context, robust_query_mass


@dataclass(frozen=True)
class CoFiConfig:
    context_panels: int = 4
    frames_per_panel: int = 4
    focus_units: int = 28
    candidate_mass: float = 0.8
    state_weight: float = 0.5
    trajectory_radius: int = 1
    trajectory_stride: int = 1
    bandwidth_multiplier: float = 1.0
    focus_noise: float = 1e-6
    jitter: float = 1e-6
    max_scale_rows: int = 2048

    @property
    def raw_context_frames(self) -> int:
        return self.context_panels * self.frames_per_panel

    @property
    def visual_units(self) -> int:
        return self.context_panels + self.focus_units


@dataclass
class CoFiSelection:
    context_rows: list[int]
    focus_rows: list[int]
    visual_units: list[dict[str, Any]]
    query_mass: np.ndarray
    initial_variance: np.ndarray
    final_variance: np.ndarray
    acquisition_scores: list[float]
    diagnostics: dict[str, Any]


def uniform_rows(total: int, count: int) -> np.ndarray:
    target = min(max(int(count), 0), int(total))
    if target == 0:
        return np.empty(0, np.int64)
    rows = np.unique(np.linspace(0, total - 1, target).round().astype(np.int64))
    if len(rows) != target:
        raise AssertionError(f"uniform selection returned {len(rows)}/{target} rows")
    return rows


def repeat_to_count(rows: list[int], count: int) -> list[int]:
    if count <= 0:
        return []
    if not rows:
        raise ValueError("cannot repeat an empty selection")
    if len(rows) >= count:
        return rows[:count]
    positions = np.linspace(0, len(rows) - 1, count).round().astype(np.int64)
    return [int(rows[position]) for position in positions]


def _candidate_pool(mass: np.ndarray, target: int, threshold: float) -> np.ndarray:
    order = np.asarray(sorted(
        range(len(mass)), key=lambda row: (-float(mass[row]), int(row))), np.int64)
    if 0.0 < float(threshold) <= 1.0:
        count = int(np.searchsorted(np.cumsum(mass[order]), threshold, side="left")) + 1
        count = min(len(order), max(int(target), count))
    else:
        count = len(order)
    return order[:count]


def select_cofi(
    embeddings: np.ndarray,
    relevance: np.ndarray,
    frame_indices: np.ndarray,
    timestamps: np.ndarray,
    context_fidelity: np.ndarray,
    config: CoFiConfig = CoFiConfig(),
) -> CoFiSelection:
    """Select C context panels followed by context-conditioned F focus frames.

    ``context_fidelity`` has one value per raw context frame. It is the cosine
    similarity between the full-resolution embedding and the embedding after
    the same downsampling used for a 2x2 panel cell.
    """
    features = np.asarray(embeddings, np.float64)
    scores = np.asarray(relevance, np.float64).reshape(-1)
    frames = np.asarray(frame_indices, np.int64)
    times = np.asarray(timestamps, np.float64)
    total = len(features)
    if not (total == len(scores) == len(frames) == len(times)) or total == 0:
        raise ValueError("embeddings, relevance, frame indices, and timestamps must align")
    if total == 0:
        raise ValueError("CoFi requires at least one candidate frame")

    context = uniform_rows(total, config.raw_context_frames)
    fidelity = np.asarray(context_fidelity, np.float64).reshape(-1)
    if len(fidelity) != len(context):
        raise ValueError(f"expected {len(context)} context fidelity values, got {len(fidelity)}")

    geometry = build_kernel_geometry(
        features,
        radius=config.trajectory_radius,
        stride=config.trajectory_stride,
        state_weight=config.state_weight,
        bandwidth_multiplier=config.bandwidth_multiplier,
        max_scale_rows=config.max_scale_rows,
    )
    all_rows = np.arange(total, dtype=np.int64)
    cross = cross_rbf(
        geometry["states"], geometry["trajectories"], all_rows, context,
        geometry["state_scale"], geometry["trajectory_scale"], config.state_weight,
    )
    observed = cross[context]
    observed = (observed + observed.T) * 0.5
    np.fill_diagonal(observed, 1.0)
    projection, variance = condition_on_context(
        cross, observed, fidelity, jitter=config.jitter)
    initial_variance = variance.copy()
    query_mass, normalization = robust_query_mass(scores)
    candidates = _candidate_pool(query_mass, config.focus_units, config.candidate_mass)

    conditional = cross_rbf(
        geometry["states"], geometry["trajectories"], all_rows, candidates,
        geometry["state_scale"], geometry["trajectory_scale"], config.state_weight,
    ) - projection @ cross[candidates].T
    available = np.ones(len(candidates), dtype=bool)
    selected: list[int] = []
    selected_scores: list[float] = []
    target = min(config.focus_units, len(candidates))
    for _ in range(target):
        diagonal = np.maximum(conditional[candidates, np.arange(len(candidates))], 0.0)
        denominator = np.maximum(diagonal + config.focus_noise, config.jitter)
        integrated_gain = np.sum(query_mass[:, None] * conditional**2, axis=0) / denominator
        acquisition = query_mass[candidates] * integrated_gain
        acquisition[~available] = -np.inf
        best = float(np.max(acquisition))
        tied = np.flatnonzero(np.isclose(acquisition, best, rtol=1e-10, atol=1e-14))
        index = int(max(tied, key=lambda idx: (float(query_mass[candidates[idx]]), -int(candidates[idx]))))
        pivot = int(candidates[index])
        selected.append(pivot)
        selected_scores.append(best)
        available[index] = False
        factor = conditional[:, index] / np.sqrt(max(float(conditional[pivot, index]) + config.focus_noise, config.jitter))
        variance = np.maximum(variance - factor * factor, 0.0)
        conditional -= factor[:, None] * factor[candidates][None, :]

    repeated_context = repeat_to_count(context.tolist(), config.raw_context_frames)
    output_focus = repeat_to_count(selected, config.focus_units)
    panels = [
        panel_unit(repeated_context[start:start + config.frames_per_panel], frames, times)
        for start in range(0, config.raw_context_frames, config.frames_per_panel)
    ]
    focuses = [focus_unit(row, frames, times) for row in output_focus]
    units = panels + focuses
    units.sort(key=lambda unit: (
        float(np.mean(unit["source_timestamps"])),
        0 if unit["role"] == "context" else 1,
        int(unit["anchor_row"]),
    ))
    return CoFiSelection(
        context_rows=context.tolist(),
        focus_rows=selected,
        visual_units=units,
        query_mass=query_mass,
        initial_variance=initial_variance,
        final_variance=variance,
        acquisition_scores=selected_scores,
        diagnostics={
            "config": asdict(config),
            "candidate_pool_size": int(len(candidates)),
            "candidate_mass_retained": float(query_mass[candidates].sum()),
            "state_scale": float(geometry["state_scale"]),
            "trajectory_scale": float(geometry["trajectory_scale"]),
            "query_normalization": normalization,
            "initial_query_weighted_variance": float(query_mass @ initial_variance),
            "final_query_weighted_variance": float(query_mass @ variance),
        },
    )
