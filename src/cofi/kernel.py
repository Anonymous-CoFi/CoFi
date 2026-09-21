"""Video-centered state--trajectory kernel used by CoFi."""

from __future__ import annotations

import numpy as np


def normalize_rows(values: np.ndarray) -> np.ndarray:
    rows = np.asarray(values, dtype=np.float64)
    return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)


def squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lhs, rhs = np.asarray(left, np.float64), np.asarray(right, np.float64)
    distances = (
        np.sum(lhs * lhs, axis=1)[:, None]
        + np.sum(rhs * rhs, axis=1)[None, :]
        - 2.0 * lhs @ rhs.T
    )
    return np.maximum(distances, 0.0)


def median_positive(values: np.ndarray) -> float:
    matrix = np.asarray(values, np.float64)
    if len(matrix) < 2:
        return 1.0
    samples = matrix[np.triu_indices(len(matrix), k=1)]
    samples = samples[np.isfinite(samples) & (samples > 1e-12)]
    return float(np.median(samples)) if len(samples) else 1.0


def state_trajectory_features(
    embeddings: np.ndarray, radius: int = 1, stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove the video-common direction and form aligned local trajectories."""
    frames = np.asarray(embeddings, np.float32)
    centered = frames - frames.mean(axis=0, keepdims=True)
    states = centered / np.maximum(
        np.linalg.norm(centered, axis=1, keepdims=True), 1e-12)
    radius, stride = max(int(radius), 0), max(int(stride), 1)
    offsets = list(range(-radius, radius + 1, stride))
    if 0 not in offsets:
        offsets.append(0)
        offsets.sort()
    rows = np.arange(len(states))
    aligned = [states[np.clip(rows + offset, 0, len(states) - 1)] for offset in offsets]
    trajectories = np.concatenate(aligned, axis=1) / np.sqrt(len(aligned))
    return np.asarray(states, np.float32), np.asarray(trajectories, np.float32)


def cross_rbf(
    states: np.ndarray,
    trajectories: np.ndarray,
    left_rows: np.ndarray,
    right_rows: np.ndarray,
    state_scale: float,
    trajectory_scale: float,
    state_weight: float = 0.5,
) -> np.ndarray:
    ds = squared_distances(states[left_rows], states[right_rows])
    dh = squared_distances(trajectories[left_rows], trajectories[right_rows])
    alpha = float(np.clip(state_weight, 0.0, 1.0))
    exponent = (
        alpha * ds / max(float(state_scale), 1e-12)
        + (1.0 - alpha) * dh / max(float(trajectory_scale), 1e-12)
    )
    return np.exp(-exponent)


def build_kernel_geometry(
    embeddings: np.ndarray,
    radius: int = 1,
    stride: int = 1,
    state_weight: float = 0.5,
    bandwidth_multiplier: float = 1.0,
    max_scale_rows: int = 2048,
) -> dict[str, object]:
    states, trajectories = state_trajectory_features(embeddings, radius, stride)
    total = len(states)
    maximum = max(int(max_scale_rows), 1)
    scale_rows = (
        np.arange(total, dtype=np.int64)
        if total <= maximum
        else np.unique(np.linspace(0, total - 1, maximum).round().astype(np.int64))
    )
    multiplier = max(float(bandwidth_multiplier), 1e-12)
    state_scale = median_positive(squared_distances(states[scale_rows], states[scale_rows])) * multiplier
    trajectory_scale = median_positive(
        squared_distances(trajectories[scale_rows], trajectories[scale_rows])
    ) * multiplier
    return {
        "states": states,
        "trajectories": trajectories,
        "state_scale": state_scale,
        "trajectory_scale": trajectory_scale,
        "state_weight": float(state_weight),
        "scale_rows": scale_rows,
    }
