"""Query mass and Gaussian-kernel posterior operations."""

from __future__ import annotations

import numpy as np


def robust_query_mass(relevance: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Median/MAD normalization followed by softplus and L1 normalization."""
    scores = np.asarray(relevance, np.float64)
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median)))
    scale = 1.4826 * mad
    if scale <= 1e-8:
        scale = float(scores.std())
    if scale <= 1e-8:
        scale = 1.0
    normalized = (scores - median) / scale
    evidence = np.logaddexp(0.0, normalized)
    mass = evidence / max(float(evidence.sum()), 1e-12)
    return mass, {"median": median, "mad": mad, "scale": scale}


def fidelity_noise(fidelity: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
    """Odds mapping: a less faithful panel cell is a noisier observation."""
    values = np.clip(np.asarray(fidelity, np.float64), epsilon, 1.0)
    return (1.0 - values) / values


def condition_on_context(
    context_cross: np.ndarray,
    context_kernel: np.ndarray,
    fidelity: np.ndarray,
    jitter: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    cross = np.asarray(context_cross, np.float64)
    observed = np.asarray(context_kernel, np.float64)
    system = observed + np.diag(fidelity_noise(fidelity) + float(jitter))
    try:
        projection = np.linalg.solve(system, cross.T).T
    except np.linalg.LinAlgError:
        projection = cross @ np.linalg.pinv(system, rcond=1e-8)
    reduction = np.clip(np.sum(projection * cross, axis=1), 0.0, 1.0)
    return projection, np.maximum(1.0 - reduction, 0.0)

