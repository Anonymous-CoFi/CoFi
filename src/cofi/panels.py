"""Panel rendering and public visual-unit schema."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from PIL import Image


def fit_square(image: Image.Image, size: int) -> Image.Image:
    source = image.convert("RGB")
    side = max(int(size), 1)
    scale = min(side / max(source.width, 1), side / max(source.height, 1))
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    resized = source.resize(
        (max(1, round(source.width * scale)), max(1, round(source.height * scale))),
        resampling,
    )
    tile = Image.new("RGB", (side, side), color=(0, 0, 0))
    tile.paste(resized, ((side - resized.width) // 2, (side - resized.height) // 2))
    return tile


def compose_2x2(images: Sequence[Image.Image], canvas_size: int = 384) -> Image.Image:
    if len(images) != 4:
        raise ValueError(f"2x2 panel needs four images, got {len(images)}")
    cell = int(canvas_size) // 2
    canvas = Image.new("RGB", (2 * cell, 2 * cell), color=(0, 0, 0))
    for index, image in enumerate(images):
        canvas.paste(fit_square(image, cell), ((index % 2) * cell, (index // 2) * cell))
    return canvas


def panel_unit(
    rows: Sequence[int], frame_indices: np.ndarray, timestamps: np.ndarray,
) -> dict[str, Any]:
    source_rows = [int(row) for row in rows]
    anchor = source_rows[(len(source_rows) - 1) // 2]
    return {
        "layout": "2x2_global",
        "role": "context",
        "anchor_row": anchor,
        "source_rows": source_rows,
        "source_frames": [int(frame_indices[row]) for row in source_rows],
        "source_timestamps": [float(timestamps[row]) for row in source_rows],
        "context_focus_role": "global_coverage_context_panel",
    }


def focus_unit(row: int, frame_indices: np.ndarray, timestamps: np.ndarray) -> dict[str, Any]:
    return {
        "layout": "1x1",
        "role": "focus",
        "anchor_row": int(row),
        "source_rows": [int(row)],
        "source_frames": [int(frame_indices[row])],
        "source_timestamps": [float(timestamps[row])],
        "context_focus_role": "residual_query_focus_1x1",
    }
