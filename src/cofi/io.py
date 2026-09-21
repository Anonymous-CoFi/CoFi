"""Manifest, cache, and route serialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(output)


def safe_video_id(video_id: str) -> str:
    return str(video_id).replace("/", "_").replace("\\", "_")


def cache_path(cache_dir: str | Path, video_id: str) -> Path:
    return Path(cache_dir) / f"{safe_video_id(video_id)}.npz"


def save_cache(
    path: str | Path,
    embeddings: np.ndarray,
    frame_indices: np.ndarray,
    timestamps: np.ndarray,
    sample_fps: float,
    cell_size: int,
    encoder_name: str,
    encoder_revision: str | None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        frame_embeddings=np.asarray(embeddings, np.float32),
        frame_indices=np.asarray(frame_indices, np.int64),
        timestamps=np.asarray(timestamps, np.float32),
        sample_fps=np.asarray(float(sample_fps), np.float32),
        cell_size=np.asarray(int(cell_size), np.int64),
        encoder_name=np.asarray(str(encoder_name)),
        encoder_revision=np.asarray(str(encoder_revision or "")),
        cache_version=np.asarray(2, np.int64),
    )
    temporary.replace(output)


def load_cache(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        required = {"frame_embeddings", "frame_indices", "timestamps"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(
                f"cache {path} predates the release preprocessing format; "
                f"missing {sorted(missing)}. Run `cofi preprocess` again.")
        return {
            "frame_embeddings": np.asarray(data["frame_embeddings"], np.float32),
            "frame_indices": np.asarray(data["frame_indices"], np.int64),
            "timestamps": np.asarray(data["timestamps"], np.float32),
            "sample_fps": float(data["sample_fps"]) if "sample_fps" in data else 1.0,
            "cell_size": int(data["cell_size"]) if "cell_size" in data else 192,
            "encoder_name": str(data["encoder_name"]) if "encoder_name" in data else "unknown",
            "encoder_revision": (
                str(data["encoder_revision"]) if "encoder_revision" in data else ""),
            "cache_version": int(data["cache_version"]) if "cache_version" in data else 1,
        }
