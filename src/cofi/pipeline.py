"""Question-conditioned CoFi selection over a preprocessed candidate cache."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from .datasets import load_dataset
from .encoder import SigLIPEncoder
from .io import cache_path, load_cache, write_jsonl
from .panels import fit_square
from .selector import CoFiConfig, select_cofi, uniform_rows
from .video import read_timestamps


def select_dataset(
    dataset: str,
    dataset_root: str,
    annotation: str | None,
    cache_dir: str,
    output_path: str,
    encoder: SigLIPEncoder,
    config: CoFiConfig,
    batch_size: int = 16,
    limit: int = 0,
) -> None:
    samples = load_dataset(dataset, dataset_root, annotation, limit)
    video_state: dict[str, dict[str, Any]] = {}
    output: list[dict[str, Any]] = []
    for sample in samples:
        video_id = sample.video_id
        if video_id not in video_state:
            cache = load_cache(cache_path(cache_dir, video_id))
            embeddings = cache["frame_embeddings"]
            frames = cache["frame_indices"]
            times = cache["timestamps"]
            context = uniform_rows(len(embeddings), config.raw_context_frames)
            normalized = embeddings / np.maximum(
                np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
            images = read_timestamps(
                sample.video_path, [float(times[row]) for row in context])
            proxies = [fit_square(image, int(cache["cell_size"])) for image in images]
            cell_embeddings = encoder.encode_images(proxies, batch_size=batch_size)
            fidelity = np.clip(
                np.sum(normalized[context] * cell_embeddings, axis=1), 0.0, 1.0)
            video_state[video_id] = {
                "embeddings": embeddings,
                "frames": frames,
                "times": times,
                "fidelity": fidelity,
                "sample_fps": cache["sample_fps"],
                "cell_size": cache["cell_size"],
                "encoder_name": cache["encoder_name"],
                "encoder_revision": cache["encoder_revision"],
            }
        state = video_state[video_id]
        query = encoder.encode_texts([sample.question])[0]
        relevance = state["embeddings"] @ query
        selection = select_cofi(
            state["embeddings"], relevance, state["frames"], state["times"],
            state["fidelity"], config,
        )
        result = {
            "dataset": sample.dataset,
            "video_id": sample.video_id,
            "question_id": sample.question_id,
            "video_path": sample.video_path,
            "question": sample.question,
            "options": sample.options,
            "answer": sample.answer,
            **sample.metadata,
            "method": "CoFi",
            "visual_units": selection.visual_units,
            "context_rows": selection.context_rows,
            "focus_rows": selection.focus_rows,
            "diagnostics": selection.diagnostics,
            "config": asdict(config),
            "candidate_sampling": {
                "fps": float(state["sample_fps"]),
                "num_candidates": int(len(state["embeddings"])),
                "cell_size": int(state["cell_size"]),
                "encoder": str(state["encoder_name"]),
                "encoder_revision": str(state["encoder_revision"]),
            },
        }
        output.append(result)
    write_jsonl(output_path, output)
