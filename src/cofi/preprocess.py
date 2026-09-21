"""Question-independent candidate-frame preprocessing."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .datasets import load_dataset
from .encoder import SigLIPEncoder
from .io import cache_path, load_cache, save_cache


def encode_candidate_video(
    video_path: str,
    encoder: SigLIPEncoder,
    sample_fps: float,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact streaming 1-fps preprocessing used in the experiments."""
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    native_fps = float(capture.get(cv2.CAP_PROP_FPS) or 1.0)
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(native_fps / max(float(sample_fps), 1e-6), 1.0)
    positions = np.arange(0, count, step).round().astype(np.int64)
    positions = np.unique(np.minimum(positions, max(0, count - 1)))
    size = max(int(batch_size), 1)
    image_batch: list[Image.Image] = []
    index_batch: list[int] = []
    encoded_parts: list[np.ndarray] = []
    kept: list[int] = []

    def flush() -> None:
        if not image_batch:
            return
        encoded_parts.append(encoder.encode_images(image_batch, batch_size=size))
        kept.extend(index_batch)
        image_batch.clear()
        index_batch.clear()

    try:
        for index in positions:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok:
                continue
            image_batch.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            index_batch.append(int(index))
            if len(image_batch) >= size:
                flush()
        flush()
    finally:
        capture.release()
    if not encoded_parts:
        raise RuntimeError(f"no candidate frames decoded from video: {video_path}")
    indices = np.asarray(kept, np.int64)
    return (np.concatenate(encoded_parts), indices,
            indices.astype(np.float32) / native_fps)


def preprocess_dataset(
    dataset: str,
    dataset_root: str,
    annotation: str | None,
    cache_dir: str,
    encoder: SigLIPEncoder,
    batch_size: int = 16,
    sample_fps: float = 1.0,
    cell_size: int = 192,
    overwrite: bool = False,
    limit: int = 0,
) -> None:
    """Build a reusable candidate cache once per unique video.

    This matches the paper preprocessing: exact 1-fps frame positions and
    standalone SigLIP embeddings are cached once per video. Panel-cell
    fidelity is evaluated later only for the selected context observations.
    """
    samples = load_dataset(dataset, dataset_root, annotation, limit)
    by_video = {sample.video_id: sample for sample in samples}

    for video_id, sample in by_video.items():
        output = cache_path(cache_dir, video_id)
        if output.exists() and not overwrite:
            cached = load_cache(output)
            expected_name = getattr(encoder, "model_name", type(encoder).__name__)
            expected_revision = str(getattr(encoder, "revision", None) or "")
            matches = (
                np.isclose(float(cached["sample_fps"]), float(sample_fps))
                and int(cached["cell_size"]) == int(cell_size)
                and str(cached["encoder_name"]) == str(expected_name)
                and str(cached["encoder_revision"]) == expected_revision
            )
            if not matches:
                raise RuntimeError(
                    f"cache settings do not match the active config: {output}. "
                    "Set OVERWRITE=1 or use a new CACHE_DIR.")
            continue
        embeddings, frames, times = encode_candidate_video(
            sample.video_path, encoder, sample_fps, batch_size)

        save_cache(
            output,
            embeddings,
            frames,
            times,
            sample_fps=sample_fps,
            cell_size=cell_size,
            encoder_name=getattr(encoder, "model_name", type(encoder).__name__),
            encoder_revision=getattr(encoder, "revision", None),
        )
