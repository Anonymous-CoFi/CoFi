"""Minimal deterministic video decoding utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def sample_video(
    path: str | Path, sample_fps: float = 1.0,
) -> tuple[list[Image.Image], np.ndarray, np.ndarray]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or frame_count <= 0:
        capture.release()
        raise RuntimeError(f"invalid video metadata: fps={fps}, frames={frame_count}")
    frame_indices, timestamps = _candidate_metadata(fps, frame_count, sample_fps)
    images: list[Image.Image] = []
    for frame in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
        ok, bgr = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"failed to decode frame {frame} from {path}")
        images.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    capture.release()
    return images, frame_indices, timestamps


def _candidate_metadata(
    source_fps: float, frame_count: int, sample_fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Match the frame-index sampling protocol used in the paper experiments."""
    rate = float(sample_fps)
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError(f"sample_fps must be positive, got {sample_fps}")
    step = max(float(source_fps) / rate, 1.0)
    indices = np.arange(0, int(frame_count), step).round().astype(np.int64)
    indices = np.unique(np.minimum(indices, int(frame_count) - 1))
    # Store the actual source-frame time, not the idealized sampling-grid time.
    return indices, indices.astype(np.float64) / float(source_fps)


def sample_metadata(
    path: str | Path, sample_fps: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return candidate frame indices and timestamps without retaining images."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if fps <= 0 or frame_count <= 0:
        raise RuntimeError(f"invalid video metadata: fps={fps}, frames={frame_count}")
    return _candidate_metadata(fps, frame_count, sample_fps)


def sample_video_1fps(path: str | Path) -> tuple[list[Image.Image], np.ndarray, np.ndarray]:
    """Backward-compatible alias for exact 1 FPS sampling."""
    return sample_video(path, sample_fps=1.0)


def sample_metadata_1fps(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible alias for exact 1 FPS candidate metadata."""
    return sample_metadata(path, sample_fps=1.0)


def read_rows(path: str | Path, frame_indices: np.ndarray, rows: np.ndarray) -> list[Image.Image]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    images: list[Image.Image] = []
    for row in rows:
        frame = int(frame_indices[int(row)])
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, bgr = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"failed to decode frame {frame} from {path}")
        images.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    capture.release()
    return images


def read_timestamps(path: str | Path, timestamps: list[float]) -> list[Image.Image]:
    """Decode the exact timestamp observations used by the paper QA runner."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    images: list[Image.Image | None] = []
    failed: list[int] = []
    if capture.isOpened():
        for index, timestamp in enumerate(timestamps):
            capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0)
            ok, frame = capture.read()
            if ok:
                images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            else:
                images.append(None)
                failed.append(index)
        capture.release()
    else:
        images = [None] * len(timestamps)
        failed = list(range(len(timestamps)))

    if failed:
        try:
            from decord import VideoReader, cpu

            reader = VideoReader(str(path), ctx=cpu(0))
            fps = max(float(reader.get_avg_fps()), 1e-6)
            indices = [
                min(max(round(float(timestamps[index]) * fps), 0), len(reader) - 1)
                for index in failed
            ]
            arrays = reader.get_batch(indices).asnumpy()
            for position, index in enumerate(failed):
                images[index] = Image.fromarray(arrays[position])
        except Exception as exc:
            first = failed[0]
            raise RuntimeError(
                f"failed reading {path} at {float(timestamps[first]):.3f}s") from exc
    return [image for image in images if image is not None]
