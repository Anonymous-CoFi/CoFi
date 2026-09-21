"""Frozen vision-language encoder adapter."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, np.float32)
    return array / np.maximum(np.linalg.norm(array, axis=-1, keepdims=True), 1e-12)


class SigLIPEncoder:
    def __init__(
        self, model_name: str = "google/siglip-so400m-patch14-384", device: str = "cuda",
        revision: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoProcessor, SiglipModel

        self.model_name = str(model_name)
        self.revision = revision
        self.torch = torch
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name, revision=revision)
        self.model = SiglipModel.from_pretrained(
            model_name, revision=revision).eval().to(device)

    def encode_images(self, images: Sequence[object], batch_size: int = 16) -> np.ndarray:
        batches = []
        for start in range(0, len(images), max(int(batch_size), 1)):
            inputs = self.processor(images=list(images[start:start + batch_size]), return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with self.torch.inference_mode():
                values = self.model.get_image_features(**inputs)
            batches.append(values.float().cpu().numpy())
        return _normalize(np.concatenate(batches))

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        inputs = self.processor(
            text=list(texts), padding="max_length", truncation=True, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            values = self.model.get_text_features(**inputs)
        return _normalize(values.float().cpu().numpy())
