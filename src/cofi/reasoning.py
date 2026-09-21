"""LLaVA-Video reasoning over CoFi visual units."""

from __future__ import annotations

import copy
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from .io import read_jsonl, write_jsonl
from .panels import compose_2x2
from .prompt import build_multiple_choice_prompt
from .video import read_timestamps


def parse_multiple_choice_answer(text: str, count: int) -> str | None:
    allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:count]
    patterns = (
        rf"(?i)(?:answer|option)(?:\s+is)?\s*[:：]?\s*\(?([{allowed}])\)?",
        rf"(?i)^\s*\(?([{allowed}])\)?(?:[.)\s]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, text.strip())
        if match:
            return match.group(1).upper()
    return None


def _add_llava_next_path(path: str) -> None:
    root = Path(path).expanduser().resolve()
    if not (root / "llava" / "model" / "builder.py").exists():
        raise RuntimeError(f"invalid LLaVA-NeXT checkout: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


class LocalLlavaVideo:
    """Official LLaVA-Video inference path used in the paper experiments."""

    def __init__(self, model_path: str, llava_next_path: str,
                 model_revision: str | None = None) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("LLaVA-Video reasoning requires a CUDA GPU")
        _add_llava_next_path(llava_next_path)
        import transformers.modeling_utils as modeling_utils
        from transformers import pytorch_utils

        for name in (
            "apply_chunking_to_forward", "find_pruneable_heads_and_indices",
            "prune_linear_layer",
        ):
            if not hasattr(modeling_utils, name) and hasattr(pytorch_utils, name):
                setattr(modeling_utils, name, getattr(pytorch_utils, name))

        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token
        from llava.model.builder import load_pretrained_model

        if model_revision and not Path(model_path).expanduser().exists():
            from huggingface_hub import snapshot_download
            model_path = snapshot_download(model_path, revision=model_revision)

        tokenizer, model, processor, _ = load_pretrained_model(
            model_path, None, "llava_qwen", device_map="auto",
            torch_dtype="bfloat16", attn_implementation="sdpa",
        )
        model.eval()
        self.torch = torch
        self.tokenizer = tokenizer
        self.model = model
        self.processor = processor
        self.device = torch.device("cuda:0")
        self.dtype = torch.bfloat16
        self.default_image_token = DEFAULT_IMAGE_TOKEN
        self.image_token_index = IMAGE_TOKEN_INDEX
        self.conv_templates = conv_templates
        self.tokenizer_image_token = tokenizer_image_token

    def generate(self, prompt: str, images: Sequence[object], max_new_tokens: int,
                 do_sample: bool = False) -> str:
        if not images:
            raise ValueError("LLaVA-Video requires at least one visual unit")
        video = self.processor.preprocess(
            list(images), return_tensors="pt")["pixel_values"]
        video = video.to(device=self.device, dtype=self.dtype)
        conversation = copy.deepcopy(self.conv_templates["qwen_1_5"])
        conversation.append_message(
            conversation.roles[0], self.default_image_token + "\n" + prompt)
        conversation.append_message(conversation.roles[1], None)
        input_ids = self.tokenizer_image_token(
            conversation.get_prompt(), self.tokenizer, self.image_token_index,
            return_tensors="pt",
        ).unsqueeze(0).to(self.device)
        input_length = int(input_ids.shape[-1])
        with self.torch.inference_mode():
            generated = self.model.generate(
                input_ids, images=[video], modalities=["video"],
                do_sample=do_sample, max_new_tokens=max_new_tokens, use_cache=True,
            )
        if (generated.shape[1] >= input_length and
                self.torch.equal(generated[:, :input_length], input_ids)):
            generated = generated[:, input_length:]
        return str(self.tokenizer.batch_decode(
            generated, skip_special_tokens=True)[0]).strip()


def render_visual_units(record: dict[str, Any], canvas_size: int) -> list[object]:
    timestamps = [
        float(timestamp)
        for unit in record["visual_units"]
        for timestamp in unit["source_timestamps"]
    ]
    decoded = read_timestamps(record["video_path"], timestamps)
    rendered: list[object] = []
    offset = 0
    for unit in record["visual_units"]:
        count = len(unit["source_timestamps"])
        images = decoded[offset:offset + count]
        offset += count
        if unit["layout"] in {"2x2", "2x2_global"}:
            rendered.append(compose_2x2(images, canvas_size))
        elif unit["layout"] == "1x1":
            if count != 1:
                raise ValueError("1x1 visual unit must contain one source frame")
            # Match the original experiment: pass the native frame directly
            # to LLaVA's image processor rather than pre-letterboxing it.
            rendered.append(images[0])
        else:
            raise ValueError(f"unsupported visual-unit layout: {unit['layout']}")
    return rendered


def reason_routes(
    routes_path: str,
    output_path: str,
    model_path: str,
    llava_next_path: str,
    *,
    model_revision: str | None = None,
    metrics_output: str | None = None,
    canvas_size: int = 384,
    max_new_tokens: int = 16,
    do_sample: bool = False,
) -> None:
    backend = LocalLlavaVideo(model_path, llava_next_path, model_revision)
    results: list[dict[str, Any]] = []
    for record in read_jsonl(routes_path):
        started = time.perf_counter()
        options = [str(value) for value in record.get("options", [])]
        images = render_visual_units(record, canvas_size)
        prompt = build_multiple_choice_prompt(
            str(record["question"]), options,
            has_context_panels=any(
                unit.get("layout") in {"2x2", "2x2_global"}
                for unit in record["visual_units"]),
        )
        raw = backend.generate(prompt, images, max_new_tokens, do_sample)
        prediction = parse_multiple_choice_answer(raw, len(options))
        answer = record.get("answer")
        result = dict(record)
        result.update({
            "status": "ok",
            "prediction": prediction,
            "raw_model_output": raw,
            "is_correct": prediction == str(answer).strip().upper() if answer is not None else None,
            "vlm_model": model_path,
            "inference_sec": time.perf_counter() - started,
        })
        results.append(result)
    write_jsonl(output_path, results)
    valid = [row for row in results if row["is_correct"] is not None]
    dataset = str(results[0].get("dataset", "")) if results else ""
    summary = {
        "num_questions": len(results),
        "num_scored": len(valid),
        "accuracy": (sum(bool(row["is_correct"]) for row in valid) / len(valid)
                     if valid else None),
        "model": model_path,
        "overall": {
            "count": len(valid),
            "correct": sum(bool(row["is_correct"]) for row in valid),
            "accuracy": (sum(bool(row["is_correct"]) for row in valid) / len(valid)
                         if valid else None),
        },
    }
    if dataset == "videomme":
        buckets: dict[str, list[bool]] = {key: [] for key in ("short", "medium", "long")}
        for row in valid:
            key = str(row.get("duration", "unknown")).lower()
            if key in buckets:
                buckets[key].append(bool(row["is_correct"]))
        summary["duration_results"] = {
            key: {
                "count": len(values),
                "correct": sum(values),
                "accuracy": sum(values) / len(values) if values else None,
            }
            for key, values in buckets.items()
        }
    metrics_path = (Path(metrics_output) if metrics_output else
                    Path(output_path).with_suffix(".metrics.json"))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
