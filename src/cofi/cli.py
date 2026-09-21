from __future__ import annotations

import argparse
import json
from pathlib import Path

from .encoder import SigLIPEncoder
from .pipeline import select_dataset
from .preprocess import preprocess_dataset
from .reasoning import reason_routes
from .selector import CoFiConfig


def _load_config(path: str | None) -> dict:
    if not path:
        return {}
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"configuration file not found: {source}")
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("configuration must be a JSON object")
    return data


def _value(args: argparse.Namespace, data: dict, argument: str,
           key: str, fallback):
    value = getattr(args, argument)
    return value if value is not None else data.get(key, fallback)


def _config(args: argparse.Namespace, data: dict) -> CoFiConfig:
    return CoFiConfig(
        context_panels=int(_value(args, data, "context_panels", "context_panels", 4)),
        frames_per_panel=int(_value(args, data, "frames_per_panel", "frames_per_panel", 4)),
        focus_units=int(_value(args, data, "focus_units", "focus_units", 28)),
        candidate_mass=float(_value(args, data, "rho", "candidate_mass", 0.8)),
        state_weight=float(_value(args, data, "state_weight", "state_weight", 0.5)),
        trajectory_radius=int(_value(args, data, "trajectory_radius", "trajectory_radius", 1)),
        trajectory_stride=int(_value(args, data, "trajectory_stride", "trajectory_stride", 1)),
        bandwidth_multiplier=float(_value(
            args, data, "bandwidth_multiplier", "bandwidth_multiplier", 1.0)),
        focus_noise=float(data.get("focus_noise", 1e-6)),
        jitter=float(data.get("jitter", 1e-6)),
        max_scale_rows=int(data.get("max_scale_rows", 2048)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="cofi")
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--dataset", required=True,
        choices=("videomme", "mlvu", "longvideobench"))
    common.add_argument("--config")
    common.add_argument("--dataset-root", required=True)
    common.add_argument("--annotation")
    common.add_argument("--limit", type=int, default=0)
    common.add_argument("--cache-dir", default="cache/siglip_1fps")
    common.add_argument("--model")
    common.add_argument("--model-revision")
    common.add_argument("--device", default="cuda")
    common.add_argument("--batch-size", type=int, default=16)

    preprocess = sub.add_parser(
        "preprocess", parents=[common],
        help="sample videos and cache question-independent candidate embeddings")
    preprocess.add_argument("--fps", type=float)
    preprocess.add_argument("--cell-size", type=int)
    preprocess.add_argument("--overwrite", action="store_true")

    select = sub.add_parser(
        "select", parents=[common], help="select and save CoFi visual units")
    run = sub.add_parser(
        "run", parents=[common], help="select units and run LLaVA-Video reasoning")
    reason = sub.add_parser(
        "reason", help="run LLaVA-Video reasoning on saved CoFi visual units")

    for command in (select, run):
        command.add_argument("--routes-output", required=True)
        command.add_argument("--context-panels", type=int)
        command.add_argument("--frames-per-panel", type=int)
        command.add_argument("--focus-units", type=int)
        command.add_argument("--rho", type=float)
        command.add_argument("--state-weight", type=float)
        command.add_argument("--trajectory-radius", type=int)
        command.add_argument("--trajectory-stride", type=int)
        command.add_argument("--bandwidth-multiplier", type=float)

    reason.add_argument("--config")
    reason.add_argument("--routes-input", required=True)
    for command in (reason, run):
        command.add_argument("--output", required=True)
        command.add_argument("--metrics-output", required=True)
        command.add_argument("--vlm-model")
        command.add_argument("--vlm-revision")
        command.add_argument("--llava-next-path", required=True)
        command.add_argument("--canvas-size", type=int)
        command.add_argument("--max-new-tokens", type=int)
        command.add_argument(
            "--do-sample", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()

    data = _load_config(args.config)
    if args.command in ("preprocess", "select", "run"):
        model = str(_value(
            args, data, "model", "encoder", "google/siglip-so400m-patch14-384"))
        model_revision = _value(
            args, data, "model_revision", "encoder_revision", None)
        encoder = SigLIPEncoder(model, args.device, model_revision)
        if args.command == "preprocess":
            preprocess_dataset(
                args.dataset, args.dataset_root, args.annotation,
                args.cache_dir, encoder, args.batch_size,
                sample_fps=float(_value(args, data, "fps", "candidate_fps", 1.0)),
                cell_size=int(_value(args, data, "cell_size", "panel_cell_size", 192)),
                overwrite=args.overwrite, limit=args.limit,
            )
        else:
            select_dataset(
                args.dataset, args.dataset_root, args.annotation,
                args.cache_dir, args.routes_output, encoder,
                _config(args, data), args.batch_size, args.limit,
            )

    if args.command in ("reason", "run"):
        vlm_model = str(_value(
            args, data, "vlm_model", "video_llm",
            "lmms-lab/LLaVA-Video-7B-Qwen2"))
        vlm_revision = _value(
            args, data, "vlm_revision", "video_llm_revision", None)
        reason_routes(
            args.routes_input if args.command == "reason" else args.routes_output,
            args.output, vlm_model,
            args.llava_next_path, model_revision=vlm_revision,
            metrics_output=args.metrics_output,
            canvas_size=int(_value(
                args, data, "canvas_size", "panel_canvas_size", 384)),
            max_new_tokens=int(_value(
                args, data, "max_new_tokens", "max_new_tokens", 16)),
            do_sample=bool(_value(
                args, data, "do_sample", "do_sample", False)),
        )


if __name__ == "__main__":
    main()
