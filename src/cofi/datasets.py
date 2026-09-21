"""Dataset adapters used by the three reported CoFi benchmarks."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QuestionSample:
    dataset: str
    question_id: str
    video_id: str
    video_path: str
    question: str
    options: list[str] = field(default_factory=list)
    answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


_VIDEO_INDEX: dict[Path, dict[str, Path]] = {}


def _option_text(value: object) -> str:
    return re.sub(r"^[A-Z]\s*[.)]\s*", "", str(value).strip())


def _resolve_video(root: Path, key: str) -> str:
    key_path = Path(key)
    filename = key_path.name if key_path.suffix.lower() == ".mp4" else f"{key_path.name}.mp4"
    candidates = [root / key, root / filename, root / "data" / filename,
                  root / "videos" / filename]
    for path in candidates:
        if path.is_file():
            return str(path.resolve())
    resolved = root.resolve()
    if resolved not in _VIDEO_INDEX:
        _VIDEO_INDEX[resolved] = {path.stem: path for path in root.glob("**/*.mp4")}
    match = _VIDEO_INDEX[resolved].get(Path(key).stem)
    return str(match.resolve()) if match else str(candidates[-1].resolve())


def _parse_embedded_options(value: str) -> tuple[str, list[str]]:
    matches = list(re.finditer(r"(?:^|\n)\s*\(([A-Z])\)\s*", value))
    if not matches:
        return value.strip(), []
    question = value[:matches[0].start()].strip()
    options = [
        value[match.end():(matches[index + 1].start()
                           if index + 1 < len(matches) else len(value))].strip()
        for index, match in enumerate(matches)
    ]
    return question, options


def load_videomme(root: str | Path, annotation: str | Path | None) -> list[QuestionSample]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Video-MME requires pandas and pyarrow") from exc
    root = Path(root)
    path = Path(annotation) if annotation else root / "videomme/test-00000-of-00001.parquet"
    result = []
    for row in pd.read_parquet(path).to_dict("records"):
        video_id = str(row.get("videoID") or row.get("video_id"))
        raw_options = row.get("options")
        result.append(QuestionSample(
            "videomme", str(row.get("question_id")), video_id,
            _resolve_video(root, video_id), str(row.get("question", "")),
            [_option_text(value) for value in ([] if raw_options is None else list(raw_options))],
            str(row.get("answer", "")).upper() or None,
            {key: row.get(key) for key in
             ("duration", "domain", "sub_category", "task_type")},
        ))
    return result


def _mlvu_paths(root: Path, annotation: str | Path | None) -> list[Path]:
    if annotation:
        path = Path(annotation)
        return sorted(path.glob("*.json")) if path.is_dir() else [path]
    annotation_root = root / "MLVU" / "json"
    if not annotation_root.is_dir():
        annotation_root = root / "json"
    return sorted(path for path in annotation_root.glob("*.json")
                  if path.name[:1].isdigit() and int(path.name.split("_", 1)[0]) <= 7)


def load_mlvu(root: str | Path, annotation: str | Path | None) -> list[QuestionSample]:
    root = Path(root)
    paths = _mlvu_paths(root, annotation)
    if not paths:
        raise FileNotFoundError(f"no MLVU-Dev annotation found under {root}")
    result, row_number = [], 0
    for path in paths:
        rows = json.loads(path.read_text(encoding="utf-8"))
        for row in rows:
            video_name = str(row.get("video_name") or row.get("video") or "").strip()
            raw_question = str(row.get("question", ""))
            parsed_question, parsed_options = _parse_embedded_options(raw_question)
            raw_options = row.get("candidates") or row.get("options") or parsed_options
            options = [_option_text(value) for value in raw_options]
            question = parsed_question if parsed_options else raw_question.strip()
            raw_answer = str(row.get("answer", "")).strip()
            answer = raw_answer.upper() if len(raw_answer) == 1 and raw_answer.isalpha() else None
            if answer is None and raw_answer:
                normalized = _option_text(raw_answer).casefold()
                for index, option in enumerate(options):
                    if option.casefold() == normalized:
                        answer = chr(65 + index)
                        break
            result.append(QuestionSample(
                "mlvu", str(row.get("question_id") or row.get("id") or f"MLVU_{row_number}"),
                Path(video_name).stem, _resolve_video(root, video_name), question, options, answer,
                {"duration": row.get("duration"),
                 "task_type": row.get("task_type") or row.get("question_type"),
                 "annotation_file": path.name},
            ))
            row_number += 1
    return result


def load_longvideobench(root: str | Path,
                        annotation: str | Path | None) -> list[QuestionSample]:
    root = Path(root)
    path = Path(annotation) if annotation else root / "lvb_val.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    result = []
    for index, row in enumerate(rows):
        video_id = str(row.get("video_id") or Path(str(row.get("video_path", ""))).stem)
        video_name = str(row.get("video_path") or f"{video_id}.mp4")
        raw_answer = row.get("correct_choice")
        answer = None if raw_answer is None else chr(65 + int(raw_answer))
        result.append(QuestionSample(
            "longvideobench", str(row.get("id") or f"LVB_{index}"), video_id,
            _resolve_video(root, video_name), str(row.get("question", "")),
            [_option_text(value) for value in row.get("candidates", [])], answer,
            {key: row.get(key) for key in
             ("duration", "duration_group", "question_category", "topic_category", "level")},
        ))
    return result


def load_dataset(name: str, root: str | Path,
                 annotation: str | Path | None = None, limit: int = 0) -> list[QuestionSample]:
    key = name.lower().replace("-", "")
    loaders = {
        "videomme": load_videomme,
        "mlvu": load_mlvu,
        "longvideobench": load_longvideobench,
    }
    if key not in loaders:
        raise ValueError(f"dataset must be one of {sorted(loaders)}, got {name!r}")
    rows = loaders[key](root, annotation)
    return rows[:limit] if limit > 0 else rows
