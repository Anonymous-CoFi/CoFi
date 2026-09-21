"""Prompt used by the default multiple-choice evaluation."""

from __future__ import annotations

from collections.abc import Sequence


def build_multiple_choice_prompt(
    question: str, options: Sequence[str], has_context_panels: bool = True,
) -> str:
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    rendered = "\n".join(f"({labels[i]}) {option}" for i, option in enumerate(options))
    base = (
        "Answer the multiple-choice video question using the chronologically ordered frames.\n"
        f"Question: {question}\n{rendered}\n"
        "Return only the option letter."
    )
    if not has_context_panels:
        return base
    description = (
        "Some video images are temporal panels. Within every panel, time proceeds "
        "in raster order: left-to-right and then top-to-bottom. "
        "The visual units themselves follow video time."
    )
    return description + "\n" + base
