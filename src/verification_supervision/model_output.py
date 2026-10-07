from __future__ import annotations

import re


_THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)


def clean_next_action_output(text: str) -> str:
    cleaned = _THINK_BLOCK.sub("", text).strip()
    if not cleaned:
        return ""

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return ""

    first = lines[0]
    for prefix in ["Action:", "action:", "Next action:", "next action:", "Answer:", "answer:"]:
        if first.startswith(prefix):
            first = first[len(prefix) :].strip()
            break

    if (
        (first.startswith('"') and first.endswith('"'))
        or (first.startswith("'") and first.endswith("'"))
    ):
        first = first[1:-1].strip()

    return first
