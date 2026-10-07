from __future__ import annotations

import json
from pathlib import Path

from structured_progress import StructuredProgressPotentialScorer
from unified_entity_bound import UnifiedEntityBoundPotentialScorer


def load_progress_scorer(path: str | Path):
    """Load one of the two scorer schemas used by the released VCPS runs."""

    model_path = Path(path)
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    schema = payload.get("schema_version")
    if schema == "structured-progress-potential-runtime-v1":
        return StructuredProgressPotentialScorer.load(model_path)
    if schema == "unified-progress-value-runtime-v1":
        return UnifiedEntityBoundPotentialScorer.load(model_path)
    raise ValueError(f"Unsupported public VCPS scorer schema: {schema!r}")
