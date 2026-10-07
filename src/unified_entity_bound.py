from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence


BRANCH = Path(__file__).resolve().parents[1]

from entity_bound import (
    EntityBoundPotentialScorer,
    EntityResidualRanker,
    ObjectLocationPrior,
    ToolLocationPrior,
    action_features,
    build_entity_graph,
)
from unified_features import lexical_features, semantic_factor_features


class UnifiedEntityResidualRanker(EntityResidualRanker):
    """Learned residual over observable entity, lexical and semantic features."""

    def score_action(
        self,
        *,
        instruction: str,
        previous_actions: Sequence[str],
        observation: str,
        action: str,
        search_prior: ObjectLocationPrior | None = None,
    ) -> tuple[float, dict[str, object]]:
        graph = build_entity_graph(instruction, previous_actions, observation)
        features = action_features(graph, action, search_prior=search_prior)
        features.update(
            lexical_features(
                instruction=instruction,
                previous_actions=previous_actions,
                observation=observation,
                action=action,
            )
        )
        features.update(
            semantic_factor_features(
                instruction=instruction,
                previous_actions=previous_actions,
                observation=observation,
                action=action,
            )
        )
        score = self.score_features(features)
        return score, {
            "entity_features": features,
            "entity_flags": graph.goal_flags(),
            "feature_view": "entity_delta_plus_learned_semantic_factors",
        }

    @classmethod
    def load(cls, path: str | Path) -> "UnifiedEntityResidualRanker":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            feature_names=[str(item) for item in payload["feature_names"]],
            weights=[float(item) for item in payload["weights"]],
            max_abs_score=float(payload.get("max_abs_score", 2.0)),
        )


class UnifiedEntityBoundPotentialScorer(EntityBoundPotentialScorer):
    """Runtime scorer for the verifier-feedback residual manifest."""

    @classmethod
    def load(cls, path: str | Path) -> "UnifiedEntityBoundPotentialScorer":
        manifest_path = Path(path).resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))

        def resolve(raw: str) -> Path:
            candidate = Path(raw)
            return candidate if candidate.is_absolute() else (manifest_path.parent / candidate).resolve()

        base_weight = float(payload.get("base_weight", 0.0))
        if base_weight != 0.0:
            raise ValueError(
                "The residual runtime requires base_weight=0."
            )

        return cls(
            base_scorer=None,
            ranker=UnifiedEntityResidualRanker.load(resolve(str(payload["entity_ranker"]))),
            base_weight=base_weight,
            residual_weight=float(payload.get("residual_weight", 1.0)),
            search_prior=(
                ObjectLocationPrior.load(resolve(str(payload["search_prior"])))
                if payload.get("search_prior")
                else None
            ),
            runtime_recovery_weight=float(payload.get("runtime_recovery_weight", 0.0)),
            search_prior_weight=float(payload.get("search_prior_weight", 0.0)),
            tool_location_prior=(
                ToolLocationPrior.load_pairs(resolve(str(payload["tool_location_prior"])))
                if payload.get("tool_location_prior")
                else None
            ),
            tool_location_weight=float(payload.get("tool_location_weight", 0.0)),
            role_calibration_weight=float(payload.get("role_calibration_weight", 0.0)),
        )
