from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


BRANCH = Path(__file__).resolve().parents[1]
PROJECT = BRANCH
sys.path[:0] = [str(PROJECT / "src"), str(BRANCH.parent / "src")]
from reproduction_contract import (model_asset_identity, source_asset_identity,
                                   execution_environment_identity, validate_output_identity)

from materials_workflow.v11.generator import build_cases  # noqa: E402
from materials_workflow.v11.skills import default_skill_registry  # noqa: E402
from materials_workflow.v11.verifier import TransitionVerifier  # noqa: E402
from vcps_transfer.evaluation import EvaluationConfig, run_episode, summarize_results  # noqa: E402
from vcps_transfer.prompting import action_json, build_messages  # noqa: E402
from vcps_transfer.protocol import build_development_manifest, build_test_manifest  # noqa: E402
from vcps_transfer.residual import LinearResidualRanker  # noqa: E402
from vcps_transfer.structural import ALF_ALIGNED_PROFILE, WorkstationStructuralPotential  # noqa: E402

PAPER_VARIANTS = ("actor", "actor_structural", "actor_residual", "full")


class HFActorScorer:
    def __init__(
        self,
        *,
        model_path: Path,
        adapter_path: Path,
        batch_size: int,
        cache_path: Path | None,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            quantization_config=quantization,
            device_map="auto",
        )
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.model.eval()
        self.batch_size = batch_size
        self.cache_path = cache_path
        self.cache = self._load_cache(cache_path)

    def __call__(self, case, state, history, legal_actions):
        messages = build_messages(
            task=case.task_text,
            state=state,
            goal=case.goal,
            history=history,
            legal_actions=legal_actions,
        )
        actions = [action_json(action) for action in legal_actions]
        key = hashlib.sha256(
            json.dumps(
                {"messages": messages, "actions": actions},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if key in self.cache:
            return dict(self.cache[key])
        scores = dict(self.score_continuations(messages, actions))
        self.cache[key] = scores
        if self.cache_path is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"key": key, "scores": scores}, ensure_ascii=False) + "\n")
                handle.flush()
        return scores

    @staticmethod
    def _load_cache(path: Path | None) -> dict[str, dict[str, float]]:
        if path is None or not path.exists():
            return {}
        cache = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                cache[row["key"]] = {str(name): float(value) for name, value in row["scores"].items()}
        return cache

    def score_continuations(self, messages, candidate_actions):
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        eos = self.tokenizer.eos_token or ""
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        scores = []
        try:
            for start in range(0, len(candidate_actions), self.batch_size):
                batch_actions = candidate_actions[start : start + self.batch_size]
                lengths = [
                    len(self.tokenizer(action + eos, add_special_tokens=False)["input_ids"])
                    for action in batch_actions
                ]
                max_length = max(lengths)
                inputs = self.tokenizer(
                    [prompt_text + action + eos for action in batch_actions],
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                ).to(self.model.device)
                with torch.inference_mode():
                    logits = self.model(**inputs, logits_to_keep=max_length + 1).logits
                log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
                targets = inputs["input_ids"][:, -max_length:]
                token_scores = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                for row, (action, length) in enumerate(zip(batch_actions, lengths, strict=True)):
                    scores.append((action, float(token_scores[row, -length:].mean().item())))
        finally:
            self.tokenizer.padding_side = original_padding_side
        return scores


def select_cases(source: str, limit: int | None):
    cases = build_cases()
    if source == "calibration":
        calibration_ids = set(build_development_manifest().calibration_episode_ids)
        selected = [case for case in cases if case.episode_id in calibration_ids]
    elif source == "tests":
        test_ids = set(build_test_manifest()["episode_ids"])
        selected = [case for case in cases if case.episode_id in test_ids]
    else:
        raise ValueError(source)
    selected = sorted(selected, key=lambda case: case.episode_id)
    return balanced_limit(selected, limit) if limit else selected


def balanced_limit(cases, limit: int):
    groups = {}
    for case in cases:
        groups.setdefault((case.metadata["scope"], case.task_family), []).append(case)
    ordered_groups = [groups[key] for key in sorted(groups)]
    selected = []
    cursor = 0
    while len(selected) < limit and ordered_groups:
        remaining = []
        for group in ordered_groups:
            if cursor < len(group) and len(selected) < limit:
                selected.append(group[cursor])
            if cursor + 1 < len(group):
                remaining.append(group)
        ordered_groups = remaining
        cursor += 1
    return sorted(selected, key=lambda case: case.episode_id)


def read_completed(path: Path) -> tuple[list[dict], set[str]]:
    if not path.exists():
        return [], set()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows, {row["episode_id"] for row in rows}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run closed-loop workstation transfer evaluation.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--residual-path", type=Path, default=BRANCH / "artifacts" / "models" / "residual_v1.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", choices=("calibration", "tests"), required=True)
    parser.add_argument("--variants", nargs="+", choices=PAPER_VARIANTS, default=list(PAPER_VARIANTS))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--action-budget", type=int, default=50)
    parser.add_argument("--structural-weight", type=float, default=0.25)
    parser.add_argument("--residual-weight", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--actor-cache", type=Path)
    args = parser.parse_args()
    if args.output_dir.resolve().is_relative_to((BRANCH / "results").resolve()):
        parser.error("Choose a new output directory under workstation/reproduced/")
    if args.action_budget <= 0 or args.batch_size <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("Action budget, batch size and limit must be positive")
    identity = {
        "adapter_sha256": hashlib.sha256((args.adapter_path / "adapter_model.safetensors").read_bytes()).hexdigest(),
        "residual_sha256": hashlib.sha256(args.residual_path.read_bytes()).hexdigest(),
        "action_budget": args.action_budget, "structural_weight": args.structural_weight,
        "residual_weight": args.residual_weight, "source": args.source, "limit": args.limit,
        "model_path": str(args.model_path),
        "base_model": model_asset_identity(args.model_path),
        "adapter_assets": model_asset_identity(args.adapter_path),
        "source_assets": source_asset_identity(BRANCH),
        "contract_helper_sha256": hashlib.sha256((BRANCH.parent / "src/reproduction_contract.py").read_bytes()).hexdigest(),
        "environment": execution_environment_identity(include_runtime=True),
        "batch_size": args.batch_size,
    }
    identity_path = args.output_dir / "run_identity.json"
    validate_output_identity(identity_path, identity, [*args.output_dir.glob("*.jsonl"),
                                                      *args.output_dir.glob("*_report.json")])
    if args.actor_cache:
        cache_identity = {key: identity[key] for key in
                          ("base_model", "adapter_assets", "source_assets", "environment", "batch_size")}
        cache_path = args.actor_cache.with_name(args.actor_cache.name + ".identity.json")
        validate_output_identity(cache_path, cache_identity, [args.actor_cache])
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache_identity, indent=2) + "\n", encoding="utf-8")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    identity_path.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    registry = default_skill_registry()
    verifier = TransitionVerifier(registry)
    structural = WorkstationStructuralPotential(
        registry,
        verifier,
        profile=ALF_ALIGNED_PROFILE,
    )
    residual = LinearResidualRanker.load(args.residual_path)
    actor = HFActorScorer(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        batch_size=args.batch_size,
        cache_path=args.actor_cache,
    )
    cases = select_cases(args.source, args.limit)
    config = EvaluationConfig(
        action_budget=args.action_budget,
        structural_weight=args.structural_weight,
        residual_weight=args.residual_weight,
        residual_application_mode="continuous_legal_candidate",
    )
    all_summaries = {}
    for variant in args.variants:
        output = args.output_dir / f"{args.source}_{variant}.jsonl"
        rows, completed = read_completed(output)
        expected = [case.episode_id for case in cases]
        actual = [row["episode_id"] for row in rows]
        if len(actual) != len(set(actual)) or actual != expected[:len(actual)] or len(actual) > len(expected):
            raise ValueError("Saved episodes must be the exact ordered case prefix")
        with output.open("a", encoding="utf-8") as handle:
            for index, case in enumerate(cases, start=1):
                if case.episode_id in completed:
                    continue
                started = time.time()
                row = run_episode(
                    case=case,
                    actor_scorer=actor,
                    structural=structural,
                    verifier=verifier,
                    registry=registry,
                    residual=residual,
                    variant=variant,
                    config=config,
                )
                row["wall_time_seconds"] = time.time() - started
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                rows.append(row)
                print(
                    f"variant={variant} episode={index}/{len(cases)} "
                    f"success={int(row['success'])} steps={row['steps']} "
                    f"elapsed={row['wall_time_seconds']:.1f}s",
                    flush=True,
                )
        all_summaries[variant] = summarize_results(rows)
    report = {
        "schema_version": "workstation-closed-loop-evaluation-report-v1",
        "adapter_path": str(args.adapter_path),
        "residual_path": str(args.residual_path),
        "source": args.source,
        "case_count": len(cases),
        "config": {
            "action_budget": args.action_budget,
            "structural_weight": args.structural_weight,
            "residual_weight": args.residual_weight,
            "candidate_protocol": "all_legal_actions",
            "structural_profile": ALF_ALIGNED_PROFILE.name,
            "residual_application_mode": config.residual_application_mode,
        },
        "summaries": all_summaries,
    }
    report_path = args.output_dir / f"{args.source}_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
