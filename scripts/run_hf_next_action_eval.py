from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from verification_supervision.model_output import clean_next_action_output  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def decode_new_tokens(*, tokenizer, generated_ids, prompt_length: int) -> str:
    token_list = generated_ids.tolist() if hasattr(generated_ids, "tolist") else list(generated_ids)
    return tokenizer.decode(token_list[prompt_length:], skip_special_tokens=True)


def load_model_and_tokenizer(*, model_path: str, adapter_path: str | None):
    tokenizer_path = adapter_path if adapter_path else model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        quantization_config=quantization,
        device_map="auto",
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def generate_action(*, model, tokenizer, messages: list[dict], max_new_tokens: int) -> tuple[str, str]:
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_length = int(inputs["input_ids"].shape[-1])
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )[0]
    raw = decode_new_tokens(tokenizer=tokenizer, generated_ids=generated, prompt_length=prompt_length)
    return raw, clean_next_action_output(raw)


def _exact_counts(results: list[dict], field: str) -> dict[str, dict[str, int]]:
    values = sorted({result[field] for result in results})
    return {
        value: {
            "correct": sum(
                1 for result in results if result[field] == value and result["exact_match"]
            ),
            "total": sum(1 for result in results if result[field] == value),
        }
        for value in values
    }


def summarize_results(
    *,
    results: list[dict],
    model_path: str,
    adapter_path: str | None,
    output: Path,
    start_time: float,
) -> dict:
    correct = sum(1 for result in results if result["exact_match"])
    return {
        "schema_version": "hf-next-action-eval-report-v1",
        "model_path": model_path,
        "adapter_path": adapter_path,
        "example_count": len(results),
        "exact_match_count": correct,
        "exact_match_rate": correct / len(results) if results else 0.0,
        "by_split": dict(Counter(result["split"] for result in results)),
        "exact_by_split": _exact_counts(results, "split"),
        "by_family": dict(Counter(result["task_family"] for result in results)),
        "exact_by_family": _exact_counts(results, "task_family"),
        "mean_latency_sec": (
            sum(result.get("latency_sec", 0.0) for result in results) / len(results)
            if results
            else 0.0
        ),
        "total_runtime_sec": round(time.perf_counter() - start_time, 4),
        "output": str(output),
    }


def evaluate(
    *,
    examples: list[dict],
    model_path: str,
    adapter_path: str | None,
    output: Path,
    report: Path,
    limit: int | None,
    max_new_tokens: int,
) -> None:
    rows = examples[:limit] if limit is not None else examples
    model, tokenizer = load_model_and_tokenizer(model_path=model_path, adapter_path=adapter_path)
    results = []
    start = time.perf_counter()

    for index, example in enumerate(rows):
        row_start = time.perf_counter()
        raw, prediction = generate_action(
            model=model,
            tokenizer=tokenizer,
            messages=example["messages"][:2],
            max_new_tokens=max_new_tokens,
        )
        target = str(example["target_action"]).strip()
        exact = prediction == target
        results.append(
            {
                "schema_version": "hf-next-action-eval-v1",
                "model_path": model_path,
                "adapter_path": adapter_path,
                "sample_id": example["sample_id"],
                "split": example["split"],
                "task_family": example["task_family"],
                "step_index": example["step_index"],
                "target_action": target,
                "raw_model_output": raw,
                "predicted_action": prediction,
                "exact_match": exact,
                "latency_sec": round(time.perf_counter() - row_start, 4),
            }
        )
        print(
            f"[{index + 1}/{len(rows)}] exact={exact} "
            f"target={target!r} pred={prediction!r}",
            flush=True,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    summary = summarize_results(
        results=results,
        model_path=model_path,
        adapter_path=adapter_path,
        output=output,
        start_time=start,
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate HF Qwen base or LoRA adapter on next-action examples.")
    parser.add_argument("--examples", required=True)
    parser.add_argument("--model-path", default=os.environ.get("VCPS_MODEL_PATH", "Qwen/Qwen3.5-9B"))
    parser.add_argument("--adapter-path")
    parser.add_argument(
        "--output",
        default=str(ROOT / "results" / "runs" / "hf_next_action_eval.jsonl"),
    )
    parser.add_argument(
        "--report",
        default=str(ROOT / "results" / "reports" / "hf_next_action_eval_report.json"),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    evaluate(
        examples=load_jsonl(Path(args.examples)),
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        output=Path(args.output),
        report=Path(args.report),
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
