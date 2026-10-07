from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from reproduction_contract import file_digest, model_asset_identity, execution_environment_identity


def restore_loader_epoch_state(generator, training_state: dict) -> None:
    if "loader_epoch_start_state" not in training_state:
        raise ValueError("Resume checkpoint requires its DataLoader epoch-start state")
    generator.set_state(training_state["loader_epoch_start_state"])


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class NextActionDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        prompt_messages = row["messages"][:2]
        full_messages = row["messages"]
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        full_text = self.tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]
        if len(full_ids) > self.max_length:
            overflow = len(full_ids) - self.max_length
            prompt_ids = prompt_ids[overflow:] if overflow < len(prompt_ids) else []
            full_ids = full_ids[overflow:]
        labels = list(full_ids)
        label_start = min(len(prompt_ids), len(labels))
        labels[:label_start] = [-100] * label_start
        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_batch(batch: list[dict], pad_token_id: int) -> dict:
    max_len = max(item["input_ids"].numel() for item in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for row_index, item in enumerate(batch):
        length = item["input_ids"].numel()
        input_ids[row_index, :length] = item["input_ids"]
        attention_mask[row_index, :length] = 1
        labels[row_index, :length] = item["labels"]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def snapshot_trainable_parameters(parameters: dict[str, torch.nn.Parameter]) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().clone() for name, parameter in parameters.items()}


def restore_anchor_reference(
    parameters: dict[str, torch.nn.Parameter],
    saved_reference: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if parameters.keys() != saved_reference.keys():
        missing = sorted(parameters.keys() - saved_reference.keys())
        unexpected = sorted(saved_reference.keys() - parameters.keys())
        raise ValueError(
            f"Anchor reference does not match trainable parameters: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return {
        name: saved_reference[name].to(device=parameter.device, dtype=parameter.dtype)
        for name, parameter in parameters.items()
    }


def relative_anchor_loss(
    parameters: dict[str, torch.nn.Parameter],
    reference: dict[str, torch.Tensor],
) -> torch.Tensor:
    numerator = None
    denominator = None
    for name, parameter in parameters.items():
        reference_parameter = reference[name]
        delta_sum = (parameter.float() - reference_parameter.float()).square().sum()
        reference_sum = reference_parameter.float().square().sum()
        numerator = delta_sum if numerator is None else numerator + delta_sum
        denominator = reference_sum if denominator is None else denominator + reference_sum
    if numerator is None or denominator is None:
        raise ValueError("Cannot compute an anchor loss without trainable parameters.")
    return numerator / denominator.clamp_min(1e-12)


def save_lora_checkpoint(
    model,
    tokenizer,
    checkpoint_dir: Path,
    meta: dict,
    *,
    optimizer=None,
    anchor_reference: dict[str, torch.Tensor] | None = None,
    loader_epoch_start_state: torch.Tensor | None = None,
    training_identity: dict | None = None,
) -> None:
    if optimizer is not None and (loader_epoch_start_state is None or training_identity is None):
        raise ValueError("Resumable checkpoints require sampler state and training identity")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    (checkpoint_dir / "checkpoint_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if optimizer is not None:
        training_state = {
                "optimizer": optimizer.state_dict(),
                "python_rng_state": random.getstate(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "epoch_index": int(meta["epoch_index"]),
                "next_batch_index": int(meta["next_batch_index"]),
                "optimizer_step": int(meta["optimizer_step"]),
                "loader_epoch_start_state": loader_epoch_start_state.clone(),
                "training_identity": training_identity,
        }
        if anchor_reference is not None:
            training_state["anchor_reference"] = {
                name: tensor.detach().cpu() for name, tensor in anchor_reference.items()
            }
        torch.save(training_state, checkpoint_dir / "training_state.pt")


def train(args) -> dict:
    if args.resume_from and args.init_adapter:
        raise ValueError("--resume-from and --init-adapter are mutually exclusive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = load_jsonl(Path(args.data))
    if args.limit:
        rows = rows[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = NextActionDataset(rows, tokenizer=tokenizer, max_length=args.max_length)
    if args.dry_run:
        sample = dataset[0]
        return {
            "schema_version": "qwen-lora-train-report-v1",
            "dry_run": True,
            "sample_count": len(dataset),
            "first_input_length": int(sample["input_ids"].numel()),
            "first_supervised_tokens": int((sample["labels"] != -100).sum().item()),
        }

    identity_keys = ("seed", "limit", "max_length", "batch_size", "epochs", "gradient_accumulation_steps",
                     "learning_rate", "weight_decay", "lora_r", "lora_alpha", "lora_dropout",
                     "target_modules", "disable_gradient_checkpointing", "lora_anchor_weight", "init_adapter")
    training_identity = {"data_sha256": file_digest(Path(args.data)),
                         "base_model": model_asset_identity(Path(args.model_path)),
                         "trainer_sha256": file_digest(Path(__file__)),
                         "environment": execution_environment_identity(include_runtime=True),
                         "parameters": {key: getattr(args, key) for key in identity_keys}}
    if args.init_adapter:
        training_identity["initial_adapter"] = model_asset_identity(Path(args.init_adapter))
    training_state = None
    if args.resume_from:
        state_path = Path(args.resume_from) / "training_state.pt"
        if not state_path.exists():
            raise FileNotFoundError(f"Resume checkpoint has no training_state.pt: {state_path}")
        training_state = torch.load(state_path, map_location="cpu", weights_only=False)
        if training_state.get("training_identity") != training_identity:
            raise ValueError("Resume requires the same training data, parameters, model and environment")
        restore_loader_epoch_state(torch.Generator(), training_state)

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        quantization_config=quantization,
        device_map="auto",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=not args.disable_gradient_checkpointing,
    )
    target_modules = args.target_modules
    if len(target_modules) == 1 and target_modules[0] == "all-linear":
        target_modules = "all-linear"

    adapter_source = args.resume_from or args.init_adapter
    if adapter_source:
        model = PeftModel.from_pretrained(model, adapter_source, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)
    model.train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
        collate_fn=lambda batch: collate_batch(batch, tokenizer.pad_token_id),
    )
    trainable_parameters = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    anchor_reference = (
        snapshot_trainable_parameters(trainable_parameters) if args.lora_anchor_weight > 0 else None
    )
    optimizer = torch.optim.AdamW(
        list(trainable_parameters.values()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    output_dir = Path(args.output_dir)
    resume_epoch_index = 0
    resume_batch_index = 0
    resume_optimizer_step = 0
    if args.resume_from:
        restore_loader_epoch_state(loader_generator, training_state)
        optimizer.load_state_dict(training_state["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(model.device)
        random.setstate(training_state["python_rng_state"])
        torch.set_rng_state(training_state["torch_rng_state"])
        if torch.cuda.is_available() and training_state.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(training_state["cuda_rng_state_all"])
        if anchor_reference is not None and training_state.get("anchor_reference") is not None:
            anchor_reference = restore_anchor_reference(
                trainable_parameters,
                training_state["anchor_reference"],
            )
        resume_epoch_index = int(training_state["epoch_index"])
        resume_batch_index = int(training_state["next_batch_index"])
        resume_optimizer_step = int(training_state["optimizer_step"])
        print(
            f"resuming_from={args.resume_from} epoch_index={resume_epoch_index} "
            f"next_batch_index={resume_batch_index} optimizer_step={resume_optimizer_step}",
            flush=True,
        )
    losses: list[float] = []
    task_losses: list[float] = []
    anchor_losses: list[float] = []
    saved_step_checkpoints: list[str] = []
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    step = resume_optimizer_step
    reached_max_optimizer_steps = False
    if args.max_optimizer_steps and step >= args.max_optimizer_steps:
        raise ValueError(
            f"Resume optimizer step {step} already reaches "
            f"--max-optimizer-steps={args.max_optimizer_steps}."
        )
    for epoch in range(args.epochs):
        if epoch < resume_epoch_index:
            continue
        # Replay this epoch's permutation, then skip the already completed batches.
        loader_epoch_start_state = loader_generator.get_state().clone()
        for batch_index, batch in enumerate(loader):
            if epoch == resume_epoch_index and batch_index < resume_batch_index:
                continue
            batch = {key: value.to(model.device) for key, value in batch.items()}
            outputs = model(**batch)
            task_loss = outputs.loss
            anchor_loss = (
                relative_anchor_loss(trainable_parameters, anchor_reference)
                if anchor_reference is not None
                else task_loss.new_zeros(())
            )
            combined_loss = task_loss + args.lora_anchor_weight * anchor_loss
            loss = combined_loss / args.gradient_accumulation_steps
            loss.backward()
            losses.append(float(combined_loss.detach().cpu()))
            task_losses.append(float(task_loss.detach().cpu()))
            anchor_losses.append(float(anchor_loss.detach().cpu()))
            if args.log_every_batches and (batch_index + 1) % args.log_every_batches == 0:
                elapsed = time.perf_counter() - start
                print(
                    f"epoch={epoch + 1} batch={batch_index + 1}/{len(loader)} "
                    f"loss={losses[-1]:.4f} task_loss={task_losses[-1]:.4f} "
                    f"anchor_loss={anchor_losses[-1]:.6f} elapsed_sec={elapsed:.1f}",
                    flush=True,
                )
            if (batch_index + 1) % args.gradient_accumulation_steps == 0 or (batch_index + 1) == len(loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                print(
                    f"epoch={epoch + 1} optimizer_step={step} "
                    f"loss={losses[-1]:.4f}",
                    flush=True,
                )
                if args.save_every_optimizer_steps and step % args.save_every_optimizer_steps == 0:
                    checkpoint_dir = output_dir.with_name(f"{output_dir.name}_step{step:04d}")
                    save_lora_checkpoint(
                        model,
                        tokenizer,
                        checkpoint_dir,
                        {
                            "schema_version": "qwen-lora-step-checkpoint-v1",
                            "output_dir": str(output_dir),
                            "step_checkpoint_dir": str(checkpoint_dir),
                            "data": args.data,
                            "epoch": epoch + 1,
                            "epoch_index": epoch,
                            "batch_index": batch_index + 1,
                            "next_batch_index": batch_index + 1,
                            "optimizer_step": step,
                            "loss": losses[-1],
                            "elapsed_sec": round(time.perf_counter() - start, 4),
                        },
                        optimizer=optimizer,
                        anchor_reference=anchor_reference,
                        loader_epoch_start_state=loader_epoch_start_state,
                        training_identity=training_identity,
                    )
                    saved_step_checkpoints.append(str(checkpoint_dir))
                    print(f"saved_step_checkpoint={checkpoint_dir}", flush=True)
                if args.max_optimizer_steps and step >= args.max_optimizer_steps:
                    reached_max_optimizer_steps = True
                    print(
                        f"reached_max_optimizer_steps={args.max_optimizer_steps}",
                        flush=True,
                    )
                    break
        if reached_max_optimizer_steps:
            break

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    elapsed = time.perf_counter() - start
    return {
        "schema_version": "qwen-lora-train-report-v1",
        "dry_run": False,
        "model_path": args.model_path,
        "data": args.data,
        "output_dir": str(output_dir),
        "sample_count": len(dataset),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_length": args.max_length,
        "optimizer_steps": step,
        "learning_rate": args.learning_rate,
        "lora_anchor_weight": args.lora_anchor_weight,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "target_modules": target_modules,
        "save_every_optimizer_steps": args.save_every_optimizer_steps,
        "max_optimizer_steps": args.max_optimizer_steps,
        "stopped_at_max_optimizer_steps": reached_max_optimizer_steps,
        "saved_step_checkpoints": saved_step_checkpoints,
        "resumed_from": args.resume_from,
        "initialized_from_adapter": args.init_adapter,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_fraction": trainable / total if total else 0.0,
        "loss_first": losses[0] if losses else math.nan,
        "loss_last": losses[-1] if losses else math.nan,
        "loss_mean": sum(losses) / len(losses) if losses else math.nan,
        "task_loss_mean": sum(task_losses) / len(task_losses) if task_losses else math.nan,
        "anchor_loss_last": anchor_losses[-1] if anchor_losses else math.nan,
        "anchor_loss_mean": sum(anchor_losses) / len(anchor_losses) if anchor_losses else math.nan,
        "elapsed_sec": round(elapsed, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Qwen LoRA on ALFWorld next-action SFT data.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--data",
        default=str(ROOT / "data" / "actor" / "d240_train.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "models" / "actor_adapter_retrained"),
    )
    parser.add_argument(
        "--report",
        default=str(ROOT / "results" / "provenance" / "d240_actor_retraining.json"),
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-anchor-weight",
        type=float,
        default=0.0,
        help="Relative L2 penalty that anchors trainable LoRA parameters to their initial values.",
    )
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
        help="LoRA target modules. Use all-linear for full linear adaptation.",
    )
    parser.add_argument("--disable-gradient-checkpointing", action="store_true")
    parser.add_argument("--log-every-batches", type=int, default=1)
    parser.add_argument(
        "--save-every-optimizer-steps",
        type=int,
        default=0,
        help="Save LoRA adapter snapshots every N optimizer steps. Use 0 to disable.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume-from",
        help="Resume an adapter checkpoint that also contains training_state.pt.",
    )
    parser.add_argument(
        "--init-adapter",
        help="Initialize trainable LoRA weights from an adapter and start a fresh optimizer/data pass.",
    )
    parser.add_argument(
        "--max-optimizer-steps",
        type=int,
        default=0,
        help="Stop normally after reaching this absolute optimizer step. Use 0 to run the full epoch.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report = train(args)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
