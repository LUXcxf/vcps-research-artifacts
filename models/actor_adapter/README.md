# VCPS D240 Actor Adapter

This directory contains the inference-only LoRA adapter used by the frozen
VCPS evaluation. It does not contain the Qwen3.5-9B base model or optimizer
state.

## Required base model

Prepare the original `Qwen/Qwen3.5-9B` model locally and pass its path to the
evaluation scripts through the runner's model argument or the documented
environment configuration. The adapter configuration intentionally uses a
portable model identifier rather than a workstation-specific path.

## Files

- `adapter_model.safetensors`: LoRA weights.
- `adapter_config.json`: PEFT configuration.
- `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`: inference
  tokenizer assets.
- `checkpoint_meta.json`: reproducibility metadata for the source checkpoint.

The omitted `training_state.pt` is not needed for inference reproduction.
