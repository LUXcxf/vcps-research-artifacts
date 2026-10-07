# Upstream Assets and Notices

## Base Model

Obtain [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) from its provider. The model card identifies Apache-2.0 licensing. `configs/base_model_assets.json` records the SHA-256 identities of the model shards, configuration, tokenizer and chat template used with the released adapters. Keep the provider's license and notices with downloaded assets.

## ALFWorld

Use the [ALFWorld repository](https://github.com/alfworld/alfworld) at commit `aaba6870f86c5be6a08a491f32a50b906227bc3e`. The repository provides its [code license](https://github.com/alfworld/alfworld/blob/aaba6870f86c5be6a08a491f32a50b906227bc3e/LICENSE) and benchmark acquisition instructions. Original game/data assets are acquired separately, with their corresponding upstream terms. The released manifests identify the exact game files used for development and test evaluation.

## Release Attribution

Original VCPS source code and original documentation are released under the MIT License in the package root. Upstream model, benchmark assets and derived ALFWorld training records remain subject to their applicable upstream terms. The MIT License grants rights to the original code and documentation, not additional rights over those upstream assets.

LoRA adapters identify the upstream base model in their configuration. Derived ALFWorld training records identify the benchmark source in the data card. Original VCPS code, derived training views and workflow contracts are distinct from separately downloaded model and benchmark assets; apply the appropriate license and attribution to each asset class when redistributing them.
