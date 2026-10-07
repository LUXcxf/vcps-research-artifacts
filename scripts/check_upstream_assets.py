"""Verify separately acquired model and benchmark files against reference identities."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from reproduction_contract import file_digest


def verify_files(root, items):
    errors = []
    for item in items:
        path = root / item["path"]
        if not path.is_file():
            errors.append(f"Missing: {item['path']}")
        elif path.stat().st_size != item["size"] or file_digest(path) != item["sha256"]:
            errors.append(f"Asset identity mismatch: {item['path']}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--benchmark-root", type=Path)
    args = parser.parse_args()
    if not args.model_path and not args.benchmark_root:
        parser.error("Provide --model-path and/or --benchmark-root")
    errors = []
    for root, name in ((args.model_path, "base_model_assets.json"), (args.benchmark_root, "benchmark_assets.json")):
        if root:
            config = json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))
            errors += verify_files(root, config["asset_files"])
    for error in errors:
        print(error)
    if not errors:
        print("OK: upstream asset identities match")
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
