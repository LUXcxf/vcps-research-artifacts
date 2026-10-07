import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_public_package import GENERATED_PREFIXES
from list_release_files import release_paths


class ReleaseFileSelectionTest(unittest.TestCase):
    def write_manifest(self, root, entries):
        (root / "RELEASE_MANIFEST.json").write_text(
            json.dumps({"payload_files": entries}), encoding="utf-8",
        )

    def entry(self, name, content=b"reference"):
        return {"path": name, "size": len(content), "sha256": hashlib.sha256(content).hexdigest()}

    def test_selects_only_manifest_files_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "README.md").write_bytes(b"reference")
            (root / "unlisted.txt").write_bytes(b"local")
            for prefix in GENERATED_PREFIXES:
                output = root / prefix / "output.json"
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(b"local")
            self.write_manifest(root, [self.entry("README.md")])
            self.assertEqual(release_paths(root), ["README.md", "RELEASE_MANIFEST.json"])

    def test_rejects_generated_and_unsafe_paths(self):
        names = [prefix + "output.json" for prefix in GENERATED_PREFIXES]
        names += ["../outside.txt", "/outside.txt", "C" + ":/outside.txt", "data\\file.json",
                  "./README.md", "RELEASE_MANIFEST.json", "__pycache__/module.pyc"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in names:
                with self.subTest(name=name):
                    self.write_manifest(root, [self.entry(name)])
                    with self.assertRaises(ValueError):
                        release_paths(root)

    def test_rejects_missing_changed_and_duplicate_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = self.entry("README.md")
            self.write_manifest(root, [entry])
            with self.assertRaises(ValueError):
                release_paths(root)
            (root / "README.md").write_bytes(b"modified")
            with self.assertRaises(ValueError):
                release_paths(root)
            (root / "README.md").write_bytes(b"reference")
            self.write_manifest(root, [entry, entry])
            with self.assertRaises(ValueError):
                release_paths(root)

    def test_rejects_symlink_outside_root(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as external:
            root = Path(directory)
            target = Path(external) / "outside.txt"
            target.write_bytes(b"reference")
            try:
                (root / "link.txt").symlink_to(target)
            except OSError as exc:
                self.skipTest(f"Symlink creation unavailable: {exc}")
            self.write_manifest(root, [self.entry("link.txt")])
            with self.assertRaises(ValueError):
                release_paths(root)


if __name__ == "__main__":
    unittest.main()
