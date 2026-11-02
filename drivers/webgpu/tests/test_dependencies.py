import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "webgpu_dependencies", Path(__file__).resolve().parents[3] / "drivers/webgpu/dependencies.py"
)
assert spec and spec.loader
dependencies = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dependencies)


class DependencyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.archive = self.cache / "fixture.zip"
        with zipfile.ZipFile(self.archive, "w") as archive:
            archive.writestr("upstream/src/test.cc", "upstream\n")
        self.digest = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        self.archive.rename(self.cache / (self.digest + ".zip"))
        lock = self.root / "drivers/webgpu/dependencies.json"
        lock.parent.mkdir(parents=True)
        self.entry = {
            "revision": "fixture",
            "url": "https://example.invalid/fixture.zip",
            "sha256": self.digest,
            "files": ["src/test.cc"],
        }
        lock.write_text(json.dumps({"tint": self.entry}), encoding="utf-8")
        self.source = self.root / "thirdparty/tint/src/test.cc"

    def prepare(self, **options):
        dependencies.prepare(self.root, self.cache, offline=True, **options)

    def write_patch(self, replacement):
        directory = self.root / "thirdparty/tint/patches"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "0001-test.patch").write_text(
            "--- a/thirdparty/tint/src/test.cc\n+++ b/thirdparty/tint/src/test.cc\n"
            f"@@ -1 +1 @@\n-upstream\n+{replacement}\n",
            encoding="utf-8",
        )

    def test_prepares_offline_and_keeps_unchanged_timestamps(self):
        with patch.object(dependencies.urllib.request, "urlopen", side_effect=AssertionError("network access")):
            self.prepare()
            self.assertEqual(self.source.read_text(), "upstream\n")
            timestamp = self.source.stat().st_mtime_ns
            self.prepare()
            self.prepare(verify=True)
            self.assertEqual(self.source.stat().st_mtime_ns, timestamp)

    def test_patch_changes_rebuild_from_upstream_without_double_application(self):
        self.write_patch("first")
        self.prepare()
        self.assertEqual(self.source.read_text(), "first\n")
        self.write_patch("second")
        self.prepare()
        self.prepare(verify=True)
        self.assertEqual(self.source.read_text(), "second\n")

    def test_preserves_crlf_files_when_normalized_content_matches(self):
        self.source.parent.mkdir(parents=True)
        self.source.write_bytes(b"upstream\r\n")
        self.prepare()
        self.prepare(verify=True)
        self.assertEqual(self.source.read_bytes(), b"upstream\r\n")

    def test_detects_local_edits_and_refuses_to_overwrite_them(self):
        self.prepare()
        self.source.write_text("local edit\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Locally modified"):
            self.prepare(verify=True)
        self.write_patch("changed")
        with self.assertRaisesRegex(ValueError, "Locally modified"):
            self.prepare()
        self.assertEqual(self.source.read_text(), "local edit\n")

    def test_missing_sources_are_restored_from_cache(self):
        self.prepare()
        self.source.unlink()
        with self.assertRaisesRegex(RuntimeError, "need preparation"):
            self.prepare(verify=True)
        self.prepare()
        self.assertEqual(self.source.read_text(), "upstream\n")

    def test_corrupt_cached_archive_fails_offline(self):
        (self.cache / (self.digest + ".zip")).write_bytes(b"corrupt")
        with self.assertRaisesRegex(RuntimeError, "corrupt cached"):
            self.prepare()
        self.assertFalse(self.source.exists())

    def test_offline_environment_blocks_downloads(self):
        (self.cache / (self.digest + ".zip")).unlink()
        with patch.dict(os.environ, {"WEBGPU_DEPS_OFFLINE": "1"}):
            with self.assertRaisesRegex(RuntimeError, "Missing or corrupt cached"):
                dependencies.prepare(self.root, self.cache)

    def test_platform_prepares_only_the_dependencies_it_uses(self):
        self.entry["platforms"] = ["web"]
        (self.root / "drivers/webgpu/dependencies.json").write_text(json.dumps({"tint": self.entry}), encoding="utf-8")
        self.prepare(platform="windows")
        self.assertFalse(self.source.exists())
        self.prepare(platform="web")
        self.assertEqual(self.source.read_text(), "upstream\n")

    def test_removed_locked_files_do_not_linger_in_incremental_checkouts(self):
        self.prepare()
        self.entry["files"] = []
        (self.root / "drivers/webgpu/dependencies.json").write_text(json.dumps({"tint": self.entry}), encoding="utf-8")
        self.prepare()
        self.prepare(verify=True)
        self.assertFalse(self.source.exists())

    def test_archive_cannot_write_outside_destination(self):
        archive_path = self.root / "unsafe.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("upstream/../../escape.txt", "escape")
        with self.assertRaisesRegex(ValueError, "Invalid dependency path"):
            dependencies.extract(archive_path, self.root / "destination", prefixes=[""])
        self.assertFalse((self.root / "escape.txt").exists())

    def test_source_selection_excludes_unused_files(self):
        archive_path = self.root / "selection.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            for name in ("src/library.cc", "src/library_test.cc", "src/readme.md", "other/library.cc"):
                archive.writestr("upstream/" + name, "content")
        files = dependencies.extract(
            archive_path,
            self.root / "selected",
            selection={"roots": ["src"], "extensions": [".cc"], "exclude": ["*_test.cc"]},
        )
        self.assertEqual(files, ["src/library.cc"])

    def test_optional_binary_assets_preserve_crlf_bytes(self):
        archive_path = self.root / "binary.zip"
        data = b"\x89PNG\r\n\x00binary"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("upstream/test.png", data)
        dependencies.extract(archive_path, self.root / "binary", files=["test.png"], normalize=False)
        self.assertEqual((self.root / "binary/test.png").read_bytes(), data)


if __name__ == "__main__":
    unittest.main()
