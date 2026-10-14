"""Small filesystem/hash regression tests; no engine process required."""
import tempfile
import unittest
from pathlib import Path

from interleaved_ab_benchmark import dependency_hashes, project_hashes, sha256


class ManifestTests(unittest.TestCase):
    def test_runtime_dependencies_require_real_tree(self):
        with tempfile.TemporaryDirectory() as temp:
            exe = Path(temp) / "engine.exe"
            with self.assertRaises(ValueError):
                dependency_hashes(exe)
            dll = Path(temp) / "GodotSharp/Api/Debug/Example.dll"
            dll.parent.mkdir(parents=True)
            dll.write_bytes(b"test")
            self.assertEqual(list(dependency_hashes(exe).values()), [sha256(dll)])

    def test_project_fingerprint_covers_loaded_assembly_not_mutable_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "bench.as").write_text("int main() {return 1;}")
            dll = root / ".godot/mono/temp/bin/Debug/Test.dll"
            dll.parent.mkdir(parents=True)
            dll.write_bytes(b"v1")
            first = project_hashes(root)
            (root / ".godot/mutable_cache").write_text("ignored cache")
            self.assertEqual(first, project_hashes(root))
            dll.write_bytes(b"v2")
            self.assertNotEqual(first, project_hashes(root))


if __name__ == "__main__":
    unittest.main()
