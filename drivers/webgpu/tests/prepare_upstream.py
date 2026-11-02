"""Restore optional, unchanged diagnostics and demos from the original WebGPU fork."""

import argparse
import importlib.util
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SPEC = {
    "url": "https://codeload.github.com/dwalter/godotwebgpu/tar.gz/f329e39ce8db7acaa5c9d6628a530fb769969228",
    "sha256": "974ac186c7717e34313e43141805087a5e1a37f639a265792dd993062ffdb718",
    "roots": [
        "webgpu_tests/benchmark",
        "webgpu_tests/driver_unit_tests",
        "webgpu_tests/resource_lifecycle",
        "webgpu_tests/scene_smoketest",
        "webgpu_tests/screenshot_comparison",
        "webgpu_tests/test_project",
    ],
    "files": ["webgpu_tests/profile_startup.mjs"],
}


def prepare(cache, offline):
    module_spec = importlib.util.spec_from_file_location("webgpu_dependencies", ROOT / "drivers/webgpu/dependencies.py")
    dependencies = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(dependencies)
    cache = cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    archive = dependencies.download(SPEC, cache, offline)
    with tempfile.TemporaryDirectory(dir=cache) as temporary:
        stage = Path(temporary)
        files = dependencies.extract(archive, stage, files=SPEC["files"], selection=SPEC, normalize=False)
        for file in files:
            source, output = stage / file, ROOT / file
            data = source.read_bytes()
            if output.exists():
                current = output.read_bytes()
                if current == data or (b"\0" not in data and current.replace(b"\r\n", b"\n") == data):
                    continue
                raise ValueError(f"Locally modified upstream tool: {output}; preserve it before restoring")
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, output)
    print(f"Prepared {len(files)} optional upstream files at f329e39ce8db.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "thirdparty/.webgpu-cache")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    prepare(args.cache, args.offline)
