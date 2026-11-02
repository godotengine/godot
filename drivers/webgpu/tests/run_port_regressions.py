"""Build and run the 2dog port regressions against an existing web archive."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--emscripten", type=Path, required=True)
parser.add_argument("--archive", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[3]
subprocess.run([sys.executable, str(root / "drivers/webgpu/dependencies.py")], check=True)
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=True)
compiler = args.emscripten.resolve() / "em++.py"
archive = args.archive.resolve()
driver = root / "drivers/webgpu"
common = [
    f"-I{driver / 'tint_cli'}",
    f"-I{driver}",
    f"-I{root}",
    f"-I{root / 'thirdparty/tint'}",
    f"-I{root / 'thirdparty/spirv-tools/include'}",
    f"-I{root / 'thirdparty/spirv-headers-webgpu/include'}",
    "-std=c++20",
    "-fwasm-exceptions",
    "-sSTACK_SIZE=5242880",
    "-sALLOW_MEMORY_GROWTH=1",
    "-sENVIRONMENT=node",
    "-sASSERTIONS=1",
]
for name in (
    "storage_access",
    "binding_array_literals",
    "tint_resource_usage",
    "tint_legacy_parse",
    "shader_portability",
    "shader_edge_cases",
):
    sources = [str(root / f"drivers/webgpu/tests/{name}_test.cpp")]
    if name not in ("tint_legacy_parse", "shader_portability"):
        sources.append(str(driver / "spirv_preprocess.cpp"))
        sources.append(str(driver / "tint_wrapper.cpp"))
        sources.append(str(root / "thirdparty/tint/src/tint/lang/wgsl/inspector/inspector.cc"))
    target = output / f"{name}.js"
    subprocess.run(
        [sys.executable, str(compiler), *sources, str(archive), *common, "-o", str(target)],
        check=True,
        env=os.environ.copy(),
    )
    subprocess.run(["node", str(target)], check=True)

# Header-only driver helpers build against the real engine headers instead of the converter shims.
target = output / "texel_convert.js"
port = root / "thirdparty/emdawnwebgpu/emdawnwebgpu.port.py"
subprocess.run(
    [
        sys.executable,
        str(compiler),
        str(root / "drivers/webgpu/tests/texel_convert_test.cpp"),
        f"-I{root}",
        f"-I{root / 'platform/web'}",
        f"-I{driver}",
        f"--use-port={port}",
        "-std=c++17",
        "-DWEBGPU_ENABLED",
        "-sENVIRONMENT=node",
        "-sASSERTIONS=1",
        "-o",
        str(target),
    ],
    check=True,
    env=os.environ.copy(),
)
subprocess.run(["node", str(target)], check=True)
