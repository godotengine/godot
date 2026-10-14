"""Fingerprint comparable build settings and dependency files before A/B sampling."""
import argparse
import json
from pathlib import Path
import subprocess

from interleaved_ab_benchmark import ROOT, dependency_hashes, project_hashes, sha256


KEYS = ("CC", "CCFLAGS", "CFLAGS", "CXXFLAGS", "LINKFLAGS", "OPTIMIZELEVEL", "WARNLEVEL", "arch", "platform", "target", "tests", "module_mono_enabled", "precision", "optimize", "debug_symbols", "lto")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / ".superpowers/bench-ab-20261013")
    args = parser.parse_args()
    base = args.root / "baseline"
    exes = {"A": base / "bin/godot.windows.editor.x86_64.mono.exe", "B": args.root / "candidate/bin/godot.windows.editor.x86_64.mono.exe"}
    settings = {"A": json.loads((base / ".scons_env.json").read_text()), "B": json.loads((ROOT / ".scons_env.json").read_text())}
    flags = {arm: {key: env.get(key) for key in KEYS} for arm, env in settings.items()}
    if flags["A"] != flags["B"]:
        raise ValueError("build/compiler flags differ")
    if settings["B"].get("angelscript_force_generic") or settings["B"].get("angelscript_pod_geometry_probe"):
        raise ValueError("candidate experimental/fallback build enabled")
    deps = {arm: dependency_hashes(exe) for arm, exe in exes.items()}
    if deps["A"] != deps["B"]:
        raise ValueError("runtime dependency trees differ")
    def git(path, *words):
        return subprocess.check_output(["git", *words], cwd=path, text=True).strip()
    if git(base, "diff", "--name-only") or git(ROOT, "diff", "--name-only", "--", "modules/angelscript"):
        raise ValueError("tracked build sources are dirty")
    result = dict(revisions={"A": git(base, "rev-parse", "HEAD"), "B": git(ROOT, "rev-parse", "HEAD")},
                  compiler_flags=flags, exes={arm: dict(path=str(exe), sha256=sha256(exe)) for arm, exe in exes.items()},
                  dependencies=deps, project=project_hashes(ROOT / "modules/angelscript/tests/bench_project"),
                  source_scope=git(ROOT, "diff", "--name-only", "20530409fc50563e19954441aa1b81f47ffcee0e", "8c0143716da82b133937d29df42ab3a4c02608c7"))
    (args.root / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("PASS: flags and dependencies match; revisions and actual runtime hashes saved")


if __name__ == "__main__":
    main()
