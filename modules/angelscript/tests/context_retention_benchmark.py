"""Measure the test-only completed-context experiment; never enables it in callp."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]


def main():
    base = ROOT / "docs/language-benchmark/data/context-retention-probe"
    if base.with_suffix(".log").exists():
        raise FileExistsError("refusing to overwrite experiment")
    exe = ROOT / "bin/godot.windows.editor.x86_64.mono.console.exe"
    runtime = exe.with_name(exe.name.replace(".console.exe", ".exe"))
    with runtime.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    command = [str(exe), "--headless", "--test", "--test-case=*retained context isolated*"]
    env = dict(os.environ, AS_BOUNDARY_TIMING="1")
    base.with_suffix(".json").write_text(json.dumps({"runtime_sha256": digest, "command": command, "runs": 3, "repetitions": 7, "note": "Global no-arg scalar function through direct context API; not ScriptInstance dispatch. Retained methods intentionally demonstrate delayed destructor."}, indent=2), encoding="utf-8")
    rows = []
    with base.with_suffix(".log").open("x", encoding="utf-8") as raw:
        for run in range(1, 4):
            done = subprocess.run(command, env=env, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", timeout=180)
            raw.write(f"=== run {run} exit {done.returncode} ===\n" + done.stdout)
            raw.flush()
            if done.returncode:
                raise RuntimeError("retention probe failed")
            found = []
            for line in done.stdout.splitlines():
                if line.startswith("RETAIN|"):
                    _, mode, n, low, med, high = line.split("|")
                    found.append([run, mode, int(n), int(low), int(med), int(high)])
            if len(found) != 2 or {r[1] for r in found} != {"unprepared", "retained"}:
                raise ValueError("missing retention timing")
            rows.extend(found)
    with base.with_suffix(".csv").open("x", encoding="utf-8", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["run", "mode", "iters", "min_us", "median_us", "max_us"])
        writer.writerows(rows)
    print("PASS: 6 timing samples and destructor/exception assertions")


if __name__ == "__main__":
    main()
