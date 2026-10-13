"""Opt-in VM boundary probes, separate from the language-parity scorecard."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if not args.stage.replace("-", "").isalnum() or args.runs < 1:
        parser.error("invalid stage or runs")
    base = ROOT / "docs/language-benchmark/data" / ("boundary-probes-" + args.stage)
    if base.with_suffix(".log").exists():
        parser.error("refusing to overwrite existing probe output")
    exe = ROOT / "bin/godot.windows.editor.x86_64.mono.console.exe"
    runtime = exe.with_name(exe.name.replace(".console.exe", ".exe"))
    with runtime.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    env = dict(os.environ, AS_BOUNDARY_TIMING="1")
    command = [str(exe), "--headless", "--test", "--test-case=*boundary probes*"]
    fields = ["stage", "run", "probe", "iters", "min_us", "median_us", "max_us", "checksum"]
    rows = []
    base.with_suffix(".json").write_text(json.dumps({"runtime_sha256": digest, "command": command, "runs": args.runs, "repetitions": 7, "warmups": 1, "note": "Native thunk counters unchanged from baseline; diagnostic iteration work differs from original language benchmark."}, indent=2), encoding="utf-8")
    with base.with_suffix(".log").open("x", encoding="utf-8") as raw:
        for run in range(1, args.runs + 1):
            print(f"probe {args.stage}: {run}/{args.runs}", flush=True)
            done = subprocess.run(command, env=env, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", timeout=180)
            raw.write(f"\n=== run {run} exit {done.returncode} ===\n" + done.stdout)
            raw.flush()
            if done.returncode:
                raise RuntimeError(f"probe failed; see {base}.log")
            run_rows = []
            for line in done.stdout.splitlines():
                if line.startswith("PROBE|"):
                    _, name, n, lo, med, hi, checksum = line.split("|")
                    run_rows.append(dict(zip(fields, [args.stage, run, name, int(n), int(lo), int(med), int(hi), int(checksum)])))
            if len(run_rows) != 19 or len({r["probe"] for r in run_rows}) != 19:
                raise ValueError("expected 19 distinct probes")
            rows.extend(run_rows)
    with base.with_suffix(".csv").open("x", newline="", encoding="utf-8") as table:
        writer = csv.DictWriter(table, fields)
        writer.writeheader()
        writer.writerows(rows)
    with Path(str(base) + "-summary.csv").open("x", newline="", encoding="utf-8") as table:
        writer = csv.writer(table)
        writer.writerow(["stage", "probe", "median_ns", "min_process_median_ns", "max_process_median_ns", "checksum"])
        for name in sorted({r["probe"] for r in rows}):
            group = [r for r in rows if r["probe"] == name]
            if len({(r["iters"], r["checksum"]) for r in group}) != 1:
                raise ValueError("unstable checksum")
            values = [r["median_us"] * 1000 / r["iters"] for r in group]
            writer.writerow([args.stage, name, statistics.median(values), min(values), max(values), group[0]["checksum"]])
    print(f"PASS: {len(rows)} probe samples", flush=True)


if __name__ == "__main__":
    main()
