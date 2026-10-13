"""Run unchanged language workloads, retain raw samples and validate checksums."""
import argparse
import csv
import hashlib
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "docs/language-benchmark/data"
FIELDS = ["stage", "run", "workload", "lang", "iters", "min_us", "median_us", "max_us", "checksum"]


def parse_output(output, stage, run):
    rows = []
    for line in output.splitlines():
        if not line.startswith("BENCH|bench_"):
            continue
        _, workload, lang, iters, low, median, high, checksum = line.split("|", 7)
        row = dict(zip(FIELDS, [stage, run, workload, lang, int(iters), int(low), int(median), int(high), checksum]))
        if row["iters"] <= 1:
            raise ValueError("BENCH_SMALL or invalid iteration count")
        if not 0 <= row["min_us"] <= row["median_us"] <= row["max_us"]:
            raise ValueError(f"invalid timings: {row}")
        rows.append(row)
    keys = {(r["workload"], r["lang"]) for r in rows}
    if len(rows) != 51 or len(keys) != 51:
        raise ValueError(f"expected 51 unique workload/language rows, got {len(rows)}/{len(keys)}")
    for workload in {r["workload"] for r in rows}:
        group = [r for r in rows if r["workload"] == workload]
        if {r["lang"] for r in group} != {"as", "cs", "gd"} or len({(r["iters"], r["checksum"]) for r in group}) != 1:
            raise ValueError(f"unequal work/checksums: {group}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--exe", type=Path, default=ROOT / "bin/godot.windows.editor.x86_64.mono.console.exe")
    args = parser.parse_args()
    if args.runs < 1 or not args.stage.replace("-", "").isalnum():
        parser.error("positive runs and alphanumeric stage required")
    prefix = DATA / f"binding-parity-{args.stage}"
    raw_path = prefix.with_suffix(".log")
    if raw_path.exists():
        parser.error(f"refusing to overwrite existing stage: {raw_path}")
    DATA.mkdir(parents=True, exist_ok=True)
    cmd = [str(args.exe), "--headless", "--path", str(ROOT / "modules/angelscript/tests/bench_project")]
    # Windows .console.exe is a launcher; also fingerprint the actual engine.
    runtime = args.exe.with_name(args.exe.name.replace(".console.exe", ".exe"))
    with args.exe.open("rb") as executable:
        launcher_hash = hashlib.file_digest(executable, "sha256").hexdigest()
    with runtime.open("rb") as executable:
        runtime_hash = hashlib.file_digest(executable, "sha256").hexdigest()
    source_diff = subprocess.check_output(["git", "diff", "--", "modules/angelscript"], cwd=ROOT)
    meta = {"stage": args.stage, "runs": args.runs, "command": cmd, "platform": platform.platform(),
            "executable_sha256": launcher_hash, "runtime_executable": str(runtime),
            "runtime_sha256": runtime_hash, "source_diff_sha256": hashlib.sha256(source_diff).hexdigest(),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "repetitions_per_process": 7,
            "csharp_assembly": "Debug (editor project)", "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    prefix.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    all_rows = []
    with raw_path.open("x", encoding="utf-8") as raw, prefix.with_suffix(".csv").open("x", newline="", encoding="utf-8") as table:
        writer = csv.DictWriter(table, FIELDS)
        writer.writeheader()
        for run in range(1, args.runs + 1):
            print(f"{args.stage}: process {run}/{args.runs}", flush=True)
            start = time.perf_counter()
            completed = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", timeout=300)
            raw.write(f"\n=== run {run}, exit {completed.returncode}, wall {time.perf_counter() - start:.3f}s ===\n")
            raw.write(completed.stdout)
            raw.flush()
            if completed.returncode:
                raise RuntimeError(f"engine exit {completed.returncode}; inspect {raw_path}")
            rows = parse_output(completed.stdout, args.stage, run)
            writer.writerows(rows)
            table.flush()
            all_rows.extend(rows)
    fields = ["stage", "workload", "lang", "median_ns", "min_process_median_ns", "max_process_median_ns", "checksum"]
    with Path(str(prefix) + "-summary.csv").open("x", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fields)
        writer.writeheader()
        for workload, lang in sorted({(r["workload"], r["lang"]) for r in all_rows}):
            group = [r for r in all_rows if r["workload"] == workload and r["lang"] == lang]
            if len({r["checksum"] for r in group}) != 1:
                raise ValueError(f"checksum changed across processes: {workload}/{lang}")
            ns = [r["median_us"] * 1000 / r["iters"] for r in group]
            writer.writerow(dict(zip(fields, [args.stage, workload, lang, statistics.median(ns), min(ns), max(ns), group[0]["checksum"]])))
    print(f"PASS: {len(all_rows)} samples, checksums match; {prefix}", flush=True)


if __name__ == "__main__":
    main()
