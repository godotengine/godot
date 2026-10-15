"""Three-language (AngelScript / C# / GDScript) comparison with the unchanged benchmark.

Reuses the deadline-bounded, idle-gated Windows process harness. Cross-language
ratios are paired inside each process, so machine load cannot silently become a
language ranking; absolute timings still require a low-load window.
"""
import argparse
import csv
import json
import math
import os
import platform
import random
import statistics
import subprocess
import time
from pathlib import Path

from binding_parity_benchmark import ROOT, parse_output
from interleaved_ab_benchmark import (ProcessMonitor, dependency_hashes, project_hashes,
                                      run_child, sha256, wait_for_idle)

LANGS = ("as", "cs", "gd")
WORKLOAD_COUNT = 17
REPETITIONS = 7
BENCHMARK_SOURCES = ("bench_driver.gd", "bench_as.as", "bench_gd.gd", "BenchCs.cs", "main.tscn", "project.godot")


def _ns(row):
    return row["median_us"] * 1000 / row["iters"]


def _median_ci(values, seed):
    rng = random.Random(seed)
    boot = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(10000))
    return boot[249], boot[9749]


def _ratio_ci(numerator, denominator, seed):
    ratios = [a / b for a, b in zip(numerator, denominator)]
    low, high = _median_ci(ratios, seed)
    return statistics.median(ratios), low, high


def _cv_percent(values):
    if len(values) < 2 or statistics.mean(values) == 0:
        return None
    return 100 * statistics.stdev(values) / statistics.mean(values)


def compute_summary(rows, runs, seed=20261014):
    """Validate the full matrix and derive per-workload language statistics.

    Ratios are computed per process and then aggregated across processes, so a
    globally busy machine moves all three languages together instead of being
    reported as a language difference.
    """
    if runs < 2:
        raise ValueError("at least two processes are required for ratio intervals")
    workloads = sorted({r["workload"] for r in rows})
    if len(workloads) != WORKLOAD_COUNT:
        raise ValueError(f"expected the original {WORKLOAD_COUNT} workloads, got {len(workloads)}")
    expected = {(run, w, lang) for run in range(1, runs + 1) for w in workloads for lang in LANGS}
    if {(r["process"], r["workload"], r["lang"]) for r in rows} != expected or len(rows) != len(expected):
        raise ValueError("missing/extra/duplicate process/workload/language rows")
    indexed = {(r["process"], r["workload"], r["lang"]): r for r in rows}
    summary = []
    for workload in workloads:
        per_lang = {}
        for lang in LANGS:
            group = [indexed[(run, workload, lang)] for run in range(1, runs + 1)]
            if len({r["checksum"] for r in group}) != 1 or len({r["iters"] for r in group}) != 1:
                raise ValueError(f"cross-process mismatch: {workload}/{lang}")
            if any(not 0 < r["median_us"] <= r["max_us"] or not 0 <= r["min_us"] <= r["median_us"] for r in group):
                raise ValueError(f"invalid timing: {workload}/{lang}")
            per_lang[lang] = group
        for run in range(1, runs + 1):
            signature = {(indexed[(run, workload, lang)]["checksum"], indexed[(run, workload, lang)]["iters"]) for lang in LANGS}
            if len(signature) != 1:
                raise ValueError(f"languages disagree on work/checksum: {workload} run {run}")
        medians = {lang: [_ns(r) for r in per_lang[lang]] for lang in LANGS}
        entry = dict(workload=workload, iters=per_lang["as"][0]["iters"], checksum=per_lang["as"][0]["checksum"])
        for lang in LANGS:
            values = medians[lang]
            entry[f"{lang}_median_ns"] = statistics.median(values)
            entry[f"{lang}_best_ns"] = min(r["min_us"] * 1000 / r["iters"] for r in per_lang[lang])
            entry[f"{lang}_worst_ns"] = max(r["max_us"] * 1000 / r["iters"] for r in per_lang[lang])
            entry[f"{lang}_process_min_ns"] = min(values)
            entry[f"{lang}_process_max_ns"] = max(values)
            entry[f"{lang}_cv_percent"] = _cv_percent(values)
        for label, numerator, denominator in (("as_gd", "as", "gd"), ("cs_gd", "cs", "gd"), ("as_cs", "as", "cs")):
            median, low, high = _ratio_ci(medians[numerator], medians[denominator], seed)
            entry[f"{label}_ratio"] = median
            entry[f"{label}_ci_low"] = low
            entry[f"{label}_ci_high"] = high
        entry["fastest_lang"] = min(LANGS, key=lambda lang: entry[f"{lang}_median_ns"])
        entry["as_faster_than_gd"] = entry["as_gd_ratio"] < 1
        entry["cs_faster_than_as"] = entry["as_cs_ratio"] > 1
        summary.append(entry)
    def geomean(key):
        return math.exp(statistics.mean(math.log(e[key]) for e in summary))
    aggregate = dict(runs=runs, workloads=len(summary),
                     as_gd_geomean=geomean("as_gd_ratio"), cs_gd_geomean=geomean("cs_gd_ratio"),
                     as_cs_geomean=geomean("as_cs_ratio"),
                     as_faster_than_gd=[e["workload"] for e in summary if e["as_faster_than_gd"]],
                     gd_faster_than_as=[e["workload"] for e in summary if not e["as_faster_than_gd"]],
                     cs_fastest=[e["workload"] for e in summary if e["fastest_lang"] == "cs"],
                     gd_fastest=[e["workload"] for e in summary if e["fastest_lang"] == "gd"],
                     as_fastest=[e["workload"] for e in summary if e["fastest_lang"] == "as"],
                     boundary_increment_ns={lang: statistics.median(
                         [_ns(indexed[(run, "bench_engine", lang)]) - _ns(indexed[(run, "bench_hoist", lang)])
                          for run in range(1, runs + 1)])
                         for lang in LANGS})
    return dict(summary=summary, aggregate=aggregate)


def _source_snapshot(project):
    return {name: sha256(project / name) for name in BENCHMARK_SOURCES if (project / name).is_file()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path, default=ROOT / "bin/godot.windows.editor.x86_64.mono.exe")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=6)
    parser.add_argument("--affinity", type=lambda s: int(s, 0), default=0xF000)
    parser.add_argument("--project", type=Path, default=ROOT / "modules/angelscript/tests/bench_project")
    args = parser.parse_args()
    if os.name != "nt" or args.runs < 2 or args.affinity <= 0 or args.affinity >= 1 << os.cpu_count():
        parser.error("Windows, at least two runs and a valid logical processor mask required")
    exe = args.exe.resolve()
    if ".console." in exe.name:
        parser.error("pass the actual engine, not the console launcher")
    args.output.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2), encoding="utf-8")

    project = args.project
    deps = dependency_hashes(exe)
    meta = dict(schema_version=1, started=time.strftime("%Y-%m-%dT%H:%M:%S%z"), platform=platform.platform(),
                source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                source_status=subprocess.check_output(["git", "status", "--short", "--", "modules/angelscript"], cwd=ROOT, text=True).splitlines(),
                module_diff_sha256=sha256_from_git("modules/angelscript"), exe=str(exe), exe_sha256=sha256(exe),
                dependencies=deps, project_hashes=project_hashes(project), benchmark_sources=_source_snapshot(project),
                runs=args.runs, repetitions=REPETITIONS, workloads=WORKLOAD_COUNT,
                affinity_mask=hex(args.affinity), priority="AboveNormal child only",
                csharp_assembly="Debug (LangBench.csproj, rebuilt from recorded source)",
                language_order="fixed by driver: as, cs, gd",
                idle_gate=dict(max_seconds=60, consecutive_samples=3, cpu_below_percent=25, sample_timeout_s=3,
                               failure_policy="abort_attempt_no_retry"),
                statistics="per-process ns/op; median across processes; paired per-process ratios with 10000-resample bootstrap CI (seed 20261014)",
                power_scheme=subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, check=True, timeout=10).stdout.decode("mbcs", errors="replace"),
                caveats=["bench_packed/typed_i64 use CLR arrays for C# but PackedInt64Array for AS/GD",
                         "typed_vec2 puts the loop in C++ for AS but in-language for C#/GD",
                         "string types differ across languages",
                         "bench_engine uses a temporary Node for AS"])
    save("metadata.json", meta)
    with (args.output / "runner-measured.py").open("xb") as snapshot:
        snapshot.write(Path(__file__).read_bytes())
    fields = ["process", "workload", "lang", "iters", "min_us", "median_us", "max_us", "checksum"]
    rows, gates, processes = [], [], []
    outcome = dict(status="running", completed_processes=0, validated_rows=0)
    save("result.json", outcome)
    save("processes.json", processes)
    save("gates.json", gates)
    monitor = ProcessMonitor()
    try:
        with (args.output / "telemetry.jsonl").open("x", encoding="utf-8") as telemetry, \
                (args.output / "samples.csv").open("x", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fields)
            writer.writeheader()
            file.flush()
            for run in range(1, args.runs + 1):
                def record(sample, pid=None, run=run):
                    sample.update(process=run, phase="idle" if pid is None else "run")
                    if pid is not None:
                        sample["pid"] = pid
                    telemetry.write(json.dumps(sample) + "\n")
                    telemetry.flush()
                gate = wait_for_idle(monitor, record=record)
                gate["process"] = run
                gates.append(gate)
                save("gates.json", gates)
                print(f"process {run} idle_gate={gate['passed']} reason={gate['reason']} elapsed={gate['elapsed_s']:.3f}s", flush=True)
                if not gate["passed"]:
                    outcome.update(status="idle_unavailable", stopped_before_process=run, reason=gate["reason"])
                    break
                raw = args.output / f"{run:02d}.log"
                entry = dict(process=run, status="started", idle_gate_passed=True, raw_log=raw.name)
                processes.append(entry)
                save("processes.json", processes)
                try:
                    entry.update(run_child(exe, project, raw, args.affinity, monitor, record, timeout=300))
                    if entry["exit_code"] != 0:
                        raise RuntimeError(f"nonzero engine exit; inspect {raw}")
                    text = raw.read_text(encoding="utf-8", errors="replace")
                    if any(marker in text for marker in ("SCRIPT ERROR:", "Parse Error:", "Nonexistent function", "Failed to load script", "Failed to load .NET")):
                        raise ValueError(f"runtime error in benchmark output: {raw}")
                    parsed = parse_output(text, "three-language", run)
                    for row in parsed:
                        row.pop("stage", None)
                        row["process"] = row.pop("run")
                    writer.writerows(parsed)
                    file.flush()
                    rows.extend(parsed)
                    entry["status"] = "validated"
                    outcome["completed_processes"] += 1
                except Exception as error:
                    entry.update(status="failed", error=f"{type(error).__name__}: {error}")
                    raise
                finally:
                    save("processes.json", processes)
            else:
                outcome["status"] = "complete"
    except Exception as error:
        outcome.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        outcome["monitor_cleanup"] = monitor.close()
        verification_start = time.monotonic()
        try:
            checks = dict(exe=sha256(exe) == meta["exe_sha256"],
                          dependencies=dependency_hashes(exe) == deps,
                          project=project_hashes(project) == meta["project_hashes"],
                          benchmark_sources=_source_snapshot(project) == meta["benchmark_sources"],
                          runner=sha256(Path(__file__)) == sha256(args.output / "runner-measured.py"))
            outcome["final_identity_checks"] = checks
            if not all(checks.values()):
                raise ValueError(f"identity changed while measuring: {checks}")
        except Exception as error:
            outcome.update(status="failed", identity_error=f"{type(error).__name__}: {error}")
        if outcome["monitor_cleanup"]["alive"]:
            outcome.update(status="failed", error="owned monitor did not exit")
        outcome.update(validated_rows=len(rows), finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                       final_verification_s=time.monotonic() - verification_start)
        save("result.json", outcome)
        save("processes.json", processes)
        save("gates.json", gates)
    if outcome["status"] == "complete":
        result = compute_summary(rows, args.runs)
        with (args.output / "summary.csv").open("x", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, list(result["summary"][0]))
            writer.writeheader()
            writer.writerows(result["summary"])
        save("analysis.json", result)
        print(f"PASS: {len(rows)} validated samples across {args.runs} processes", flush=True)
        return 0
    print("STOP: " + json.dumps(outcome), flush=True)
    return 2 if outcome["status"] == "idle_unavailable" else 1


def sha256_from_git(path):
    import hashlib
    diff = subprocess.check_output(["git", "diff", "--", path], cwd=ROOT)
    return hashlib.sha256(diff).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
