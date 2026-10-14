"""Summarize retained telemetry separately from the predeclared paired timing analysis."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return dict(n=0, median=None, minimum=None, maximum=None)
    return dict(n=len(values), median=statistics.median(values), minimum=min(values), maximum=max(values))


def analyze(root):
    meta = json.loads((root / "metadata.json").read_text())
    processes = json.loads((root / "processes.json").read_text())
    telemetry_path = root / "telemetry.jsonl"
    telemetry = [json.loads(line) for line in telemetry_path.read_text().splitlines()] if telemetry_path.exists() else []
    outcome_path = root / "result.json"
    outcome = json.loads(outcome_path.read_text()) if outcome_path.exists() else {"status": "complete"}
    if outcome["status"] not in ("complete", "idle_unavailable", "failed"):
        raise ValueError("experiment is not finalized")
    gates_path = root / "gates.json"
    gates = json.loads(gates_path.read_text()) if gates_path.exists() else []
    idle = [r for r in telemetry if r["phase"] == "idle"]
    gate_analysis = dict(gates=gates, idle_gates_passed=sum(g["passed"] for g in gates),
                         idle_samples=len(idle), idle_aggregate={key: stats([r.get(key) for r in idle])
                         for key in ("cpu_percent", "cpu_window_s", "processor_queue", "available_mb")},
                         valid_idle_samples=sum(r.get("gate_sample_valid", False) for r in idle))
    if outcome["status"] != "complete":
        if (root / "summary.csv").exists():
            raise ValueError("aborted experiment must not have a complete summary")
        return dict(status=outcome["status"], outcome=outcome, planned_processes=2*meta["pairs"],
                    process_records=processes, performance_comparison_available=False, **gate_analysis,
                    cautions=["No new complete performance comparison; do not merge partial samples into earlier runs",
                              "An idle gate timeout is not a performance regression",
                              "All recorded data retained; no automatic replacement attempt"])
    run = [r for r in telemetry if r["phase"] == "run"]
    with (root / "summary.csv").open(newline="", encoding="utf-8") as file:
        summary = list(csv.DictReader(file))
    if len(processes) != 2 * meta["pairs"] or len(summary) != 17:
        raise ValueError("incomplete experiment")
    records = []
    for p in processes:
        group = [r for r in run if r["pair"] == p["pair"] and r["arm"] == p["arm"]]
        idle = [r for r in telemetry if r["pair"] == p["pair"] and r["arm"] == p["arm"] and r["phase"] == "idle"]
        intervals = [b["timestamp"] - a["timestamp"] for a, b in zip(group, group[1:])]
        records.append(dict(pair=p["pair"], arm=p["arm"], idle_gate=p["idle_gate_passed"],
                            idle_samples=len(idle), idle_observed_span_s=idle[-1]["timestamp"]-idle[0]["timestamp"] if idle else None,
                            actual_sample_interval_s=stats(intervals),
                            wall_s=p["wall_s"], cpu_s=p["cpu_s"], samples=len(group),
                            **{key: stats([r.get(key) for r in group]) for key in ("cpu_percent", "available_mb", "frequency_mhz_proxy", "frequency_percent_proxy", "processor_queue")}))
    result = dict(processes=records, idle_gates_passed=sum(p["idle_gate_passed"] for p in processes),
                  telemetry_samples=len(run), aggregate={key: stats([r.get(key) for r in run]) for key in ("cpu_percent", "processor_queue", "frequency_mhz_proxy", "available_mb")},
                  classifications={kind: [r["workload"] for r in summary if r["classification"] == kind] for kind in ("supported_improvement", "supported_regression", "inconclusive")},
                  numerical_as_gd_parity=[r["workload"] for r in summary if float(r["b_as_gd_ratio"]) <= 1],
                  cautions=["No samples removed", "Intervals exploratory: six pairs, no multiple-comparison correction", "PDH frequency proxy is not effective per-core frequency", "Tested pair is combined optimization, not remaining-nine isolated delta"])
    result.update(status="complete", performance_comparison_available=True)
    if gates:
        result["gate_analysis"] = gate_analysis
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = analyze(args.directory)
    with (args.directory / "analysis.json").open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
