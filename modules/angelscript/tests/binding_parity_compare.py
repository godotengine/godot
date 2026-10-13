"""Compare process-median summaries; never substitute best-of-run timings."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "docs/language-benchmark/data"
STAGES = ("before", "auxiliary", "callbacks", "final")


def main():
    stages = {}
    for stage in STAGES:
        with (DATA / f"binding-parity-{stage}-summary.csv").open(encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source))
        indexed = {(r["workload"], r["lang"]): r for r in rows}
        if len(rows) != 51 or len(indexed) != 51:
            raise ValueError(f"invalid summary {stage}")
        stages[stage] = indexed
    baseline = stages["before"]
    for stage, entries in stages.items():
        if entries.keys() != baseline.keys():
            raise ValueError(f"workload mismatch: {stage}")
        for key, row in entries.items():
            if row["checksum"] != baseline[key]["checksum"]:
                raise ValueError(f"cross-stage checksum mismatch: {stage}/{key}")
    output = []
    for workload, lang in sorted(baseline):
        if lang != "as":
            continue
        values = {s: float(stages[s][workload, "as"]["median_ns"]) for s in STAGES}
        final = stages["final"][workload, "as"]
        gd = float(stages["final"][workload, "gd"]["median_ns"])
        cs = float(stages["final"][workload, "cs"]["median_ns"])
        low = float(final["min_process_median_ns"])
        high = float(final["max_process_median_ns"])
        ratio = values["final"] / gd
        output.append({
            "workload": workload,
            **{f"as_{s}_ns": v for s, v in values.items()},
            "gd_final_ns": gd, "cs_final_ns": cs,
            "as_final_min_process_median_ns": low,
            "as_final_max_process_median_ns": high,
            "as_final_range_pct": 100 * (high - low) / values["final"],
            "as_speedup_before_over_final": values["before"] / values["final"],
            "as_time_reduction_pct": 100 * (1 - values["final"] / values["before"]),
            "as_gd_before": values["before"] / float(stages["before"][workload, "gd"]["median_ns"]),
            "as_gd_final": ratio,
            "parity": ratio <= 1.0,
        })
    target = DATA / "binding-parity-comparison.csv"
    with target.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(json.dumps({"parity_count": sum(r["parity"] for r in output), "total": len(output), "rows": output}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
