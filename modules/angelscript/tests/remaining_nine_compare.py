"""Cross-check the unchanged workloads for the remaining-nine optimization round."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "docs/language-benchmark/data"
STAGES = ["nine-before", "nine-direct", "nine-geometry", "nine-final"]


def main():
    tables = {}
    for stage in STAGES:
        with (DATA / f"binding-parity-{stage}-summary.csv").open(encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        table = {(r["workload"], r["lang"]): r for r in rows}
        if len(rows) != 51 or len(table) != 51:
            raise ValueError(f"invalid summary {stage}")
        tables[stage] = table
    reference = tables[STAGES[0]]
    for stage, table in tables.items():
        if table.keys() != reference.keys():
            raise ValueError(f"different workloads {stage}")
        for key, row in table.items():
            if row["checksum"] != reference[key]["checksum"]:
                raise ValueError(f"checksum mismatch {key} {stage}")
    raw = []
    for stage in STAGES:
        with (DATA / f"binding-parity-{stage}.csv").open(encoding="utf-8") as source:
            raw.extend(csv.DictReader(source))
    for key in reference:
        group = [r for r in raw if (r["workload"], r["lang"]) == key]
        if len(group) != 3 * len(STAGES) or len({(r["iters"], r["checksum"]) for r in group}) != 1:
            raise ValueError(f"raw sample mismatch {key}")
    fields = ["workload", "as_before_ns", "as_direct_ns", "as_geometry_experiment_ns", "as_final_ns", "gd_final_ns", "cs_final_ns", "as_gd_ratio", "as_reduction_pct", "as_process_min_ns", "as_process_max_ns", "gd_process_min_ns", "gd_process_max_ns", "numerical_parity", "gd_geometry_experiment_ns", "geometry_experiment_ratio"]
    output = []
    for workload in sorted({key[0] for key in reference}):
        a = [float(tables[s][workload, "as"]["median_ns"]) for s in STAGES]
        final = tables[STAGES[-1]]
        as_row, gd_row = final[workload, "as"], final[workload, "gd"]
        gd, cs = float(gd_row["median_ns"]), float(final[workload, "cs"]["median_ns"])
        values = [workload, *a, gd, cs, a[-1]/gd, (1-a[-1]/a[0])*100,
                  as_row["min_process_median_ns"], as_row["max_process_median_ns"],
                  gd_row["min_process_median_ns"], gd_row["max_process_median_ns"], a[-1]<=gd,
                  float(tables["nine-geometry"][workload, "gd"]["median_ns"]),
                  a[2]/float(tables["nine-geometry"][workload, "gd"]["median_ns"])]
        output.append(dict(zip(fields, values)))
    target = DATA / "remaining-nine-comparison.csv"
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(output)
    print(f"PASS: {len(raw)} rows consistent; numerical parity {sum(r['numerical_parity'] for r in output)}/17")


if __name__ == "__main__":
    main()
