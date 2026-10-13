"""Pool both final-binary runs without discarding the first noisy measurements."""
import csv
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "docs/language-benchmark/data"
STAGES = ["nine-final", "nine-confirm"]


def main():
    samples = {}
    metadata = []
    for stage in STAGES:
        base = DATA / f"binding-parity-{stage}"
        metadata.append(json.loads(base.with_suffix(".json").read_text(encoding="utf-8")))
        with base.with_suffix(".csv").open(encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        keys = {(r["workload"], r["lang"]) for r in rows}
        if len(rows) != 153 or len(keys) != 51 or len({(r["run"], r["workload"], r["lang"]) for r in rows}) != 153:
            raise ValueError(f"invalid stage {stage}")
        samples[stage] = rows
    for field in ("runtime_sha256", "source_diff_sha256", "source_revision", "repetitions_per_process"):
        if metadata[0][field] != metadata[1][field]:
            raise ValueError(f"not the same final build: {field}")
    rows = [r for stage in STAGES for r in samples[stage]]
    keys = {(r["workload"], r["lang"]) for r in rows}
    summary = {}
    for key in keys:
        group = [r for r in rows if (r["workload"], r["lang"]) == key]
        if len(group) != 6 or len({(r["iters"], r["checksum"]) for r in group}) != 1:
            raise ValueError(f"inconsistent samples: {key}")
        ns = [int(r["median_us"]) * 1000 / int(r["iters"]) for r in group]
        summary[key] = (statistics.median(ns), min(ns), max(ns))
    output = []
    fields = ["workload", "as_first_ns", "as_confirm_ns", "as_pooled_ns", "gd_pooled_ns", "cs_pooled_ns", "as_gd_ratio", "as_min_ns", "as_max_ns", "gd_min_ns", "gd_max_ns", "numerical_parity"]
    for workload in sorted({k[0] for k in keys}):
        group = [r for r in rows if r["workload"] == workload]
        if {r["lang"] for r in group} != {"as", "cs", "gd"} or len({(r["iters"], r["checksum"]) for r in group}) != 1:
            raise ValueError(f"language mismatch: {workload}")
        stage_as = [statistics.median(int(r["median_us"]) * 1000 / int(r["iters"]) for r in samples[s] if r["workload"] == workload and r["lang"] == "as") for s in STAGES]
        a, gd, cs = [summary[workload, lang] for lang in ("as", "gd", "cs")]
        output.append(dict(zip(fields, [workload, *stage_as, a[0], gd[0], cs[0], a[0]/gd[0], a[1], a[2], gd[1], gd[2], a[0] <= gd[0]])))
    with (DATA / "remaining-nine-stability.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fields)
        writer.writeheader()
        writer.writerows(output)
    print(f"PASS: {len(rows)} samples from identical final binary; pooled numerical parity {sum(r['numerical_parity'] for r in output)}/17")


if __name__ == "__main__":
    main()
