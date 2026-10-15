"""Diagnostic control: is the AS speed change caused by the binary or the machine state?

Alternates the current build and the preserved candidate build so that any drift in
machine state is shared by both. Not a formal measurement; no idle gate is used.
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modules/angelscript/tests"))
from binding_parity_benchmark import ROOT, parse_output  # noqa: E402

EXES = {
    "current": ROOT / "bin/godot.windows.editor.x86_64.mono.exe",
    "preserved": ROOT / ".superpowers/bench-ab-20261013/candidate/bin/godot.windows.editor.x86_64.mono.exe",
}
ORDER = ["current", "preserved", "current", "preserved"]
PROJECT = ROOT / "modules/angelscript/tests/bench_project"
OUT = Path(__file__).resolve().parent
PREFIX = sys.argv[1] if len(sys.argv) > 1 else "binary-control"
ORDER = ["preserved", "current", "preserved", "current"] if len(sys.argv) > 2 and sys.argv[2] == "reverse" else ORDER


def run_once(label, index, exe):
    log = OUT / f"{PREFIX}-{label}-{index}.log"
    with log.open("xb") as output:
        done = subprocess.run([str(exe), "--headless", "--path", str(PROJECT)], stdout=output,
                              stderr=subprocess.STDOUT, timeout=300)
    text = log.read_text(encoding="utf-8", errors="replace")
    if done.returncode or "SCRIPT ERROR" in text:
        raise RuntimeError(f"{label} run {index} failed: exit {done.returncode}")
    rows = parse_output(text, label, index)
    return rows


records = []
for index, label in enumerate(ORDER, start=1):
    rows = run_once(label, index, EXES[label])
    for row in rows:
        row["build"] = label
        row["iteration"] = index
    records.extend(rows)
    print(f"ran {label} #{index}", flush=True)

# per build/iteration medians per workload/language
agg = {}
for row in records:
    key = (row["build"], row["iteration"], row["workload"], row["lang"])
    agg[key] = row["median_us"] * 1000 / row["iters"]
result = {}
for (build, iteration, workload, lang), ns in agg.items():
    result.setdefault(build, {}).setdefault(workload, {}).setdefault(lang, []).append(ns)

table = []
for workload in sorted({r["workload"] for r in records}):
    entry = {"workload": workload}
    for build in EXES:
        for lang in ("as", "cs", "gd"):
            values = result[build][workload][lang]
            entry[f"{build}_{lang}"] = __import__("statistics").median(values)
    entry["current_as_gd"] = entry["current_as"] / entry["current_gd"]
    entry["preserved_as_gd"] = entry["preserved_as"] / entry["preserved_gd"]
    entry["as_speedup"] = entry["preserved_as"] / entry["current_as"]
    entry["gd_speedup"] = entry["preserved_gd"] / entry["current_gd"]
    table.append(entry)

with (OUT / f"{PREFIX}-20261014.json").open("x", encoding="utf-8") as handle:
    json.dump(dict(table=table), handle, indent=2)
with (OUT / f"{PREFIX}-20261014.csv").open("x", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, list(table[0]))
    writer.writeheader()
    writer.writerows(table)
header = f"{'workload':<20}{'AS cur':>9}{'AS old':>9}{'AS x':>7}{'GD cur':>9}{'GD old':>9}{'GD x':>7}{'CS cur':>9}{'CS old':>9}"
print(header)
for entry in table:
    print(f"{entry['workload']:<20}{entry['current_as']:>9.2f}{entry['preserved_as']:>9.2f}{entry['as_speedup']:>7.2f}"
          f"{entry['current_gd']:>9.2f}{entry['preserved_gd']:>9.2f}{entry['gd_speedup']:>7.2f}"
          f"{entry['current_cs']:>9.2f}{entry['preserved_cs']:>9.2f}")
