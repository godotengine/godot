"""Reproducible Windows process-paired A/B using the unchanged language benchmark.

No production source changes. Windows APIs constrain ONLY child benchmark processes.
All process samples are retained, including failed idle gates and noisy measurements.
"""
import argparse
import csv
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import time

from binding_parity_benchmark import ROOT, parse_output


def build_order(pairs):
    if pairs < 1:
        raise ValueError("positive pair count required")
    return [(pair, arm) for pair in range(1, pairs + 1) for arm in ("AB" if pair % 2 else "BA")]


def validate_samples(rows, pairs):
    expected = {(pair, arm) for pair, arm in build_order(pairs)}
    if len(rows) != pairs * 102 or {(r["pair"], r["arm"]) for r in rows} != expected:
        raise ValueError("missing/extra process samples")
    keys = {(r["pair"], r["arm"], r["workload"], r["lang"]) for r in rows}
    if len(keys) != len(rows):
        raise ValueError("duplicate process/workload/language row")
    workloads = {r["workload"] for r in rows}
    if len(workloads) != 17:
        raise ValueError("expected original 17 workloads")
    expected_keys = {(pair, arm, workload, lang) for pair, arm in expected for workload in workloads for lang in ("as", "cs", "gd")}
    if keys != expected_keys:
        raise ValueError("missing process/workload/language combination")
    if any(not 0 < r["median_us"] <= r["max_us"] or not 0 <= r["min_us"] <= r["median_us"] for r in rows):
        raise ValueError("invalid or zero process timings")
    for workload in workloads:
        group = [r for r in rows if r["workload"] == workload]
        if len(group) != pairs * 6 or {r["lang"] for r in group} != {"as", "cs", "gd"}:
            raise ValueError(f"missing workload/language: {workload}")
        if len({(r["iters"], r["checksum"]) for r in group}) != 1:
            raise ValueError(f"cross-version/language work mismatch: {workload}")


def paired_summary(rows, pairs):
    validate_samples(rows, pairs)
    result = []
    for workload in sorted({r["workload"] for r in rows}):
        group = [r for r in rows if r["workload"] == workload]
        ns = {(r["pair"], r["arm"], r["lang"]): r["median_us"] * 1000 / r["iters"] for r in group}
        ratios = [ns[p, "B", "as"] / ns[p, "A", "as"] for p in range(1, pairs + 1)]
        # Resample whole pairs, not seven within-process repetitions. Exploratory only.
        rng = random.Random(20261013)
        boot = sorted(statistics.median(rng.choices(ratios, k=pairs)) for _ in range(10000))
        low, high = boot[249], boot[9749]
        med = statistics.median(ratios)
        a, b, gd, cs = [[ns[p, arm, lang] for p in range(1, pairs + 1)] for arm, lang in (("A", "as"), ("B", "as"), ("B", "gd"), ("B", "cs"))]
        classification = "supported_regression" if low > 1.05 else "supported_improvement" if high < 0.95 else "inconclusive"
        gd_controls = [ns[p, "B", "gd"] / ns[p, "A", "gd"] for p in range(1, pairs + 1)]
        normalized = [ratios[i] / gd_controls[i] for i in range(pairs)]
        result.append(dict(workload=workload, a_as_ns=statistics.median(a), b_as_ns=statistics.median(b),
                           a_gd_ns=statistics.median(ns[p, "A", "gd"] for p in range(1, pairs + 1)),
                           paired_gd_ba_median=statistics.median(gd_controls),
                           paired_normalized_ba_median=statistics.median(normalized),
                           b_gd_ns=statistics.median(gd), b_cs_ns=statistics.median(cs),
                           paired_ba_median=med, paired_ba_ci_low=low, paired_ba_ci_high=high,
                           b_as_gd_ratio=statistics.median(b)/statistics.median(gd),
                           a_min_ns=min(a), a_max_ns=max(a), b_min_ns=min(b), b_max_ns=max(b),
                           gd_min_ns=min(gd), gd_max_ns=max(gd), classification=classification))
    return result


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def dependency_hashes(exe):
    root = exe.parent / "GodotSharp"
    if not root.is_dir():
        raise ValueError(f"missing managed runtime dependencies: {root}")
    return {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*")) if p.is_file()}


def project_hashes(project):
    paths = [p for p in project.iterdir() if p.is_file() and p.suffix in (".as", ".gd", ".cs", ".tscn", ".godot", ".csproj")]
    # Fingerprint the actually loaded Debug assembly plus its runtime/dependency files.
    paths += [p for p in (project / ".godot/mono/temp/bin/Debug").rglob("*") if p.is_file()]
    return {str(p.relative_to(project)): sha256(p) for p in sorted(paths)}


class WinMonitor:
    class Memory(ctypes.Structure):
        _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [(name, ctypes.c_ulonglong) for name in ("total_phys", "avail_phys", "total_page", "avail_page", "total_virtual", "avail_virtual", "avail_extended")]

    class CounterValue(ctypes.Structure):
        _fields_ = [("status", wintypes.DWORD), ("value", ctypes.c_double)]

    def __init__(self):
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.GetSystemTimes.argtypes = [ctypes.POINTER(wintypes.FILETIME)] * 3
        self.kernel.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(self.Memory)]
        self.pdh = ctypes.WinDLL("pdh")
        self.pdh.PdhOpenQueryW.argtypes = [wintypes.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(wintypes.HANDLE)]
        self.pdh.PdhAddEnglishCounterW.argtypes = [wintypes.HANDLE, wintypes.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(wintypes.HANDLE)]
        self.pdh.PdhCollectQueryData.argtypes = [wintypes.HANDLE]
        self.pdh.PdhGetFormattedCounterValue.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.c_void_p, ctypes.POINTER(self.CounterValue)]
        self.pdh.PdhCloseQuery.argtypes = [wintypes.HANDLE]
        self.query = wintypes.HANDLE()
        self.counters = {}
        self.errors = {}
        status = self.pdh.PdhOpenQueryW(None, 0, ctypes.byref(self.query))
        if status == 0:
            for key, path in {"frequency_mhz_proxy": r"\Processor Information(_Total)\Processor Frequency", "frequency_percent_proxy": r"\Processor Information(_Total)\% of Maximum Frequency", "processor_queue": r"\System\Processor Queue Length"}.items():
                handle = wintypes.HANDLE()
                code = self.pdh.PdhAddEnglishCounterW(self.query, path, 0, ctypes.byref(handle))
                if code == 0:
                    self.counters[key] = handle
                else:
                    self.errors[key] = code
            self.pdh.PdhCollectQueryData(self.query)
        else:
            self.errors["open"] = status
        self.previous = self.system_times()

    @staticmethod
    def ft(value):
        return (value.dwHighDateTime << 32) | value.dwLowDateTime

    def system_times(self):
        idle, kernel, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
        if not self.kernel.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
            raise ctypes.WinError(ctypes.get_last_error())
        return tuple(self.ft(v) for v in (idle, kernel, user))

    def sample(self):
        now = self.system_times()
        di, dk, du = [a-b for a, b in zip(now, self.previous)]
        self.previous = now
        data = dict(timestamp=time.time(), cpu_percent=(100 * (dk + du - di) / (dk + du)) if dk + du > 0 else None)
        memory = self.Memory()
        memory.length = ctypes.sizeof(memory)
        data["available_mb"] = memory.avail_phys / 2**20 if self.kernel.GlobalMemoryStatusEx(ctypes.byref(memory)) else None
        if self.query:
            self.pdh.PdhCollectQueryData(self.query)
        for name in ("frequency_mhz_proxy", "frequency_percent_proxy", "processor_queue"):
            data[name] = None
            if name in self.counters:
                value = self.CounterValue()
                status = self.pdh.PdhGetFormattedCounterValue(self.counters[name], 0x200, None, ctypes.byref(value))
                if status == 0 and value.status in (0, 1):
                    data[name] = value.value
        return data

    def close(self):
        if self.query:
            self.pdh.PdhCloseQuery(self.query)
            self.query = None


def run_child(exe, project, raw_path, mask, monitor, record, timeout=600, small=False):
    import _winapi
    import msvcrt
    kernel = monitor.kernel
    kernel.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    kernel.GetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    kernel.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.GetPriorityClass.argtypes = [wintypes.HANDLE]
    kernel.ResumeThread.argtypes = [wintypes.HANDLE]
    kernel.ResumeThread.restype = wintypes.DWORD
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    flags = 0x00000004 | subprocess.CREATE_NO_WINDOW  # CREATE_SUSPENDED is not exported by subprocess.
    env = os.environ.copy()
    env.pop("BENCH_SMALL", None)
    if small:
        env["BENCH_SMALL"] = "1"
    args = [str(exe), "--headless", "--path", str(project)]
    with raw_path.open("xb") as output, open(os.devnull, "rb") as stdin:
        out_handle, in_handle = msvcrt.get_osfhandle(output.fileno()), msvcrt.get_osfhandle(stdin.fileno())
        os.set_handle_inheritable(out_handle, True)
        os.set_handle_inheritable(in_handle, True)
        si = subprocess.STARTUPINFO()
        si.dwFlags = subprocess.STARTF_USESTDHANDLES
        si.hStdInput, si.hStdOutput, si.hStdError = in_handle, out_handle, out_handle
        ph, th, pid, tid = _winapi.CreateProcess(None, subprocess.list2cmdline(args), None, None, True, flags, env, str(ROOT), si)
        start = time.perf_counter()
        try:
            if not kernel.SetProcessAffinityMask(ph, mask) or not kernel.SetPriorityClass(ph, 0x8000):
                raise ctypes.WinError(ctypes.get_last_error())
            actual, system = ctypes.c_size_t(), ctypes.c_size_t()
            if not kernel.GetProcessAffinityMask(ph, ctypes.byref(actual), ctypes.byref(system)) or actual.value != mask:
                raise RuntimeError("child affinity was not applied")
            if kernel.ResumeThread(th) == 0xFFFFFFFF:
                raise ctypes.WinError(ctypes.get_last_error())
            while _winapi.WaitForSingleObject(ph, 1000) == _winapi.WAIT_TIMEOUT:
                record(monitor.sample(), pid)
                if time.perf_counter() - start > timeout:
                    raise TimeoutError(f"benchmark child {pid} exceeded {timeout}s")
            code = _winapi.GetExitCodeProcess(ph)
            times = [wintypes.FILETIME() for _ in range(4)]
            valid_times = kernel.GetProcessTimes(ph, *(ctypes.byref(t) for t in times))
            return dict(pid=pid, exit_code=code, wall_s=time.perf_counter()-start,
                        cpu_s=(monitor.ft(times[2])+monitor.ft(times[3]))/1e7 if valid_times else None,
                        affinity_mask=hex(actual.value), priority_class=kernel.GetPriorityClass(ph), command=args)
        finally:
            if _winapi.WaitForSingleObject(ph, 0) == _winapi.WAIT_TIMEOUT:
                _winapi.TerminateProcess(ph, 1)
                _winapi.WaitForSingleObject(ph, 10000)
            _winapi.CloseHandle(th)
            _winapi.CloseHandle(ph)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=Path, required=True, help="actual engine .exe, NOT console launcher")
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--a-revision", required=True)
    parser.add_argument("--b-revision", required=True)
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--affinity", type=lambda s: int(s, 0), default=0xF000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=ROOT / "modules/angelscript/tests/bench_project")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if os.name != "nt" or args.affinity <= 0 or args.affinity >= 1 << os.cpu_count():
        parser.error("Windows and valid logical processor mask required")
    exes = {"A": args.a.resolve(), "B": args.b.resolve()}
    if any(".console." in p.name for p in exes.values()):
        parser.error("pass actual engine, not launcher")
    order = build_order(args.pairs) if not args.smoke else [(1, "A"), (1, "B")]
    deps = {arm: dependency_hashes(exe) for arm, exe in exes.items()}
    if deps["A"] != deps["B"]:
        raise ValueError("A/B managed runtime dependencies differ")
    args.output.mkdir(parents=True, exist_ok=False)
    monitor = WinMonitor()
    meta = dict(started=time.strftime("%Y-%m-%dT%H:%M:%S%z"), platform=platform.platform(), pairs=args.pairs,
                order=order, affinity_mask=hex(args.affinity), priority="AboveNormal child only", repetitions=7,
                smoke=args.smoke, source_revisions={"A": args.a_revision, "B": args.b_revision},
                exes={arm: dict(path=str(p), sha256=sha256(p)) for arm, p in exes.items()},
                project_hashes=project_hashes(args.project), dependency_hashes=deps, monitor_errors=monitor.errors,
                runner_sha256=sha256(Path(__file__)), parser_sha256=sha256(Path(__file__).with_name("binding_parity_benchmark.py")),
                power_scheme=subprocess.run(["powercfg", "/getactivescheme"], capture_output=True).stdout.decode("mbcs", errors="replace"),
                sampling_interval_s=1, idle_gate=dict(max_seconds=60, consecutive_samples=3, cpu_below_percent=25),
                frequency_warning="PDH aggregate frequency proxy, not per-core effective clocks or scheduling trace")
    (args.output / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    all_rows, process_records = [], []
    fields = ["pair", "arm", "stage", "run", "workload", "lang", "iters", "min_us", "median_us", "max_us", "checksum"]
    try:
        with (args.output / "samples.csv").open("x", newline="", encoding="utf-8") as file, (args.output / "telemetry.jsonl").open("x", encoding="utf-8") as telemetry:
            writer = csv.DictWriter(file, fields)
            writer.writeheader()
            for pair, arm in order:
                idle_ok, consecutive = False, 0
                for _ in range(60):
                    time.sleep(1)
                    sample = monitor.sample()
                    sample.update(pair=pair, arm=arm, phase="idle")
                    telemetry.write(json.dumps(sample) + "\n")
                    consecutive = consecutive + 1 if sample["cpu_percent"] is not None and sample["cpu_percent"] < 25 else 0
                    if consecutive >= 3:
                        idle_ok = True
                        break
                print(f"pair {pair} arm {arm}, idle_gate={idle_ok}", flush=True)
                def record(sample, pid):
                    sample.update(pair=pair, arm=arm, phase="run", pid=pid)
                    telemetry.write(json.dumps(sample) + "\n")
                    telemetry.flush()
                raw = args.output / f"{pair:02d}-{arm}.log"
                result = run_child(exes[arm], args.project, raw, args.affinity, monitor, record, small=args.smoke)
                result.update(pair=pair, arm=arm, idle_gate_passed=idle_ok)
                process_records.append(result)
                (args.output / "processes.json").write_text(json.dumps(process_records, indent=2), encoding="utf-8")
                if result["exit_code"] != 0:
                    raise RuntimeError(f"nonzero engine exit; inspect {raw}")
                text = raw.read_text(encoding="utf-8", errors="replace")
                if any(marker in text for marker in ("SCRIPT ERROR:", "Parse Error:", "Nonexistent function", "Failed to load script", "Failed to load .NET", "ERROR:")):
                    raise ValueError(f"runtime error in benchmark output: {raw}")
                if args.smoke:
                    lines = [line for line in text.splitlines() if line.startswith("BENCH|bench_")]
                    if len(lines) != 51 or any("Nonexistent function" in line or "SCRIPT ERROR" in line for line in text.splitlines()):
                        raise ValueError(f"invalid smoke output: {raw}")
                else:
                    rows = parse_output(text, arm, pair)
                    for row in rows:
                        row.update(pair=pair, arm=arm)
                    writer.writerows(rows)
                    file.flush()
                    all_rows.extend(rows)
                if sha256(exes[arm]) != meta["exes"][arm]["sha256"]:
                    raise RuntimeError("runtime binary changed while measuring")
        if project_hashes(args.project) != meta["project_hashes"]:
            raise RuntimeError("benchmark or C# assembly changed while measuring")
        if {arm: dependency_hashes(exe) for arm, exe in exes.items()} != deps:
            raise RuntimeError("managed runtime dependencies changed while measuring")
        if not args.smoke:
            summary = paired_summary(all_rows, args.pairs)
            with (args.output / "summary.csv").open("x", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, list(summary[0]))
                writer.writeheader()
                writer.writerows(summary)
        print(f"PASS: {'smoke' if args.smoke else len(all_rows)} validated samples; {args.output}", flush=True)
    finally:
        monitor.close()


if __name__ == "__main__":
    main()
