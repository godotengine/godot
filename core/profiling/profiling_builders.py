"""Functions used to generate source files during build time"""

import argparse
import sys

try:
    sys.path.insert(0, "./")
    import methods
except ImportError:
    raise SystemExit(f'Generator script "{__file__}" must be run from repository root!')


def profiling(target: str, profiler: str, sample_callstack: bool, track_memory: bool, record_on_demand: bool) -> None:
    with methods.generated_wrapper(target) as file:
        if profiler == "tracy":
            file.write("#define GODOT_USE_TRACY\n")
            if sample_callstack:
                file.write("#define TRACY_CALLSTACK 62\n")
            if track_memory:
                file.write("#define GODOT_PROFILER_TRACK_MEMORY\n")
            if record_on_demand:
                file.write("#define TRACY_ON_DEMAND\n")
        if profiler == "perfetto":
            file.write("#define GODOT_USE_PERFETTO\n")
        if profiler == "instruments":
            file.write("#define GODOT_USE_INSTRUMENTS\n")
            if sample_callstack:
                file.write("#define INSTRUMENTS_SAMPLE_CALLSTACKS\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    profiling_parser = subparsers.add_parser("profiling")
    profiling_parser.add_argument("target")
    profiling_parser.add_argument("profiler")
    profiling_parser.add_argument("--sample_callstack", action="store_true")
    profiling_parser.add_argument("--track_memory", action="store_true")
    profiling_parser.add_argument("--record_on_demand", action="store_true")

    args = vars(parser.parse_args())
    command = globals().get(args.pop("command"), {})
    command(**args)


if __name__ == "__main__":
    main()
