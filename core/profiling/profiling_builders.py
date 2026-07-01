"""Functions used to generate source files during build time"""

import argparse
import os
import sys

# Add parent directory to path so we can import methods
sys.path.insert(0, root_directory := os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))

import methods


def profiler_gen_builder(target, profiler, sample_callstack, track_memory, record_on_demand):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="Target file")
    parser.add_argument("--profiler", help="Profiler type")
    parser.add_argument("--sample-callstack", action="store_true", help="Enable callstack sampling")
    parser.add_argument("--track-memory", action="store_true", help="Enable memory tracking")
    parser.add_argument("--record-on-demand", action="store_true", help="Enable on-demand recording")
    args = parser.parse_args()

    profiler_gen_builder(args.target, args.profiler, args.sample_callstack, args.track_memory, args.record_on_demand)


if __name__ == "__main__":
    main()
