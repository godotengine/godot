"""Functions used to generate source files during build time"""

import methods


def profiler_gen_builder(target, source, env):
    with methods.generated_wrapper(str(target[0])) as file:
        if env["profiler"] == "tracy":
            file.write("#define GODOT_USE_TRACY\n")
            if env["profiler_sample_callstack"]:
                file.write("#define TRACY_CALLSTACK 62\n")
            if env["profiler_track_memory"]:
                file.write("#define GODOT_PROFILER_TRACK_MEMORY\n")
        if env["profiler"] == "perfetto":
            file.write("#define GODOT_USE_PERFETTO\n")
        if env["profiler"] == "instruments":
            file.write("#define GODOT_USE_INSTRUMENTS\n")
            if env["profiler_sample_callstack"]:
                file.write("#define INSTRUMENTS_SAMPLE_CALLSTACKS\n")
