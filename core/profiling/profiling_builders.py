"""Functions used to generate source files during build time"""

import methods


def tracy_gen(file, env):
    file.write("#define GODOT_USE_TRACY\n")
    file.write("#define TRACY_ENABLE\n")

    if env["profiler_sample_callstack"]:
        file.write("#define TRACY_CALLSTACK 62\n")
    else:
        file.write("#define TRACY_NO_SAMPLING\n")

    if env["profiler_track_memory"]:
        file.write("#define GODOT_PROFILER_TRACK_MEMORY\n")

    if env["profiler_record_on_demand"]:
        file.write("#define TRACY_ON_DEMAND\n")


def perfetto_gen(file, env):
    file.write("#define GODOT_USE_PERFETTO\n")


def instruments_gen(file, env):
    file.write("#define GODOT_USE_INSTRUMENTS\n")
    if env["profiler_sample_callstack"]:
        file.write("#define INSTRUMENTS_SAMPLE_CALLSTACKS\n")


def profiler_gen_builder(target, source, env):
    with methods.generated_wrapper(str(target[0])) as file:
        if env["profiler"] == "tracy":
            tracy_gen(file, env)
        elif env["profiler"] == "perfetto":
            perfetto_gen(file, env)
        elif env["profiler"] == "instruments":
            instruments_gen(file, env)
