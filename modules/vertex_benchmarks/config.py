"""Configuration for the Vertex benchmarks module.

Provides a benchmark runner that creates synthetic workloads (sprites,
particles, tilemaps, physics bodies, large scenes, textures, UI controls,
mobile workloads) and measures FPS, frame time, RAM, draw calls and startup
time using the engine Performance singleton. Intended to be run after every
major change to catch regressions.
"""


def can_build(env, platform):
    return True


def configure(env):
    pass


def get_doc_classes():
    return [
        "VertexBenchmarkRunner",
        "VertexBenchmarkResult",
    ]


def get_doc_path():
    return "doc_classes"
