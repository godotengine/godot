"""Configuration for the Vertex Project Optimizer module.

Analyzes a project for performance risk (large textures/assets, expensive
shaders, excessive particles, high draw calls, memory usage, unnecessary
processing, physics load) and produces recommendations plus safe automatic
optimizations. Exposed via the VertexProjectOptimizer class.
"""


def can_build(env, platform):
    return True


def configure(env):
    pass


def get_doc_classes():
    return [
        "VertexProjectOptimizer",
        "VertexOptimizationReport",
    ]


def get_doc_path():
    return "doc_classes"
