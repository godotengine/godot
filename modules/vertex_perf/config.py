"""Configuration for the Vertex performance module.

Provides performance profiles (Ultra Low / Low / Balanced / High / Ultra /
Custom), adaptive quality that scales expensive effects down on sustained
frame-time spikes, and memory/asset-cache budgets. Exposed to scripting via
the VertexPerformanceManager singleton (registered in register_types).
"""


def can_build(env, platform):
    return True


def configure(env):
    pass


def get_doc_classes():
    return [
        "VertexPerformanceManager",
        "VertexPerformanceProfile",
    ]


def get_doc_path():
    return "doc_classes"
