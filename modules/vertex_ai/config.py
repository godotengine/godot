"""Configuration for the Vertex AI assistant module.

Provides an extensible architecture that can understand project structure,
scenes, scripts, errors, logs, profiler results and performance problems.
Implements a command registry (Create a player, Create a scene, Fix this
error, Optimize this project, Reduce memory usage, Explain this node, Create
animation, Create UI, Optimize this shader) and requires confirmation before
destructive changes. The LLM backend is pluggable via a Callable so a real
provider can be wired in without changing the engine.
"""


def can_build(env, platform):
    return True


def configure(env):
    pass


def get_doc_classes():
    return [
        "VertexAIAssistant",
        "VertexAICommand",
        "VertexAIContext",
    ]


def get_doc_path():
    return "doc_classes"
