"""Configuration for the Vertex mobile editor module.

Provides a dedicated responsive/mobile editor layout (large touch targets,
compact toolbar, collapsible panels, touch gestures, pinch zoom, mobile scene
tree, virtual-keyboard-friendly controls) rather than simply shrinking the
desktop UI. Runtime settings live in VertexMobileSettings; the editor layout
plugin is built only under editor_build.
"""


def can_build(env, platform):
    return True


def configure(env):
    pass


def get_doc_classes():
    return [
        "VertexMobileSettings",
    ]


def get_doc_path():
    return "doc_classes"
