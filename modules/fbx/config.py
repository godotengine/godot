def can_build(env, platform):
    return env["tools"]


def configure(env):
    pass


def get_doc_classes():
    return [
        "EditorSceneImporterFBX",
    ]


def get_doc_path():
    return "doc_classes"
