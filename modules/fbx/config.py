def can_build(env, platform):
    env.module_add_dependencies("fbx", ["gltf"])
    return not env["disable_3d"]


def configure(env):
    pass


def get_doc_classes():
    return [
        "EditorSceneFormatImporterFBX2GLTF",
        "EditorSceneFormatImporterUFBX",
        "FBXDocument",
        "FBXState",
    ]


def get_doc_path():
    return "doc_classes"
