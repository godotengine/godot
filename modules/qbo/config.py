def can_build(env, platform):
    env.module_add_dependencies("qbo", ["gltf"])
    return not env["disable_3d"]


def configure(env):
    pass


def get_doc_classes():
    return ["QBODocument", "EditorSceneFormatImporterQBO"]


def get_doc_path():
    return "doc_classes"
