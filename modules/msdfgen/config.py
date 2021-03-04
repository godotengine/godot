def can_build(env, platform):
    return env.module_check_dependencies("msdfgen", ["freetype"])


def configure(env):
    pass


def get_doc_classes():
    return [
        "MSDFLoader",
    ]


def get_doc_path():
    return "doc_classes"
