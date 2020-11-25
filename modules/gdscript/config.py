def can_build(env, platform):
    return True


def configure(env):
    env.use_ptrcall = True


def get_doc_classes():
    return [
        "@GDScript",
        "GDScript",
        "GDScriptFunctionState",
    ]


def get_doc_path():
    return "doc_classes"
