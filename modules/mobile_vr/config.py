def can_build(env, platform):
    return not env["disable_xr"]


def configure(env):
    pass


def get_doc_classes():
    return [
        "MobileVRInterface",
    ]


def get_doc_path():
    return "doc_classes"
