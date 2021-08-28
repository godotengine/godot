def can_build(env, platform):
    if env["arch"].startswith("rv"):
        return False
    return env.module_check_dependencies("theora", ["ogg", "vorbis"])


def configure(env):
    pass


def get_doc_classes():
    return [
        "VideoStreamTheora",
    ]


def get_doc_path():
    return "doc_classes"
