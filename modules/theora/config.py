def can_build(env, platform):
    if env["arch"].startswith("rv"):
        return False
    env.module_add_dependencies("theora", ["ogg", "vorbis"])
    return True


def configure(env):
    pass


def get_doc_classes():
    return [
        "VideoStreamTheora",
    ]


def get_doc_path():
    return "doc_classes"
