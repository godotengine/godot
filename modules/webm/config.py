def can_build(env, platform):
    env.module_add_dependencies("webm", ["ogg", "opus", "vorbis"])
    return True


def configure(env):
    pass


def get_doc_classes():
    return [
        "VideoStreamWebm",
    ]


def get_doc_path():
    return "doc_classes"
