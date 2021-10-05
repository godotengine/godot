def can_build(env, platform):
    if platform in ["iphone"]:
        return False

    return env.module_check_dependencies("webm", ["ogg", "opus", "vorbis"])


def configure(env):
    pass


def get_doc_classes():
    return [
        "VideoStreamWebm",
    ]


def get_doc_path():
    return "doc_classes"


def get_module_dependencies():
    return [
        "ogg",
        "opus",
        "vorbis",
    ]
