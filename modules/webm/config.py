def can_build(env, platform):
<<<<<<< HEAD
    if platform in ["iphone"]:
        return False

    return env.module_check_dependencies("webm", ["ogg", "opus", "vorbis"])
=======
    return platform not in ["iphone"]
>>>>>>> master


def configure(env):
    pass


def get_doc_classes():
    return [
        "VideoStreamWebm",
    ]


def get_doc_path():
    return "doc_classes"
