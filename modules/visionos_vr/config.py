def can_build(env, platform):
    return platform == "visionos" and not env["disable_3d"]


def configure(env):
    pass


def get_doc_classes():
    return [
        "VisionOSVRInterface",
    ]


def get_doc_path():
    return "doc_classes"
