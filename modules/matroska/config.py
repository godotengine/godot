def can_build(env, platform):
    return platform == "linuxbsd" or platform == "windows"


def configure(env):
    pass


def get_doc_classes():
    return [
        "VideoStreamMatroska",
    ]


def get_doc_path():
    return "doc_classes"
