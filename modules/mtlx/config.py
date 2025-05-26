def can_build(env, platform):
    return platform == "linuxbsd"


def configure(env):
    pass


def get_doc_classes():
    return ["ResourceFormatLoaderMtlx"]


def get_doc_path():
    return "doc_classes"
