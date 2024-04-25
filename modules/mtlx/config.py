def can_build(env, platform):
    if platform == "android":
        return False
    return True


def configure(env):
    pass


def get_doc_classes():
    return ["MTLXLoader"]


def get_doc_path():
    return "doc_classes"
