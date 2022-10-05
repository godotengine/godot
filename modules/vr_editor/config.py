from operator import truediv


def can_build(env, platform):
    # We rely on our OpenXR module so only include if that is available and we're building an editor build
    if platform in ("linuxbsd", "windows", "android") and env["openxr"] and env.editor_build:
        return True
    else:
        return False


def configure(env):
    pass


def get_doc_classes():
    return []


def get_doc_path():
    return "doc_classes"
