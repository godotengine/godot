def can_build(env, platform):
    if not env["openxr"] or env["disable_3d"]:
        return False

    if platform not in ("linuxbsd", "windows", "android"):
        return False

    if platform == "linuxbsd" and env["opengl3"] and not env["x11"]:  # Needs Xlib for OpenGL
        return False

    return True


def configure(env):
    pass


def get_doc_classes():
    return [
        "OpenXRInterface",
        "OpenXRAction",
        "OpenXRActionSet",
        "OpenXRActionMap",
        "OpenXRInteractionProfile",
        "OpenXRIPBinding",
        "OpenXRHand",
    ]


def get_doc_path():
    return "doc_classes"
