def can_build(env, platform):
    if (
        platform == "linuxbsd" or platform == "windows"
    ):  # or platform == "android" -- temporarily disabled android support
        return env["openxr"]
    else:
        # not supported on these platforms
        return False


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
    ]


def get_doc_path():
    return "doc_classes"
