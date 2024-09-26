def can_build(env, platform):
    if platform in ("linuxbsd", "windows", "android", "macos"):
        return not env["disable_xr"]
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
        "OpenXRAPIExtension",
        "OpenXRExtensionWrapperExtension",
        "OpenXRInteractionProfile",
        "OpenXRInteractionProfileMetadata",
        "OpenXRIPBinding",
        "OpenXRHand",
        "OpenXRVisibilityMask",
        "OpenXRCompositionLayer",
        "OpenXRCompositionLayerQuad",
        "OpenXRCompositionLayerCylinder",
        "OpenXRCompositionLayerEquirect",
    ]


def get_doc_path():
    return "doc_classes"
