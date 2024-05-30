def can_build(env, platform):
    return True


def configure(env):
    pass


def get_doc_classes():
    return [
        "RetargetAnimationPlayer",
        "RetargetAnimationTree",
        "RetargetPoseTransporter",
        "RetargetProfile",
        "RetargetProfileGlobalAll",
        "RetargetProfileLocalAll",
        "RetargetProfileAbsoluteAll",
        "RetargetProfileLocalFingersGlobalOthers",
        "RetargetProfileLocalLimbsGlobalOthers",
        "RetargetProfileAbsoluteFingersGlobalOthers",
        "RetargetProfileAbsoluteLimbsGlobalOthers",
        "RetargetProfileAbsoluteFingersLocalLimbsGlobalOthers",
        "RetargetUtility",
    ]


def get_doc_path():
    return "doc_classes"
