def can_build(env, platform):
    return True


def configure(env):
    pass


def get_doc_classes():
    return [
        "SkeletonModification3DNBoneIK",
        "IKBone3D",
        "IKEffector3D",
        "IKBoneSegment",
        "IKEffectorTemplate",
        "IKKusudama",
        "IKRay3D",
        "IKNode3D",
        "IKLimitCone",
    ]


def get_doc_path():
    return "doc_classes"
