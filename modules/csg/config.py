def can_build(env, platform):
    return not env["disable_3d"]


def configure(env):
    pass


def get_doc_classes():
    return [
        "CSGCombiner3D",
        "CSGMesh3D",
        "CSGPolygon3D",
        "CSGPrimitive3D",
        "CSGShape3D",
    ]


def get_doc_path():
    return "doc_classes"
