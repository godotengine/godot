def can_build(env, platform):
    return not env["disable_3d"]


def configure(env):
    pass


def get_doc_classes():
    return [
        "CSGManifoldBox3D",
        "CSGManifoldCombiner3D",
        "CSGManifoldCylinder3D",
        "CSGManifoldMesh3D",
        "CSGManifoldPolygon3D",
        "CSGManifoldPrimitive3D",
        "CSGManifoldShape3D",
        "CSGManifoldSphere3D",
        "CSGManifoldTorus3D",
    ]


def get_doc_path():
    return "doc_classes"
