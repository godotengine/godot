def can_build(env, platform):
    return True


def configure(env):
    pass


def get_doc_classes():
    return [
        "CSGBox",
        "CSGCombiner",
        "CSGCylinder",
        "CSGMesh",
        "CSGPolygon",
        "CSGPrimitive",
        "CSGShape",
        "CSGSphere",
        "CSGTorus",
    ]


def get_doc_path():
    return "doc_classes"
