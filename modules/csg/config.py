def can_build(env: dict) -> bool:
    return True


def get_doc_classes() -> [str]:
    return [
        "CSGBox3D",
        "CSGCombiner3D",
        "CSGCylinder3D",
        "CSGMesh3D",
        "CSGPolygon3D",
        "CSGPrimitive3D",
        "CSGShape3D",
        "CSGSphere3D",
        "CSGTorus3D",
    ]


def get_doc_path() -> str:
    return "doc_classes"
