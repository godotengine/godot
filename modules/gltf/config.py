def can_build(env, platform):
    return not env["disable_3d"]


def configure(env):
    pass


def get_doc_classes():
    return [
        "EditorSceneFormatImporterBlend",
        "EditorSceneFormatImporterGLTF",
        "GLTFAccessor",
        "GLTFAnimation",
        "GLTFBufferView",
        "GLTFCamera",
        "GLTFDocument",
        "GLTFDocumentExtension",
        "GLTFDocumentExtensionConvertImporterMesh",
        "GLTFLight",
        "GLTFMesh",
        "GLTFNode",
        "GLTFPhysicsBody",
        "GLTFPhysicsShape",
        "GLTFSkeleton",
        "GLTFSkin",
        "GLTFSpecGloss",
        "GLTFState",
        "GLTFTexture",
        "GLTFTextureSampler",
        # FIXME: Remove once those classes are decoupled from the gltf module.
        "ModelDocument3D",
        "ModelState3D",
    ]


def get_doc_path():
    return "doc_classes"
