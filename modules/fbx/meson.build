_module_fbx_srcs = files([
    'data/fbx_bone.cpp',
    'data/fbx_material.cpp',
    'data/fbx_mesh_data.cpp',
    'data/fbx_skeleton.cpp',
    'data/pivot_transform.cpp',
    'editor_scene_importer_fbx.cpp',
    'fbx_parser/FBXAnimation.cpp',
    'fbx_parser/FBXBinaryTokenizer.cpp',
    'fbx_parser/FBXDeformer.cpp',
    'fbx_parser/FBXDocument.cpp',
    'fbx_parser/FBXDocumentUtil.cpp',
    'fbx_parser/FBXMaterial.cpp',
    'fbx_parser/FBXMeshGeometry.cpp',
    'fbx_parser/FBXModel.cpp',
    'fbx_parser/FBXNodeAttribute.cpp',
    'fbx_parser/FBXParser.cpp',
    'fbx_parser/FBXPose.cpp',
    'fbx_parser/FBXProperties.cpp',
    'fbx_parser/FBXTokenizer.cpp',
    'fbx_parser/FBXUtil.cpp',
    'register_types.cpp',
    'tools/import_utils.cpp',
    'tools/validation_tools.cpp',
])

# add our current directory into the include path
_module_fbx_incdirs = include_directories('.')

lib_module_fbx = library('module_fbx', _module_fbx_srcs, include_directories: _module_fbx_incdirs, dependencies: [DEP_GODOT_ALL])

DEP_MODULE_FBX = declare_dependency(link_with: lib_module_fbx)

MODULE_DEPENDENCIES += DEP_MODULE_FBX