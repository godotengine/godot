_module_gltf_srcs = files([
    'editor_scene_exporter_gltf_plugin.cpp',
    'editor_scene_importer_gltf.cpp',
    'gltf_accessor.cpp',
    'gltf_animation.cpp',
    'gltf_buffer_view.cpp',
    'gltf_camera.cpp',
    'gltf_document.cpp',
    'gltf_light.cpp',
    'gltf_mesh.cpp',
    'gltf_node.cpp',
    'gltf_skeleton.cpp',
    'gltf_skin.cpp',
    'gltf_spec_gloss.cpp',
    'gltf_state.cpp',
    'gltf_texture.cpp',
    'register_types.cpp',
])

lib_module_gltf = library('module_gltf', _module_gltf_srcs, dependencies: [DEP_GODOT_ALL])

DEP_MODULE_GLTF = declare_dependency(link_with: lib_module_gltf)

MODULE_DEPENDENCIES += DEP_MODULE_GLTF