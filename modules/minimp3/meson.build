_module_minimp3_srcs = files([
    'audio_stream_mp3.cpp',
    'register_types.cpp',
    'resource_importer_mp3.cpp'
])

lib_module_minimp3 = library('module_minimp3', _module_minimp3_srcs, dependencies: [DEP_GODOT_ALL, DEP_MINIMP3])

DEP_MODULE_MINIMP3 = declare_dependency(link_with: lib_module_minimp3)

MODULE_DEPENDENCIES += DEP_MODULE_MINIMP3