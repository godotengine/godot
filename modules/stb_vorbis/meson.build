_module_stb_vorbis_srcs = files([
    # thirdparty single misc file
    '../../thirdparty/misc/stb_vorbis.c',

    # module files
    'audio_stream_ogg_vorbis.cpp',
    'register_types.cpp',
    'resource_importer_ogg_vorbis.cpp'
])

lib_module_stb_vorbis = library('module_stb_vorbis', _module_stb_vorbis_srcs,
    dependencies: [DEP_GODOT_ALL]
)

DEP_MODULE_STB_VORBIS = declare_dependency(link_with: lib_module_stb_vorbis)

MODULE_DEPENDENCIES += DEP_MODULE_STB_VORBIS