_module_stb_vorbis_srcs = files([
    # thirdparty single misc file
    '../../thirdparty/misc/stb_vorbis.c'
])
_module_stb_vorbis_srcs += files(run_command(GLOB, '*.cpp').stdout().split('\n'))

lib_module_stb_vorbis = library('module_stb_vorbis', _module_stb_vorbis_srcs,
    dependencies: [DEP_GODOT_ALL]
)

DEP_MODULE_STB_VORBIS = declare_dependency(link_with: lib_module_stb_vorbis)

MODULE_DEPENDENCIES += DEP_MODULE_STB_VORBIS
