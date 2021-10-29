_module_vorbis_sourcs = files(run_command(GLOB, '*.cpp').stdout().split('\n'))
_lib_module_vorbis = library('module_vorbis', _module_vorbis_sourcs, dependencies: [DEP_GODOT_ALL, DEP_LIBVORBIS, DEP_LIBOGG])

DEP_MODULE_VORBIS = declare_dependency(link_with: _lib_module_vorbis)

MODULE_DEPENDENCIES += DEP_MODULE_VORBIS
