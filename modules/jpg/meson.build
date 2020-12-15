_module_jpg_srcs = files(run_command(GLOB, '*.cpp').stdout().split('\n'))

_lib_module_jpg = library('module_jpg', _module_jpg_srcs,
    dependencies: [DEP_GODOT_ALL, DEP_JPEG_COMPRESSOR]
)

DEP_MODULE_JPG = declare_dependency(link_with: _lib_module_jpg)

MODULE_DEPENDENCIES += DEP_MODULE_JPG
