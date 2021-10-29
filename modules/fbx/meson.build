_module_fbx_srcs = files(run_command(GLOB, '*.cpp').stdout().split('\n'))
_module_fbx_srcs += files(run_command(GLOB, 'data/*.cpp').stdout().split('\n'))
_module_fbx_srcs += files(run_command(GLOB, 'fbx_parser/*.cpp').stdout().split('\n'))
_module_fbx_srcs += files(run_command(GLOB, 'tools/*.cpp').stdout().split('\n'))

# add our current directory into the include path
_module_fbx_incdirs = include_directories('.')

lib_module_fbx = library('module_fbx',
                         _module_fbx_srcs,
                         include_directories: _module_fbx_incdirs,
                         dependencies: [DEP_GODOT_ALL, DEP_ZLIB])

DEP_MODULE_FBX = declare_dependency(link_with: lib_module_fbx)

MODULE_DEPENDENCIES += DEP_MODULE_FBX
