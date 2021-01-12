_modules_glslang_srcs = files([
    'register_types.cpp'
])

_lib_modules_glslang = library('modules_glslang', _modules_glslang_srcs, dependencies: [DEP_GODOT_ALL, DEP_GLSLANG], build_by_default:true)

DEP_MODULE_GLSLANG = declare_dependency(link_with: _lib_modules_glslang)

MODULE_DEPENDENCIES += [DEP_MODULE_GLSLANG]