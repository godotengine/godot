_spirv_reflect_incdirs = include_directories('include')

_spirv_reflect_sources = files([
    'spirv_reflect.c'
])

_lib_spirv_reflect = static_library('spirv-reflect', _spirv_reflect_sources, include_directories: _spirv_reflect_incdirs, build_by_default: false)

DEP_SPIRV_REFLECT = declare_dependency(link_with: _lib_spirv_reflect, include_directories: _spirv_reflect_incdirs)