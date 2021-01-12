_builtin_tinyexr_srcs = files([
    'tinyexr.cc'
])

_builtin_tinyexr_incdirs = include_directories('.')

_builtin_tinyexr_compile_args = ['-DTINYEXR_USE_THREAD']

lib_builtin_tinyexr = static_library('builtin_tinyexr', _builtin_tinyexr_srcs,
    include_directories: _builtin_tinyexr_incdirs,
    cpp_args: _builtin_tinyexr_compile_args,
    build_by_default: false
)

DEP_TINYEXR = declare_dependency(
    link_with: lib_builtin_tinyexr,
    include_directories: _builtin_tinyexr_incdirs,
    compile_args: _builtin_tinyexr_compile_args
)