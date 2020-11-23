_module_denoise_srcs = files([
    'denoise_wrapper.cpp',
    'lightmap_denoiser.cpp',
    'register_types.cpp'
])

_lib_module_denoise = library('module_denoise', _module_denoise_srcs,
    dependencies: [DEP_GODOT_ALL, DEP_OIDN]
)

DEP_MODULE_DENOISE = declare_dependency(link_with: _lib_module_denoise)

MODULE_DEPENDENCIES += DEP_MODULE_DENOISE