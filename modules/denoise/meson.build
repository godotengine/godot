_module_denoise_srcs = files(run_command(GLOB, '*.cpp', check: true).stdout().split('\n'))
_lib_module_denoise = library('module_denoise', _module_denoise_srcs,
    dependencies: [DEP_GODOT_ALL, DEP_OIDN]
)

DEP_MODULE_DENOISE = declare_dependency(link_with: _lib_module_denoise)

MODULE_DEPENDENCIES += DEP_MODULE_DENOISE
MODULES_ENABLED += 'denoise'
