_module_hdr_srcs = files(run_command(GLOB, '*.cpp', check: true).stdout().split('\n'))

_lib_module_hdr = library('module_hdr', _module_hdr_srcs,
    dependencies: [godotcore_dep]
)

DEP_MODULE_HDR = declare_dependency(link_with: _lib_module_hdr)

MODULE_DEPENDENCIES += DEP_MODULE_HDR
MODULES_ENABLED += 'hdr'
