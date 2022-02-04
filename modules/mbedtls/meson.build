_module_mbedtls_srcs = files(run_command(GLOB, '*.cpp', check: true).stdout().split('\n'))
_module_mbedtls_deps = [DEP_GODOT_ALL, DEP_MBEDTLS]

if get_option('tests')
    _module_mbedtls_srcs += files(run_command(GLOB, 'tests/*.cpp', check: true).stdout().split('\n'))
    _module_mbedtls_deps += DEP_TESTS
endif

lib_module_mbedtls = library('module_mbedtls',
    _module_mbedtls_srcs,
    _certs_target,
    dependencies: [DEP_GODOT_ALL, DEP_MBEDTLS]
)

DEP_MODULE_MBEDTLS = declare_dependency(link_with: lib_module_mbedtls)

MODULE_DEPENDENCIES += DEP_MODULE_MBEDTLS
