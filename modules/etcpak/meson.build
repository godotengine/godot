_module_etcpak_srcs = files(run_command(GLOB, '*.cpp', check: true).stdout().split('\n'))
_module_etcpak_deps = [DEP_ETCPAK]

lib_module_etcpak = library('module_etcpak',
    _module_etcpak_srcs,
    _certs_target,
    dependencies: [godotcore_dep, DEP_ETCPAK]
)

DEP_MODULE_ETCPAK = declare_dependency(link_with: lib_module_etcpak)

MODULE_DEPENDENCIES += DEP_MODULE_ETCPAK
MODULES_ENABLED += 'etcpak'
