lib_module_zip = library('module_zip',
    files(run_command(GLOB, '*.cpp', check: true).stdout().split('\n')),
    dependencies: [godotcore_dep, DEP_ZLIB]
)

DEP_MODULE_ZIP = declare_dependency(link_with: lib_module_zip)

MODULE_DEPENDENCIES += DEP_MODULE_ZIP
MODULES_ENABLED += 'zip'
