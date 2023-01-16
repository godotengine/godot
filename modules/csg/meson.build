_module_csg_srcs = files(run_command(GLOB, '*.cpp', check: true).stdout().split('\n'))
_lib_module_csg = library('module_csg',
                _module_csg_srcs,
                'editor/csg_gizmos.cpp',
                dependencies: [godotcore_dep, DEP_ZLIB])

DEP_MODULE_CSG = declare_dependency(link_with: _lib_module_csg)

MODULE_DEPENDENCIES += DEP_MODULE_CSG
MODULES_ENABLED += 'csg'
