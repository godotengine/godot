_module_basis_universal_srcs = files([
    'register_types.cpp',
    'texture_basisu.cpp'
])

if not DEP_BASIS_UNIVERSAL.found()
    error('module basis_universal requires DEP_BASIS_UNIVERSAL')
endif

_lib_module_basis_universal = library('module_basis_universal', _module_basis_universal_srcs,
    dependencies: [DEP_GODOT_ALL, DEP_BASIS_UNIVERSAL]
)

DEP_MODULE_BASIS_UNIVERSAL = declare_dependency(link_with: _lib_module_basis_universal)

MODULE_DEPENDENCIES += DEP_MODULE_BASIS_UNIVERSAL