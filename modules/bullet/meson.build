# Currently always disabled.
MODULES_DISABLED += 'bullet'
subdir_done()

_module_bullet_srcs = files(run_command(GLOB, '*.cpp', check: true).stdout().split('\n'))
if not DEP_BULLET.found()
    error('module bullet requires bullet as a dependency')
endif

_lib_module_bullet = library('module_bullet', _module_bullet_srcs, dependencies: [DEP_GODOT_ALL, DEP_BULLET])

DEP_MODULE_BULLET = declare_dependency(link_with: _lib_module_bullet)

MODULE_DEPENDENCIES += [DEP_MODULE_BULLET]
MODULES_ENABLED += 'bullet'
