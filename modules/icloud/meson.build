if host_machine.system() != 'darwin'
  subdir_done()
endif

_module_icloud_srcs = files(run_command(GLOB, '*.cpp', check: true).stdout().split('\n'))
_module_icloud_srcs = files(run_command(GLOB, '*.mm', check: true).stdout().split('\n'))

_module_icloud_cpp_args = ['-fmodules', '-fcxx-modules']

# static lib for ios
_lib_module_icloud = static_library('module_icloud', _module_icloud_srcs,
    dependencies: DEP_GODOT_ALL,
    cpp_args: _module_icloud_cpp_args
)

DEP_MODULE_ICLOUD = declare_dependency(link_with: _lib_module_icloud)

MODULE_DEPENDENCIES += DEP_MODULE_ICLOUD
