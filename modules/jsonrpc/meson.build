_module_jsonrpc_srcs = files(run_command(GLOB, '*.cpp', check: true).stdout().split('\n'))
_lib_module_jsonrpc = library('module_jsonrpc', _module_jsonrpc_srcs, dependencies: [DEP_GODOT_ALL])

DEP_MODULE_JSONRPC = declare_dependency(link_with: _lib_module_jsonrpc)

MODULE_DEPENDENCIES += DEP_MODULE_JSONRPC
