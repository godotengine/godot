_module_jsonrpc_srcs = files([
    'jsonrpc.cpp',
    'register_types.cpp'
])

_lib_module_jsonrpc = library('module_jsonrpc', _module_jsonrpc_srcs, dependencies: [DEP_GODOT_ALL])

DEP_MODULE_JSONRPC = declare_dependency(link_with: _lib_module_jsonrpc)

MODULE_DEPENDENCIES += DEP_MODULE_JSONRPC