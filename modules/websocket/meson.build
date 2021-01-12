_module_websocket_srcs = files([
    'editor_debugger_server_websocket.cpp',
    'emws_client.cpp',
    'emws_peer.cpp',
    'emws_server.cpp',
    'register_types.cpp',
    'remote_debugger_peer_websocket.cpp',
    'websocket_client.cpp',
    'websocket_multiplayer_peer.cpp',
    'websocket_peer.cpp',
    'websocket_server.cpp',
    'wsl_client.cpp',
    'wsl_peer.cpp',
    'wsl_server.cpp',
])

_module_websocket_deps = [DEP_GODOT_ALL]

if PLATFORM == 'javascript'
    error('javascript not setup for websocket')
    # library_godot_websocket.js
else
    _module_websocket_deps += DEP_WSLAY
endif

lib_module_websocket = library('module_websocket', _module_websocket_srcs, dependencies: _module_websocket_deps)

DEP_MODULE_WEBSOCKET = declare_dependency(link_with: lib_module_websocket)

MODULE_DEPENDENCIES += DEP_MODULE_WEBSOCKET