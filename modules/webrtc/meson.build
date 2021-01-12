_module_webrtc_srcs = files([
    'register_types.cpp',
    'webrtc_data_channel.cpp',
    'webrtc_data_channel_gdnative.cpp',
    'webrtc_data_channel_js.cpp',
    'webrtc_multiplayer.cpp',
    'webrtc_peer_connection.cpp',
    'webrtc_peer_connection_gdnative.cpp',
    'webrtc_peer_connection_js.cpp',
])

_module_webrtc_compile_args = []
_module_webrtc_deps = [DEP_GODOT_ALL]

if 'gdnative' in MODULES_ENABLED
    _module_webrtc_compile_args += '-DWEBRTC_GDNATIVE_ENABLED'
    _module_webrtc_deps += DEP_MODULE_GDNATIVE
endif

if PLATFORM == 'javascript'
    error('javascript not setup for webrtc')
    # library_godot_webrtc.js
endif

lib_module_webrtc = library('module_webrtc', _module_webrtc_srcs, dependencies: _module_webrtc_deps, cpp_args: _module_webrtc_compile_args)

DEP_MODULE_WEBRTC = declare_dependency(link_with: lib_module_webrtc)

MODULE_DEPENDENCIES += DEP_MODULE_WEBRTC