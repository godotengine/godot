module_gdnative_srcs += files([
    'multiplayer_peer_gdnative.cpp',
    'packet_peer_gdnative.cpp',
    'register_types.cpp',
    'stream_peer_gdnative.cpp',
    'webrtc_gdnative.cpp',
])

if 'webrtx' in MODULES_ENABLED
    module_gdnative_cpp_args += ['-DWEBRTC_GDNATIVE_ENABLED']
endif