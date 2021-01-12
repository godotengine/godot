_module_mbedtls_srcs = files([
    'crypto_mbedtls.cpp',
    'dtls_server_mbedtls.cpp',
    'packet_peer_mbed_dtls.cpp',
    'register_types.cpp',
    'ssl_context_mbedtls.cpp',
    'stream_peer_mbedtls.cpp',
])

lib_module_mbedtls = library('module_mbedtls', _module_mbedtls_srcs, 
    dependencies: [DEP_GODOT_ALL, DEP_MBEDTLS]
)

DEP_MODULE_MBEDTLS = declare_dependency(link_with: lib_module_mbedtls)

MODULE_DEPENDENCIES += DEP_MODULE_MBEDTLS