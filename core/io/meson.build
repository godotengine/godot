# generated certs

_certs_target = custom_target(
    'certs',
    output: ['certs_compressed.gen.h'],
    input: [meson.project_source_root() + '/thirdparty/certs/ca-certificates.crt'],
    command: [SCRIPT_CERTS, '@INPUT@', '@OUTPUT@', get_option('system_certs_path'), get_option('builtin_certs') ? 'true' : 'false'],
    build_by_default: true
)

SOURCES += [_certs_target]

SOURCES += files([
    'compression.cpp',
    'config_file.cpp',
    'dtls_server.cpp',
    'file_access_compressed.cpp',
    'file_access_encrypted.cpp',
    'file_access_memory.cpp',
    'file_access_network.cpp',
    'file_access_pack.cpp',
    'file_access_zip.cpp',
    'http_client.cpp',
    'image.cpp',
    'image_loader.cpp',
    'ip.cpp',
    'ip_address.cpp',
    'json.cpp',
    'logger.cpp',
    'marshalls.cpp',
    'multiplayer_api.cpp',
    'networked_multiplayer_peer.cpp',
    'net_socket.cpp',
    'packed_data_container.cpp',
    'packet_peer.cpp',
    'packet_peer_dtls.cpp',
    'packet_peer_udp.cpp',
    'pck_packer.cpp',
    'resource.cpp',
    'resource_format_binary.cpp',
    'resource_importer.cpp',
    'resource_loader.cpp',
    'resource_saver.cpp',
    'stream_peer.cpp',
    'stream_peer_ssl.cpp',
    'stream_peer_tcp.cpp',
    'tcp_server.cpp',
    'translation_loader_po.cpp',
    'udp_server.cpp',
    'xml_parser.cpp',
    'zip_io.cpp',
])