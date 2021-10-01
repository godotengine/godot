# the include directory
_etcpak_srcs = files(
    'Dither.cpp',
    'ProcessDxtc.cpp',
    'ProcessRGB.cpp',
    'Tables.cpp',
)

_lib_etcpak = static_library('etcpak',
    _etcpak_srcs,
    cpp_args: ['-DNOMINMAX'])
DEP_ETCPAK = declare_dependency(link_with: _lib_etcpak)
