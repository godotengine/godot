# the include directory
_etcpak_srcs = files([
    'Dither.cpp',
    'ProcessDxtc.cpp',
    'ProcessRGB.cpp',
    'Tables.cpp',
])

_lib_etcpak = static_library('etcpak', _etcpak_srcs, build_by_default: false)
DEP_ETCPAK = declare_dependency(link_with: _lib_etcpak)
