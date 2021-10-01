# the include directory
_etcpak_srcs = files(
    'Dither.cpp',
    'ProcessDxtc.cpp',
    'ProcessRGB.cpp',
    'Tables.cpp',
)

_etcpak_cpp_args = []
if PLATFORM == 'windows'
    _etcpak_cpp_args += ['-DNOMINMAX']
endif

_lib_etcpak = static_library('etcpak',
    _etcpak_srcs,
    cpp_args: _etcpak_cpp_args)
DEP_ETCPAK = declare_dependency(link_with: _lib_etcpak)
