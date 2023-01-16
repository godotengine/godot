# the include directory
etcpak_srcs = files(
    'Dither.cpp',
    'ProcessDxtc.cpp',
    'ProcessRGB.cpp',
    'Tables.cpp',
)

etcpak_cpp_args = []
if PLATFORM == 'windows'
    etcpak_cpp_args += ['-DNOMINMAX']
endif

lib_etcpak = static_library('etcpak',
    etcpak_srcs,
    cpp_args: etcpak_cpp_args)

DEP_ETCPAK = declare_dependency(link_with: lib_etcpak,
  include_directories: '.'
)
