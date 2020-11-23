_basis_universal_incdirs = include_directories('.')

_basis_universal_srcs = files([
    'basisu_astc_decomp.cpp',
    'basisu_backend.cpp',
    'basisu_basis_file.cpp',
    'basisu_comp.cpp',
    'basisu_enc.cpp',
    'basisu_etc.cpp',
    'basisu_frontend.cpp',
    'basisu_global_selector_palette_helpers.cpp',
    'basisu_gpu_texture.cpp',
    'basisu_pvrtc1_4.cpp',
    'basisu_resample_filters.cpp',
    'basisu_resampler.cpp',
    'basisu_ssim.cpp',
    'lodepng.cpp',

    # this was guarded by tools enabled, but it seems to be incorrect
    'transcoder/basisu_transcoder.cpp'
])

_basis_universal_cpp_args = []

if meson.get_compiler('cpp').get_id() == 'msvc'
    _basis_universal_cpp_args += ['-DBASISU_NO_ITERATOR_DEBUG_LEVEL', '-DNOMINMAX']
elif meson.get_compiler('cpp').get_id() == 'gcc'
    _basis_universal_cpp_args += '-Wno-multichar'
endif

if get_option('debug')
    _basis_universal_cpp_args += ['-DBASISU_DEVEL_MESSAGES=1', '-DBASISD_ENABLE_DEBUG_FLAGS=1']
endif


_lib_basis_universal = static_library('builtin_basis_universal', _basis_universal_srcs,
    include_directories: _basis_universal_incdirs,
    cpp_args: _basis_universal_cpp_args,
    build_by_default: false
)

DEP_BASIS_UNIVERSAL = declare_dependency(link_with: _lib_basis_universal, include_directories: _basis_universal_incdirs)