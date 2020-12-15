_basis_universal_incdirs = include_directories('.')

_basis_universal_srcs = files([
'encoder/basisu_astc_decomp.cpp',
'encoder/basisu_backend.cpp',
'encoder/basisu_basis_file.cpp',
'encoder/basisu_bc7enc.cpp',
'encoder/basisu_comp.cpp',
'encoder/basisu_enc.cpp',
'encoder/basisu_etc.cpp',
'encoder/basisu_frontend.cpp',
'encoder/basisu_global_selector_palette_helpers.cpp',
'encoder/basisu_gpu_texture.cpp',
'encoder/basisu_kernels_sse.cpp',
'encoder/basisu_pvrtc1_4.cpp',
'encoder/basisu_resample_filters.cpp',
'encoder/basisu_resampler.cpp',
'encoder/basisu_ssim.cpp',
'encoder/basisu_uastc_enc.cpp',
'encoder/jpgd.cpp',
'encoder/lodepng.cpp',

    'encoder/apg_bmp.c',

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
