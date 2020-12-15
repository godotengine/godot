_msdf_incdirs = include_directories('.')

_msdf_srcs = files(
    'core/contour-combiners.cpp',
    'core/Contour.cpp',
    'core/edge-coloring.cpp',
    'core/EdgeHolder.cpp',
    'core/edge-segments.cpp',
    'core/edge-selectors.cpp',
    'core/equation-solver.cpp',
    'core/msdf-error-correction.cpp',
    'core/MSDFErrorCorrection.cpp',
    'core/msdfgen.cpp',
    'core/Projection.cpp',
    'core/rasterization.cpp',
    'core/render-sdf.cpp',
    'core/save-bmp.cpp',
    'core/save-tiff.cpp',
    'core/Scanline.cpp',
    'core/sdf-error-estimation.cpp',
    'core/Shape.cpp',
    'core/shape-description.cpp',
    'core/SignedDistance.cpp',
    'core/Vector2.cpp',
)

_lib_msdf = static_library('msdf',
    _msdf_srcs,
    include_directories: _msdf_incdirs,
    c_args: _freetype_dep_defines + _freetype_c_defines,
    dependencies: [DEP_ZLIB, DEP_LIBPNG],
)

DEP_MSDF = declare_dependency(link_with: _lib_msdf,
         include_directories: _msdf_incdirs,
)

DEPENDENCIES += DEP_MSDF
