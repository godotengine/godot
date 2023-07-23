thorvg_sources = []
subdir('src')

thorvg_lib = library('thorvg', thorvg_sources,
    include_directories: ['inc',
                          'src/lib',
                          'src/loaders/svg',
                          'src/loaders/raw',
                          'src/lib/sw_engine',
                          ],
    cpp_args: ['-DTVG_STATIC'],
    dependencies: [dependency('threads'), DEP_LIBPNG],
    )

thorvg_dep = declare_dependency(
    link_with: thorvg_lib,
    compile_args: ['-DTVG_STATIC'],
    include_directories: ['inc', 'src/lib'],
)
