thorvg_sources = []
subdir('src')

thorvg_lib = library('thorvg', thorvg_sources,
    include_directories: ['inc',
                          'src/common',
                          'src/renderer',
                          'src/renderer/sw_engine',
                          'src/loaders/svg',
                          'src/loaders/png',
                          'src/loaders/raw',
                          'src/loaders/jpg',
                          ],
    cpp_args: ['-DTVG_STATIC'],
    dependencies: [dependency('threads'), DEP_LIBPNG],
    )

thorvg_dep = declare_dependency(
    link_with: thorvg_lib,
    compile_args: ['-DTVG_STATIC'],
    include_directories: ['inc',
        'src/renderer',
        'src/common',
        'src/loaders/png/',
        'src/loaders/jpg',
        ],
)
