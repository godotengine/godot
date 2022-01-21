thorvg_sources = []
subdir('src')

thorvg_lib = library('thorvg', thorvg_sources,
    include_directories: ['inc',
                          'src/lib',
                          'src/loaders/tvg',
                          'src/loaders/svg',
                          'src/loaders/png',
                          'src/loaders/jpg',
                          'src/loaders/raw',
                          'src/savers/tvg',
                          'src/lib/sw_engine'],
    dependencies: dependency('threads'),
    )

thorvg_dep = declare_dependency(
    link_with: thorvg_lib,
    include_directories: 'inc',
)
