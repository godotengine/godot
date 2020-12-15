if get_option('disable_3d')
    SOURCES += files([
        'node_3d.cpp',
        'skeleton_3d.cpp'
    ])
else
    SOURCES += files(run_command(GLOB, '*.cpp').stdout().split('\n'))
endif
