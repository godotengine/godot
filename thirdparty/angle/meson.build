if PLATFORM == 'windows'
    DEP_ANGLE = []
    if get_option('opengl3')
        DEP_ANGLE = declare_dependency(
            include_directories: include_directories('include')
        )
    endif
endif
