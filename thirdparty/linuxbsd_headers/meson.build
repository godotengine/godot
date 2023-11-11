if PLATFORM == 'linuxbsd'
    DEP_LINUXBSD_HEADERS = []
    if get_option('use_sowrap')
        DEP_LINUXBSD_HEADERS = declare_dependency(
            include_directories: include_directories('.')
        )
    endif
endif
