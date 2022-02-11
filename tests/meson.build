if get_option('tests')
    _tests_srcs = files(run_command(GLOB, '*.cpp', check: true).stdout().split('\n'))

    _tests_cpp_args = []
    if PLATFORM == 'windows'
        _tests_cpp_args += '/bigobj'
    endif

    #_tests_incdirs = include_directories('.')

    lib_tests = static_library('tests', _tests_srcs, #builtin_fonts_gen,
                               cpp_args: _tests_cpp_args,
                               include_directories: INCDIRS,
                               dependencies: godotcore_dep)
    # include_directories: _builtin_libogg_incdirs,

    DEP_TESTS = declare_dependency(link_with: lib_tests)
endif
