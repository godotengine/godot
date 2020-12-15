if get_option('tests')
    _tests_srcs = files(run_command(GLOB, '*.cpp').stdout().split('\n'))

    _tests_cpp_args = []
    if PLATFORM == 'windows'
        _tests_cpp_args += '-DDOCTEST_THREAD_LOCAL'
    endif

    #_tests_incdirs = include_directories('.')

    lib_tests = static_library('tests', _tests_srcs,
        cpp_args: _tests_cpp_args,
	include_directories: INCDIRS,
	dependencies: DEP_GODOT_ALL,
        build_by_default: false)
    # include_directories: _builtin_libogg_incdirs,

    DEP_TESTS = declare_dependency(link_with: lib_tests)
endif
