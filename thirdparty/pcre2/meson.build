if get_option('builtin_pcre2')
    _pcre2_incdirs = include_directories('src')

    _pcre2_srcs = files([
        'src/pcre2_auto_possess.c',
        'src/pcre2_chartables.c',
        'src/pcre2_compile.c',
        'src/pcre2_config.c',
        'src/pcre2_context.c',
        'src/pcre2_convert.c',
        'src/pcre2_dfa_match.c',
        'src/pcre2_error.c',
        'src/pcre2_extuni.c',
        'src/pcre2_find_bracket.c',
        'src/pcre2_jit_compile.c',
        # 'src/pcre2_jit_match.c', 'src/pcre2_jit_misc.c', # these files are included in src/pcre2_jit_compile.c.
        'src/pcre2_maketables.c',
        'src/pcre2_match.c',
        'src/pcre2_match_data.c',
        'src/pcre2_newline.c',
        'src/pcre2_ord2utf.c',
        'src/pcre2_pattern_info.c',
        'src/pcre2_script_run.c',
        'src/pcre2_serialize.c',
        'src/pcre2_string_utils.c',
        'src/pcre2_study.c',
        'src/pcre2_substitute.c',
        'src/pcre2_substring.c',
        'src/pcre2_tables.c',
        'src/pcre2_ucd.c',
        'src/pcre2_valid_utf.c',
        'src/pcre2_xclass.c',
    ])

    _pcre2_c_args = [ '-DHAVE_CONFIG_H', '-DSUPPORT_UNICODE']
    _pcre2_dep_args = ['-DPCRE2_STATIC','-DPCRE2_CODE_UNIT_WIDTH=32']

    if get_option('builtin_pcre2_with_jit')
        _pcre2_c_args += ['-DSUPPORT_JIT']
    endif

    _lib_pcre2 = static_library('libpcre2', _pcre2_srcs, include_directories: _pcre2_incdirs, c_args: _pcre2_c_args + _pcre2_dep_args, build_by_default: false)

    DEP_PCRE2 = declare_dependency(link_with: _lib_pcre2, include_directories: _pcre2_incdirs, compile_args: _pcre2_dep_args)
else
    DEP_PCRE2 = dependency('libpcre2-32')
endif