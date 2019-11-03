import os


def is_active():
    return True


def get_name():
    return 'JavaScript'


def can_build():
    return 'EM_CONFIG' in os.environ or os.path.exists(os.path.expanduser('~/.emscripten'))


def get_opts():
    from SCons.Variables import BoolVariable
    return [
        # eval() can be a security concern, so it can be disabled.
        BoolVariable('javascript_eval', 'Enable JavaScript eval interface', True),
    ]


def get_flags():
    return [
        ('tools', False),
        # Disabling the mbedtls module reduces file size.
        # The module has little use due to the limited networking functionality
        # in this platform. For the available networking methods, the browser
        # manages TLS.
        ('module_mbedtls_enabled', False),
    ]


def configure(env):

    ## Build type

    if env['target'] != 'debug':
        # Use -Os to prioritize optimizing for reduced file size. This is
        # particularly valuable for the web platform because it directly
        # decreases download time.
        # -Os reduces file size by around 5 MiB over -O3. -Oz only saves about
        # 100 KiB over -Os, which does not justify the negative impact on
        # run-time performance.
        env.Append(CCFLAGS=['-Os'])
        env.Append(LINKFLAGS=['-Os'])
        if env['target'] == 'release_debug':
            env.Append(CPPDEFINES=['DEBUG_ENABLED'])
            # Retain function names for backtraces at the cost of file size.
            env.Append(LINKFLAGS=['--profiling-funcs'])
    else:
        env.Append(CPPDEFINES=['DEBUG_ENABLED'])
        env.Append(CCFLAGS=['-O1', '-g'])
        env.Append(LINKFLAGS=['-O1', '-g'])
        env.Append(LINKFLAGS=['-s', 'ASSERTIONS=1'])

    ## Compiler configuration

    env['ENV'] = os.environ

    em_config_file = os.getenv('EM_CONFIG') or os.path.expanduser('~/.emscripten')
    if not os.path.exists(em_config_file):
        raise RuntimeError("Emscripten configuration file '%s' does not exist" % em_config_file)
    with open(em_config_file) as f:
        em_config = {}
        try:
            # Emscripten configuration file is a Python file with simple assignments.
            exec(f.read(), em_config)
        except StandardError as e:
            raise RuntimeError("Emscripten configuration file '%s' is invalid:\n%s" % (em_config_file, e))
    if 'BINARYEN_ROOT' in em_config and os.path.isdir(os.path.join(em_config.get('BINARYEN_ROOT'), 'emscripten')):
        # New style, emscripten path as a subfolder of BINARYEN_ROOT
        env.PrependENVPath('PATH', os.path.join(em_config.get('BINARYEN_ROOT'), 'emscripten'))
    elif 'EMSCRIPTEN_ROOT' in em_config:
        # Old style (but can be there as a result from previous activation, so do last)
        env.PrependENVPath('PATH', em_config.get('EMSCRIPTEN_ROOT'))
    else:
        raise RuntimeError("'BINARYEN_ROOT' or 'EMSCRIPTEN_ROOT' missing in Emscripten configuration file '%s'" % em_config_file)

    env['CC'] = 'emcc'
    env['CXX'] = 'em++'
    env['LINK'] = 'emcc'

    # Emscripten's ar has issues with duplicate file names, so use cc.
    env['AR'] = 'emcc'
    env['ARFLAGS'] = '-o'
    # emranlib is a noop, so it's safe to use with AR=emcc.
    env['RANLIB'] = 'emranlib'

    # Use TempFileMunge since some AR invocations are too long for cmd.exe.
    # Use POSIX-style paths, required with TempFileMunge.
    env['ARCOM_POSIX'] = env['ARCOM'].replace(
        '$TARGET', '$TARGET.posix').replace(
        '$SOURCES', '$SOURCES.posix')
    env['ARCOM'] = '${TEMPFILE(ARCOM_POSIX)}'

    # All intermediate files are just LLVM bitcode.
    env['OBJPREFIX'] = ''
    env['OBJSUFFIX'] = '.bc'
    env['PROGPREFIX'] = ''
    # Program() output consists of multiple files, so specify suffixes manually at builder.
    env['PROGSUFFIX'] = ''
    env['LIBPREFIX'] = 'lib'
    env['LIBSUFFIX'] = '.bc'
    env['LIBPREFIXES'] = ['$LIBPREFIX']
    env['LIBSUFFIXES'] = ['$LIBSUFFIX']

    ## Compile flags

    env.Prepend(CPPPATH=['#platform/javascript'])
    env.Append(CPPDEFINES=['JAVASCRIPT_ENABLED', 'UNIX_ENABLED'])

    # No multi-threading (SharedArrayBuffer) available yet,
    # once feasible also consider memory buffer size issues.
    env.Append(CPPDEFINES=['NO_THREADS'])

    # Disable exceptions and rtti on non-tools (template) builds
    if not env['tools']:
        # These flags help keep the file size down.
        env.Append(CCFLAGS=['-fno-exceptions', '-fno-rtti'])
        # Don't use dynamic_cast, necessary with no-rtti.
        env.Append(CPPDEFINES=['NO_SAFE_CAST'])

    if env['javascript_eval']:
        env.Append(CPPDEFINES=['JAVASCRIPT_EVAL_ENABLED'])

    ## Link flags

    env.Append(LINKFLAGS=['-s', 'BINARYEN=1'])

    # Allow increasing memory buffer size during runtime. This is efficient
    # when using WebAssembly (in comparison to asm.js) and works well for
    # us since we don't know requirements at compile-time.
    env.Append(LINKFLAGS=['-s', 'ALLOW_MEMORY_GROWTH=1'])

    # This setting just makes WebGL 2 APIs available, it does NOT disable WebGL 1.
    env.Append(LINKFLAGS=['-s', 'USE_WEBGL2=1'])

    env.Append(LINKFLAGS=['-s', 'INVOKE_RUN=0'])

    # TODO: Reevaluate usage of this setting now that engine.js manages engine runtime.
    env.Append(LINKFLAGS=['-s', 'NO_EXIT_RUNTIME=1'])

    #adding flag due to issue with emscripten 1.38.41 callMain method https://github.com/emscripten-core/emscripten/blob/incoming/ChangeLog.md#v13841-08072019
    env.Append(LINKFLAGS=['-s', 'EXTRA_EXPORTED_RUNTIME_METHODS=["callMain"]'])
