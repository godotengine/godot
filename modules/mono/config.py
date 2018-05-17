
import imp
import os
import sys
import subprocess

from SCons.Script import BoolVariable, Dir, Environment, File, PathVariable, SCons, Variables


monoreg = imp.load_source('mono_reg_utils', 'modules/mono/mono_reg_utils.py')


def find_file_in_dir(directory, files, prefix='', extension=''):
    if not extension.startswith('.'):
        extension = '.' + extension
    for curfile in files:
        if os.path.isfile(os.path.join(directory, prefix + curfile + extension)):
            return curfile
    return ''


def can_build(platform):
    if platform in ["javascript"]:
        return False # Not yet supported
    return True


def is_enabled():
    # The module is disabled by default. Use module_mono_enabled=yes to enable it.
    return False


def copy_file(src_dir, dst_dir, name):
    from shutil import copyfile

    src_path = os.path.join(src_dir, name)
    dst_path = os.path.join(dst_dir, name)

    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)

    copyfile(src_path, dst_path)


def custom_path_is_dir_create(key, val, env):
    """Validator to check if Path is a directory, creating it if it does not exist.
       Similar to PathIsDirCreate, except it uses SCons.Script.Dir() and
       SCons.Script.File() in order to support the '#' top level directory token.
       """
    # Dir constructor will throw an error if the path points to a file
    fsDir = Dir(val)
    if not fsDir.exists:
        os.makedirs(fsDir.abspath)


def configure(env):
    env.use_ptrcall = True
    env.add_module_version_string("mono")

    envvars = Variables()
    envvars.Add(BoolVariable('mono_static', 'Statically link mono', False))
    envvars.Add(PathVariable('mono_assemblies_output_dir', 'Path to the assemblies output directory', '#bin', custom_path_is_dir_create))
    envvars.Update(env)

    bits = env['bits']

    mono_static = env['mono_static']
    assemblies_output_dir = Dir(env['mono_assemblies_output_dir']).abspath

    mono_lib_names = ['mono-2.0-sgen', 'monosgen-2.0']

    if env['platform'] == 'windows':
        if bits == '32':
            if os.getenv('MONO32_PREFIX'):
                mono_root = os.getenv('MONO32_PREFIX')
            elif os.name == 'nt':
                mono_root = monoreg.find_mono_root_dir(bits)
        else:
            if os.getenv('MONO64_PREFIX'):
                mono_root = os.getenv('MONO64_PREFIX')
            elif os.name == 'nt':
                mono_root = monoreg.find_mono_root_dir(bits)

        if not mono_root:
            raise RuntimeError('Mono installation directory not found')

        mono_lib_path = os.path.join(mono_root, 'lib')

        env.Append(LIBPATH=mono_lib_path)
        env.Append(CPPPATH=os.path.join(mono_root, 'include', 'mono-2.0'))

        if mono_static:
            lib_suffix = Environment()['LIBSUFFIX']
            mono_static_lib_name = 'libmono-static-sgen'

            if not os.path.isfile(os.path.join(mono_lib_path, mono_static_lib_name + lib_suffix)):
                raise RuntimeError('Could not find static mono library in: ' + mono_lib_path)

            if env.msvc:
                env.Append(LINKFLAGS=mono_static_lib_name + lib_suffix)

                env.Append(LINKFLAGS='Mincore' + lib_suffix)
                env.Append(LINKFLAGS='msvcrt' + lib_suffix)
                env.Append(LINKFLAGS='LIBCMT' + lib_suffix)
                env.Append(LINKFLAGS='Psapi' + lib_suffix)
            else:
                env.Append(LIBS=mono_static_lib_name)
        else:
            mono_lib_name = find_file_in_dir(mono_lib_path, mono_lib_names, extension='.lib')

            if not mono_lib_name:
                raise RuntimeError('Could not find mono library in: ' + mono_lib_path)

            if env.msvc:
                env.Append(LINKFLAGS=mono_lib_name + Environment()['LIBSUFFIX'])
            else:
                env.Append(LIBS=mono_lib_name)

            mono_bin_path = os.path.join(mono_root, 'bin')

            mono_dll_name = find_file_in_dir(mono_bin_path, mono_lib_names, extension='.dll')

            if not mono_dll_name:
                raise RuntimeError('Could not find mono shared library in: ' + mono_bin_path)

            copy_file(mono_bin_path, 'bin', mono_dll_name + '.dll')

        copy_file(os.path.join(mono_lib_path, 'mono', '4.5'), assemblies_output_dir, 'mscorlib.dll')
    else:
        sharedlib_ext = '.dylib' if sys.platform == 'darwin' else '.so'

        mono_root = ''
        mono_lib_path = ''

        if bits == '32':
            if os.getenv('MONO32_PREFIX'):
                mono_root = os.getenv('MONO32_PREFIX')
        else:
            if os.getenv('MONO64_PREFIX'):
                mono_root = os.getenv('MONO64_PREFIX')

        # We can't use pkg-config to link mono statically,
        # but we can still use it to find the mono root directory
        if not mono_root and mono_static:
            def pkgconfig_try_find_mono_root():
                tmpenv = Environment()
                tmpenv.AppendENVPath('PKG_CONFIG_PATH', os.getenv('PKG_CONFIG_PATH'))
                tmpenv.ParseConfig('pkg-config monosgen-2 --libs-only-L')
                for hint_dir in tmpenv['LIBPATH']:
                    name_found = find_file_in_dir(hint_dir, mono_lib_names, prefix='lib', extension=sharedlib_ext)
                    if name_found and os.path.isdir(os.path.join(hint_dir, '..', 'include', 'mono-2.0')):
                        return os.path.join(hint_dir, '..')
                return ''
            mono_root = pkgconfig_try_find_mono_root()
            if not mono_root:
                raise RuntimeError('Building with mono_static=yes, but failed to find the mono prefix with pkg-config. Specify one manually')

        if mono_root:
            mono_lib_path = os.path.join(mono_root, 'lib')

            env.Append(LIBPATH=mono_lib_path)
            env.Append(CPPPATH=os.path.join(mono_root, 'include', 'mono-2.0'))

            mono_lib = find_file_in_dir(mono_lib_path, mono_lib_names, prefix='lib', extension='.a')

            if not mono_lib:
                raise RuntimeError('Could not find mono library in: ' + mono_lib_path)

            env.Append(CPPFLAGS=['-D_REENTRANT'])

            if mono_static:
                mono_lib_file = os.path.join(mono_lib_path, 'lib' + mono_lib + '.a')

                if sys.platform == "darwin":
                    env.Append(LINKFLAGS=['-Wl,-force_load,' + mono_lib_file])
                elif sys.platform == "linux" or sys.platform == "linux2":
                    env.Append(LINKFLAGS=['-Wl,-whole-archive', mono_lib_file, '-Wl,-no-whole-archive'])
                else:
                    raise RuntimeError('mono-static: Not supported on this platform')
            else:
                env.Append(LIBS=[mono_lib])

            if sys.platform == "darwin":
                env.Append(LIBS=['iconv', 'pthread'])
            elif sys.platform == "linux" or sys.platform == "linux2":
                env.Append(LIBS=['m', 'rt', 'dl', 'pthread'])

            if not mono_static:
                mono_so_name = find_file_in_dir(mono_lib_path, mono_lib_names, prefix='lib', extension=sharedlib_ext)

                if not mono_so_name:
                    raise RuntimeError('Could not find mono shared library in: ' + mono_lib_path)

                copy_file(mono_lib_path, 'bin', 'lib' + mono_so_name + sharedlib_ext)

            copy_file(os.path.join(mono_lib_path, 'mono', '4.5'), assemblies_output_dir, 'mscorlib.dll')
        else:
            assert not mono_static

            env.ParseConfig('pkg-config monosgen-2 --cflags --libs')

            mono_lib_path = ''
            mono_so_name = ''
            mono_prefix = subprocess.check_output(["pkg-config", "mono-2", "--variable=prefix"]).decode("utf8").strip()

            tmpenv = Environment()
            tmpenv.AppendENVPath('PKG_CONFIG_PATH', os.getenv('PKG_CONFIG_PATH'))
            tmpenv.ParseConfig('pkg-config monosgen-2 --libs-only-L')

            for hint_dir in tmpenv['LIBPATH']:
                name_found = find_file_in_dir(hint_dir, mono_lib_names, prefix='lib', extension=sharedlib_ext)
                if name_found:
                    mono_lib_path = hint_dir
                    mono_so_name = name_found
                    break

            if not mono_so_name:
                raise RuntimeError('Could not find mono shared library in: ' + str(tmpenv['LIBPATH']))

            copy_file(mono_lib_path, 'bin', 'lib' + mono_so_name + sharedlib_ext)
            copy_file(os.path.join(mono_prefix, 'lib', 'mono', '4.5'), assemblies_output_dir, 'mscorlib.dll')

        env.Append(LINKFLAGS='-rdynamic')


def get_doc_classes():
    return [
        "@C#",
        "CSharpScript",
        "GodotSharp",
    ]


def get_doc_path():
    return "doc_classes"
