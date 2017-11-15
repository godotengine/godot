
import imp
import os
import sys

from SCons.Script import BoolVariable, Environment, Variables


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


def copy_file_no_replace(src_dir, dst_dir, name):
    from shutil import copyfile

    src_path = os.path.join(src_dir, name)
    dst_path = os.path.join(dst_dir, name)
    need_copy = True

    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)
    elif os.path.exists(dst_path):
        need_copy = False

    if need_copy:
        copyfile(src_path, dst_path)


def configure(env):
    env.use_ptrcall = True
    env.add_module_version_string("mono")

    envvars = Variables()
    envvars.Add(BoolVariable('mono_static', 'Statically link mono', False))
    envvars.Update(env)

    bits = env['bits']

    mono_static = env['mono_static']

    mono_lib_names = ['mono-2.0-sgen', 'monosgen-2.0']

    if env['platform'] == 'windows':
        if mono_static:
            raise RuntimeError('mono-static: Not supported on Windows')

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

        mono_lib_name = find_file_in_dir(mono_lib_path, mono_lib_names, extension='.lib')

        if not mono_lib_name:
            raise RuntimeError('Could not find mono library in: ' + mono_lib_path)

        if os.getenv('VCINSTALLDIR'):
            env.Append(LINKFLAGS=mono_lib_name + Environment()['LIBSUFFIX'])
        else:
            env.Append(LIBS=mono_lib_name)

        mono_bin_path = os.path.join(mono_root, 'bin')

        mono_dll_name = find_file_in_dir(mono_bin_path, mono_lib_names, extension='.dll')

        if not mono_dll_name:
            raise RuntimeError('Could not find mono shared library in: ' + mono_bin_path)

        copy_file_no_replace(mono_bin_path, 'bin', mono_dll_name + '.dll')
    else:
        sharedlib_ext = '.dylib' if sys.platform == 'darwin' else '.so'

        mono_root = ''

        if bits == '32':
            if os.getenv('MONO32_PREFIX'):
                mono_root = os.getenv('MONO32_PREFIX')
        else:
            if os.getenv('MONO64_PREFIX'):
                mono_root = os.getenv('MONO64_PREFIX')

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

                copy_file_no_replace(mono_lib_path, 'bin', 'lib' + mono_so_name + sharedlib_ext)
        else:
            if mono_static:
                raise RuntimeError('mono-static: Not supported with pkg-config. Specify a mono prefix manually')

            env.ParseConfig('pkg-config monosgen-2 --cflags --libs')

            mono_lib_path = ''
            mono_so_name = ''

            tmpenv = Environment()
            tmpenv.ParseConfig('pkg-config monosgen-2 --libs-only-L')

            for hint_dir in tmpenv['LIBPATH']:
                name_found = find_file_in_dir(hint_dir, mono_lib_names, prefix='lib', extension=sharedlib_ext)
                if name_found:
                    mono_lib_path = hint_dir
                    mono_so_name = name_found
                    break

            if not mono_so_name:
                raise RuntimeError('Could not find mono shared library in: ' + str(tmpenv['LIBPATH']))

            copy_file_no_replace(mono_lib_path, 'bin', 'lib' + mono_so_name + sharedlib_ext)

        env.Append(LINKFLAGS='-rdynamic')


def get_doc_classes():
    return [
        "@C#",
        "CSharpScript",
        "GodotSharp",
    ]


def get_doc_path():
    return "doc_classes"
