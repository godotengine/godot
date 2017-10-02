
import imp
import os
import sys
from shutil import copyfile

from SCons.Script import BoolVariable, Environment, Variables


monoreg = imp.load_source('mono_reg_utils', 'modules/mono/mono_reg_utils.py')


def find_file_in_dir(directory, files, prefix='', extension=''):
    if not extension.startswith('.'):
        extension = '.' + extension
    for curfile in files:
        if os.path.isfile(os.path.join(directory, prefix + curfile + extension)):
            return curfile

    return None


def can_build(platform):
    if platform in ["javascript"]:
        return False # Not yet supported
    return True


def is_enabled():
    # The module is disabled by default. Use module_mono_enabled=yes to enable it.
    return False


def configure(env):
    env.use_ptrcall = True

    envvars = Variables()
    envvars.Add(BoolVariable('mono_static', 'Statically link mono', False))
    envvars.Update(env)

    mono_static = env['mono_static']

    mono_lib_names = ['mono-2.0-sgen', 'monosgen-2.0']

    if env['platform'] == 'windows':
        if mono_static:
            raise RuntimeError('mono-static: Not supported on Windows')

        if env['bits'] == '32':
            if os.getenv('MONO32_PREFIX'):
                mono_root = os.getenv('MONO32_PREFIX')
            elif os.name == 'nt':
                mono_root = monoreg.find_mono_root_dir()
        else:
            if os.getenv('MONO64_PREFIX'):
                mono_root = os.getenv('MONO64_PREFIX')
            elif os.name == 'nt':
                mono_root = monoreg.find_mono_root_dir()

        if mono_root is None:
            raise RuntimeError('Mono installation directory not found')

        mono_lib_path = os.path.join(mono_root, 'lib')

        env.Append(LIBPATH=mono_lib_path)
        env.Append(CPPPATH=os.path.join(mono_root, 'include', 'mono-2.0'))

        mono_lib_name = find_file_in_dir(mono_lib_path, mono_lib_names, extension='.lib')

        if mono_lib_name is None:
            raise RuntimeError('Could not find mono library in: ' + mono_lib_path)

        if os.getenv('VCINSTALLDIR'):
            env.Append(LINKFLAGS=mono_lib_name + Environment()['LIBSUFFIX'])
        else:
            env.Append(LIBS=mono_lib_name)

        mono_bin_path = os.path.join(mono_root, 'bin')

        mono_dll_name = find_file_in_dir(mono_bin_path, mono_lib_names, extension='.dll')

        mono_dll_src = os.path.join(mono_bin_path, mono_dll_name + '.dll')
        mono_dll_dst = os.path.join('bin', mono_dll_name + '.dll')
        copy_mono_dll = True

        if not os.path.isdir('bin'):
            os.mkdir('bin')
        elif os.path.exists(mono_dll_dst):
            copy_mono_dll = False

        if copy_mono_dll:
            copyfile(mono_dll_src, mono_dll_dst)
    else:
        mono_root = None

        if env['bits'] == '32':
            if os.getenv('MONO32_PREFIX'):
                mono_root = os.getenv('MONO32_PREFIX')
        else:
            if os.getenv('MONO64_PREFIX'):
                mono_root = os.getenv('MONO64_PREFIX')

        if mono_root is not None:
            mono_lib_path = os.path.join(mono_root, 'lib')

            env.Append(LIBPATH=mono_lib_path)
            env.Append(CPPPATH=os.path.join(mono_root, 'include', 'mono-2.0'))

            mono_lib = find_file_in_dir(mono_lib_path, mono_lib_names, prefix='lib', extension='.a')

            if mono_lib is None:
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

            env.Append(LIBS=['m', 'rt', 'dl', 'pthread'])
        else:
            if mono_static:
                raise RuntimeError('mono-static: Not supported with pkg-config. Specify a mono prefix manually')

            env.ParseConfig('pkg-config mono-2 --cflags --libs')

        env.Append(LINKFLAGS='-rdynamic')


def get_doc_classes():
    return ["@C#", "CSharpScript", "GodotSharp"]


def get_doc_path():
    return "doc_classes"
