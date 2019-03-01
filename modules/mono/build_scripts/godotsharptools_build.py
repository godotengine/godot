# Build GodotSharpTools solution


import os

from SCons.Script import Builder, Dir


def find_nuget_unix():
    import os

    if 'NUGET_PATH' in os.environ:
        hint_path = os.environ['NUGET_PATH']
        if os.path.isfile(hint_path) and os.access(hint_path, os.X_OK):
            return hint_path
        hint_path = os.path.join(hint_path, 'nuget')
        if os.path.isfile(hint_path) and os.access(hint_path, os.X_OK):
            return hint_path

    import os.path
    import sys

    hint_dirs = ['/opt/novell/mono/bin']
    if sys.platform == 'darwin':
        hint_dirs = ['/Library/Frameworks/Mono.framework/Versions/Current/bin', '/usr/local/var/homebrew/linked/mono/bin'] + hint_dirs

    for hint_dir in hint_dirs:
        hint_path = os.path.join(hint_dir, 'nuget')
        if os.path.isfile(hint_path):
            return hint_path
        elif os.path.isfile(hint_path + '.exe'):
            return hint_path + '.exe'

    for hint_dir in os.environ['PATH'].split(os.pathsep):
        hint_dir = hint_dir.strip('"')
        hint_path = os.path.join(hint_dir, 'nuget')
        if os.path.isfile(hint_path) and os.access(hint_path, os.X_OK):
            return hint_path
        if os.path.isfile(hint_path + '.exe') and os.access(hint_path + '.exe', os.X_OK):
            return hint_path + '.exe'

    return None


def find_nuget_windows(env):
    import os

    if 'NUGET_PATH' in os.environ:
        hint_path = os.environ['NUGET_PATH']
        if os.path.isfile(hint_path) and os.access(hint_path, os.X_OK):
            return hint_path
        hint_path = os.path.join(hint_path, 'nuget.exe')
        if os.path.isfile(hint_path) and os.access(hint_path, os.X_OK):
            return hint_path

    from . import mono_reg_utils as monoreg

    mono_root = ''
    bits = env['bits']

    if bits == '32':
        if os.getenv('MONO32_PREFIX'):
            mono_root = os.getenv('MONO32_PREFIX')
        else:
            mono_root = monoreg.find_mono_root_dir(bits)
    else:
        if os.getenv('MONO64_PREFIX'):
            mono_root = os.getenv('MONO64_PREFIX')
        else:
            mono_root = monoreg.find_mono_root_dir(bits)

    if mono_root:
        mono_bin_dir = os.path.join(mono_root, 'bin')
        nuget_mono = os.path.join(mono_bin_dir, 'nuget.bat')

        if os.path.isfile(nuget_mono):
            return nuget_mono

    # Standalone NuGet

    for hint_dir in os.environ['PATH'].split(os.pathsep):
        hint_dir = hint_dir.strip('"')
        hint_path = os.path.join(hint_dir, 'nuget.exe')
        if os.path.isfile(hint_path) and os.access(hint_path, os.X_OK):
            return hint_path

    return None


def find_msbuild_unix(filename):
    import os.path
    import sys

    hint_dirs = ['/opt/novell/mono/bin']
    if sys.platform == 'darwin':
        hint_dirs = ['/Library/Frameworks/Mono.framework/Versions/Current/bin', '/usr/local/var/homebrew/linked/mono/bin'] + hint_dirs

    for hint_dir in hint_dirs:
        hint_path = os.path.join(hint_dir, filename)
        if os.path.isfile(hint_path):
            return hint_path
        elif os.path.isfile(hint_path + '.exe'):
            return hint_path + '.exe'

    for hint_dir in os.environ['PATH'].split(os.pathsep):
        hint_dir = hint_dir.strip('"')
        hint_path = os.path.join(hint_dir, filename)
        if os.path.isfile(hint_path) and os.access(hint_path, os.X_OK):
            return hint_path
        if os.path.isfile(hint_path + '.exe') and os.access(hint_path + '.exe', os.X_OK):
            return hint_path + '.exe'

    return None


def find_msbuild_windows(env):
    from . import mono_reg_utils as monoreg

    mono_root = ''
    bits = env['bits']

    if bits == '32':
        if os.getenv('MONO32_PREFIX'):
            mono_root = os.getenv('MONO32_PREFIX')
        else:
            mono_root = monoreg.find_mono_root_dir(bits)
    else:
        if os.getenv('MONO64_PREFIX'):
            mono_root = os.getenv('MONO64_PREFIX')
        else:
            mono_root = monoreg.find_mono_root_dir(bits)

    if not mono_root:
        raise RuntimeError('Cannot find mono root directory')

    framework_path = os.path.join(mono_root, 'lib', 'mono', '4.5')
    mono_bin_dir = os.path.join(mono_root, 'bin')
    msbuild_mono = os.path.join(mono_bin_dir, 'msbuild.bat')

    if os.path.isfile(msbuild_mono):
        # The (Csc/Vbc/Fsc)ToolExe environment variables are required when
        # building with Mono's MSBuild. They must point to the batch files
        # in Mono's bin directory to make sure they are executed with Mono.
        mono_msbuild_env = {
            'CscToolExe': os.path.join(mono_bin_dir, 'csc.bat'),
            'VbcToolExe': os.path.join(mono_bin_dir, 'vbc.bat'),
            'FscToolExe': os.path.join(mono_bin_dir, 'fsharpc.bat')
        }
        return (msbuild_mono, framework_path, mono_msbuild_env)

    msbuild_tools_path = monoreg.find_msbuild_tools_path_reg()

    if msbuild_tools_path:
        return (os.path.join(msbuild_tools_path, 'MSBuild.exe'), framework_path, {})

    return None


def mono_build_solution(source, target, env):
    import subprocess
    from shutil import copyfile

    sln_path = os.path.abspath(str(source[0]))
    target_path = os.path.abspath(str(target[0]))

    framework_path = ''
    msbuild_env = os.environ.copy()

    # Needed when running from Developer Command Prompt for VS
    if 'PLATFORM' in msbuild_env:
        del msbuild_env['PLATFORM']

    # Find MSBuild
    if os.name == 'nt':
        msbuild_info = find_msbuild_windows(env)
        if msbuild_info is None:
            raise RuntimeError('Cannot find MSBuild executable')
        msbuild_path = msbuild_info[0]
        framework_path = msbuild_info[1]
        msbuild_env.update(msbuild_info[2])
    else:
        msbuild_path = find_msbuild_unix('msbuild')
        if msbuild_path is None:
            xbuild_fallback = env['xbuild_fallback']

            if xbuild_fallback and os.name == 'nt':
                print('Option \'xbuild_fallback\' not supported on Windows')
                xbuild_fallback = False

            if xbuild_fallback:
                print('Cannot find MSBuild executable, trying with xbuild')
                print('Warning: xbuild is deprecated')

                msbuild_path = find_msbuild_unix('xbuild')

                if msbuild_path is None:
                    raise RuntimeError('Cannot find xbuild executable')
            else:
                raise RuntimeError('Cannot find MSBuild executable')

    print('MSBuild path: ' + msbuild_path)

    # Find NuGet
    nuget_path = find_nuget_windows(env) if os.name == 'nt' else find_nuget_unix()
    if nuget_path is None:
        raise RuntimeError('Cannot find NuGet executable')

    print('NuGet path: ' + nuget_path)

    # Do NuGet restore

    try:
        subprocess.check_call([nuget_path, 'restore', sln_path])
    except subprocess.CalledProcessError:
        raise RuntimeError('GodotSharpTools: NuGet restore failed')

    # Build solution

    build_config = 'Release'

    msbuild_args = [
        msbuild_path,
        sln_path,
        '/p:Configuration=' + build_config,
    ]

    if framework_path:
        msbuild_args += ['/p:FrameworkPathOverride=' + framework_path]

    try:
        subprocess.check_call(msbuild_args, env=msbuild_env)
    except subprocess.CalledProcessError:
        raise RuntimeError('GodotSharpTools: Build failed')

    # Copy files

    src_dir = os.path.abspath(os.path.join(sln_path, os.pardir, 'bin', build_config))
    dst_dir = os.path.abspath(os.path.join(target_path, os.pardir))
    asm_file = 'GodotSharpTools.dll'

    if not os.path.isdir(dst_dir):
        if os.path.exists(dst_dir):
            raise RuntimeError('Target directory is a file')
        os.makedirs(dst_dir)

    copyfile(os.path.join(src_dir, asm_file), os.path.join(dst_dir, asm_file))

    # Dependencies
    copyfile(os.path.join(src_dir, "DotNet.Glob.dll"), os.path.join(dst_dir, "DotNet.Glob.dll"))

def build(env_mono):
    if not env_mono['tools']:
        return

    output_dir = Dir('#bin').abspath
    editor_tools_dir = os.path.join(output_dir, 'GodotSharp', 'Tools')

    mono_sln_builder = Builder(action=mono_build_solution)
    env_mono.Append(BUILDERS={'MonoBuildSolution': mono_sln_builder})
    env_mono.MonoBuildSolution(
        os.path.join(editor_tools_dir, 'GodotSharpTools.dll'),
        'editor/GodotSharpTools/GodotSharpTools.sln'
    )
