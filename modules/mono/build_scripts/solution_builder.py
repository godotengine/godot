import os


verbose = False


def find_dotnet_cli():
    import os.path

    if os.name == "nt":
        for hint_dir in os.environ["PATH"].split(os.pathsep):
            hint_dir = hint_dir.strip('"')
            hint_path = os.path.join(hint_dir, "dotnet")
            if os.path.isfile(hint_path) and os.access(hint_path, os.X_OK):
                return hint_path
            if os.path.isfile(hint_path + ".exe") and os.access(hint_path + ".exe", os.X_OK):
                return hint_path + ".exe"
    else:
        for hint_dir in os.environ["PATH"].split(os.pathsep):
            hint_dir = hint_dir.strip('"')
            hint_path = os.path.join(hint_dir, "dotnet")
            if os.path.isfile(hint_path) and os.access(hint_path, os.X_OK):
                return hint_path


def find_msbuild_unix():
    import os.path
    import sys

    hint_dirs = []
    if sys.platform == "darwin":
        hint_dirs[:0] = [
            "/Library/Frameworks/Mono.framework/Versions/Current/bin",
            "/usr/local/var/homebrew/linked/mono/bin",
        ]

    for hint_dir in hint_dirs:
        hint_path = os.path.join(hint_dir, "msbuild")
        if os.path.isfile(hint_path):
            return hint_path
        elif os.path.isfile(hint_path + ".exe"):
            return hint_path + ".exe"

    for hint_dir in os.environ["PATH"].split(os.pathsep):
        hint_dir = hint_dir.strip('"')
        hint_path = os.path.join(hint_dir, "msbuild")
        if os.path.isfile(hint_path) and os.access(hint_path, os.X_OK):
            return hint_path
        if os.path.isfile(hint_path + ".exe") and os.access(hint_path + ".exe", os.X_OK):
            return hint_path + ".exe"

    return None


def find_msbuild_windows(env):
    from .mono_reg_utils import find_mono_root_dir, find_msbuild_tools_path_reg

    mono_root = env["mono_prefix"] or find_mono_root_dir(env["bits"])

    if not mono_root:
        raise RuntimeError("Cannot find mono root directory")

    mono_bin_dir = os.path.join(mono_root, "bin")
    msbuild_mono = os.path.join(mono_bin_dir, "msbuild.bat")

    msbuild_tools_path = find_msbuild_tools_path_reg()

    if msbuild_tools_path:
        return (os.path.join(msbuild_tools_path, "MSBuild.exe"), {})

    if os.path.isfile(msbuild_mono):
        # The (Csc/Vbc/Fsc)ToolExe environment variables are required when
        # building with Mono's MSBuild. They must point to the batch files
        # in Mono's bin directory to make sure they are executed with Mono.
        mono_msbuild_env = {
            "CscToolExe": os.path.join(mono_bin_dir, "csc.bat"),
            "VbcToolExe": os.path.join(mono_bin_dir, "vbc.bat"),
            "FscToolExe": os.path.join(mono_bin_dir, "fsharpc.bat"),
        }
        return (msbuild_mono, mono_msbuild_env)

    return None


def run_command(command, args, env_override=None, name=None):
    def cmd_args_to_str(cmd_args):
        return " ".join([arg if not " " in arg else '"%s"' % arg for arg in cmd_args])

    args = [command] + args

    if name is None:
        name = os.path.basename(command)

    if verbose:
        print("Running '%s': %s" % (name, cmd_args_to_str(args)))

    import subprocess

    try:
        if env_override is None:
            subprocess.check_call(args)
        else:
            subprocess.check_call(args, env=env_override)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("'%s' exited with error code: %s" % (name, e.returncode))


def build_solution(env, solution_path, build_config, extra_msbuild_args=[]):
    global verbose
    verbose = env["verbose"]

    msbuild_env = os.environ.copy()

    # Needed when running from Developer Command Prompt for VS
    if "PLATFORM" in msbuild_env:
        del msbuild_env["PLATFORM"]

    msbuild_args = []

    dotnet_cli = find_dotnet_cli()

    if dotnet_cli:
        msbuild_path = dotnet_cli
        msbuild_args += ["msbuild"]  # `dotnet msbuild` command
    else:
        # Find MSBuild
        if os.name == "nt":
            msbuild_info = find_msbuild_windows(env)
            if msbuild_info is None:
                raise RuntimeError("Cannot find MSBuild executable")
            msbuild_path = msbuild_info[0]
            msbuild_env.update(msbuild_info[1])
        else:
            msbuild_path = find_msbuild_unix()
            if msbuild_path is None:
                raise RuntimeError("Cannot find MSBuild executable")

    print("MSBuild path: " + msbuild_path)

    # Build solution

    msbuild_args += [solution_path, "/restore", "/t:Build", "/p:Configuration=" + build_config]
    msbuild_args += extra_msbuild_args

    run_command(msbuild_path, msbuild_args, env_override=msbuild_env, name="msbuild")
