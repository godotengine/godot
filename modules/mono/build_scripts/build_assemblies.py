#!/usr/bin/python3

import os
import os.path
import shlex
import subprocess
from dataclasses import dataclass


def find_dotnet_cli():
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


def find_msbuild_standalone_windows():
    msbuild_tools_path = find_msbuild_tools_path_reg()

    if msbuild_tools_path:
        return os.path.join(msbuild_tools_path, "MSBuild.exe")

    return None


def find_msbuild_mono_windows(mono_prefix):
    assert mono_prefix is not None

    mono_bin_dir = os.path.join(mono_prefix, "bin")
    msbuild_mono = os.path.join(mono_bin_dir, "msbuild.bat")

    if os.path.isfile(msbuild_mono):
        return msbuild_mono

    return None


def find_msbuild_mono_unix():
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


def find_msbuild_tools_path_reg():
    import subprocess

    program_files = os.getenv("PROGRAMFILES(X86)")
    if not program_files:
        program_files = os.getenv("PROGRAMFILES")
    vswhere = os.path.join(program_files, "Microsoft Visual Studio", "Installer", "vswhere.exe")

    vswhere_args = ["-latest", "-products", "*", "-requires", "Microsoft.Component.MSBuild"]

    try:
        lines = subprocess.check_output([vswhere] + vswhere_args).splitlines()

        for line in lines:
            parts = line.decode("utf-8").split(":", 1)

            if len(parts) < 2 or parts[0] != "installationPath":
                continue

            val = parts[1].strip()

            if not val:
                raise ValueError("Value of `installationPath` entry is empty")

            # Since VS2019, the directory is simply named "Current"
            msbuild_dir = os.path.join(val, "MSBuild", "Current", "Bin")
            if os.path.isdir(msbuild_dir):
                return msbuild_dir

            # Directory name "15.0" is used in VS 2017
            return os.path.join(val, "MSBuild", "15.0", "Bin")

        raise ValueError("Cannot find `installationPath` entry")
    except ValueError as e:
        print("Error reading output from vswhere: " + str(e))
    except OSError:
        pass  # Fine, vswhere not found
    except (subprocess.CalledProcessError, OSError):
        pass


@dataclass
class ToolsLocation:
    dotnet_cli: str = ""
    msbuild_standalone: str = ""
    msbuild_mono: str = ""
    mono_bin_dir: str = ""


def find_any_msbuild_tool(mono_prefix):
    # Preference order: dotnet CLI > Standalone MSBuild > Mono's MSBuild

    # Find dotnet CLI
    dotnet_cli = find_dotnet_cli()
    if dotnet_cli:
        return ToolsLocation(dotnet_cli=dotnet_cli)

    # Find standalone MSBuild
    if os.name == "nt":
        msbuild_standalone = find_msbuild_standalone_windows()
        if msbuild_standalone:
            return ToolsLocation(msbuild_standalone=msbuild_standalone)

    if mono_prefix:
        # Find Mono's MSBuild
        if os.name == "nt":
            msbuild_mono = find_msbuild_mono_windows(mono_prefix)
            if msbuild_mono:
                return ToolsLocation(msbuild_mono=msbuild_mono)
        else:
            msbuild_mono = find_msbuild_mono_unix()
            if msbuild_mono:
                return ToolsLocation(msbuild_mono=msbuild_mono)

    return None


def run_msbuild(tools: ToolsLocation, sln: str, msbuild_args: [str] = None):
    if msbuild_args is None:
        msbuild_args = []

    using_msbuild_mono = False

    # Preference order: dotnet CLI > Standalone MSBuild > Mono's MSBuild
    if tools.dotnet_cli:
        args = [tools.dotnet_cli, "msbuild"]
    elif tools.msbuild_standalone:
        args = [tools.msbuild_standalone]
    elif tools.msbuild_mono:
        args = [tools.msbuild_mono]
        using_msbuild_mono = True
    else:
        raise RuntimeError("Path to MSBuild or dotnet CLI not provided.")

    args += [sln]

    if len(msbuild_args) > 0:
        args += msbuild_args

    print("Running MSBuild: ", " ".join(shlex.quote(arg) for arg in args), flush=True)

    msbuild_env = os.environ.copy()

    # Needed when running from Developer Command Prompt for VS
    if "PLATFORM" in msbuild_env:
        del msbuild_env["PLATFORM"]

    if using_msbuild_mono:
        # The (Csc/Vbc/Fsc)ToolExe environment variables are required when
        # building with Mono's MSBuild. They must point to the batch files
        # in Mono's bin directory to make sure they are executed with Mono.
        msbuild_env.update(
            {
                "CscToolExe": os.path.join(tools.mono_bin_dir, "csc.bat"),
                "VbcToolExe": os.path.join(tools.mono_bin_dir, "vbc.bat"),
                "FscToolExe": os.path.join(tools.mono_bin_dir, "fsharpc.bat"),
            }
        )

    return subprocess.call(args, env=msbuild_env)


def build_godot_api(msbuild_tool, module_dir, output_dir):
    target_filenames = [
        "GodotSharp.dll",
        "GodotSharp.pdb",
        "GodotSharp.xml",
        "GodotSharpEditor.dll",
        "GodotSharpEditor.pdb",
        "GodotSharpEditor.xml",
    ]

    for build_config in ["Debug", "Release"]:
        editor_api_dir = os.path.join(output_dir, "GodotSharp", "Api", build_config)

        targets = [os.path.join(editor_api_dir, filename) for filename in target_filenames]

        sln = os.path.join(module_dir, "glue/GodotSharp/GodotSharp.sln")
        exit_code = run_msbuild(
            msbuild_tool,
            sln=sln,
            msbuild_args=["/restore", "/t:Build", "/p:Configuration=" + build_config, "/p:NoWarn=1591"],
        )
        if exit_code != 0:
            return exit_code

        # Copy targets

        core_src_dir = os.path.abspath(os.path.join(sln, os.pardir, "GodotSharp", "bin", build_config))
        editor_src_dir = os.path.abspath(os.path.join(sln, os.pardir, "GodotSharpEditor", "bin", build_config))

        if not os.path.isdir(editor_api_dir):
            assert not os.path.isfile(editor_api_dir)
            os.makedirs(editor_api_dir)

        def copy_target(target_path):
            from shutil import copy

            filename = os.path.basename(target_path)

            src_path = os.path.join(core_src_dir, filename)
            if not os.path.isfile(src_path):
                src_path = os.path.join(editor_src_dir, filename)

            print(f"Copying assembly to {target_path}...")
            copy(src_path, target_path)

        for scons_target in targets:
            copy_target(scons_target)

    return 0


def build_all(msbuild_tool, module_dir, output_dir, godot_target, godot_platform):
    # Godot API
    exit_code = build_godot_api(msbuild_tool, module_dir, output_dir)
    if exit_code != 0:
        return exit_code

    # GodotTools
    sln = os.path.join(module_dir, "editor/GodotTools/GodotTools.sln")
    args = ["/restore", "/t:Build", "/p:Configuration=" + ("Debug" if godot_target == "debug" else "Release")] + (
        ["/p:GodotPlatform=" + godot_platform] if godot_platform else []
    )
    exit_code = run_msbuild(msbuild_tool, sln=sln, msbuild_args=args)
    if exit_code != 0:
        return exit_code

    # Godot.NET.Sdk
    sln = os.path.join(module_dir, "editor/Godot.NET.Sdk/Godot.NET.Sdk.sln")
    exit_code = run_msbuild(msbuild_tool, sln=sln, msbuild_args=["/restore", "/t:Build", "/p:Configuration=Release"])
    if exit_code != 0:
        return exit_code

    return 0


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Builds all Godot .NET solutions")
    parser.add_argument("--godot-output-dir", type=str, required=True)
    parser.add_argument("--godot-target", choices=["debug", "release_debug", "release"], type=str, required=True)
    parser.add_argument("--godot-platform", type=str, default="")
    parser.add_argument("--mono-prefix", type=str, default="")

    args = parser.parse_args()

    this_script_dir = os.path.dirname(os.path.realpath(__file__))
    module_dir = os.path.abspath(os.path.join(this_script_dir, os.pardir))

    output_dir = os.path.abspath(args.godot_output_dir)

    msbuild_tool = find_any_msbuild_tool(args.mono_prefix)

    if msbuild_tool is None:
        print("Unable to find MSBuild")
        sys.exit(1)

    exit_code = build_all(msbuild_tool, module_dir, output_dir, args.godot_target, args.godot_platform)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
