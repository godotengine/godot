#!/usr/bin/python3

"""Functions used to generate source files during build time

All such functions are invoked in a subprocess on Windows to prevent build flakiness.

"""

import os
import os.path
import shlex
import subprocess
from dataclasses import dataclass
from typing import Optional, List

target_filenames = [
    "GodotSharp.dll",
    "GodotSharp.pdb",
    "GodotSharp.xml",
    "GodotSharpEditor.dll",
    "GodotSharpEditor.pdb",
    "GodotSharpEditor.xml",
    "GodotPlugins.dll",
    "GodotPlugins.pdb",
    "GodotPlugins.runtimeconfig.json",
]


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
    nuget_tool: str = ""


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
                parts = os.path.split(msbuild_mono)
                nuget_tool = os.path.join(parts[0], "nuget" + os.path.splitext(parts[1])[1])
                return ToolsLocation(msbuild_mono=msbuild_mono, nuget_tool=nuget_tool)

        else:
            msbuild_mono = find_msbuild_mono_unix()
            if msbuild_mono:
                parts = os.path.split(msbuild_mono)
                nuget_tool = os.path.join(parts[0], "nuget" + os.path.splitext(parts[1])[1])
                return ToolsLocation(msbuild_mono=msbuild_mono, nuget_tool=nuget_tool)

    return None


def run_msbuild(tools: ToolsLocation, sln: str, msbuild_args: Optional[List[str]] = None):
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

    if msbuild_args:
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


def set_default_args(args: List[str], precision, push_nupkgs_local, clear_nuget_cache):
    if clear_nuget_cache:
        args += ["/p:ClearNuGetLocalCache=true"]
    if push_nupkgs_local:
        args += ["/p:PushNuGetToLocalSource=" + push_nupkgs_local]
    if precision == "double":
        args += ["/p:GodotFloat64=true"]
    return args


def build_godot_api(msbuild_tool, module_dir, output_dir, precision, push_nupkgs_local, clear_nuget_cache):
    for build_config in ["Debug", "Release"]:
        editor_api_dir = os.path.join(output_dir, "GodotSharp", "Api", build_config)

        targets = [os.path.join(editor_api_dir, filename) for filename in target_filenames]

        args = ["/restore", "/t:Build", "/p:Configuration=" + build_config, "/p:NoWarn=1591"]
        args = set_default_args(args, precision, push_nupkgs_local, clear_nuget_cache)

        sln = os.path.join(module_dir, "glue/GodotSharp/GodotSharp.sln")
        exit_code = run_msbuild(
            msbuild_tool,
            sln=sln,
            msbuild_args=args,
        )
        if exit_code != 0:
            return exit_code

        # Copy targets

        core_src_dir = os.path.abspath(os.path.join(sln, os.pardir, "GodotSharp", "bin", build_config))
        editor_src_dir = os.path.abspath(os.path.join(sln, os.pardir, "GodotSharpEditor", "bin", build_config))
        plugins_src_dir = os.path.abspath(os.path.join(sln, os.pardir, "GodotPlugins", "bin", build_config, "net6.0"))

        if not os.path.isdir(editor_api_dir):
            assert not os.path.isfile(editor_api_dir)
            os.makedirs(editor_api_dir)

        def copy_target(target_path):
            from shutil import copy

            filename = os.path.basename(target_path)

            src_path = os.path.join(core_src_dir, filename)
            if not os.path.isfile(src_path):
                src_path = os.path.join(editor_src_dir, filename)
            if not os.path.isfile(src_path):
                src_path = os.path.join(plugins_src_dir, filename)

            print(f"Copying assembly to {target_path}...")
            copy(src_path, target_path)

        for scons_target in targets:
            copy_target(scons_target)

    return 0


def generate_sdk_package_versions():
    # I can't believe importing files in Python is so convoluted when not
    # following the golden standard for packages/modules.
    import os
    import sys
    from os.path import dirname

    # We want ../../../methods.py.
    script_path = dirname(os.path.abspath(__file__))
    root_path = dirname(dirname(dirname(script_path)))

    sys.path.insert(0, root_path)
    from methods import get_version_info

    version_info = get_version_info("")
    sys.path.remove(root_path)

    version_str = "{major}.{minor}.{patch}".format(**version_info)
    version_status = version_info["status"]
    if version_status != "stable":  # Pre-release
        # If version was overridden to be e.g. "beta3", we insert a dot between
        # "beta" and "3" to follow SemVer 2.0.
        import re

        match = re.search(r"[\d]+$", version_status)
        if match:
            pos = match.start()
            version_status = version_status[:pos] + "." + version_status[pos:]
        version_str += "-" + version_status

    import version

    version_defines = (
        [
            f"GODOT{version.major}",
            f"GODOT{version.major}_{version.minor}",
            f"GODOT{version.major}_{version.minor}_{version.patch}",
        ]
        + [f"GODOT{v}_OR_GREATER" for v in range(4, version.major + 1)]
        + [f"GODOT{version.major}_{v}_OR_GREATER" for v in range(0, version.minor + 1)]
        + [f"GODOT{version.major}_{version.minor}_{v}_OR_GREATER" for v in range(0, version.patch + 1)]
    )

    props = """<Project>
  <PropertyGroup>
    <PackageVersion_GodotSharp>{0}</PackageVersion_GodotSharp>
    <PackageVersion_Godot_NET_Sdk>{0}</PackageVersion_Godot_NET_Sdk>
    <PackageVersion_Godot_SourceGenerators>{0}</PackageVersion_Godot_SourceGenerators>
    <GodotVersionConstants>{1}</GodotVersionConstants>
  </PropertyGroup>
</Project>
""".format(
        version_str, ";".join(version_defines)
    )

    # We write in ../SdkPackageVersions.props.
    with open(os.path.join(dirname(script_path), "SdkPackageVersions.props"), "w") as f:
        f.write(props)
        f.close()


def build_all(
    msbuild_tool, module_dir, output_dir, godot_platform, dev_debug, precision, push_nupkgs_local, clear_nuget_cache
):
    # Generate SdkPackageVersions.props
    generate_sdk_package_versions()

    # Godot API
    exit_code = build_godot_api(msbuild_tool, module_dir, output_dir, precision, push_nupkgs_local, clear_nuget_cache)
    if exit_code != 0:
        return exit_code

    # GodotTools
    sln = os.path.join(module_dir, "editor/GodotTools/GodotTools.sln")
    args = ["/restore", "/t:Build", "/p:Configuration=" + ("Debug" if dev_debug else "Release")] + (
        ["/p:GodotPlatform=" + godot_platform] if godot_platform else []
    )
    args = set_default_args(args, precision, push_nupkgs_local, clear_nuget_cache)
    exit_code = run_msbuild(msbuild_tool, sln=sln, msbuild_args=args)
    if exit_code != 0:
        return exit_code

    # Godot.NET.Sdk
    args = ["/restore", "/t:Build", "/p:Configuration=Release"]
    args = set_default_args(args, precision, push_nupkgs_local, clear_nuget_cache)
    sln = os.path.join(module_dir, "editor/Godot.NET.Sdk/Godot.NET.Sdk.sln")
    exit_code = run_msbuild(msbuild_tool, sln=sln, msbuild_args=args)
    if exit_code != 0:
        return exit_code

    return 0


def run_quiet(args):
    try:
        subprocess.check_output(args)
    except Exception as e:
        pass


def make_assemblies(target, source, env):
    exit_code = start_build(
        godot_output_dir="bin/",
        godot_platform=env["platform"],
        dev_debug=env["dev_build"],
        mono_prefix="",
        precision=env["precision"],
        push_nupkgs_local=False,
        clear_nuget_cache=True,
    )

    if exit_code != 0:
        return exit_code

    nupkgs_dir = os.path.abspath("bin/GodotSharp/Tools/nupkgs")
    tools = find_any_msbuild_tool("")

    # configure a default nuget source pointing to the bin/GodotSharp/Tools/nupkgs dir, so packages can be picked up automatically
    if tools.dotnet_cli:
        # add/replace a nuget source using the dotnet nuget tool
        args = [tools.dotnet_cli, "nuget"]
        run_quiet(args + ["remove", "source", "GodotSourceBuild"])
        run_quiet(args + ["add", "source", nupkgs_dir, "--name", "GodotSourceBuild"])
    elif tools.nuget_tool:
        # add/replace a nuget source using the .net framework nuget tool
        args = [tools.nuget_tool, "sources"]
        run_quiet(args + ["remove", "-Name", "GodotSourceBuild", "-NonInteractive"])
        run_quiet(args + ["add", "-Name", "GodotSourceBuild", "-Source", nupkgs_dir, "-NonInteractive"])

    return exit_code


def start_build(
    godot_output_dir, godot_platform, dev_debug, mono_prefix, precision, push_nupkgs_local, clear_nuget_cache=False
):
    this_script_dir = os.path.dirname(os.path.realpath(__file__))
    module_dir = os.path.abspath(os.path.join(this_script_dir, os.pardir))

    output_dir = os.path.abspath(godot_output_dir)

    if push_nupkgs_local:
        clear_nuget_cache = True
        push_nupkgs_local = os.path.abspath(push_nupkgs_local)
    else:
        push_nupkgs_local = None

    msbuild_tool = find_any_msbuild_tool(mono_prefix)

    if msbuild_tool is None:
        print("Unable to find MSBuild")
        return 1

    return build_all(
        msbuild_tool,
        module_dir,
        output_dir,
        godot_platform,
        dev_debug,
        precision,
        push_nupkgs_local,
        clear_nuget_cache,
    )


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Builds all Godot .NET solutions")
    parser.add_argument("--godot-output-dir", type=str, required=True)
    parser.add_argument(
        "--dev-debug",
        action="store_true",
        default=False,
        help="Build GodotTools and Godot.NET.Sdk with 'Configuration=Debug'",
    )
    parser.add_argument("--godot-platform", type=str, default="")
    parser.add_argument("--mono-prefix", type=str, default="")
    parser.add_argument("--push-nupkgs-local", type=str, default="")
    parser.add_argument(
        "--precision", type=str, default="single", choices=["single", "double"], help="Floating-point precision level"
    )

    args = parser.parse_args()

    exit_code = start_build(
        args.godot_output_dir,
        args.godot_platform,
        args.dev_debug,
        args.mono_prefix,
        args.precision,
        args.push_nupkgs_local,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    try:
        # if this fails, it's because it's being invoked directly from the shell
        from platform_methods import subprocess_main

        subprocess_main(globals())
    except Exception as e:
        main()
