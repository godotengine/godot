# Build Godot.NET.Sdk solution

import os

from SCons.Script import Dir


def build_godot_net_sdk(source, target, env):
    # source and target elements are of type SCons.Node.FS.File, hence why we convert them to str

    module_dir = env["module_dir"]

    solution_path = os.path.join(module_dir, "editor/Godot.NET.Sdk/Godot.NET.Sdk.sln")
    build_config = "Release"

    from .solution_builder import build_solution

    extra_msbuild_args = ["/p:GodotPlatform=" + env["platform"]]

    build_solution(env, solution_path, build_config, extra_msbuild_args)
    # No need to copy targets. The Godot.NET.Sdk csproj takes care of copying them.


def build(env_mono):
    assert env_mono["tools"]

    output_dir = Dir("#bin").abspath
    editor_tools_dir = os.path.join(output_dir, "GodotSharp", "Tools")
    nupkgs_dir = os.path.join(editor_tools_dir, "nupkgs")

    module_dir = os.getcwd()

    package_version_file = os.path.join(
        module_dir, "editor", "Godot.NET.Sdk", "Godot.NET.Sdk", "Godot.NET.Sdk_PackageVersion.txt"
    )

    with open(package_version_file, mode="r") as f:
        version = f.read().strip()

    target_filenames = ["Godot.NET.Sdk.%s.nupkg" % version]

    targets = [os.path.join(nupkgs_dir, filename) for filename in target_filenames]

    cmd = env_mono.CommandNoCache(targets, [], build_godot_net_sdk, module_dir=module_dir)
    env_mono.AlwaysBuild(cmd)
