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


def get_nupkgs_versions(props_file):
    import xml.etree.ElementTree as ET

    tree = ET.parse(props_file)
    root = tree.getroot()

    return {
        "Godot.NET.Sdk": root.find("./PropertyGroup/PackageVersion_Godot_NET_Sdk").text.strip(),
        "Godot.SourceGenerators": root.find("./PropertyGroup/PackageVersion_Godot_SourceGenerators").text.strip(),
    }


def build(env_mono):
    assert env_mono["tools"]

    output_dir = Dir("#bin").abspath
    editor_tools_dir = os.path.join(output_dir, "GodotSharp", "Tools")
    nupkgs_dir = os.path.join(editor_tools_dir, "nupkgs")

    module_dir = os.getcwd()

    nupkgs_versions = get_nupkgs_versions(os.path.join(module_dir, "SdkPackageVersions.props"))

    target_filenames = [
        "Godot.NET.Sdk.%s.nupkg" % nupkgs_versions["Godot.NET.Sdk"],
        "Godot.SourceGenerators.%s.nupkg" % nupkgs_versions["Godot.SourceGenerators"],
    ]

    targets = [os.path.join(nupkgs_dir, filename) for filename in target_filenames]

    cmd = env_mono.CommandNoCache(targets, [], build_godot_net_sdk, module_dir=module_dir)
    env_mono.AlwaysBuild(cmd)
