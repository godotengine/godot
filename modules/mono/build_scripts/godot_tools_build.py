# Build GodotTools solution

import os

from SCons.Script import Dir


def build_godot_tools(source, target, env):
    # source and target elements are of type SCons.Node.FS.File, hence why we convert them to str

    module_dir = env["module_dir"]

    solution_path = os.path.join(module_dir, "editor/GodotTools/GodotTools.sln")
    build_config = "Debug" if env["target"] == "debug" else "Release"

    from .solution_builder import build_solution

    extra_msbuild_args = ["/p:GodotPlatform=" + env["platform"]]

    build_solution(env, solution_path, build_config, extra_msbuild_args)
    # No need to copy targets. The GodotTools csproj takes care of copying them.


def build(env_mono, api_sln_cmd):
    assert env_mono["tools"]

    output_dir = Dir("#bin").abspath
    editor_tools_dir = os.path.join(output_dir, "GodotSharp", "Tools")

    target_filenames = ["GodotTools.dll"]

    if env_mono["target"] == "debug":
        target_filenames += ["GodotTools.pdb"]

    targets = [os.path.join(editor_tools_dir, filename) for filename in target_filenames]

    cmd = env_mono.CommandNoCache(targets, api_sln_cmd, build_godot_tools, module_dir=os.getcwd())
    env_mono.AlwaysBuild(cmd)
