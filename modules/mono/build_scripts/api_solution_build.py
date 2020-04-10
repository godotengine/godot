# Build the Godot API solution

import os

from SCons.Script import Dir


def build_api_solution(source, target, env):
    # source and target elements are of type SCons.Node.FS.File, hence why we convert them to str

    module_dir = env["module_dir"]

    solution_path = os.path.join(module_dir, "glue/GodotSharp/GodotSharp.sln")

    build_config = env["solution_build_config"]

    extra_msbuild_args = ["/p:NoWarn=1591"]  # Ignore missing documentation warnings

    from .solution_builder import build_solution

    build_solution(env, solution_path, build_config, extra_msbuild_args=extra_msbuild_args)

    # Copy targets

    core_src_dir = os.path.abspath(os.path.join(solution_path, os.pardir, "GodotSharp", "bin", build_config))
    editor_src_dir = os.path.abspath(os.path.join(solution_path, os.pardir, "GodotSharpEditor", "bin", build_config))

    dst_dir = os.path.abspath(os.path.join(str(target[0]), os.pardir))

    if not os.path.isdir(dst_dir):
        assert not os.path.isfile(dst_dir)
        os.makedirs(dst_dir)

    def copy_target(target_path):
        from shutil import copy

        filename = os.path.basename(target_path)

        src_path = os.path.join(core_src_dir, filename)
        if not os.path.isfile(src_path):
            src_path = os.path.join(editor_src_dir, filename)

        copy(src_path, target_path)

    for scons_target in target:
        copy_target(str(scons_target))


def build(env_mono):
    assert env_mono["tools"]

    target_filenames = [
        "GodotSharp.dll",
        "GodotSharp.pdb",
        "GodotSharp.xml",
        "GodotSharpEditor.dll",
        "GodotSharpEditor.pdb",
        "GodotSharpEditor.xml",
    ]

    depend_cmd = []

    for build_config in ["Debug", "Release"]:
        output_dir = Dir("#bin").abspath
        editor_api_dir = os.path.join(output_dir, "GodotSharp", "Api", build_config)

        targets = [os.path.join(editor_api_dir, filename) for filename in target_filenames]

        cmd = env_mono.CommandNoCache(
            targets, depend_cmd, build_api_solution, module_dir=os.getcwd(), solution_build_config=build_config
        )
        env_mono.AlwaysBuild(cmd)

        # Make the Release build of the API solution depend on the Debug build.
        # We do this in order to prevent SCons from building them in parallel,
        # which can freak out MSBuild. In many cases, one of the builds would
        # hang indefinitely requiring a key to be pressed for it to continue.
        depend_cmd = cmd

    return depend_cmd
