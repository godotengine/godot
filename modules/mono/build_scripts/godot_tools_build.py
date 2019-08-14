# Build GodotTools solution

import os

from SCons.Script import Dir


def build_godot_tools(source, target, env):
    # source and target elements are of type SCons.Node.FS.File, hence why we convert them to str

    module_dir = env['module_dir']

    solution_path = os.path.join(module_dir, 'editor/GodotTools/GodotTools.sln')
    build_config = 'Debug' if env['target'] == 'debug' else 'Release'

    from . solution_builder import build_solution, nuget_restore
    nuget_restore(env, solution_path)
    build_solution(env, solution_path, build_config)

    # Copy targets

    solution_dir = os.path.abspath(os.path.join(solution_path, os.pardir))

    src_dir = os.path.join(solution_dir, 'GodotTools', 'bin', build_config)
    dst_dir = os.path.abspath(os.path.join(str(target[0]), os.pardir))

    if not os.path.isdir(dst_dir):
        assert not os.path.isfile(dst_dir)
        os.makedirs(dst_dir)

    def copy_target(target_path):
        from shutil import copy
        filename = os.path.basename(target_path)
        copy(os.path.join(src_dir, filename), target_path)

    for scons_target in target:
        copy_target(str(scons_target))


def build_godot_tools_project_editor(source, target, env):
    # source and target elements are of type SCons.Node.FS.File, hence why we convert them to str

    module_dir = env['module_dir']

    project_name = 'GodotTools.ProjectEditor'

    csproj_dir = os.path.join(module_dir, 'editor/GodotTools', project_name)
    csproj_path = os.path.join(csproj_dir, project_name + '.csproj')
    build_config = 'Debug' if env['target'] == 'debug' else 'Release'

    from . solution_builder import build_solution, nuget_restore

    # Make sure to restore NuGet packages in the project directory for the project to find it
    nuget_restore(env, os.path.join(csproj_dir, 'packages.config'), '-PackagesDirectory',
                  os.path.join(csproj_dir, 'packages'))

    build_solution(env, csproj_path, build_config)

    # Copy targets

    src_dir = os.path.join(csproj_dir, 'bin', build_config)
    dst_dir = os.path.abspath(os.path.join(str(target[0]), os.pardir))

    if not os.path.isdir(dst_dir):
        assert not os.path.isfile(dst_dir)
        os.makedirs(dst_dir)

    def copy_target(target_path):
        from shutil import copy
        filename = os.path.basename(target_path)
        copy(os.path.join(src_dir, filename), target_path)

    for scons_target in target:
        copy_target(str(scons_target))


def build(env_mono):
    assert env_mono['tools']

    output_dir = Dir('#bin').abspath
    editor_tools_dir = os.path.join(output_dir, 'GodotSharp', 'Tools')
    editor_api_dir = os.path.join(output_dir, 'GodotSharp', 'Api', 'Debug')

    source_filenames = ['GodotSharp.dll', 'GodotSharpEditor.dll']
    sources = [os.path.join(editor_api_dir, filename) for filename in source_filenames]

    target_filenames = [
        'GodotTools.dll', 'GodotTools.IdeConnection.dll', 'GodotTools.BuildLogger.dll',
        'GodotTools.ProjectEditor.dll', 'DotNet.Glob.dll', 'GodotTools.Core.dll'
    ]

    if env_mono['target'] == 'debug':
        target_filenames += [
            'GodotTools.pdb', 'GodotTools.IdeConnection.pdb', 'GodotTools.BuildLogger.pdb',
            'GodotTools.ProjectEditor.pdb', 'GodotTools.Core.pdb'
        ]

    targets = [os.path.join(editor_tools_dir, filename) for filename in target_filenames]

    cmd = env_mono.CommandNoCache(targets, sources, build_godot_tools, module_dir=os.getcwd())
    env_mono.AlwaysBuild(cmd)


def build_project_editor_only(env_mono):
    assert env_mono['tools']

    output_dir = Dir('#bin').abspath
    editor_tools_dir = os.path.join(output_dir, 'GodotSharp', 'Tools')

    target_filenames = ['GodotTools.ProjectEditor.dll', 'DotNet.Glob.dll', 'GodotTools.Core.dll']

    if env_mono['target'] == 'debug':
        target_filenames += ['GodotTools.ProjectEditor.pdb', 'GodotTools.Core.pdb']

    targets = [os.path.join(editor_tools_dir, filename) for filename in target_filenames]

    cmd = env_mono.CommandNoCache(targets, [], build_godot_tools_project_editor, module_dir=os.getcwd())
    env_mono.AlwaysBuild(cmd)
