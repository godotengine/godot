"""
In this file we're making our best effort to leverage the convenience Xcode project included in MetalANGLE.
The goal is to get as much information from it about the targets (and groups) we need.
However, some files we need for the build are not discovered from the project, and that's why we have a code section
where we explicitly copy some stuff out from the MetalANGLE source tree.
"""

from pbxproj import XcodeProject
import os
import os.path
import re
import shutil
import sys

FIRST_LEVEL_GROUP_WHITELIST = [
    "MGLKit",
    "third_party",
    "angle_common",
    "angle_util",
    "angle_image_util",
    "angle",
    "angle_metal_backend",
    "angle_gl_backend",
]

TARGETS_BLACKLIST = [
    "angle_commit_id",
]


def main():
    if len(sys.argv) != 2:
        print("Usage: %s <metalangle_repo_dir>" % os.path.basename(__file__))
        sys.exit(1)
    global metalangle_module_dir
    metalangle_module_dir = os.path.dirname(__file__)
    global metalangle_repo_dir
    metalangle_repo_dir = sys.argv[1]

    assert os.path.isdir(metalangle_repo_dir)

    global my_thirdparty_dir
    my_thirdparty_dir = os.path.normpath(os.path.dirname(__file__) + "/../../thirdparty/metalangle")
    if os.path.exists(my_thirdparty_dir):
        shutil.rmtree(my_thirdparty_dir)
    os.mkdir(my_thirdparty_dir)

    global xcode_project_dir
    xcode_project_dir = metalangle_repo_dir + "/ios/xcode"
    xcode_project_path = xcode_project_dir + "/OpenGLES.xcodeproj/project.pbxproj"
    global project_archive
    project_archive = XcodeProject.load(xcode_project_path)
    global project
    project = project_archive.get_object(project_archive["rootObject"])
    main_group = project_archive.get_object(project["mainGroup"])

    first_level_groups = _collect_first_level_groups(project_archive, main_group)
    global relevant_groups
    relevant_groups = first_level_groups + _collect_subgroups(project_archive, first_level_groups)

    _copy_project_files()

    # I don't wanna miss a thing (these are needed, but weren't discovered in the project)
    _copy_raw("src/common/apple_platform_utils.h")
    _copy_raw("src/common/third_party/base/anglebase/no_destructor.h")
    _copy_raw("src/compiler/translator/ASTMetadataHLSL.h")
    _copy_raw("src/compiler/translator/OutputHLSL.h")
    _copy_raw("src/compiler/translator/ParseContext_interm.h")
    _copy_raw("src/compiler/translator/ParseContext_complete_autogen.h")
    _copy_raw("src/compiler/translator/ShaderStorageBlockOutputHLSL.h")
    _copy_raw("src/compiler/translator/tree_util/BuiltIn.h")
    _copy_raw("src/compiler/translator/tree_util/BuiltIn_complete_autogen.h")
    _copy_raw("src/libANGLE/renderer/d3d/RenderbufferD3D.h")
    _copy_raw("src/libANGLE/renderer/d3d/RenderTargetD3D.h")
    _copy_raw("src/libANGLE/renderer/OverlayImpl.h")
    _copy_raw("src/libANGLE/Context_gl_3_2_autogen.h")
    _copy_raw("src/libANGLE/Context_gl_3_3_autogen.h")
    _copy_raw("src/libANGLE/Context_gl_4_0_autogen.h")
    _copy_raw("src/libANGLE/Context_gl_4_1_autogen.h")
    _copy_raw("src/libANGLE/Context_gl_4_2_autogen.h")
    _copy_raw("src/libANGLE/Context_gl_4_3_autogen.h")
    _copy_raw("src/libANGLE/Context_gl_4_4_autogen.h")
    _copy_raw("src/libANGLE/Context_gl_4_5_autogen.h")
    _copy_raw("src/libANGLE/Context_gl_4_6_autogen.h")
    _copy_raw("third_party/glslang/src/glslang/Public/ShaderLang.h")
    _copy_raw("third_party/glslang/src/SPIRV/NonSemanticDebugPrintf.h")
    _copy_raw("third_party/jsoncpp/source/src/lib_json/json_valueiterator.inl")
    _copy_raw("third_party/spirv-cross/src/GLSL.std.450.h")

    # Copy the whole public MetalANGLE include directory
    shutil.copytree(metalangle_repo_dir + "/include", my_thirdparty_dir + "/include", dirs_exist_ok=True)
    # Copy the commit id information that the Xcode project generates
    shutil.copytree(metalangle_repo_dir + "/ios/xcode/gen", my_thirdparty_dir + "/ios/xcode/gen", dirs_exist_ok=True)

    _process_target("MetalANGLE_static", "iphone")
    _process_target("MetalANGLE_static_mac", "osx")


def _copy_raw(p_rel_path):
    src = metalangle_repo_dir + "/" + p_rel_path
    dst = my_thirdparty_dir + "/" + p_rel_path
    if not os.path.exists(os.path.dirname(dst)):
        os.mkdir(os.path.dirname(dst))
    shutil.copyfile(src, dst)


def _collect_first_level_groups(p_project_archive, p_main_group):
    first_level_groups = [p_project_archive.get_object(v) for v in p_main_group.children]
    first_level_groups = [v for v in first_level_groups if v.get_name() in FIRST_LEVEL_GROUP_WHITELIST]
    return first_level_groups


def _collect_subgroups(p_project_archive, p_groups):
    subgroups = []
    for group in p_groups:
        children = [p_project_archive.get_object(v) for v in group.children]
        child_groups = [v for v in children if v.isa == "PBXGroup"]
        subgroups += child_groups + _collect_subgroups(p_project_archive, child_groups)
    return subgroups


def _copy_project_files():
    for group in relevant_groups:
        children = [project_archive.get_object(v) for v in group.children]
        children_files = [v for v in children if v.isa == "PBXFileReference"]
        for file in children_files:
            src_path = _get_file_full_path(file, group)
            if not os.path.isfile(src_path):
                continue
            dst_path = my_thirdparty_dir + "/" + os.path.relpath(src_path, metalangle_repo_dir)
            dst_dir = os.path.dirname(dst_path)
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            shutil.copyfile(src_path, dst_path)


def _get_file_full_path(p_file, p_group):
    if "/" in p_file.path:
        full_path = p_file.path
    else:
        full_group_path = p_group.get_path()
        if not "/" in full_group_path:
            group = p_group
            while True:
                group = next(filter(lambda v: group.get_id() in v.children, relevant_groups), None)
                if group is None:
                    break
                curr_group_path = group.get_path()
                full_group_path = curr_group_path + "/" + full_group_path
                if "/" in curr_group_path:
                    break
        full_path = full_group_path + "/" + p_file.path

    full_path = xcode_project_dir + "/" + full_path
    return os.path.normpath(full_path)


def _get_source_file_thirdparty_rel_path(p_file, p_group):
    src_path = _get_file_full_path(p_file, p_group)
    path = os.path.relpath(src_path, metalangle_repo_dir)
    return path.replace("\\", "/")


def _process_target(p_target_name, p_platform):
    target = project_archive.get_target_by_name(p_target_name)
    assert target

    # Collect global and local options

    files_build_info = []
    _collect_target_files_build_info(target, files_build_info)

    project_options = _get_compiler_options_from_object(project)
    for fbi in files_build_info:
        fbi["options"] = project_options + " " + fbi["options"]

    # Clean

    for fbi in files_build_info:
        if re.search(r"-std=gnu[^+][^+]", fbi["options"]) and not fbi["path"].endswith(".c"):
            fbi["options"] = re.sub(r"-std=gnu[^+][^+]", "", fbi["options"])
        if re.search(r"-std=gnu\+\+\S\S", fbi["options"]) and not fbi["path"].endswith(".cpp"):
            fbi["options"] = re.sub(r"-std=gnu\+\+\S\S", "", fbi["options"])

    # Group by options

    unique_options = set()
    for fbi in files_build_info:
        unique_options.add(fbi["options"])
    build_groups = []
    for uo in unique_options:
        build_groups.append(
            {
                "paths": [v["path"] for v in files_build_info if v["options"] == uo],
                "options": uo,
            }
        )

    # Write out the results

    filename = "%s_generated.py" % p_platform
    file = open(metalangle_module_dir + "/" + filename, "w")
    file.write("def setup_%s(p_env_thirdparty):\n" % p_platform)
    file.write("    thirdparty_obj = []\n\n")
    for i, bg in enumerate(build_groups):
        file.write("    env_thirdparty_%d = p_env_thirdparty.Clone()\n" % i)
        file.write("    env_thirdparty_%d.disable_warnings()\n" % i)
        options_list = bg["options"].strip().split(" ")
        options_list = [v.strip() for v in options_list if v.strip()]
        options_str = '"' + ('",\n            "'.join(options_list)) + '"'
        file.write("    env_thirdparty_%d.Append(\n" % i)
        file.write("        CPPFLAGS=[\n")
        file.write("            %s,\n" % options_str)
        file.write("        ],\n")
        file.write("    )\n")
        file.write("    env_thirdparty_%d.add_source_files(\n" % i)
        file.write("        thirdparty_obj,\n")
        file.write("        [\n")
        for p in bg["paths"]:
            file.write('            "#thirdparty/metalangle/%s",\n' % p)
        file.write("        ],\n")
        file.write("    )\n\n")
    file.write("    return thirdparty_obj\n")
    file.close()


def _collect_target_files_build_info(p_target, r_files_build_info):
    if p_target.name in TARGETS_BLACKLIST:
        return

    assert p_target.isa == "PBXNativeTarget"

    options = _get_compiler_options_from_object(p_target)

    # Sources
    sources_phase = p_target.get_or_create_build_phase("PBXSourcesBuildPhase")
    for build_file_id in sources_phase[0].files:
        build_file = project_archive.get_object(build_file_id)
        group = next(filter(lambda v: build_file.fileRef in v.children, relevant_groups), None)
        assert group

        file = project_archive.get_object(build_file.fileRef)
        rel_path = _get_source_file_thirdparty_rel_path(file, group)

        r_files_build_info.append(
            {
                "path": rel_path,
                "options": options + " " + (build_file.get_compiler_flags() or ""),
            }
        )

    # Dependencies
    for dependency_id in p_target.dependencies:
        dep = project_archive.get_object(dependency_id)
        assert dep.isa == "PBXTargetDependency"

        dep_target = project_archive.get_object(dep.target)
        _collect_target_files_build_info(dep_target, r_files_build_info)


def _get_compiler_options_from_object(p_object):
    build_configs = project_archive.get_object(p_object.buildConfigurationList).buildConfigurations
    build_configs = [project_archive.get_object(v) for v in build_configs]
    return _get_compiler_options_from_build_config(build_configs, "Release")


def _get_compiler_options_from_build_config(p_build_configs, p_config_name):
    build_config = next(filter(lambda v: v.name == p_config_name, p_build_configs), None)
    assert build_config
    build_settings = build_config.buildSettings

    options = ""

    if "HEADER_SEARCH_PATHS" in build_settings:
        # Easier than using SCons' CPPPATH and compatible enough for the case of Apple
        paths = [v for v in build_settings.HEADER_SEARCH_PATHS if v != "${inherited}"]
        paths = [v.replace("${PROJECT_DIR}/gen", "thirdparty/metalangle/ios/xcode/gen") for v in paths]
        paths = [v.replace("${PROJECT_DIR}/../..", "thirdparty/metalangle") for v in paths]
        options += " " + " ".join(["-I%s" % v for v in paths])

    if "CLANG_ENABLE_OBJC_ARC" in build_settings:
        options += " -fobjc-arc" if build_settings.CLANG_ENABLE_OBJC_ARC == "YES" else "-fno-objc-arc"

    if "GCC_C_LANGUAGE_STANDARD" in build_settings:
        options += " -std=" + build_settings.GCC_C_LANGUAGE_STANDARD

    if "CLANG_CXX_LANGUAGE_STANDARD" in build_settings:
        options += " -std=" + build_settings.CLANG_CXX_LANGUAGE_STANDARD

    if "GCC_PREPROCESSOR_DEFINITIONS" in build_settings:
        defs = [v for v in build_settings.GCC_PREPROCESSOR_DEFINITIONS if not "inherited" in v]
        options += " " + " ".join(["-D%s" % v for v in defs])

    if "OTHER_CFLAGS" in build_settings:
        if type(build_settings.OTHER_CFLAGS) is list:
            cflags = [v for v in build_settings.OTHER_CFLAGS if v != "$(inherited)"]
            options += " " + " ".join(cflags)
        else:
            options += " " + build_settings.OTHER_CFLAGS

    return options


main()
