"""Functions used to generate source files during build time"""

import os
import shutil

from platform_methods import detect_mvk, lipo


def combine_libs(target, source, env):
    lib_path = target[0].srcnode().abspath
    if "osxcross" in env:
        libtool = "$IOS_TOOLCHAIN_PATH/usr/bin/${ios_triple}libtool"
    else:
        libtool = "$IOS_TOOLCHAIN_PATH/usr/bin/libtool"
    env.Execute(
        libtool + ' -static -o "' + lib_path + '" ' + " ".join([('"' + lib.srcnode().abspath + '"') for lib in source])
    )


def generate_bundle(target, source, env):
    bin_dir = env.Dir("#bin").abspath

    # Template bundle.
    app_prefix = "godot." + env["platform"]
    rel_prefix = "libgodot." + env["platform"] + "." + "template_release"
    dbg_prefix = "libgodot." + env["platform"] + "." + "template_debug"
    if env.dev_build:
        app_prefix += ".dev"
        rel_prefix += ".dev"
        dbg_prefix += ".dev"
    if env["precision"] == "double":
        app_prefix += ".double"
        rel_prefix += ".double"
        dbg_prefix += ".double"

    # Lipo template libraries.
    rel_target_bin = lipo(bin_dir + "/" + rel_prefix, env.extra_suffix + ".a")
    dbg_target_bin = lipo(bin_dir + "/" + dbg_prefix, env.extra_suffix + ".a")
    rel_target_bin_sim = lipo(bin_dir + "/" + rel_prefix, ".simulator" + env.extra_suffix + ".a")
    dbg_target_bin_sim = lipo(bin_dir + "/" + dbg_prefix, ".simulator" + env.extra_suffix + ".a")

    # Assemble Xcode project bundle.
    app_dir = env.Dir("#bin/ios_xcode").abspath
    templ = env.Dir("#misc/dist/ios_xcode").abspath
    if os.path.exists(app_dir):
        shutil.rmtree(app_dir)
    shutil.copytree(templ, app_dir)
    if rel_target_bin != "":
        shutil.copy(rel_target_bin, app_dir + "/libgodot.ios.release.xcframework/ios-arm64/libgodot.a")
    if dbg_target_bin != "":
        shutil.copy(dbg_target_bin, app_dir + "/libgodot.ios.debug.xcframework/ios-arm64/libgodot.a")
    if rel_target_bin_sim != "":
        shutil.copy(
            rel_target_bin_sim, app_dir + "/libgodot.ios.release.xcframework/ios-arm64_x86_64-simulator/libgodot.a"
        )
    if dbg_target_bin_sim != "":
        shutil.copy(
            dbg_target_bin_sim, app_dir + "/libgodot.ios.debug.xcframework/ios-arm64_x86_64-simulator/libgodot.a"
        )
    mvk_path = detect_mvk(env, "ios-arm64")
    if mvk_path != "":
        shutil.copytree(mvk_path, app_dir + "/MoltenVK.xcframework")

    # ZIP Xcode project bundle.
    zip_dir = env.Dir("#bin/" + (app_prefix + env.extra_suffix).replace(".", "_")).abspath
    shutil.make_archive(zip_dir, "zip", root_dir=app_dir)
    shutil.rmtree(app_dir)
