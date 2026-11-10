"""Functions used to generate source files during build time"""

import os
import shutil
import subprocess

from platform_methods import lipo


def generate_bundle(target, source, env):
    env = source[1]
    bin_dir = os.path.abspath("bin")
    dist_dir = os.path.abspath("misc/dist")
    suffix = env["extra_suffix"] + env["module_version_string"]

    if env["editor_build"]:
        # Editor bundle.
        prefix = "godot." + env["platform"] + "." + env["target"]
        if env["dev_build"]:
            prefix += ".dev"
        if env["precision"] == "double":
            prefix += ".double"

        # Lipo editor executable.
        target_bin = lipo(bin_dir + "/" + prefix, suffix)

        # Assemble .app bundle and update version info.
        app_dir = bin_dir + "/" + (prefix + suffix).replace(".", "_") + ".app"
        templ = dist_dir + "/macos_tools.app"
        if os.path.exists(app_dir):
            shutil.rmtree(app_dir)

        # Create the .app bundle directory itself from scratch so that the creation
        # date is accurate, but copy the rest of the template over.
        os.mkdir(app_dir)
        shutil.copytree(
            os.path.join(templ, "Contents"),
            os.path.join(app_dir, "Contents"),
            ignore=shutil.ignore_patterns("Info.plist"),
        )

        if not os.path.isdir(app_dir + "/Contents/MacOS"):
            os.mkdir(app_dir + "/Contents/MacOS")
        if target_bin != "":
            shutil.copy(target_bin, app_dir + "/Contents/MacOS/Godot")
        if "mono" in env["module_version_string"]:
            shutil.copytree(bin_dir + "/GodotSharp", app_dir + "/Contents/Resources/GodotSharp")
        version = env["version"]
        with open(dist_dir + "/macos" + "/editor_info_plist.template", "rt", encoding="utf-8") as fin:
            with open(app_dir + "/Contents/Info.plist", "wt", encoding="utf-8", newline="\n") as fout:
                for line in fin:
                    line = line.replace("$version", "{major}.{minor}.{patch}.{status}.{build}".format(**version))
                    line = line.replace("$short_version", "{major}.{minor}.{patch}".format(**version))
                    if version["build"] != "official" and version["build"] != "steam":
                        line = line.replace("org.godotengine.godot", "org.godotengine.godot." + version["build"])
                    fout.write(line)

        # Sign .app bundle.
        if env["bundle_sign_identity"] != "":
            sign_command = [
                "codesign",
                "-s",
                env["bundle_sign_identity"],
                "--deep",
                "--force",
                "--options=runtime",
                "--entitlements",
            ]
            if env["dev_build"]:
                sign_command += [dist_dir + "/macos" + "/editor_debug.entitlements"]
            else:
                sign_command += [dist_dir + "/macos" + "/editor.entitlements"]
            sign_command += [app_dir]
            subprocess.run(sign_command)

    else:
        # Template bundle.
        app_prefix = "godot." + env["platform"]
        rel_prefix = "godot." + env["platform"] + "." + "template_release"
        dbg_prefix = "godot." + env["platform"] + "." + "template_debug"
        if env["dev_build"]:
            app_prefix += ".dev"
            rel_prefix += ".dev"
            dbg_prefix += ".dev"
        if env["precision"] == "double":
            app_prefix += ".double"
            rel_prefix += ".double"
            dbg_prefix += ".double"

        # Lipo template executables.
        rel_target_bin = lipo(bin_dir + "/" + rel_prefix, suffix)
        dbg_target_bin = lipo(bin_dir + "/" + dbg_prefix, suffix)

        # Assemble .app bundle.
        app_dir = bin_dir + "/macos_template.app"
        templ = dist_dir + "/macos_template.app"
        if os.path.exists(app_dir):
            shutil.rmtree(app_dir)
        shutil.copytree(templ, app_dir)
        if not os.path.isdir(app_dir + "/Contents/MacOS"):
            os.mkdir(app_dir + "/Contents/MacOS")
        if rel_target_bin != "":
            shutil.copy(rel_target_bin, app_dir + "/Contents/MacOS/godot_macos_release.universal")
        if dbg_target_bin != "":
            shutil.copy(dbg_target_bin, app_dir + "/Contents/MacOS/godot_macos_debug.universal")

        # ZIP .app bundle.
        zip_dir = bin_dir + "/" + (app_prefix + suffix).replace(".", "_")
        shutil.make_archive(zip_dir, "zip", root_dir=bin_dir, base_dir="macos_template.app")
        shutil.rmtree(app_dir)


def make_debug_macos(target, source, env):
    dst = str(target[0])
    dsymutil, llvm_strip = source
    subprocess.run([dsymutil, dst, "-o", f"{dst}.dSYM"], check=True)
    if llvm_strip:
        subprocess.run(["llvm-strip", dst], check=True)
    else:
        subprocess.run(["strip", "-u", "-r", dst], check=True)
