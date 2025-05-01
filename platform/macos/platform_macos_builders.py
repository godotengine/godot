"""Functions used to generate source files during build time"""

import os
import shutil
import subprocess

from platform_methods import get_build_version, lipo


def generate_bundle(target, source, env):
    bin_dir = env.Dir("#bin").abspath

    if env.editor_build:
        # Editor bundle.
        prefix = "godot." + env["platform"] + "." + env["target"]
        if env.dev_build:
            prefix += ".dev"
        if env["precision"] == "double":
            prefix += ".double"

        # Lipo editor executable.
        target_bin = lipo(bin_dir + "/" + prefix, env.extra_suffix + env.module_version_string)

        # Assemble .app bundle and update version info.
        app_dir = env.Dir(
            "#bin/" + (prefix + env.extra_suffix + env.module_version_string).replace(".", "_") + ".app"
        ).abspath
        templ = env.Dir("#misc/dist/macos_tools.app").abspath
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
        if "mono" in env.module_version_string:
            shutil.copytree(env.Dir("#bin/GodotSharp").abspath, app_dir + "/Contents/Resources/GodotSharp")
        version = get_build_version(False)
        short_version = get_build_version(True)
        with open(env.Dir("#misc/dist/macos").abspath + "/editor_info_plist.template", "rt", encoding="utf-8") as fin:
            with open(app_dir + "/Contents/Info.plist", "wt", encoding="utf-8", newline="\n") as fout:
                for line in fin:
                    line = line.replace("$version", version)
                    line = line.replace("$short_version", short_version)
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
            if env.dev_build:
                sign_command += [env.Dir("#misc/dist/macos").abspath + "/editor_debug.entitlements"]
            else:
                sign_command += [env.Dir("#misc/dist/macos").abspath + "/editor.entitlements"]
            sign_command += [app_dir]
            subprocess.run(sign_command)

    else:
        # Template bundle.
        app_prefix = "godot." + env["platform"]
        rel_prefix = "godot." + env["platform"] + "." + "template_release"
        dbg_prefix = "godot." + env["platform"] + "." + "template_debug"
        if env.dev_build:
            app_prefix += ".dev"
            rel_prefix += ".dev"
            dbg_prefix += ".dev"
        if env["precision"] == "double":
            app_prefix += ".double"
            rel_prefix += ".double"
            dbg_prefix += ".double"

        # Lipo template executables.
        rel_target_bin = lipo(bin_dir + "/" + rel_prefix, env.extra_suffix + env.module_version_string)
        dbg_target_bin = lipo(bin_dir + "/" + dbg_prefix, env.extra_suffix + env.module_version_string)

        # Assemble .app bundle.
        app_dir = env.Dir("#bin/macos_template.app").abspath
        templ = env.Dir("#misc/dist/macos_template.app").abspath
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
        zip_dir = env.Dir(
            "#bin/" + (app_prefix + env.extra_suffix + env.module_version_string).replace(".", "_")
        ).abspath
        shutil.make_archive(zip_dir, "zip", root_dir=bin_dir, base_dir="macos_template.app")
        shutil.rmtree(app_dir)


def make_debug_macos(target, source, env):
    dst = str(target[0])
    if env["macports_clang"] != "no":
        mpprefix = os.environ.get("MACPORTS_PREFIX", "/opt/local")
        mpclangver = env["macports_clang"]
        os.system(mpprefix + "/libexec/llvm-" + mpclangver + "/bin/llvm-dsymutil {0} -o {0}.dSYM".format(dst))
    else:
        os.system("dsymutil {0} -o {0}.dSYM".format(dst))
    os.system("strip -u -r {0}".format(dst))
