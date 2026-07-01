import os
import platform
import shutil
import subprocess
import sys

from methods import print_error, print_warning

# NOTE: The multiprocessing module is not compatible with SCons due to conflict on cPickle


compatibility_platform_aliases = {
    "osx": "macos",
    "iphone": "ios",
    "x11": "linuxbsd",
    "javascript": "web",
}

# CPU architecture options.
architectures = ["x86_32", "x86_64", "arm32", "arm64", "rv64", "ppc64", "wasm32", "wasm64", "loongarch64"]
architecture_aliases = {
    "x86": "x86_32",
    "x64": "x86_64",
    "amd64": "x86_64",
    "armv7": "arm32",
    "armv8": "arm64",
    "arm64v8": "arm64",
    "aarch64": "arm64",
    "rv": "rv64",
    "riscv": "rv64",
    "riscv64": "rv64",
    "ppc64le": "ppc64",
    "loong64": "loongarch64",
}


def detect_arch():
    host_machine = platform.machine().lower()
    if host_machine in architectures:
        return host_machine
    elif host_machine in architecture_aliases.keys():
        return architecture_aliases[host_machine]
    elif "86" in host_machine:
        # Catches x86, i386, i486, i586, i686, etc.
        return "x86_32"
    else:
        print_warning(f'Unsupported CPU architecture: "{host_machine}". Falling back to x86_64.')
        return "x86_64"


def validate_arch(arch, platform_name, supported_arches):
    if arch not in supported_arches:
        print_error(
            'Unsupported CPU architecture "%s" for %s. Supported architectures are: %s.'
            % (arch, platform_name, ", ".join(supported_arches))
        )
        sys.exit(255)


def get_build_version(short):
    import version

    name = "custom_build"
    if os.getenv("BUILD_NAME") is not None:
        name = os.getenv("BUILD_NAME")
    v = "%d.%d" % (version.major, version.minor)
    if version.patch > 0:
        v += ".%d" % version.patch
    status = version.status
    if not short:
        if os.getenv("GODOT_VERSION_STATUS") is not None:
            status = str(os.getenv("GODOT_VERSION_STATUS"))
        v += ".%s.%s" % (status, name)
    return v


def lipo(prefix, suffix):
    from pathlib import Path

    target_bin = ""
    lipo_command = ["lipo", "-create"]
    arch_found = 0

    for arch in architectures:
        bin_name = prefix + "." + arch + suffix
        if Path(bin_name).is_file():
            target_bin = bin_name
            lipo_command += [bin_name]
            arch_found += 1

    if arch_found > 1:
        target_bin = prefix + ".fat" + suffix
        lipo_command += ["-output", target_bin]
        subprocess.run(lipo_command)

    return target_bin


def get_mvk_sdk_path(osname):
    def int_or_zero(i):
        try:
            return int(i)
        except (TypeError, ValueError):
            return 0

    def ver_parse(a):
        return [int_or_zero(i) for i in a.split(".")]

    dirname = os.path.expanduser("~/VulkanSDK")
    if not os.path.exists(dirname):
        return ""

    ver_min = ver_parse("1.3.231.0")
    ver_num = ver_parse("0.0.0.0")
    files = os.listdir(dirname)
    lib_name_out = dirname
    for file in files:
        if os.path.isdir(os.path.join(dirname, file)):
            ver_comp = ver_parse(file)
            if ver_comp > ver_num and ver_comp >= ver_min:
                # Try new SDK location.
                lib_name = os.path.join(os.path.join(dirname, file), "macOS/lib/MoltenVK.xcframework/" + osname + "/")
                if os.path.isfile(os.path.join(lib_name, "libMoltenVK.a")):
                    ver_num = ver_comp
                    lib_name_out = os.path.join(os.path.join(dirname, file), "macOS/lib/MoltenVK.xcframework")
                else:
                    # Try old SDK location.
                    lib_name = os.path.join(
                        os.path.join(dirname, file), "MoltenVK/MoltenVK.xcframework/" + osname + "/"
                    )
                    if os.path.isfile(os.path.join(lib_name, "libMoltenVK.a")):
                        ver_num = ver_comp
                        lib_name_out = os.path.join(os.path.join(dirname, file), "MoltenVK/MoltenVK.xcframework")

    return lib_name_out


def detect_mvk(env, osname):
    mvk_list = [
        get_mvk_sdk_path(osname),
        "/opt/homebrew/Frameworks/MoltenVK.xcframework",
        "/usr/local/homebrew/Frameworks/MoltenVK.xcframework",
        "/opt/local/Frameworks/MoltenVK.xcframework",
    ]
    if env["vulkan_sdk_path"] != "":
        mvk_list.insert(0, os.path.expanduser(env["vulkan_sdk_path"]))
        mvk_list.insert(
            0,
            os.path.join(os.path.expanduser(env["vulkan_sdk_path"]), "macOS/lib/MoltenVK.xcframework"),
        )
        mvk_list.insert(
            0,
            os.path.join(os.path.expanduser(env["vulkan_sdk_path"]), "MoltenVK/MoltenVK.xcframework"),
        )

    for mvk_path in mvk_list:
        if mvk_path and os.path.isfile(os.path.join(mvk_path, f"{osname}/libMoltenVK.a")):
            print(f"MoltenVK found at: {mvk_path}")
            return mvk_path

    return ""


def combine_libs_apple_embedded(target, source, env):
    lib_path = target[0].srcnode().abspath
    if "osxcross" in env:
        libtool = "$APPLE_TOOLCHAIN_PATH/usr/bin/${apple_target_triple}libtool"
    else:
        libtool = "$APPLE_TOOLCHAIN_PATH/usr/bin/libtool"
    env.Execute(
        libtool + ' -static -o "' + lib_path + '" ' + " ".join([('"' + lib.srcnode().abspath + '"') for lib in source])
    )


def lipo_and_copy_apple_embedded(
    platform, framework_dir, framework_dir_sim, rel_prefix, dbg_prefix, module_prefix, app_dir, env
):
    bin_dir = env.Dir("#bin").abspath

    # Lipo template libraries.
    #
    # env.extra_suffix contains ".simulator" when building for simulator,
    # but it's undesired when calling lipo()
    extra_suffix = env.extra_suffix.replace(".simulator", "")
    rel_target_bin = lipo(bin_dir + "/libgodot" + module_prefix + "." + rel_prefix, extra_suffix + ".a")
    dbg_target_bin = lipo(bin_dir + "/libgodot" + module_prefix + "." + dbg_prefix, extra_suffix + ".a")
    rel_target_bin_sim = lipo(
        bin_dir + "/libgodot" + module_prefix + "." + rel_prefix, ".simulator" + extra_suffix + ".a"
    )
    dbg_target_bin_sim = lipo(
        bin_dir + "/libgodot" + module_prefix + "." + dbg_prefix, ".simulator" + extra_suffix + ".a"
    )
    # Assemble Xcode project bundle.
    if rel_target_bin != "":
        print(f' Copying "{platform}" release framework')
        shutil.copy(
            rel_target_bin,
            app_dir
            + "/libgodot"
            + module_prefix
            + "."
            + platform
            + ".release.xcframework/"
            + framework_dir
            + "/libgodot"
            + module_prefix
            + ".a",
        )
    if dbg_target_bin != "":
        print(f' Copying "{platform}" debug framework')
        shutil.copy(
            dbg_target_bin,
            app_dir
            + "/libgodot"
            + module_prefix
            + "."
            + platform
            + ".debug.xcframework/"
            + framework_dir
            + "/libgodot"
            + module_prefix
            + ".a",
        )
    if rel_target_bin_sim != "":
        print(f' Copying "{platform}" (simulator) release framework')
        shutil.copy(
            rel_target_bin_sim,
            app_dir
            + "/libgodot"
            + module_prefix
            + "."
            + platform
            + ".release.xcframework/"
            + framework_dir_sim
            + "/libgodot"
            + module_prefix
            + ".a",
        )
    if dbg_target_bin_sim != "":
        print(f' Copying "{platform}" (simulator) debug framework')
        shutil.copy(
            dbg_target_bin_sim,
            app_dir
            + "/libgodot"
            + module_prefix
            + "."
            + platform
            + ".debug.xcframework/"
            + framework_dir_sim
            + "/libgodot"
            + module_prefix
            + ".a",
        )


def generate_bundle_apple_embedded(platform, framework_dir, framework_dir_sim, use_mkv, target, source, env):
    # Template bundle.
    extra_suffix = env.extra_suffix.replace(".simulator", "")
    app_prefix = "godot." + platform
    rel_prefix = platform + "." + "template_release"
    dbg_prefix = platform + "." + "template_debug"
    if env.dev_build:
        app_prefix += ".dev"
        rel_prefix += ".dev"
        dbg_prefix += ".dev"
    if env["precision"] == "double":
        app_prefix += ".double"
        rel_prefix += ".double"
        dbg_prefix += ".double"

    app_dir = env.Dir("#bin/" + platform + "_xcode").abspath
    templ = env.Dir("#misc/dist/apple_embedded_xcode").abspath
    if os.path.exists(app_dir):
        shutil.rmtree(app_dir)
    shutil.copytree(templ, app_dir)

    lipo_and_copy_apple_embedded(platform, framework_dir, framework_dir_sim, rel_prefix, dbg_prefix, "", app_dir, env)
    if "MODULES_EXTERNAL" in env:
        for mod in env["MODULES_EXTERNAL"]:
            lipo_and_copy_apple_embedded(
                platform, framework_dir, framework_dir_sim, rel_prefix, dbg_prefix, mod, app_dir, env
            )

    # Remove other platform xcframeworks
    for entry in os.listdir(app_dir):
        if (entry.startswith("libgodot.") or entry.startswith("libgodot_")) and entry.endswith(".xcframework"):
            parts = entry.split(".")
            if len(parts) >= 3 and parts[1] != platform:
                full_path = os.path.join(app_dir, entry)
                shutil.rmtree(full_path)

    if use_mkv:
        mvk_path = detect_mvk(env, "ios-arm64")
        if mvk_path != "":
            shutil.copytree(mvk_path + "/ios-arm64", app_dir + "/MoltenVK.xcframework/ios-arm64")
            shutil.copytree(
                mvk_path + "/ios-arm64_x86_64-simulator", app_dir + "/MoltenVK.xcframework/ios-arm64_x86_64-simulator"
            )
            shutil.copy(mvk_path + "/Info.plist", app_dir + "/MoltenVK.xcframework/Info.plist")

    # ZIP Xcode project bundle.
    zip_dir = env.Dir("#bin/" + (app_prefix + extra_suffix).replace(".", "_")).abspath
    shutil.make_archive(zip_dir, "zip", root_dir=app_dir)
    shutil.rmtree(app_dir)


def setup_swift_builder(env, apple_platform, sdk_path, current_path, bridging_header_filename, all_swift_files):
    from SCons.Script import Action, Builder

    if apple_platform == "macos":
        target_suffix = "macosx10.9"

    elif apple_platform == "ios":
        target_suffix = "ios14.0"  # iOS 14.0 needed for SwiftUI lifecycle

    elif apple_platform == "iossimulator":
        target_suffix = "ios14.0-simulator"  # iOS 14.0 needed for SwiftUI lifecycle

    elif apple_platform == "visionos":
        target_suffix = "xros26.0"

    elif apple_platform == "visionossimulator":
        target_suffix = "xros26.0-simulator"

    else:
        raise Exception("Invalid platform argument passed to detect_darwin_sdk_path")

    swiftc_target = env["arch"] + "-apple-" + target_suffix

    env["ALL_SWIFT_FILES"] = all_swift_files
    env["CURRENT_PATH"] = current_path
    if "SWIFT_FRONTEND" in env and env["SWIFT_FRONTEND"] != "":
        frontend_path = env["SWIFT_FRONTEND"]
    elif "osxcross" not in env:
        frontend_path = "$APPLE_TOOLCHAIN_PATH/usr/bin/swift-frontend"
    else:
        frontend_path = None

    if frontend_path is None:
        raise Exception("Swift frontend path is not set. Please set SWIFT_FRONTEND.")

    bridging_header_path = current_path + "/" + bridging_header_filename
    env["SWIFTC"] = frontend_path + " -frontend -c"  # Swift compiler
    env["SWIFTCFLAGS"] = [
        "-cxx-interoperability-mode=default",
        "-emit-object",
        "-target",
        swiftc_target,
        "-sdk",
        sdk_path,
        "-import-objc-header",
        bridging_header_path,
        "-swift-version",
        "6",
        "-parse-as-library",
        "-module-name",
        "godot_swift_module",
        "-I./",  # Pass the current directory as the header root so bridging headers can include files from any point of the hierarchy
    ]

    if "osxcross" in env:
        env.Append(
            SWIFTCFLAGS=[
                "-resource-dir",
                "/root/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/swift",
            ]
        )

    if env["debug_symbols"]:
        env.Append(SWIFTCFLAGS=["-g"])

    if env["optimize"] in ["speed", "speed_trace"]:
        env.Append(SWIFTCFLAGS=["-O"])

    elif env["optimize"] == "size":
        env.Append(SWIFTCFLAGS=["-Osize"])

    elif env["optimize"] in ["debug", "none"]:
        env.Append(SWIFTCFLAGS=["-Onone"])

    def generate_swift_action(source, target, env, for_signature):
        fullpath_swift_files = [env["CURRENT_PATH"] + "/" + file for file in env["ALL_SWIFT_FILES"]]
        fullpath_swift_files.remove(source[0].abspath)

        fullpath_swift_files_string = '"' + '" "'.join(fullpath_swift_files) + '"'
        compile_command = "$SWIFTC " + fullpath_swift_files_string + " -primary-file $SOURCE -o $TARGET $SWIFTCFLAGS"

        swift_comdstr = env.get("SWIFTCOMSTR")
        if swift_comdstr is not None:
            swift_action = Action(compile_command, cmdstr=swift_comdstr)
        else:
            swift_action = Action(compile_command)

        return swift_action

    from methods import redirect_emitter

    # Define Builder for Swift files
    swift_builder = Builder(
        generator=generate_swift_action, suffix=env["OBJSUFFIX"], src_suffix=".swift", emitter=redirect_emitter
    )

    env.Append(BUILDERS={"Swift": swift_builder})
    env["BUILDERS"]["Library"].add_src_builder("Swift")
    env["BUILDERS"]["Object"].add_action(".swift", Action(generate_swift_action, generator=1))


# Custom Visual Studio project generation logic that supports any platform that has a msvs.py
# script, so Visual Studio can be used to run scons for any platform, with the right defines per target.
# Invoked with scons vsproj=yes
#
# Only platforms that opt in to vs proj generation by having a msvs.py file in the platform folder are included.
# Platforms with a msvs.py file will be added to the solution, but only the current active platform+target+arch
# will have a build configuration generated, because we only know what the right defines/includes/flags/etc are
# on the active build target.
#
# Platforms that don't support an editor target will have a dummy editor target that won't do anything on build,
# but will have the files and configuration for the windows editor target.
#
# To generate build configuration files for all platforms+targets+arch combinations, users can call
#   scons vsproj=yes
# for each combination of platform+target+arch. This will generate the relevant vs project files but
# skip the build process. This lets project files be quickly generated even if there are build errors.
#
# To generate AND build from the command line:
#   scons vsproj=yes vsproj_gen_only=no
def generate_vs_project(env, original_args, project_name="godot"):
    import glob
    import re

    # Augmented glob_recursive that also fills the dirs argument with traversed directories that have content.
    def glob_recursive_2(pattern, dirs, node="."):
        from SCons import Node
        from SCons.Script import Glob

        results = []
        for f in Glob(str(node) + "/*", source=True):
            if type(f) is Node.FS.Dir:
                results += glob_recursive_2(pattern, dirs, f)
        r = Glob(str(node) + "/" + pattern, source=True)
        if len(r) > 0 and str(node) not in dirs:
            d = ""
            for part in str(node).split("\\"):
                d += part
                if d not in dirs:
                    dirs.append(d)
                d += "\\"
        results += r
        return results

    def get_bool(args, option, default):
        from SCons.Variables.BoolVariable import _text2bool

        val = args.get(option, default)
        if val is not None:
            try:
                return _text2bool(val)
            except (ValueError, AttributeError):
                return default
        else:
            return default

    def format_key_value(v):
        if type(v) in [tuple, list]:
            return v[0] if len(v) == 1 else f"{v[0]}={v[1]}"
        return v

    def get_dependencies(file, env, exts, headers, sources, others):
        for child in file.children():
            if isinstance(child, str):
                child = env.File(x)
            fname = ""
            try:
                fname = child.path
            except AttributeError:
                # It's not a file.
                pass

            if fname:
                parts = os.path.splitext(fname)
                if len(parts) > 1:
                    ext = parts[1].lower()
                    if ext in exts["sources"]:
                        sources += [fname]
                    elif ext in exts["headers"]:
                        headers += [fname]
                    elif ext in exts["others"]:
                        others += [fname]

            get_dependencies(child, env, exts, headers, sources, others)

    def get_default_include_paths(env):
        if env.msvc:
            return []
        compiler = env.subst("$CXX")
        target = os.path.join(env.Dir("#main").abspath, "main.cpp")
        args = [compiler, target, "-x", "c++", "-v"]
        ret = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        output = ret.stdout
        match = re.search(r"#include <\.\.\.> search starts here:([\S\s]*)End of search list.", output)
        if not match:
            print_warning("Failed to find the include paths in the compiler output.")
            return []
        return [x.strip() for x in match[1].strip().splitlines()]

    filtered_args = original_args.copy()

    # Ignore the "vsproj" option to not regenerate the VS project on every build
    filtered_args.pop("vsproj", None)

    # This flag allows users to regenerate the proj files but skip the building process.
    # This lets projects be regenerated even if there are build errors.
    filtered_args.pop("vsproj_gen_only", None)

    # This flag allows users to regenerate only the props file without touching the sln or vcxproj files.
    # This preserves any customizations users have done to the solution, while still updating the file list
    # and build commands.
    filtered_args.pop("vsproj_props_only", None)

    # The "progress" option is ignored as the current compilation progress indication doesn't work in VS
    filtered_args.pop("progress", None)

    # We add these three manually because they might not be explicitly passed in, and it's important to always set them.
    filtered_args.pop("platform", None)
    filtered_args.pop("target", None)
    filtered_args.pop("arch", None)

    platform = env["platform"]
    target = env["target"]
    arch = env["arch"]
    host_arch = detect_arch()

    host_platform = "windows"
    if (
        sys.platform.startswith("linux")
        or sys.platform.startswith("dragonfly")
        or sys.platform.startswith("freebsd")
        or sys.platform.startswith("netbsd")
        or sys.platform.startswith("openbsd")
    ):
        host_platform = "linuxbsd"
    elif sys.platform == "darwin":
        host_platform = "macos"

    vs_configuration = {}
    host_vs_configuration = {}
    common_build_prefix = []
    confs = []
    for x in sorted(glob.glob("platform/*")):
        # Only platforms that opt in to vs proj generation are included.
        if not os.path.isdir(x) or not os.path.exists(x + "/msvs.py"):
            continue
        tmppath = "./" + x
        sys.path.insert(0, tmppath)
        import msvs

        vs_plats = []
        vs_confs = []
        try:
            platform_name = x[9:]
            vs_plats = msvs.get_platforms()
            vs_confs = msvs.get_configurations()
            val = []
            for plat in vs_plats:
                val += [{"platform": plat[0], "architecture": plat[1]}]

            vsconf = {"platform": platform_name, "targets": vs_confs, "arches": val}
            confs += [vsconf]

            # Save additional information about the configuration for the actively selected platform,
            # so we can generate the platform-specific props file with all the build commands/defines/etc
            if platform == platform_name:
                common_build_prefix = msvs.get_build_prefix(env)
                vs_configuration = vsconf
            if platform_name == host_platform:
                host_vs_configuration = vsconf
                for a in vsconf["arches"]:
                    if host_arch == a["architecture"]:
                        host_arch = a["platform"]
                        break
        except Exception:
            pass

        sys.path.remove(tmppath)
        sys.modules.pop("msvs")

    extensions = {}
    extensions["headers"] = [".h", ".hh", ".hpp", ".hxx", ".inc", ".inl"]
    extensions["sources"] = [".c", ".cc", ".cpp", ".cxx", ".m", ".mm", ".java"]
    extensions["others"] = [".natvis", ".glsl", ".rc"]

    headers = []
    headers_dirs = []
    for ext in extensions["headers"]:
        for file in glob_recursive_2("*" + ext, headers_dirs):
            headers.append(str(file).replace("/", "\\"))

    sources = []
    sources_dirs = []
    for ext in extensions["sources"]:
        for file in glob_recursive_2("*" + ext, sources_dirs):
            sources.append(str(file).replace("/", "\\"))

    others = []
    others_dirs = []
    for ext in extensions["others"]:
        for file in glob_recursive_2("*" + ext, others_dirs):
            others.append(str(file).replace("/", "\\"))

    skip_filters = False
    import hashlib
    import json

    md5 = hashlib.md5(
        json.dumps(sorted(headers + headers_dirs + sources + sources_dirs + others + others_dirs)).encode("utf-8")
    ).hexdigest()

    if os.path.exists(f"{project_name}.vcxproj.filters"):
        with open(f"{project_name}.vcxproj.filters", "r", encoding="utf-8") as file:
            existing_filters = file.read()
        match = re.search(r"(?ms)^<!-- CHECKSUM$.([0-9a-f]{32})", existing_filters)
        if match is not None and md5 == match.group(1):
            skip_filters = True

    import uuid

    # Don't regenerate the filters file if nothing has changed, so we keep the existing UUIDs.
    if not skip_filters:
        print(f"Regenerating {project_name}.vcxproj.filters")

        with open("misc/msvs/vcxproj.filters.template", "r", encoding="utf-8") as file:
            filters_template = file.read()
        for i in range(1, 10):
            filters_template = filters_template.replace(f"%%UUID{i}%%", str(uuid.uuid4()))

        filters = ""

        for d in headers_dirs:
            filters += f'<Filter Include="Header Files\\{d}"><UniqueIdentifier>{{{str(uuid.uuid4())}}}</UniqueIdentifier></Filter>\n'
        for d in sources_dirs:
            filters += f'<Filter Include="Source Files\\{d}"><UniqueIdentifier>{{{str(uuid.uuid4())}}}</UniqueIdentifier></Filter>\n'
        for d in others_dirs:
            filters += f'<Filter Include="Other Files\\{d}"><UniqueIdentifier>{{{str(uuid.uuid4())}}}</UniqueIdentifier></Filter>\n'

        filters_template = filters_template.replace("%%FILTERS%%", filters)

        filters = ""
        for file in headers:
            filters += (
                f'<ClInclude Include="{file}"><Filter>Header Files\\{os.path.dirname(file)}</Filter></ClInclude>\n'
            )
        filters_template = filters_template.replace("%%INCLUDES%%", filters)

        filters = ""
        for file in sources:
            filters += (
                f'<ClCompile Include="{file}"><Filter>Source Files\\{os.path.dirname(file)}</Filter></ClCompile>\n'
            )

        filters_template = filters_template.replace("%%COMPILES%%", filters)

        filters = ""
        for file in others:
            filters += f'<None Include="{file}"><Filter>Other Files\\{os.path.dirname(file)}</Filter></None>\n'
        filters_template = filters_template.replace("%%OTHERS%%", filters)

        filters_template = filters_template.replace("%%HASH%%", md5)

        with open(f"{project_name}.vcxproj.filters", "w", encoding="utf-8", newline="\r\n") as f:
            f.write(filters_template)

    headers_active = []
    sources_active = []
    others_active = []

    get_dependencies(
        env.File(f"#bin/godot{env['PROGSUFFIX']}"), env, extensions, headers_active, sources_active, others_active
    )

    all_items = []
    properties = []
    activeItems = []
    extraItems = []

    set_headers = set(headers_active)
    set_sources = set(sources_active)
    set_others = set(others_active)
    for file in headers:
        base_path = os.path.dirname(file).replace("\\", "_")
        all_items.append(f'<ClInclude Include="{file}">')
        all_items.append(
            f"  <ExcludedFromBuild Condition=\"!$(ActiveProjectItemList_{base_path}.Contains(';{file};'))\">true</ExcludedFromBuild>"
        )
        all_items.append("</ClInclude>")
        if file in set_headers:
            activeItems.append(file)

    for file in sources:
        base_path = os.path.dirname(file).replace("\\", "_")
        all_items.append(f'<ClCompile Include="{file}">')
        all_items.append(
            f"  <ExcludedFromBuild Condition=\"!$(ActiveProjectItemList_{base_path}.Contains(';{file};'))\">true</ExcludedFromBuild>"
        )
        all_items.append("</ClCompile>")
        if file in set_sources:
            activeItems.append(file)

    for file in others:
        base_path = os.path.dirname(file).replace("\\", "_")
        all_items.append(f'<None Include="{file}">')
        all_items.append(
            f"  <ExcludedFromBuild Condition=\"!$(ActiveProjectItemList_{base_path}.Contains(';{file};'))\">true</ExcludedFromBuild>"
        )
        all_items.append("</None>")
        if file in set_others:
            activeItems.append(file)

    if vs_configuration:
        vsconf = ""
        for a in vs_configuration["arches"]:
            if arch == a["architecture"]:
                vsconf = f"{target}|{a['platform']}"
                break

        condition = "'$(GodotConfiguration)|$(GodotPlatform)'=='" + vsconf + "'"
        itemlist = {}
        for item in activeItems:
            key = os.path.dirname(item).replace("\\", "_")
            if key not in itemlist:
                itemlist[key] = [item]
            else:
                itemlist[key] += [item]

        for x in itemlist.keys():
            properties.append(
                "<ActiveProjectItemList_%s>;%s;</ActiveProjectItemList_%s>" % (x, ";".join(itemlist[x]), x)
            )
        output = os.path.join("bin", f"godot{env['PROGSUFFIX']}")

        # The modules_enabled.gen.h header containing the defines is only generated on build, and only for the most recently built
        # platform, which means VS can't properly render code that's inside module-specific ifdefs. This adds those defines to the
        # platform-specific VS props file, so that VS knows which defines are enabled for the selected platform.
        env.Append(VSHINT_DEFINES=[f"MODULE_{module.upper()}_ENABLED" for module in env.module_list])

        with open("misc/msvs/props.template", "r", encoding="utf-8") as file:
            props_template = file.read()

        props_template = props_template.replace("%%CONDITION%%", condition)
        props_template = props_template.replace("%%PROPERTIES%%", "\n    ".join(properties))
        props_template = props_template.replace("%%EXTRA_ITEMS%%", "\n    ".join(extraItems))

        props_template = props_template.replace("%%OUTPUT%%", output)

        proplist = [format_key_value(j) for j in list(env["CPPDEFINES"])]
        proplist += [format_key_value(j) for j in env.get("VSHINT_DEFINES", [])]
        props_template = props_template.replace("%%DEFINES%%", ";".join(proplist))

        proplist = [str(j) for j in env["CPPPATH"]]
        proplist += [str(j) for j in env.get("VSHINT_INCLUDES", [])]
        proplist += [str(j) for j in get_default_include_paths(env)]
        props_template = props_template.replace("%%INCLUDES%%", ";".join(proplist))

        proplist = [env.subst("$CCFLAGS")]
        proplist += [env.subst("$CXXFLAGS")]
        proplist += [env.subst("$VSHINT_OPTIONS")]
        props_template = props_template.replace("%%OPTIONS%%", " ".join(proplist))

        # Windows allows us to have spaces in paths, so we need
        # to double quote off the directory. However, the path ends
        # in a backslash, so we need to remove this, lest it escape the
        # last double quote off, confusing MSBuild
        common_build_postfix = [
            "--directory=&quot;$(ProjectDir.TrimEnd(&apos;\\&apos;))&quot;",
            "progress=no",
            f"platform={platform}",
            f"target={target}",
            f"arch={arch}",
        ]

        for arg, value in filtered_args.items():
            common_build_postfix.append(f"{arg}={value}")

        cmd_rebuild = [
            "vsproj=yes",
            "vsproj_props_only=yes",
            "vsproj_gen_only=no",
            f"vsproj_name={project_name}",
        ] + common_build_postfix

        cmd_clean = [
            "--clean",
        ] + common_build_postfix

        commands = "scons"
        if len(common_build_prefix) == 0:
            commands = "echo Starting SCons &amp; " + commands
        else:
            common_build_prefix[0] = "echo Starting SCons &amp; " + common_build_prefix[0]

        cmd = " ".join(common_build_prefix + [" ".join([commands] + common_build_postfix)])
        props_template = props_template.replace("%%BUILD%%", cmd)

        cmd = " ".join(common_build_prefix + [" ".join([commands] + cmd_rebuild)])
        props_template = props_template.replace("%%REBUILD%%", cmd)

        cmd = " ".join(common_build_prefix + [" ".join([commands] + cmd_clean)])
        props_template = props_template.replace("%%CLEAN%%", cmd)

        with open(
            f"{project_name}.{platform}.{target}.{arch}.generated.props", "w", encoding="utf-8", newline="\r\n"
        ) as f:
            f.write(props_template)

    proj_uuid = str(uuid.uuid4())
    sln_uuid = str(uuid.uuid4())

    if os.path.exists(f"{project_name}.sln"):
        for line in open(f"{project_name}.sln", "r", encoding="utf-8").read().splitlines():
            if line.startswith('Project("{8BC9CEB8-8B4A-11D0-8D11-00A0C91BC942}")'):
                proj_uuid = re.search(
                    r"\"{(\b[0-9a-fA-F]{8}\b-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-\b[0-9a-fA-F]{12}\b)}\"$",
                    line,
                ).group(1)
            elif line.strip().startswith("SolutionGuid ="):
                sln_uuid = re.search(
                    r"{(\b[0-9a-fA-F]{8}\b-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-\b[0-9a-fA-F]{12}\b)}", line
                ).group(1)
                break

    configurations = []
    imports = []
    properties = []
    section1 = []
    section2 = []
    for conf in confs:
        godot_platform = conf["platform"]
        has_editor = "editor" in conf["targets"]

        # Skip any platforms that can build the editor and don't match the host platform.
        #
        # When both Windows and Mac define an editor target, it's defined as platform+target+arch (windows+editor+x64 for example).
        # VS only supports two attributes, a "Configuration" and a "Platform", and we currently map our target to the Configuration
        # (i.e. editor/template_debug/template_release), and our architecture to the "Platform" (i.e. x64, arm64, etc).
        # Those two are not enough to disambiguate multiple godot targets for different godot platforms with the same architecture,
        # i.e. editor|x64 would currently match both windows editor intel 64 and linux editor intel 64.
        #
        # TODO: More work is needed in order to support generating VS projects that unambiguously support all platform+target+arch variations.
        # The VS "Platform" has to be a known architecture that VS recognizes, so we can only play around with the "Configuration" part of the combo.
        if has_editor and godot_platform != host_vs_configuration["platform"]:
            continue

        for p in conf["arches"]:
            sln_plat = p["platform"]
            proj_plat = sln_plat
            godot_arch = p["architecture"]

            # Redirect editor configurations for platforms that don't support the editor target to the default editor target on the
            # active host platform, so the solution has all the permutations and VS doesn't complain about missing project configurations.
            # These configurations are disabled, so they show up but won't build.
            if not has_editor:
                section1 += [f"editor|{sln_plat} = editor|{proj_plat}"]
                section2 += [f"{{{proj_uuid}}}.editor|{proj_plat}.ActiveCfg = editor|{host_arch}"]

                configurations += [
                    f'<ProjectConfiguration Include="editor|{proj_plat}">',
                    "  <Configuration>editor</Configuration>",
                    f"  <Platform>{proj_plat}</Platform>",
                    "</ProjectConfiguration>",
                ]

                properties += [
                    f"<PropertyGroup Condition=\"'$(Configuration)|$(Platform)'=='editor|{proj_plat}'\">",
                    "  <GodotConfiguration>editor</GodotConfiguration>",
                    f"  <GodotPlatform>{proj_plat}</GodotPlatform>",
                    "</PropertyGroup>",
                ]

            for t in conf["targets"]:
                godot_target = t

                # Windows x86 is a special little flower that requires a project platform == Win32 but a solution platform == x86.
                if godot_platform == "windows" and godot_target == "editor" and godot_arch == "x86_32":
                    sln_plat = "x86"

                configurations += [
                    f'<ProjectConfiguration Include="{godot_target}|{proj_plat}">',
                    f"  <Configuration>{godot_target}</Configuration>",
                    f"  <Platform>{proj_plat}</Platform>",
                    "</ProjectConfiguration>",
                ]

                properties += [
                    f"<PropertyGroup Condition=\"'$(Configuration)|$(Platform)'=='{godot_target}|{proj_plat}'\">",
                    f"  <GodotConfiguration>{godot_target}</GodotConfiguration>",
                    f"  <GodotPlatform>{proj_plat}</GodotPlatform>",
                    "</PropertyGroup>",
                ]

                p = f"{project_name}.{godot_platform}.{godot_target}.{godot_arch}.generated.props"
                imports += [
                    f'<Import Project="$(MSBuildProjectDirectory)\\{p}" Condition="Exists(\'$(MSBuildProjectDirectory)\\{p}\')"/>'
                ]

                section1 += [f"{godot_target}|{sln_plat} = {godot_target}|{sln_plat}"]

                section2 += [
                    f"{{{proj_uuid}}}.{godot_target}|{sln_plat}.ActiveCfg = {godot_target}|{proj_plat}",
                    f"{{{proj_uuid}}}.{godot_target}|{sln_plat}.Build.0 = {godot_target}|{proj_plat}",
                ]

    # Add an extra import for a local user props file at the end, so users can add more overrides.
    imports += [
        f'<Import Project="$(MSBuildProjectDirectory)\\{project_name}.vs.user.props" Condition="Exists(\'$(MSBuildProjectDirectory)\\{project_name}.vs.user.props\')"/>'
    ]
    section1 = sorted(section1)
    section2 = sorted(section2)

    if not get_bool(original_args, "vsproj_props_only", False):
        with open("misc/msvs/vcxproj.template", "r", encoding="utf-8") as file:
            proj_template = file.read()
        proj_template = proj_template.replace("%%UUID%%", proj_uuid)
        proj_template = proj_template.replace("%%CONFS%%", "\n    ".join(configurations))
        proj_template = proj_template.replace("%%IMPORTS%%", "\n  ".join(imports))
        proj_template = proj_template.replace("%%DEFAULT_ITEMS%%", "\n    ".join(all_items))
        proj_template = proj_template.replace("%%PROPERTIES%%", "\n  ".join(properties))

        with open(f"{project_name}.vcxproj", "w", encoding="utf-8", newline="\r\n") as f:
            f.write(proj_template)

    if not get_bool(original_args, "vsproj_props_only", False):
        with open("misc/msvs/sln.template", "r", encoding="utf-8") as file:
            sln_template = file.read()
        sln_template = sln_template.replace("%%NAME%%", project_name)
        sln_template = sln_template.replace("%%UUID%%", proj_uuid)
        sln_template = sln_template.replace("%%SLNUUID%%", sln_uuid)
        sln_template = sln_template.replace("%%SECTION1%%", "\n\t\t".join(section1))
        sln_template = sln_template.replace("%%SECTION2%%", "\n\t\t".join(section2))

        with open(f"{project_name}.sln", "w", encoding="utf-8", newline="\r\n") as f:
            f.write(sln_template)

    if get_bool(original_args, "vsproj_gen_only", True):
        sys.exit()
