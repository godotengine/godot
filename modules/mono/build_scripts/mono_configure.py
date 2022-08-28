import os
import os.path


def is_desktop(platform):
    return platform in ["windows", "macos", "linuxbsd", "uwp", "haiku"]


def is_unix_like(platform):
    return platform in ["macos", "linuxbsd", "android", "haiku", "ios"]


def module_supports_tools_on(platform):
    return is_desktop(platform)


def configure(env, env_mono):
    # is_android = env["platform"] == "android"
    # is_javascript = env["platform"] == "javascript"
    # is_ios = env["platform"] == "ios"
    # is_ios_sim = is_ios and env["arch"] in ["x86_32", "x86_64"]

    tools_enabled = env["tools"]

    if tools_enabled and not module_supports_tools_on(env["platform"]):
        raise RuntimeError("This module does not currently support building for this platform with tools enabled")

    if env["tools"]:
        env_mono.Append(CPPDEFINES=["GD_MONO_HOT_RELOAD"])

    app_host_dir = find_dotnet_app_host_dir(env)

    def check_app_host_file_exists(file):
        file_path = os.path.join(app_host_dir, file)
        if not os.path.isfile(file_path):
            raise RuntimeError("File not found: " + file_path)

    # TODO:
    # All libnethost does for us is provide a function to find hostfxr.
    # If we could handle that logic ourselves we could void linking it.

    # nethost file names:
    #   static: libnethost.a/lib
    #   shared: libnethost.a/dylib and nethost.dll
    check_app_host_file_exists("libnethost.lib" if os.name == "nt" else "libnethost.a")
    check_app_host_file_exists("nethost.h")
    check_app_host_file_exists("hostfxr.h")
    check_app_host_file_exists("coreclr_delegates.h")

    env_mono.Prepend(CPPPATH=app_host_dir)

    env.Append(LIBPATH=[app_host_dir])

    # Only the editor build  links nethost, which is needed to find hostfxr.
    # Exported games don't need this logic as hostfxr is bundled with them.
    if tools_enabled:
        libnethost_path = os.path.join(app_host_dir, "libnethost.lib" if os.name == "nt" else "libnethost.a")

        if env["platform"] == "windows":
            env_mono.Append(CPPDEFINES=["NETHOST_USE_AS_STATIC"])

            if env.msvc:
                env.Append(LINKFLAGS="libnethost.lib")
            else:
                env.Append(LINKFLAGS=["-Wl,-whole-archive", libnethost_path, "-Wl,-no-whole-archive"])
        else:
            is_apple = env["platform"] in ["macos", "ios"]
            # is_macos = is_apple and not is_ios

            # if is_ios and not is_ios_sim:
            #     env_mono.Append(CPPDEFINES=["IOS_DEVICE"])

            if is_apple:
                env.Append(LINKFLAGS=["-Wl,-force_load," + libnethost_path])
            else:
                env.Append(LINKFLAGS=["-Wl,-whole-archive", libnethost_path, "-Wl,-no-whole-archive"])


def find_dotnet_app_host_dir(env):
    dotnet_version = "6.0"

    dotnet_root = env["dotnet_root"]

    if not dotnet_root:
        dotnet_cmd = find_dotnet_executable(env["arch"])
        if dotnet_cmd:
            sdk_path = find_dotnet_sdk(dotnet_cmd, dotnet_version)
            if sdk_path:
                dotnet_root = os.path.abspath(os.path.join(sdk_path, os.pardir))

    if not dotnet_root:
        raise RuntimeError("Cannot find .NET Core Sdk")

    print("Found .NET Core Sdk root directory: " + dotnet_root)

    dotnet_cmd = os.path.join(dotnet_root, "dotnet.exe" if os.name == "nt" else "dotnet")

    runtime_identifier = determine_runtime_identifier(env)

    # TODO: In the future, if it can't be found this way, we want to obtain it
    # from the runtime.{runtime_identifier}.Microsoft.NETCore.DotNetAppHost NuGet package.
    app_host_version = find_app_host_version(dotnet_cmd, dotnet_version)
    if not app_host_version:
        raise RuntimeError("Cannot find .NET app host for version: " + dotnet_version)

    def get_runtime_path():
        return os.path.join(
            dotnet_root,
            "packs",
            "Microsoft.NETCore.App.Host." + runtime_identifier,
            app_host_version,
            "runtimes",
            runtime_identifier,
            "native",
        )

    app_host_dir = get_runtime_path()

    # Some Linux distros use their distro name as the RID in these paths.
    # If the initial generic path doesn't exist, try to get the RID from `dotnet --info`.
    # The generic RID should still be the first choice. Some platforms like Windows 10
    # define the RID as `win10-x64` but still use the generic `win-x64` for directory names.
    if not app_host_dir or not os.path.isdir(app_host_dir):
        runtime_identifier = find_dotnet_cli_rid(dotnet_cmd)
        app_host_dir = get_runtime_path()

    return app_host_dir


def determine_runtime_identifier(env):
    # The keys are Godot's names, the values are the Microsoft's names.
    # List: https://docs.microsoft.com/en-us/dotnet/core/rid-catalog
    names_map = {
        "windows": "win",
        "macos": "osx",
        "linuxbsd": "linux",
    }
    arch_map = {
        "x86_64": "x64",
        "x86_32": "x86",
        "arm64": "arm64",
        "arm32": "arm",
    }
    platform = env["platform"]
    if is_desktop(platform):
        return "%s-%s" % (names_map[platform], arch_map[env["arch"]])
    else:
        raise NotImplementedError()


def find_app_host_version(dotnet_cmd, search_version_str):
    import subprocess
    from distutils.version import LooseVersion

    search_version = LooseVersion(search_version_str)
    found_match = False

    try:
        env = dict(os.environ, DOTNET_CLI_UI_LANGUAGE="en-US")
        lines = subprocess.check_output([dotnet_cmd, "--list-runtimes"], env=env).splitlines()

        for line_bytes in lines:
            line = line_bytes.decode("utf-8")
            if not line.startswith("Microsoft.NETCore.App "):
                continue

            parts = line.split(" ", 2)
            if len(parts) < 3:
                continue

            version_str = parts[1]

            version = LooseVersion(version_str)

            if version >= search_version:
                search_version = version
                found_match = True
        if found_match:
            return str(search_version)
    except (subprocess.CalledProcessError, OSError) as e:
        import sys

        print(e, file=sys.stderr)

    return ""


def find_dotnet_arch(dotnet_cmd):
    import subprocess

    try:
        env = dict(os.environ, DOTNET_CLI_UI_LANGUAGE="en-US")
        lines = subprocess.check_output([dotnet_cmd, "--info"], env=env).splitlines()

        for line_bytes in lines:
            line = line_bytes.decode("utf-8")

            parts = line.split(":", 1)
            if len(parts) < 2:
                continue

            arch_str = parts[0].strip()
            if arch_str != "Architecture":
                continue

            arch_value = parts[1].strip()
            arch_map = {"x64": "x86_64", "x86": "x86_32", "arm64": "arm64", "arm32": "arm32"}
            return arch_map[arch_value]
    except (subprocess.CalledProcessError, OSError) as e:
        import sys

        print(e, file=sys.stderr)

    return ""


def find_dotnet_sdk(dotnet_cmd, search_version_str):
    import subprocess
    from distutils.version import LooseVersion

    search_version = LooseVersion(search_version_str)

    try:
        env = dict(os.environ, DOTNET_CLI_UI_LANGUAGE="en-US")
        lines = subprocess.check_output([dotnet_cmd, "--list-sdks"], env=env).splitlines()

        for line_bytes in lines:
            line = line_bytes.decode("utf-8")

            parts = line.split(" ", 1)
            if len(parts) < 2:
                continue

            version_str = parts[0]

            version = LooseVersion(version_str)

            if version < search_version:
                continue

            path_part = parts[1]
            return path_part[1 : path_part.find("]")]
    except (subprocess.CalledProcessError, OSError) as e:
        import sys

        print(e, file=sys.stderr)

    return ""


def find_dotnet_cli_rid(dotnet_cmd):
    import subprocess

    try:
        env = dict(os.environ, DOTNET_CLI_UI_LANGUAGE="en-US")
        lines = subprocess.check_output([dotnet_cmd, "--info"], env=env).splitlines()

        for line_bytes in lines:
            line = line_bytes.decode("utf-8")
            if not line.startswith(" RID:"):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            return parts[1]
    except (subprocess.CalledProcessError, OSError) as e:
        import sys

        print(e, file=sys.stderr)

    return ""


ENV_PATH_SEP = ";" if os.name == "nt" else ":"


def find_dotnet_executable(arch):
    is_windows = os.name == "nt"
    windows_exts = os.environ["PATHEXT"].split(ENV_PATH_SEP) if is_windows else None
    path_dirs = os.environ["PATH"].split(ENV_PATH_SEP)

    search_dirs = path_dirs + [os.getcwd()]  # cwd is last in the list

    for dir in path_dirs:
        search_dirs += [
            os.path.join(dir, "x64"),
            os.path.join(dir, "x86"),
            os.path.join(dir, "arm64"),
            os.path.join(dir, "arm32"),
        ]  # search subfolders for cross compiling

    # `dotnet --info` may not specify architecture. In such cases,
    # we fallback to the first one we find without architecture.
    sdk_path_unknown_arch = ""

    for dir in search_dirs:
        path = os.path.join(dir, "dotnet")

        if is_windows:
            for extension in windows_exts:
                path_with_ext = path + extension

                if os.path.isfile(path_with_ext) and os.access(path_with_ext, os.X_OK):
                    sdk_arch = find_dotnet_arch(path_with_ext)
                    if sdk_arch == arch or arch == "":
                        return path_with_ext
                    elif sdk_arch == "":
                        sdk_path_unknown_arch = path_with_ext
        else:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                sdk_arch = find_dotnet_arch(path)
                if sdk_arch == arch or arch == "":
                    return path
                elif sdk_arch == "":
                    sdk_path_unknown_arch = path

    return sdk_path_unknown_arch
