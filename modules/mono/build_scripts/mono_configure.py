import os
import os.path


def is_desktop(platform):
    return platform in ["windows", "osx", "linuxbsd", "server", "uwp", "haiku"]


def is_unix_like(platform):
    return platform in ["osx", "linuxbsd", "server", "android", "haiku", "iphone"]


def module_supports_tools_on(platform):
    return is_desktop(platform)


def configure(env, env_mono):
    # is_android = env["platform"] == "android"
    # is_javascript = env["platform"] == "javascript"
    # is_ios = env["platform"] == "iphone"
    # is_ios_sim = is_ios and env["arch"] in ["x86", "x86_64"]

    tools_enabled = env["tools"]

    if tools_enabled and not module_supports_tools_on(env["platform"]):
        raise RuntimeError("This module does not currently support building for this platform with tools enabled")

    if env["tools"]:
        env_mono.Append(CPPDEFINES=["GD_MONO_HOT_RELOAD"])

    dotnet_version = "5.0"

    dotnet_root = env["dotnet_root"]

    if not dotnet_root:
        dotnet_cmd = find_executable("dotnet")
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

    app_host_dir = os.path.join(
        dotnet_root,
        "packs",
        "Microsoft.NETCore.App.Host." + runtime_identifier,
        app_host_version,
        "runtimes",
        runtime_identifier,
        "native",
    )

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
            if env.msvc:
                env.Append(LINKFLAGS="libnethost.lib")
            else:
                env.Append(LINKFLAGS=["-Wl,-whole-archive", libnethost_path, "-Wl,-no-whole-archive"])
        else:
            is_apple = env["platform"] in ["osx", "iphone"]
            # is_macos = is_apple and not is_ios

            # if is_ios and not is_ios_sim:
            #     env_mono.Append(CPPDEFINES=["IOS_DEVICE"])

            if is_apple:
                env.Append(LINKFLAGS=["-Wl,-force_load," + libnethost_path])
            else:
                env.Append(LINKFLAGS=["-Wl,-whole-archive", libnethost_path, "-Wl,-no-whole-archive"])


def determine_runtime_identifier(env):
    names_map = {
        "windows": "win",
        "osx": "osx",
        "linuxbsd": "linux",
        "server": "linux",  # FIXME: Is server linux only, or also macos?
    }

    # architectures names: x86, x64, arm, or arm64

    platform = env["platform"]

    if is_desktop(platform):
        bits = env["bits"]
        bit_arch_map = {"64": "x64", "32": "x86"}
        return "%s-%s" % (names_map[platform], bit_arch_map[bits])
    else:
        raise NotImplementedError()


def find_app_host_version(dotnet_cmd, search_version_str):
    import subprocess
    from distutils.version import LooseVersion

    search_version = LooseVersion(search_version_str)

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
                return version_str
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


ENV_PATH_SEP = ";" if os.name == "nt" else ":"


def find_executable(name):
    is_windows = os.name == "nt"
    windows_exts = os.environ["PATHEXT"].split(ENV_PATH_SEP) if is_windows else None
    path_dirs = os.environ["PATH"].split(ENV_PATH_SEP)

    search_dirs = path_dirs + [os.getcwd()]  # cwd is last in the list

    for dir in search_dirs:
        path = os.path.join(dir, name)

        if is_windows:
            for extension in windows_exts:
                path_with_ext = path + extension

                if os.path.isfile(path_with_ext) and os.access(path_with_ext, os.X_OK):
                    return path_with_ext
        else:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path

    return ""
