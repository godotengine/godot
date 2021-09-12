# Prior to .NET Core, we supported these: ["windows", "macos", "linuxbsd", "server", "android", "haiku", "javascript", "ios"]
# Eventually support for each them should be added back (except Haiku if not supported by .NET Core)
supported_platforms = ["windows", "macos", "linuxbsd", "server"]


def can_build(env, platform):
    return not env["arch"].startswith("rv")


def get_opts(platform):
    from SCons.Variables import BoolVariable, PathVariable

    default_mono_static = platform in ["ios", "javascript"]
    default_mono_bundles_zlib = platform in ["javascript"]

    return [
        PathVariable(
            "dotnet_root",
            "Path to the .NET Sdk installation directory for the target platform and architecture",
            "",
            PathVariable.PathAccept,
        ),
    ]


def configure(env):
    platform = env["platform"]

    if platform not in supported_platforms:
        raise RuntimeError("This module does not currently support building for this platform")

    env.add_module_version_string("mono")


def get_doc_classes():
    return [
        "CSharpScript",
        "GodotSharp",
    ]


def get_doc_path():
    return "doc_classes"


def is_enabled():
    # The module is disabled by default. Use module_mono_enabled=yes to enable it.
    return False
