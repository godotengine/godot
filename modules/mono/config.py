# Prior to .NET Core, we supported these: ["windows", "osx", "linuxbsd", "server", "android", "haiku", "javascript", "iphone"]
# Eventually support for each them should be added back (except Haiku if not supported by .NET Core)
supported_platforms = ["windows", "osx", "linuxbsd", "server"]


def can_build(env, platform):
    return env["module_gdnative_enabled"] and not env["arch"].startswith("rv")


def configure(env):
    platform = env["platform"]

    if platform not in supported_platforms:
        raise RuntimeError("This module does not currently support building for this platform")

    env.add_module_version_string("mono")

    from SCons.Script import PathVariable, Variables, Help

    envvars = Variables()
    envvars.Add(
        PathVariable(
            "dotnet_root",
            "Path to the .NET Sdk installation directory for the target platform and architecture",
            "",
            PathVariable.PathAccept,
        )
    )

    envvars.Update(env)
    Help(envvars.GenerateHelpText(env))


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
