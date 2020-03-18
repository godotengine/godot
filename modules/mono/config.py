supported_platforms = ["windows", "osx", "linuxbsd", "server", "android", "haiku", "javascript", "iphone"]


def can_build(env, platform):
    return True


def configure(env):
    platform = env["platform"]

    if platform not in supported_platforms:
        raise RuntimeError("This module does not currently support building for this platform")

    env.use_ptrcall = True
    env.add_module_version_string("mono")

    from SCons.Script import BoolVariable, PathVariable, Variables, Help

    default_mono_static = platform in ["iphone", "javascript"]
    default_mono_bundles_zlib = platform in ["javascript"]

    envvars = Variables()
    envvars.Add(
        PathVariable(
            "mono_prefix",
            "Path to the mono installation directory for the target platform and architecture",
            "",
            PathVariable.PathAccept,
        )
    )
    envvars.Add(BoolVariable("mono_static", "Statically link mono", default_mono_static))
    envvars.Add(BoolVariable("mono_glue", "Build with the mono glue sources", True))
    envvars.Add(
        BoolVariable(
            "copy_mono_root", "Make a copy of the mono installation directory to bundle with the editor", False
        )
    )
    envvars.Add(BoolVariable("xbuild_fallback", "If MSBuild is not found, fallback to xbuild", False))

    # TODO: It would be great if this could be detected automatically instead
    envvars.Add(
        BoolVariable(
            "mono_bundles_zlib", "Specify if the Mono runtime was built with bundled zlib", default_mono_bundles_zlib
        )
    )

    envvars.Update(env)
    Help(envvars.GenerateHelpText(env))

    if env["mono_bundles_zlib"]:
        # Mono may come with zlib bundled for WASM or on newer version when built with MinGW.
        print("This Mono runtime comes with zlib bundled. Disabling 'builtin_zlib'...")
        env["builtin_zlib"] = False
        thirdparty_zlib_dir = "#thirdparty/zlib/"
        env.Prepend(CPPPATH=[thirdparty_zlib_dir])


def get_doc_classes():
    return [
        "@C#",
        "CSharpScript",
        "GodotSharp",
    ]


def get_doc_path():
    return "doc_classes"


def is_enabled():
    # The module is disabled by default. Use module_mono_enabled=yes to enable it.
    return False
