import sys

from methods import print_error


def can_build(env, platform):
    if env.editor_build:
        env.module_add_dependencies("mono", ["regex"])

    return True


def configure(env):
    # Check if the platform has marked mono as supported.
    supported = env.get("supported", [])
    if "mono" not in supported:
        print("The 'mono' module does not currently support building for this platform. Aborting.")
        sys.exit(255)

    if env["library_type"] != "executable" and not env["disable_crash_handler"]:
        print_error(".NET installs its own crash handler.")
        sys.exit(255)

    if env["platform"] == "web":
        if env["library_type"] == "executable":
            print_error(".NET needs to be an entry point on web.")
            sys.exit(255)

        if env["library_type"] == "shared_library":
            print_error("Can't build .NET with MAIN_MODULE, which would have made it possible to load shared library.")
            sys.exit(255)

        if env["threads"] and not env["proxy_to_pthread"]:
            print_error(
                '.NET runtime moves to worker thread when multi-threading is enabled, because of this godot needs to be compiled with "proxy_to_pthread" support.'
            )
            # https://github.com/dotnet/runtime/issues/126438.
            sys.exit(255)

        if env["lto"] != "none":
            print_error(".NET can't work with lto library on web.")
            sys.exit(255)

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
