def can_build(env, platform):
    return True


def configure(env):
    import sys

    # Check if the platform has marked mono as supported.
    supported = env.get("supported", [])
    if "mono" not in supported:
        print("The 'mono' module does not currently support building for this platform. Aborting.")
        sys.exit(255)
    if env["platform"] == "web" and env["library_type"] != "static_library":
        print("The 'mono' module only supports \"library_type=static_library\" on web. Aborting.")
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
