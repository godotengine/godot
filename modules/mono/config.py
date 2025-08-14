def can_build(env, platform):
    if env["arch"].startswith("rv"):
        return False

    if env.editor_build:
        env.module_add_dependencies("mono", ["regex"])

    return True


def configure(env):
    # Check if the platform has marked mono as supported.
    supported = env.get("supported", [])
    if "mono" not in supported:
        import sys

        print("The 'mono' module does not currently support building for this platform. Aborting.")
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
