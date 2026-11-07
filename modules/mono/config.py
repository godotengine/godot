def can_build(env, platform):
    if "mono" not in env.get("supported", []):
        return False
    if env.editor_build:
        return platform in ["windows", "macos", "linuxbsd"]
    return True


def configure(env):
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
