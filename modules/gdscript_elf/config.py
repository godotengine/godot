def can_build(env, platform):
    # This module depends on gdscript and sandbox modules
    env.module_add_dependencies("gdscript_elf", ["gdscript", "sandbox"], True)
    return True


def configure(env):
    pass


def get_doc_classes():
    return [
        "GDScriptLanguageWrapper",
        "GDScriptWrapper",
        "GDScriptFunctionWrapper",
    ]


def get_doc_path():
    return "doc_classes"

