def can_build(env, platform):
    return True


def configure(env):
    pass


def get_doc_classes():
    return [
        "AIAgent",
        "GeminiClient",
        "AIConversation",
        "TypeScriptRunner",
        "AIAgentEditor",
    ]


def get_doc_path():
    return "doc_classes"


def get_opts(platform):
    from SCons.Variables import BoolVariable, PathVariable
    
    return [
        BoolVariable("builtin_typescript", "Use the built-in TypeScript compiler", True),
        PathVariable("typescript_path", "Path to TypeScript compiler (tsc)", "", PathVariable.PathAccept),
    ]


def configure_env(env):
    # Configure TypeScript support
    if env.get("builtin_typescript", True):
        env.Append(CPPDEFINES=["BUILTIN_TYPESCRIPT_ENABLED"])