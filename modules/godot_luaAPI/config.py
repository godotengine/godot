def can_build(env, platform):
    return True

def configure(env):
    from SCons.Script import BoolVariable, EnumVariable, Variables, Help

    env_vars = Variables()

    env_vars.Add(BoolVariable("luaapi_luajit_build",
    "When LuaAPI is using luaJIT, be default it will attempt to build it automatically. if you prefer you can build it manually and disable auto building with this flag. Make sure to build statically and that the libs are in external/luaJIT/src",
    True))

    env_vars.Add(EnumVariable("luaapi_host_cc",
    "LuaJIT builds some tools to assist with the rest of the build. You can set the host CC to be used here in the case of cross compilation.", "gcc", ("gcc", "clang")))

    env_vars.Add(EnumVariable("luaapi_luaver",
    "Build the LuaAPI module with the following lua VM", "5.4", ("5.4", "5.1", "jit")))

    env_vars.Update(env)
    Help(env_vars.GenerateHelpText(env))

def get_doc_classes():
    return [
        "LuaAPI",
        "LuaCoroutine",
        "LuaError",
        "LuaTuple",
        "LuaCallableExtra",
        "LuaFunctionRef",
        "LuaObjectMetatable",
        "LuaDefaultObjectMetatable",
    ]

def get_doc_path():
    return "doc_classes"
