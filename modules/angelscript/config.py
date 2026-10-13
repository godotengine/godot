def can_build(env, platform):
    return True


def get_opts(platform):
    from SCons.Variables import BoolVariable

    return [
        BoolVariable("angelscript_force_generic", "Force generic AngelScript value bindings for parity tests", False),
        BoolVariable("angelscript_pod_geometry_probe", "Experimental VM trivial geometry lifecycle (MSVC x64 only)", False),
    ]


def configure(env):
    pass
