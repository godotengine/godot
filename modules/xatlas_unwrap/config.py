def can_build(env, platform):
    return env.editor_build and platform not in ["android", "ios"] or env["module_xatlas_unwrap_enabled"]


def configure(env):
    from SCons.Script import BoolVariable, Variables, Help

    envvars = Variables()
    envvars.Add(
        BoolVariable(
            "xatlas_unwrap",
            "Enable xatlas unwrapping functionality in export template builds (increases binary size)",
            False,
        )
    )
    envvars.Update(env)
    Help(envvars.GenerateHelpText(env))
