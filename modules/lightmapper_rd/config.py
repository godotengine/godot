def can_build(env, platform):
    return (env.editor_build and platform not in ["android", "ios"]) or env["module_lightmapper_rd_enabled"]


def configure(env):
    from SCons.Script import BoolVariable, Variables, Help

    envvars = Variables()
    envvars.Add(
        BoolVariable(
            "lightmapper_rd",
            "Enable Lightmapper functionality in export template builds (increases binary size)",
            False,
        )
    )
    envvars.Update(env)
    Help(envvars.GenerateHelpText(env))
