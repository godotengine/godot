def can_build(env, platform):
    return env.editor_build or env["tinyexr_export_templates"]


def configure(env):
    from SCons.Script import BoolVariable, Variables, Help

    envvars = Variables()
    envvars.Add(
        BoolVariable(
            "tinyexr_export_templates",
            "Enable saving and loading OpenEXR images in export template builds (increases binary size)",
            False,
        )
    )
    envvars.Update(env)
    Help(envvars.GenerateHelpText(env))
