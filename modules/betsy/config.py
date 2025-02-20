def can_build(env, platform):
    return (env.editor_build or env["betsy_export_templates"]) and env["rendering_device"]


def get_opts(platform):
    from SCons.Variables import BoolVariable

    return [
        BoolVariable(
            "betsy_export_templates",
            "Enable Betsy image compression in export template builds (increases binary size)",
            False,
        ),
    ]


def configure(env):
    pass
