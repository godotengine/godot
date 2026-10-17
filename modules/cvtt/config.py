def can_build(env, platform):
    return env.editor_build or env["cvtt_export_templates"]


def get_opts(platform):
    from SCons.Variables import BoolVariable

    return [
        BoolVariable(
            "cvtt_export_templates",
            "Enable CVTT image compression in export template builds (increases binary size)",
            False,
        ),
    ]


def configure(env):
    pass
