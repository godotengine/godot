def can_build(env, platform):
    return True


def get_opts(platform):
    from SCons.Variables import BoolVariable

    return [
        BoolVariable(
            "etcpak_export_templates",
            "Enable S3TC and ETC image compression in export template builds (increases binary size)",
            False,
        ),
    ]


def configure(env):
    pass
