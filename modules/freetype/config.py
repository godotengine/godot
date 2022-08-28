def can_build(env, platform):
    return True


def get_opts(platform):
    from SCons.Variables import BoolVariable

    return [
        BoolVariable("brotli", "Enable Brotli decompressor for WOFF2 fonts support", True),
    ]


def configure(env):
    pass
