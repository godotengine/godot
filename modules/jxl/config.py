def can_build(env, platform):
    return True


def get_opts(platform):
    from SCons.Variables import BoolVariable

    return [
        BoolVariable(
            "jxl_all_simd_targets",
            "Build libjxl SIMD code for AVX-512, SVE and NEON BF16 too, not just SSE4/AVX2 and NEON (increases binary size)",
            False,
        ),
    ]


def configure(env):
    pass
