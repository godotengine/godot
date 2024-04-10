def can_build(env, platform):
    return env["platform"] in ["ios"]


def configure(env):
    if env["platform"] in ["ios"]:
        env.Append(LINKFLAGS=["-ObjC"])
        env.Append(CPPPATH=["#core"])
