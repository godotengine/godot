def can_build(env, platform):
    return True


def configure(env):
    from SCons.Script import BoolVariable, Help, Variables

    env_vars = Variables()
    env_vars.Add(BoolVariable("lottie", "Enable Lottie support using thorvg", True))

    env_vars.Update(env)
    Help(env_vars.GenerateHelpText(env))
