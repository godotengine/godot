def can_build(env, platform):
    return (env.editor_build or env["module_xatlas_unwrap_enabled"]) and platform not in ["android", "ios"]


def configure(env):
    pass
