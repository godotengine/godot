def can_build(env, platform):
    return env.editor_build or env["tinyexr_export_templates"]


def configure(env):
    pass
