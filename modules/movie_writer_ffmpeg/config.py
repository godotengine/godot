def can_build(env, platform):
    # Editor and Desktop platforms where execute_with_pipe are available only.
    return env.editor_build and env["platform"] in ["linuxbsd", "macos", "windows"]


def configure(env):
    pass
