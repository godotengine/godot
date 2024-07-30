def can_build(env, platform):
    # Godot only uses it in the editor, but ANGLE depends on it and we had
    # to remove the copy from prebuilt ANGLE libs to solve symbol clashes.
    return env.editor_build or env.get("angle_libs")


def configure(env):
    pass
