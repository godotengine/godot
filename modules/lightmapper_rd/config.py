from __future__ import annotations


def can_build(env, platform):
    return env.editor_build and platform not in ["android", "ios"]


def configure(env):
    pass
