from __future__ import annotations


def can_build(env, platform):
    env.module_add_dependencies("basis_universal", ["jpg"])
    return True


def configure(env):
    pass
