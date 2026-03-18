from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from SCons.Script.SConscript import SConsEnvironment


def can_build(env, platform):
    env.module_add_dependencies("text_server_adv", ["freetype", "msdfgen", "svg"], True)
    return True


def get_opts(env: "SConsEnvironment"):
    from SCons.Variables import BoolVariable

    return [
        BoolVariable("graphite", "Enable SIL Graphite smart fonts support", True),
    ]


def configure(env):
    pass


def get_doc_classes():
    return [
        "TextServerAdvanced",
    ]


def get_doc_path():
    return "doc_classes"
