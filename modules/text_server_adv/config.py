def can_build(env, platform):
    return True


def get_opts(platform):
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
