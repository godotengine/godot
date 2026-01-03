# Tuples with the name of the arch
def get_platforms():
    return [("x64", "x86_64")]


def get_configurations():
    return ["editor", "template_debug", "template_release"]


def get_build_prefix(env):
    return []
