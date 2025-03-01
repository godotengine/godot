# Tuples with the name of the arch
def get_platforms():
    return [("arm64", "arm64"), ("x86_64", "x86_64")]


def get_configurations():
    return ["editor", "template_debug", "template_release"]


# results are not needed on Mac and Linux
def get_build_prefix(env):
    return []
