def can_build(env, platform):
    if env["arch"].startswith("rv"):
        return False
    if platform == "iphone":
        return False
    # Can work in theory but our libvpx/SCsub is too broken to compile NEON .s
    # files properly on Linux arm32. Could be fixed by someone motivated.
    if platform in ["x11", "server"] and env["arch"] in ["arm", "arm32"]:
        return False
    return True


def configure(env):
    pass


def get_doc_classes():
    return [
        "VideoStreamWebm",
    ]


def get_doc_path():
    return "doc_classes"
