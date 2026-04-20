def can_build(env, platform):
    if platform == "web":
        # WebXR is incompatible with proxy_to_pthread.
        return not env["proxy_to_pthread"]

    return env["opengl3"] and not env["disable_xr"]


def configure(env):
    pass


def get_doc_classes():
    return ["WebXRInterface"]


def get_doc_path():
    return "doc_classes"
