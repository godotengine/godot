def can_build(env, platform):
    if platform == "web" and env["proxy_to_pthread"]:
        # WebXR is incompatible with proxy_to_pthread.
        return False

    if platform == "web" and env["module_mono_enabled"]:
        # NOTE: Remove when C# emscripten updates from 3.1.56.
        # Single-threaded build should support it.
        return False

    return env["opengl3"] and not env["disable_xr"]


def configure(env):
    pass


def get_doc_classes():
    return ["WebXRInterface"]


def get_doc_path():
    return "doc_classes"
