def can_build(env, platform):
    if platform == "web" and env["proxy_to_pthread"]:
        # WebXR is incompatible with proxy_to_pthread.
        return False

    # 2dog: web+mono is allowed; library_godot_webxr.js picks emscripten 3.1.56's Browser.mainLoop
    # (pinned by the .NET runtime pack) or 4.x's MainLoop at link time.
    return env["opengl3"] and not env["disable_xr"]


def configure(env):
    pass


def get_doc_classes():
    return ["WebXRInterface"]


def get_doc_path():
    return "doc_classes"
