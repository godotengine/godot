def can_build(env, platform):
    return True


def configure(env):
    pass


def is_enabled():
    # The module is disabled by default. Use module_text_server_fb_enabled=yes to enable it.
    return False


def get_doc_classes():
    return [
        "TextServerFallback",
    ]


def get_doc_path():
    return "doc_classes"
