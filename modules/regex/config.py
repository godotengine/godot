def can_build(env, platform):
    return not env["arch"].startswith("rv")


def configure(env):
    pass


def get_doc_classes():
    return [
        "RegEx",
        "RegExMatch",
    ]


def get_doc_path():
    return "doc_classes"
