def can_build(env, platform):
    env.module_add_dependencies("svg", ["jpg", "webp"], True)
    return True


def configure(env):
    pass


def get_doc_classes():
    return [
        "ResourceImporterLottie",
    ]


def get_doc_path():
    return "doc_classes"
