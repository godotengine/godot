def can_build(env, platform):
    # should probably change this to only be true on iOS and Android
    return False

def configure(env):
    pass

def get_doc_classes():
    return [
        "MobileVRInterface",
    ]

def get_doc_path():
    return "doc_classes"
