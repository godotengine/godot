def can_build(env, platform):
    return platform not in ['iphone']

def configure(env):
    pass

def get_doc_classes():
    return [
        "ResourceImporterWebm",
        "VideoStreamWebm",
    ]

def get_doc_path():
    return "doc_classes"
