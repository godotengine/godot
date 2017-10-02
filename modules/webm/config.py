def can_build(platform):
    return platform != 'iphone'

def configure(env):
    pass

def get_doc_classes():
    return [
        "ResourceImporterWebm",
        "VideoStreamWebm",
    ]

def get_doc_path():
    return "doc_classes"
