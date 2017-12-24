def can_build(platform):
    # Sorry guys, do not enable this unless you can figure out a way
    # to get Opus to not do any memory allocation or system calls
    # in the audio thread.
    # Currently the implementation even reads files from the audio thread,
    # and this is not how audio programming works.
    return False

def configure(env):
    pass

def get_doc_classes():
    return [
        "AudioStreamOpus",
    ]

def get_doc_path():
    return "doc_classes"
