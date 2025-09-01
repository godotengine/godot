def can_build(env, _platform):
    env.module_add_dependencies("opus", ["ogg"])
    return True

def configure(env):
    _ = env
    return


def get_doc_classes():
    return [
        "AudioStreamOggOpus",
        "AudioStreamPlaybackOggOpus",
        "ResourceImporterOggOpus",
    ]


def get_doc_path():
    return "doc_classes"