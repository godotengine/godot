def can_build(env, platform):
    env.module_add_dependencies("opus", ["ogg"])
    return True


def configure(env):
    pass


def get_doc_classes():
    return [
        "AudioStreamOggOpus",
        "AudioStreamPlaybackOggOpus",
        "ResourceImporterOggOpus",
    ]


def get_doc_path():
    return "doc_classes"
