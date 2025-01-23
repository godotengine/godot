def can_build(env, platform):
    env.module_add_dependencies("vorbis", ["ogg"])
    return True


def configure(env):
    pass


def get_doc_classes():
    return [
        "AudioStreamOggVorbis",
        "AudioStreamPlaybackOggVorbis",
        "ResourceImporterOggVorbis",
    ]


def get_doc_path():
    return "doc_classes"
