def get_name() -> str:
    return 'stb_vorbis'
    
def can_build(env: dict) -> bool:
    return True


def get_doc_classes() -> [str]:
    return [
        "AudioStreamOGGVorbis",
    ]


def get_doc_path() -> str:
    return "doc_classes"
