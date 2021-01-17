def get_name() -> str:
    return 'minimp3'
    
def can_build(env: dict) -> bool:
    return True


def get_doc_classes() -> [str]:
    return [
        "AudioStreamMP3",
    ]


def get_doc_path() -> str:
    return "doc_classes"
