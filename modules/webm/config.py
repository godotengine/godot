def can_build(env: dict) -> bool:
    return env.platform == 'iphone'


def get_doc_classes() -> [str]:
    return [
        "VideoStreamWebm",
    ]


def get_doc_path() -> str:
    return "doc_classes"
