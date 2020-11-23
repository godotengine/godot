def can_build(env: dict) -> bool:
    return True


def get_doc_classes() -> [str]:
    return [
        "@GDScript",
        "GDScript",
    ]


def get_doc_path() -> str:
    return "doc_classes"


def module_dependencies() -> dict:
    return {
        'jsonrpc': {'required': False}
    }
