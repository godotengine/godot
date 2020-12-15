# Required
def can_build(env: dict, platform: str) -> bool:
    return True


def get_name() -> str:
    return "some_module"


def get_doc_classes() -> [str]:
    return [
        "SomeModule",
    ]


def get_doc_path() -> str:
    return "doc_classes"


def module_dependencies() -> dict:
    return {"some_dependency": {"required": False}}
