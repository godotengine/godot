def can_build(env: dict) -> bool:
    return env.tools_enabled


def get_doc_classes() -> [str]:
    return [
        "EditorSceneImporterFBX",
    ]


def get_doc_path() -> str:
    return "doc_classes"
