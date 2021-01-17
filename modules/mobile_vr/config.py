def get_name() -> str:
    return 'mobile_vr'
    
def can_build(env: dict) -> bool:
    return True


def get_doc_classes() -> [str]:
    return [
        "MobileVRInterface",
    ]


def get_doc_path() -> str:
    return "doc_classes"
