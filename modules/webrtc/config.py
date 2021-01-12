def can_build(env: dict) -> bool:
    return True


def get_doc_classes() -> [str]:
    return [
        "WebRTCPeerConnection",
        "WebRTCDataChannel",
        "WebRTCMultiplayer",
    ]


def get_doc_path() -> str:
    return "doc_classes"


def module_dependencies() -> dict:
    return {
        'gdnative': {'required': False}
    }
