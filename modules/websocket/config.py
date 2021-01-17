def get_name() -> str:
    return 'websocket'
    
def can_build(env: dict) -> bool:
    return True


def get_doc_classes() -> [str]:
    return [
        "WebSocketClient",
        "WebSocketMultiplayerPeer",
        "WebSocketPeer",
        "WebSocketServer",
    ]


def get_doc_path() -> str:
    return "doc_classes"
