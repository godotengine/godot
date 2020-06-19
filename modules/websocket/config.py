def can_build(env, platform):
    return True


def configure(env):
    pass


def get_doc_classes():
<<<<<<< HEAD
    return [
        "WebSocketClient",
        "WebSocketMultiplayerPeer",
        "WebSocketPeer",
        "WebSocketServer",
    ]
=======
    return ["WebSocketClient", "WebSocketMultiplayerPeer", "WebSocketPeer", "WebSocketServer"]

>>>>>>> master


def get_doc_path():
    return "doc_classes"
