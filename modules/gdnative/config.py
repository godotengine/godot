def can_build(env: dict) -> bool:
    return True


def get_doc_classes() -> [str]:
    return [
        "XRInterfaceGDNative",
        "GDNative",
        "GDNativeLibrary",
        "MultiplayerPeerGDNative",
        "NativeScript",
        "PacketPeerGDNative",
        "PluginScript",
        "StreamPeerGDNative",
        "VideoStreamGDNative",
        "WebRTCPeerConnectionGDNative",
        "WebRTCDataChannelGDNative",
    ]


def get_doc_path() -> str:
    return "doc_classes"
