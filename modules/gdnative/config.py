def can_build(env, platform):
    return True


def configure(env):
    env.use_ptrcall = True


def get_doc_classes():
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


def get_doc_path():
    return "doc_classes"
