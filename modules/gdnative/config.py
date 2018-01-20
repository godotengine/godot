def can_build(env, platform):
    return False

def configure(env):
    env.use_ptrcall = True

def get_doc_classes():
    return [
        "@NativeScript",
        "ARVRInterfaceGDNative",
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
