def can_build(env, platform):
    return True

def configure(env):
    env.use_ptrcall = True

def get_doc_classes():
    return [
        "ARVRInterfaceGDNative",
        "GDNative",
        "GDNativeLibrary",
        "MultiplayerPeerGDNative",
        "NativeScript",
        "PacketPeerGDNative",
        "PluginScript",
        "ResourceFormatLoaderVideoStreamGDNative",
        "StreamPeerGDNative",
        "VideoStreamGDNative",
    ]

def get_doc_path():
    return "doc_classes"
