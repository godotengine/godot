
def can_build(platform):
    return True

def configure(env):
    env.use_ptrcall = True

def get_doc_classes():
  return ["GDNative", "GDNativeClass", "GDNativeLibrary", "NativeScript"]

def get_doc_path():
  return "doc_classes"