import methods


# Tuples with the name of the arch that will be used in VS, mapped to our internal arch names.
# For Windows platforms, Win32 is what VS wants. For other platforms, it can be different.
def get_platforms():
    return [("Win32", "x86_32"), ("x64", "x86_64")]


def get_configurations():
    return ["editor", "template_debug", "template_release"]


def get_build_prefix(env):
    if not env.msvc:
        return []
    batch_file = methods.find_visual_c_batch_file(env)
    return [
        "cmd /V /C",
        "set &quot;plat=$(PlatformTarget)&quot;",
        "^&amp; (if &quot;$(PlatformTarget)&quot;==&quot;x64&quot; (set &quot;plat=x86_amd64&quot;))",
        f"^&amp; call &quot;{batch_file}&quot; !plat!",
        "^&amp;",
    ]
