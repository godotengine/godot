def can_build(env, platform):
    if platform in ['javascript']:
        return False # Not yet supported
    return True


def configure(env):
    env.use_ptrcall = True
    env.add_module_version_string('mono')


def get_doc_classes():
    return [
        '@C#',
        'CSharpScript',
        'GodotSharp',
    ]


def get_doc_path():
    return 'doc_classes'


def is_enabled():
    # The module is disabled by default. Use module_mono_enabled=yes to enable it.
    return False
