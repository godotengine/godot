def can_build(env, platform):
    if platform in ['javascript']:
        return False # Not yet supported
    return True


def configure(env):
    env.use_ptrcall = True
    env.add_module_version_string('mono')

    from SCons.Script import BoolVariable, PathVariable, Variables

    envvars = Variables()
    envvars.Add(PathVariable('mono_prefix', 'Path to the mono installation directory for the target platform and architecture', '', PathVariable.PathAccept))
    envvars.Add(BoolVariable('mono_static', 'Statically link mono', False))
    envvars.Add(BoolVariable('mono_glue', 'Build with the mono glue sources', True))
    envvars.Add(BoolVariable('copy_mono_root', 'Make a copy of the mono installation directory to bundle with the editor', False))
    envvars.Add(BoolVariable('xbuild_fallback', 'If MSBuild is not found, fallback to xbuild', False))
    envvars.Update(env)


def is_enabled():
    # The module is disabled by default. Use module_mono_enabled=yes to enable it.
    return False
