"""SCons.Tool.wayland_scanner

SCons support for generating Wayland protocols with wayland-scanner.
"""

import SCons.Action
import SCons.Builder
import SCons.Util

class ToolWLScannerWarning(SCons.Warnings.Warning):
    pass

class WLScannerNotFound(ToolWLScannerWarning):
    pass

SCons.Warnings.enableWarningClass(ToolWLScannerWarning)

def _detect(env):
    try:
        return env['WAYLAND_SCANNER']
    except KeyError:
        pass

    wlscanner = env.WhereIs('wayland-scanner')
    if wlscanner:
        return wlscanner

    raise SCons.Errors.StopError(
            WLScannerNotFound,
            "Could not detect wayland-scanner")

suffixes = {
    'client-header': '.h',
    'server-header': '.h',
    'private-code': '.c',
    'public-code': '.c',
}

builders = dict()
actions = ['client-header', 'server-header', 'private-code', 'public-code']
for action in actions:
    builders[action] = SCons.Builder.Builder(
        action = SCons.Action.Action(
            '$WAYLAND_SCANNER {} $SOURCE $TARGET'.format(action),
            'Emitting Wayland {}'.format(action)),
        src_suffix = '.xml',
        suffix = suffixes[action],
        single_source = 1)

def WaylandScanner(env, action, target, source, **kwargs):
    result = builders[action].__call__(env, target, source, **kwargs)
    env.Clean(result, [str(source)[:-len(".xml")] + suffixes[action]])
    return [result]

_wl_client_header = SCons.Builder.Builder(
        action = SCons.Action.Action(
            '$WAYLAND_SCANNER client-header $SOURCE $TARGET',
            'Emitting Wayland client header'),
        src_suffix = '.xml',
        suffix = '.h',
        single_source = 1)

def exists(env):
    return _detect(env)

def generate(env):
    env['WAYLAND_SCANNER'] = _detect(env)
    env.AddMethod(WaylandScanner, 'WaylandScanner')
