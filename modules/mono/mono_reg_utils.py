import os

if os.name == 'nt':
    import sys
    if sys.version_info < (3,):
        import _winreg as winreg
    else:
        import winreg


def _reg_open_key(key, subkey):
    try:
        return winreg.OpenKey(key, subkey)
    except (WindowsError, EnvironmentError) as e:
        import platform
        if platform.architecture()[0] == '32bit':
            bitness_sam = winreg.KEY_WOW64_64KEY
        else:
            bitness_sam = winreg.KEY_WOW64_32KEY
        return winreg.OpenKey(key, subkey, 0, winreg.KEY_READ | bitness_sam)


def _find_mono_in_reg(subkey):
    try:
        with _reg_open_key(winreg.HKEY_LOCAL_MACHINE, subkey) as hKey:
            value, regtype = winreg.QueryValueEx(hKey, 'SdkInstallRoot')
            return value
    except (WindowsError, EnvironmentError) as e:
        return None

def _find_mono_in_reg_old(subkey):
    try:
        with _reg_open_key(winreg.HKEY_LOCAL_MACHINE, subkey) as hKey:
            default_clr, regtype = winreg.QueryValueEx(hKey, 'DefaultCLR')
            if default_clr:
                return _find_mono_in_reg(subkey + '\\' + default_clr)
            return None
    except (WindowsError, EnvironmentError):
        return None


def find_mono_root_dir():
    dir = _find_mono_in_reg(r'SOFTWARE\Mono')
    if dir:
        return dir
    dir = _find_mono_in_reg_old(r'SOFTWARE\Novell\Mono')
    if dir:
        return dir
    return None


def find_msbuild_tools_path_reg():
    try:
        with _reg_open_key(winreg.HKEY_LOCAL_MACHINE, r'SOFTWARE\Microsoft\MSBuild\ToolsVersions\4.0') as hKey:
            value, regtype = winreg.QueryValueEx(hKey, 'MSBuildToolsPath')
            return value
    except (WindowsError, EnvironmentError) as e:
        return None
