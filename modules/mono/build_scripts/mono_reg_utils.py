import os
import platform

from compat import decode_utf8

if os.name == "nt":
    import sys

    if sys.version_info < (3,):
        import _winreg as winreg
    else:
        import winreg


def _reg_open_key(key, subkey):
    try:
        return winreg.OpenKey(key, subkey)
    except (WindowsError, OSError):
        if platform.architecture()[0] == "32bit":
            bitness_sam = winreg.KEY_WOW64_64KEY
        else:
            bitness_sam = winreg.KEY_WOW64_32KEY
        return winreg.OpenKey(key, subkey, 0, winreg.KEY_READ | bitness_sam)


def _reg_open_key_bits(key, subkey, bits):
    sam = winreg.KEY_READ

    if platform.architecture()[0] == "32bit":
        if bits == "64":
            # Force 32bit process to search in 64bit registry
            sam |= winreg.KEY_WOW64_64KEY
    else:
        if bits == "32":
            # Force 64bit process to search in 32bit registry
            sam |= winreg.KEY_WOW64_32KEY

    return winreg.OpenKey(key, subkey, 0, sam)


def _find_mono_in_reg(subkey, bits):
    try:
        with _reg_open_key_bits(winreg.HKEY_LOCAL_MACHINE, subkey, bits) as hKey:
            value = winreg.QueryValueEx(hKey, "SdkInstallRoot")[0]
            return value
    except (WindowsError, OSError):
        return None


def _find_mono_in_reg_old(subkey, bits):
    try:
        with _reg_open_key_bits(winreg.HKEY_LOCAL_MACHINE, subkey, bits) as hKey:
            default_clr = winreg.QueryValueEx(hKey, "DefaultCLR")[0]
            if default_clr:
                return _find_mono_in_reg(subkey + "\\" + default_clr, bits)
            return None
    except (WindowsError, EnvironmentError):
        return None


def find_mono_root_dir(bits):
    root_dir = _find_mono_in_reg(r"SOFTWARE\Mono", bits)
    if root_dir is not None:
        return str(root_dir)
    root_dir = _find_mono_in_reg_old(r"SOFTWARE\Novell\Mono", bits)
    if root_dir is not None:
        return str(root_dir)
    return ""


def find_msbuild_tools_path_reg():
    import subprocess

    vswhere = os.getenv("PROGRAMFILES(X86)")
    if not vswhere:
        vswhere = os.getenv("PROGRAMFILES")
    vswhere += r"\Microsoft Visual Studio\Installer\vswhere.exe"

    vswhere_args = ["-latest", "-products", "*", "-requires", "Microsoft.Component.MSBuild"]

    try:
        lines = subprocess.check_output([vswhere] + vswhere_args).splitlines()

        for line in lines:
            parts = decode_utf8(line).split(":", 1)

            if len(parts) < 2 or parts[0] != "installationPath":
                continue

            val = parts[1].strip()

            if not val:
                raise ValueError("Value of `installationPath` entry is empty")

            # Since VS2019, the directory is simply named "Current"
            msbuild_dir = os.path.join(val, "MSBuild\\Current\\Bin")
            if os.path.isdir(msbuild_dir):
                return msbuild_dir

            # Directory name "15.0" is used in VS 2017
            return os.path.join(val, "MSBuild\\15.0\\Bin")

        raise ValueError("Cannot find `installationPath` entry")
    except ValueError as e:
        print("Error reading output from vswhere: " + e.message)
    except WindowsError:
        pass  # Fine, vswhere not found
    except (subprocess.CalledProcessError, OSError):
        pass

    # Try to find 14.0 in the Registry

    try:
        subkey = r"SOFTWARE\Microsoft\MSBuild\ToolsVersions\14.0"
        with _reg_open_key(winreg.HKEY_LOCAL_MACHINE, subkey) as hKey:
            value = winreg.QueryValueEx(hKey, "MSBuildToolsPath")[0]
            return value
    except (WindowsError, OSError):
        return ""
