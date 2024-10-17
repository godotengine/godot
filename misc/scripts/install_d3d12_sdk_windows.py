#!/usr/bin/env python

import argparse
import os
import platform
import shutil
import subprocess
import sys
import urllib.request


def setup(compiler: str):
    # Enable ANSI escape code support on Windows 10 and later (for colored console output).
    # <https://github.com/python/cpython/issues/73245>
    if sys.platform == "win32":
        from ctypes import byref, c_int, windll

        stdout_handle = windll.kernel32.GetStdHandle(c_int(-11))
        mode = c_int(0)
        windll.kernel32.GetConsoleMode(c_int(stdout_handle), byref(mode))
        mode = c_int(mode.value | 4)
        windll.kernel32.SetConsoleMode(c_int(stdout_handle), mode)

    # Base Godot dependencies path
    # If cross-compiling (no LOCALAPPDATA), we install in `bin`
    deps_folder = os.getenv("LOCALAPPDATA")
    if deps_folder:
        deps_folder = os.path.join(deps_folder, "Godot", "build_deps")
    else:
        deps_folder = os.path.join("bin", "build_deps")

    # Mesa NIR
    # Check for latest version: https://github.com/godotengine/godot-nir-static/releases/latest
    mesa_version = "23.1.9-1"
    mesa_archs = {
        "msvc": ["arm64", "x86_32", "x86_64"],
        "llvm": ["arm64", "x86_32", "x86_64"],
        "gcc": ["x86_32", "x86_64"],
    }
    mesa_filename = "godot-nir-static-%s-release.zip"
    mesa_folder = os.path.join(deps_folder, "mesa")
    # WinPixEventRuntime
    # Check for latest version: https://www.nuget.org/api/v2/package/WinPixEventRuntime (check downloaded filename)
    pix_version = "1.0.240308001"
    pix_archive = os.path.join(deps_folder, f"WinPixEventRuntime_{pix_version}.nupkg")
    pix_folder = os.path.join(deps_folder, "pix")
    # DirectX 12 Agility SDK
    # Check for latest version: https://www.nuget.org/api/v2/package/Microsoft.Direct3D.D3D12 (check downloaded filename)
    # After updating this, remember to change the default value of the `rendering/rendering_device/d3d12/agility_sdk_version`
    # project setting to match the minor version (e.g. for `1.613.3`, it should be `613`).
    agility_sdk_version = "1.613.3"
    agility_sdk_archive = os.path.join(deps_folder, f"Agility_SDK_{agility_sdk_version}.nupkg")
    agility_sdk_folder = os.path.join(deps_folder, "agility_sdk")

    # Create dependencies folder
    if not os.path.exists(deps_folder):
        os.makedirs(deps_folder)

    # Mesa NIR
    print("\x1b[1m[1/3] Mesa NIR\x1b[0m")
    if os.path.exists(mesa_folder):
        print(f"Removing existing local Mesa NIR installation in {mesa_folder} ...")
        shutil.rmtree(mesa_folder)

    for mesa_arch in mesa_archs[compiler]:
        filename = mesa_filename % f"{mesa_arch}-{compiler}"
        print(f"Downloading Mesa NIR {filename} ...")
        archive = os.path.join(deps_folder, filename)
        urllib.request.urlretrieve(
            f"https://github.com/godotengine/godot-nir-static/releases/download/{mesa_version}/{filename}",
            archive,
        )
        print(f"Extracting Mesa NIR {filename} to {mesa_folder} ...")
        shutil.unpack_archive(archive, mesa_folder)
        os.remove(archive)
    print("Mesa NIR installed successfully.\n")

    # WinPixEventRuntime

    # MinGW needs DLLs converted with dlltool.
    # We rely on finding gendef/dlltool to detect if we have MinGW.
    # Check existence of needed tools for generating mingw library.
    gendef = shutil.which("gendef") or ""
    dlltool = shutil.which("dlltool") or ""
    if dlltool == "":
        dlltool = shutil.which("x86_64-w64-mingw32-dlltool") or ""
    has_mingw = compiler != "msvc" and gendef != "" and dlltool != ""

    print("\x1b[1m[2/3] WinPixEventRuntime\x1b[0m")
    if os.path.isfile(pix_archive):
        os.remove(pix_archive)
    print(f"Downloading WinPixEventRuntime {pix_version} ...")
    urllib.request.urlretrieve(f"https://www.nuget.org/api/v2/package/WinPixEventRuntime/{pix_version}", pix_archive)
    if os.path.exists(pix_folder):
        print(f"Removing existing local WinPixEventRuntime installation in {pix_folder} ...")
        shutil.rmtree(pix_folder)
    print(f"Extracting WinPixEventRuntime {pix_version} to {pix_folder} ...")
    shutil.unpack_archive(pix_archive, pix_folder, "zip")
    os.remove(pix_archive)
    if has_mingw:
        print("Adapting WinPixEventRuntime to also support MinGW alongside MSVC.")
        cwd = os.getcwd()
        os.chdir(pix_folder)
        subprocess.run([gendef, "./bin/x64/WinPixEventRuntime.dll"])
        subprocess.run(
            [dlltool]
            + "--machine i386:x86-64 --no-leading-underscore -d WinPixEventRuntime.def -D WinPixEventRuntime.dll -l ./bin/x64/libWinPixEventRuntime.a".split()
        )
        subprocess.run([gendef, "./bin/ARM64/WinPixEventRuntime.dll"])
        subprocess.run(
            [dlltool]
            + "--machine arm64 --no-leading-underscore -d WinPixEventRuntime.def -D WinPixEventRuntime.dll -l ./bin/ARM64/libWinPixEventRuntime.a".split()
        )
        os.chdir(cwd)
    elif compiler != "msvc":
        print("MinGW wasn't found, so only MSVC support is provided for WinPixEventRuntime.")
    print(f"WinPixEventRuntime {pix_version} installed successfully.\n")

    # DirectX 12 Agility SDK
    print("\x1b[1m[3/3] DirectX 12 Agility SDK\x1b[0m")
    if os.path.isfile(agility_sdk_archive):
        os.remove(agility_sdk_archive)
    print(f"Downloading DirectX 12 Agility SDK {agility_sdk_version} ...")
    urllib.request.urlretrieve(
        f"https://www.nuget.org/api/v2/package/Microsoft.Direct3D.D3D12/{agility_sdk_version}", agility_sdk_archive
    )
    if os.path.exists(agility_sdk_folder):
        print(f"Removing existing local DirectX 12 Agility SDK installation in {agility_sdk_folder} ...")
        shutil.rmtree(agility_sdk_folder)
    print(f"Extracting DirectX 12 Agility SDK {agility_sdk_version} to {agility_sdk_folder} ...")
    shutil.unpack_archive(agility_sdk_archive, agility_sdk_folder, "zip")
    os.remove(agility_sdk_archive)
    print(f"DirectX 12 Agility SDK {agility_sdk_version} installed successfully.\n")

    # Complete message
    print(f'\x1b[92mAll Direct3D 12 SDK components were installed to "{deps_folder}" successfully!\x1b[0m')
    print('\x1b[92mYou can now build Godot with Direct3D 12 support enabled by running "scons d3d12=yes".\x1b[0m')


def detect_platform():
    system = platform.system()
    if system != "Windows":
        print("Current system is not Windows. Aborting cowardly.")
        exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Install d3d12 SDK for Windows.")
    parser.add_argument(
        "--compiler",
        help="sets the compiler used",
        type=str,
        choices=["msvc", "llvm", "gcc"],
        dest="compiler",
    )
    parser.set_defaults(compiler="msvc")

    args = parser.parse_args()

    detect_platform()
    setup(args.compiler)
