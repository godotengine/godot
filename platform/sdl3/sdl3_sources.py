# Source list for the in-tree full SDL3, imported by platform/sdl3/SCsub.
#
# This lives here rather than in thirdparty/sdl3-full/ so the submodule stays
# pristine: git submodule update would otherwise wipe it.

import fnmatch
import os

from SCons.Errors import UserError

# Full SDL3, built as part of the engine for 2dog's `sdl3` platform.
# The reduced copy in thirdparty/sdl stays joypad-only for every other platform.
# Console builds replace this directory's contents with the platform's SDL fork
# and pass their own source list through env["sdl3_external_sources"].
#
# Sources are globbed the same way SDL's own CMakeLists.txt globs them, so this
# list does not rot every time SDL adds a file. The per-target groups below are
# the dirs CMakeLists.txt pulls in for Windows.

sdl_dir = "#thirdparty/sdl3-full/"
src = "src/"

# Global groups: built on every platform, with backend selection done inside the
# files by SDL_build_config.h.
common_dirs = [
    "",
    "atomic",
    "audio",
    "camera",
    "core",
    "cpuinfo",
    "dialog",
    "dynapi",
    "events",
    "filesystem",
    "gpu",
    "haptic",
    "hidapi",
    "io",
    "io/generic",
    "libm",
    "joystick",
    "locale",
    "main",
    "misc",
    "power",
    "render",
    "sensor",
    "stdlib",
    "storage",
    "thread",
    "tray",
    "process",
    "time",
    "timer",
    "video",
    "video/yuv2rgb",
]

windows_dirs = [
    "audio/directsound",
    "audio/disk",
    "audio/dummy",
    "audio/wasapi",
    "core/windows",
    "haptic/windows",
    "main/generic",
    "filesystem/windows",
    "io/windows",
    "joystick/gdk",
    "joystick/hidapi",
    "joystick/virtual",
    "joystick/windows",
    "loadso/windows",
    "locale/windows",
    "misc/windows",
    "power/windows",
    "sensor/windows",
    "process/windows",
    "storage/generic",
    "storage/steam",
    "thread/windows",
    "time/windows",
    "timer/windows",
    "tray/windows",
    "video/dummy",
    "video/offscreen",
    "video/windows",
    # Renderer backends this platform's SDL_build_config_windows.h enables.
    "render/direct3d",
    "render/direct3d11",
    "render/direct3d12",
    "render/gpu",
    "render/opengl",
    "render/opengles2",
    "render/software",
    "render/vulkan",
    "gpu/d3d12",
    "gpu/vulkan",
]


# CMake reaches into thread/generic for just these two: the Windows build takes
# mutex/semaphore/thread/TLS from thread/windows, and both halves define the same
# symbols. Building the whole generic directory would be a duplicate definition.
extra_sources = [
    "thread/generic/SDL_syscond.c",
    "thread/generic/SDL_sysrwlock.c",
]


def _glob_dirs(root_dir, dirs):
    out = []
    root = root_dir
    for d in dirs:
        path = os.path.join(root, *d.split("/")) if d else root
        if not os.path.isdir(path):
            continue
        for name in sorted(os.listdir(path)):
            if fnmatch.fnmatch(name, "*.c"):
                out.append(sdl_dir + src + (d + "/" if d else "") + name)
    return out


def build(env):
    """Returns (sources, env_sdl): the SDL sources to compile and the environment for them."""
    env_sdl = env.Clone()
    # /fp:strict stops MSVC constant-folding SDL's float lookup tables (C2099).
    env_sdl["CCFLAGS"] = [f for f in env_sdl["CCFLAGS"] if f != "/fp:strict"]

    if env.get("sdl3_host") == "windows":
        sources = _glob_dirs(env.Dir(sdl_dir + src).abspath, common_dirs + windows_dirs)
        sources += [sdl_dir + src + f for f in extra_sources]
        # SDL_MAIN_HANDLED: the engine owns main; SDL_main.h would otherwise redefine it.
        env_sdl.Append(CPPDEFINES=["SDL_PLATFORM_WINDOWS", "SDL_MAIN_HANDLED"])
    elif env.get("sdl3_host") == "linux":
        # Exactly the files SDL's CMake compiles for Linux with X11 + Vulkan (regenerate with
        # cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON); the per-backend choices live in linux/SDL_build_config.h.
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "linux", "sources.txt")) as f:
            sources = [sdl_dir + line.strip() for line in f if line.strip()]
        env_sdl.Append(CPPDEFINES=["SDL_MAIN_HANDLED", "USING_GENERATED_CONFIG_H", "_REENTRANT"])
        env_sdl.Append(CPPPATH=["/usr/include/dbus-1.0", "/usr/lib/x86_64-linux-gnu/dbus-1.0/include"])
    elif env.get("sdl3_external_sources"):
        sources = env["sdl3_external_sources"]
    else:
        raise UserError(
            "thirdparty/sdl3-full has no source list for this target. "
            "Set sdl3_host=windows or linux for desktop development, or point the build at a console SDL fork."
        )

    env_sdl.Prepend(
        CPPPATH=[
            sdl_dir + "include",
            sdl_dir + "include/build_config",
            sdl_dir + "src",
            sdl_dir + "src/video",
            sdl_dir + "src/video/khronos",
            sdl_dir + "src/events",
            sdl_dir + "src/joystick",
        ]
    )

    if env.get("sdl3_host") == "linux":
        # Must come before include/build_config, whose static SDL_build_config.h rejects generated builds.
        env_sdl.Prepend(CPPPATH=["#platform/sdl3/linux"])

    env_sdl.Append(CPPDEFINES=["SDL_BUILDING_LIBRARY"])
    # Audio, camera and dialogs are not built; SDL.c's init calls are guarded by these.
    env_sdl.Append(CPPDEFINES=["SDL_AUDIO_DISABLED", "SDL_CAMERA_DISABLED", "SDL_DIALOG_DISABLED"])
    return sources, env_sdl
