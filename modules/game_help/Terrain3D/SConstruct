#!/usr/bin/env python
from glob import glob
from pathlib import Path
import os

# TODO: Do not copy environment after godot-cpp/test is updated <https://github.com/godotengine/godot-cpp/blob/master/test/SConstruct>.
env = SConscript("godot-cpp/SConstruct")

# Add source files.
env.Append(CPPPATH=["src/"])
sources = Glob("src/*.cpp")

# Find gdextension path even if the directory or extension is renamed (e.g. project/addons/example/example.gdextension).
(extension_path,) = glob("project/addons/terrain_3d/*.gdextension")

# Find the addon path (e.g. project/addons/example).
addon_path = Path(extension_path).parent

# Find the project name from the gdextension file (e.g. example).
project_name = Path(extension_path).stem

scons_cache_path = os.environ.get("SCONS_CACHE")
if scons_cache_path != None:
    CacheDir(scons_cache_path)
    print("Scons cache enabled... (path: '" + scons_cache_path + "')")

# Create the library target (e.g. libexample.linux.debug.x86_64.so).
debug_or_release = "release" if env["target"] == "template_release" else "debug"
if env["platform"] == "macos":
    library = env.SharedLibrary(
        "{0}/bin/lib{1}.{2}.{3}.framework/{1}.{2}.{3}".format(
            addon_path,
            project_name,
            env["platform"],
            debug_or_release,
        ),
        source=sources,
    )
else:
    library = env.SharedLibrary(
        "{}/bin/lib{}.{}.{}.{}{}".format(
            addon_path,
            project_name,
            env["platform"],
            debug_or_release,
            env["arch"],
            env["SHLIBSUFFIX"],
        ),
        source=sources,
    )

## Option to use C++20 for this extension by replacing CXXFLAGS
#if env.get("is_msvc", False):
#    env.Replace(CXXFLAGS=["/std:c++20"])
#else:
#    env.Replace(CXXFLAGS=["-std=c++20"])

## Reenable CXXFLAGS removed by the above from godot-cpp/tools/godotcpp.py
# Disable exception handling. Godot doesn't use exceptions anywhere, and this
# saves around 20% of binary size and very significant build time.
#if env["disable_exceptions"]:
#    if env.get("is_msvc", False):
#        env.Append(CPPDEFINES=[("_HAS_EXCEPTIONS", 0)])
#    else:
#        env.Append(CXXFLAGS=["-fno-exceptions"])
#elif env.get("is_msvc", False):
#    env.Append(CXXFLAGS=["/EHsc"])

Default(library)
