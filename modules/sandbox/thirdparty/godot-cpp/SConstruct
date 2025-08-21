#!/usr/bin/env python

import os
import sys

# Add godot-cpp folder to sys.path, so that we can import local modules.
sys.path.append(Dir(".").srcnode().abspath)


EnsureSConsVersion(4, 0)
EnsurePythonVersion(3, 8)

try:
    Import("env")
except Exception:
    # Default tools with no platform defaults to gnu toolchain.
    # We apply platform specific toolchains via our custom tools.
    env = Environment(tools=["default"], PLATFORM="")

env.PrependENVPath("PATH", os.getenv("PATH"))

# Custom options and profile flags.
customs = ["custom.py"]
try:
    customs += Import("customs")
except Exception:
    pass
profile = ARGUMENTS.get("profile", "")
if profile:
    if os.path.isfile(profile):
        customs.append(profile)
    elif os.path.isfile(profile + ".py"):
        customs.append(profile + ".py")
opts = Variables(customs, ARGUMENTS)
cpp_tool = Tool("godotcpp", toolpath=[Dir("tools").srcnode().abspath])
cpp_tool.options(opts, env)
opts.Update(env)

Help(opts.GenerateHelpText(env))

# Detect and print a warning listing unknown SCons variables to ease troubleshooting.
unknown = opts.UnknownVariables()
if unknown:
    print("WARNING: Unknown SCons variables were passed and will be ignored:")
    for item in unknown.items():
        print("    " + item[0] + "=" + item[1])

scons_cache_path = os.environ.get("SCONS_CACHE")
if scons_cache_path is not None:
    CacheDir(scons_cache_path)
    Decider("MD5")

cpp_tool.generate(env)
library = env.GodotCPP()

Return("env")
