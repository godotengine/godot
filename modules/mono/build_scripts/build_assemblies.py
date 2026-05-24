"""
modules/mono/build_scripts/build_assemblies.py
Builds the managed (.NET) assemblies that ship with Godot.

Change for issue #70796:
  When env_mono["MONO_PUBLISH_MSBUILD_PROPS"] is set (populated by
  mono_configure.py for the web platform), those extra MSBuild properties
  are appended to every `dotnet publish` command so that WasmEnableThreads
  (and any future per-platform props) are forwarded correctly.
"""

import os
import subprocess
import sys

from SCons.Script import *


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_dotnet(args, env=None):
    """
    Run a `dotnet` sub-command, printing the command first.
    Raises subprocess.CalledProcessError on failure.
    """
    print("Running:", " ".join(str(a) for a in args))
    subprocess.check_call(args, env=env)


def _get_publish_extra_props(env_mono):
    """
    Return the list of extra MSBuild /p: flags injected by mono_configure.py.
    Returns an empty list when none are set (all platforms except web for now).
    """
    return env_mono.get("MONO_PUBLISH_MSBUILD_PROPS", [])


# ---------------------------------------------------------------------------
# Public SCons builder action
# ---------------------------------------------------------------------------

def build_assemblies(env, env_mono):
    """
    Entry point called from modules/mono/SCsub.
    Drives `dotnet build` / `dotnet publish` for GodotSharp and GodotTools.
    """
    dotnet = env_mono.get("MONO_DOTNET_CLI", "dotnet")
    if not env_mono.get("MONO_DOTNET_FOUND", False):
        print("Skipping managed assembly build: dotnet not found.")
        return

    extra_props = _get_publish_extra_props(env_mono)

    platform = env["platform"]
    target   = env["target"]
    arch     = env.get("arch", "x86_64")

    # Paths — adjust if the repo layout changes.
    mono_dir      = os.path.join(env.Dir("#").abspath, "modules", "mono")
    glue_dir      = os.path.join(mono_dir, "glue", "GodotSharp")
    godottools_dir = os.path.join(mono_dir, "editor", "GodotTools")

    # ------------------------------------------------------------------
    # 1. Build GodotSharp (the runtime library shipped to games)
    # ------------------------------------------------------------------
    godotsharp_csproj = os.path.join(glue_dir, "GodotSharp", "GodotSharp.csproj")
    if os.path.isfile(godotsharp_csproj):
        cmd = [dotnet, "build", godotsharp_csproj, "--nologo", "-c", "Release"]
        _run_dotnet(cmd)

    # ------------------------------------------------------------------
    # 2. Publish for the target platform (creates self-contained output)
    #    The extra_props list contains e.g. ["/p:WasmEnableThreads=true"]
    #    when building for web with threads=yes.
    # ------------------------------------------------------------------
    if platform == "web":
        # dotnet publish for WASM targets the browser-wasm RID.
        rid = "browser-wasm"
        publish_cmd = [
            dotnet, "publish", godotsharp_csproj,
            "--nologo",
            "-c", "Release",
            "-r", rid,
            "--no-self-contained",  # We link Godot's own runtime
        ] + extra_props            # <-- injects /p:WasmEnableThreads=...

        _run_dotnet(publish_cmd)

    # ------------------------------------------------------------------
    # 3. Build GodotTools (editor-side C# tooling — desktop only)
    # ------------------------------------------------------------------
    if platform not in ("web", "android", "ios", "visionos"):
        godottools_csproj = os.path.join(
            godottools_dir, "GodotTools", "GodotTools.csproj"
        )
        if os.path.isfile(godottools_csproj):
            cmd = [dotnet, "build", godottools_csproj, "--nologo", "-c", "Release"]
            _run_dotnet(cmd)
