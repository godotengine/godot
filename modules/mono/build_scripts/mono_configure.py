"""
modules/mono/build_scripts/mono_configure.py
Godot .NET/Mono build configuration helper.

Key change for issue #70796 (Web C# export support):
  When building for the `web` platform, we now dynamically inject the
  WasmEnableThreads MSBuild property into the dotnet publish command so that
  the Mono WASM runtime pack is compiled with the same threading ABI as the
  Emscripten host (which uses -sUSE_PTHREADS=1 when threads=yes).
  Mismatching this property was the root cause of the linker failure described
  in the issue.
"""

import os
import sys
import subprocess

from SCons.Script import *


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_dotnet_cli():
    """Return the path to the `dotnet` CLI, or None if not found."""
    import shutil
    return shutil.which("dotnet")


def _get_api_version(env):
    """Return the Godot API version string used for the GodotSharp nupkg."""
    # version.py lives at the repo root.
    import version
    return "{}.{}.{}".format(version.major, version.minor, version.patch)


# ---------------------------------------------------------------------------
# Public entry point called from modules/mono/SCsub
# ---------------------------------------------------------------------------

def configure(env, env_mono):
    """
    Configure the Mono/C# build for the current platform.

    This function is called by modules/mono/SCsub during the SCons
    configuration phase.  It sets environment variables and build flags
    needed to compile GodotSharp, run `dotnet publish`, and (on web)
    link the resulting WASM object into the Godot binary.
    """

    dotnet = _find_dotnet_cli()
    if not dotnet:
        print("WARNING: `dotnet` CLI not found in PATH. C# support will be disabled.")
        env_mono["MONO_DOTNET_FOUND"] = False
        return

    env_mono["MONO_DOTNET_FOUND"] = True
    env_mono["MONO_DOTNET_CLI"] = dotnet

    # ------------------------------------------------------------------
    # Web platform: synchronise WasmEnableThreads with the SCons threads
    # option so that the runtime pack ABI matches the Emscripten link.
    #
    # Background (issue #70796):
    #   The GetRuntimePack C# project hard-coded <WasmEnableThreads>false</WasmEnableThreads>.
    #   When SCons builds with threads=yes (passing -sUSE_PTHREADS=1 to
    #   Emscripten), the resulting pinvoke-table.h is compiled with atomics
    #   support, but the runtime pack was not, causing a binary-signature
    #   mismatch at link time on macOS arm64 and Linux x86_64.
    #
    #   Fix: read the `threads` SCons option and pass the matching value as
    #   an MSBuild property on every `dotnet publish` invocation for web.
    # ------------------------------------------------------------------
    if env["platform"] == "web":
        # SCons stores boolean options as "yes"/"no" strings after they have
        # been processed by BoolVariable; fall back to True (threads on) to
        # match the Emscripten default when the option is absent.
        threads_enabled = env.get("threads", True)
        if isinstance(threads_enabled, str):
            threads_enabled = threads_enabled.lower() not in ("no", "false", "0")

        wasm_threads_value = "true" if threads_enabled else "false"

        # Append to the list of extra MSBuild properties that will be
        # forwarded to every `dotnet publish` call in build_assemblies.py.
        existing = env_mono.get("MONO_PUBLISH_MSBUILD_PROPS", [])
        existing.append("/p:WasmEnableThreads={}".format(wasm_threads_value))
        env_mono["MONO_PUBLISH_MSBUILD_PROPS"] = existing

        print(
            "mono_configure: web platform detected — "
            "WasmEnableThreads={} (threads={})".format(
                wasm_threads_value, threads_enabled
            )
        )

    # ------------------------------------------------------------------
    # iOS: set the correct target architecture flags for dotnet publish.
    # (Unchanged from upstream behaviour — kept here for completeness.)
    # ------------------------------------------------------------------
    if env["platform"] in ("ios", "visionos"):
        arch = env.get("arch", "arm64")
        env_mono["MONO_IOS_ARCH"] = arch

    # ------------------------------------------------------------------
    # Android: nothing extra required here; handled by Gradle.
    # ------------------------------------------------------------------

    # Record the API version for downstream consumers.
    env_mono["GODOT_API_VERSION"] = _get_api_version(env)
