#!/usr/bin/env python3
"""
Cross‑platform build script for the CrossRuntime C# worker.

Usage:
    python build.py

Optional environment variables:
    GODOT_WEB_DIR   – path to the Godot web export bundle (default: ~/godot/bin/.web_zip)
    DOTNET_VERSION  – the .NET version to target (default: net8.0)
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


# Where this script lives (the cs/ folder of the module)
SCRIPT_DIR = Path(__file__).resolve().parent

# Where the .csproj file is expected
PROJECT_FILE = SCRIPT_DIR / "CrossRuntime.csproj"

# Output of dotnet publish
DOTNET_TFM = os.environ.get("DOTNET_VERSION", "net8.0")
PUBLISH_DIR = SCRIPT_DIR / "bin" / "Release" / DOTNET_TFM / "browser-wasm"


# Source of the _framework folder inside the publish output.
# The browser-wasm template often places it under AppBundle/.
FRAMEWORK_SRC = PUBLISH_DIR / "AppBundle" / "_framework"
# Fall back to direct _framework if the AppBundle layout is not used
if not FRAMEWORK_SRC.exists():
    FRAMEWORK_SRC = PUBLISH_DIR / "_framework"

# Destination: where the web export bundle lives.
GODOT_WEB_DIR = Path(os.environ.get("GODOT_WEB_DIR", Path.home() / "godot" / "bin" / ".web_zip"))
DEST_FRAMEWORK = GODOT_WEB_DIR / "cs" / "_framework"

# Helper functions
def run(cmd, description):
    """Print and run a command, aborting on failure."""
    print(f"\n[BUILD] {description}")
    print("  " + " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)

def main() -> None:
    #Verify dotnet is available
    if shutil.which("dotnet") is None:
        sys.exit("ERROR: 'dotnet' not found. Install the .NET SDK first.")

    #Clean old build artifacts (bin/ and obj/)
    for folder in (SCRIPT_DIR / "bin", SCRIPT_DIR / "obj"):
        if folder.exists():
            print(f"[BUILD] Removing {folder}")
            shutil.rmtree(folder)

    #Publish the C# project for browser-wasm
    run([
        "dotnet", "publish", str(PROJECT_FILE),
        "-c", "Release",
        "-r", "browser-wasm",
        "-p:SelfContained=true",
        "-p:PublishTrimmed=false",
        "-p:InvariantGlobalization=true",
        "-o", str(PUBLISH_DIR),
    ], description="Publishing C# project for browser-wasm")

    #Verify that _framework was produced
    if not FRAMEWORK_SRC.is_dir():
        sys.exit(
            f"ERROR: _framework not found.\n"
            f"  Looked in: {FRAMEWORK_SRC}\n"
            f"  Check the publish output above for errors."
        )

    #Remove old _framework from the web bundle
    if DEST_FRAMEWORK.exists():
        print(f"[BUILD] Removing old {DEST_FRAMEWORK}")
        shutil.rmtree(DEST_FRAMEWORK)

    #Copy new _framework into the web bundle
    DEST_FRAMEWORK.parent.mkdir(parents=True, exist_ok=True)
    print(f"[BUILD] Copying framework → {DEST_FRAMEWORK}")
    shutil.copytree(FRAMEWORK_SRC, DEST_FRAMEWORK)

    # 7) Copy the interop.js if it's next to this script
    interop_js = SCRIPT_DIR / "interop.js"    
    dest_js = GODOT_WEB_DIR / "cs" / "interop.js"
    shutil.copy2(interop_js, dest_js)
    print(f"[BUILD] Copied interop.js → {dest_js}")

    print("\n[BUILD] Complete!")
    print(f"  Web bundle is at: {GODOT_WEB_DIR}")
    print("   Please copy the contents of web export files which include dot.pck, host.html, server.py into GODOT_WEB_DIR. After that, you can edit them as you deem fit.")
    print("  Serve it with: python server.py (from that directory)")
    print("  Then open http://localhost:8000/host.html")

if __name__ == "__main__":
    main()