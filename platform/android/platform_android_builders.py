"""Functions used to generate source files during build time"""

import subprocess
import sys


def generate_android_binaries(target, source, env):
    gradle_process = []

    if sys.platform.startswith("win"):
        gradle_process = [
            "cmd",
            "/c",
            "gradlew.bat",
        ]
    else:
        gradle_process = ["./gradlew"]

    if env["target"] == "editor":
        gradle_process += ["generateGodotEditor", "generateGodotHorizonOSEditor", "generateGodotPicoOSEditor"]
    else:
        gradle_process += ["generateGodotTemplates"]
    gradle_process += ["--quiet"]

    if env["gradle_do_not_strip"]:
        gradle_process += ["-PdoNotStrip=true"]

    subprocess.run(
        gradle_process,
        cwd="platform/android/java",
    )
