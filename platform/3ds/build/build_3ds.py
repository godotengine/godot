#!/usr/bin/env python3

import os
import subprocess
import sys


def main():
    devkitpro = os.environ.get("DEVKITPRO")
    devkitarm = os.environ.get("DEVKITARM")

    if not devkitpro:
        print("DEVKITPRO não encontrado.")
        return 1

    if not devkitarm:
        print("DEVKITARM não encontrado.")
        return 1

    print("================================")
    print(" Godot 4.7 Nintendo 3DS Builder")
    print("================================")
    print("DEVKITPRO :", devkitpro)
    print("DEVKITARM :", devkitarm)

    result = subprocess.run(
        ["make"],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
