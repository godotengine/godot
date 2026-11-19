#!/usr/bin/env python3

import argparse
import os
import shutil
import signal
from pathlib import Path

if __name__ != "__main__":
    raise ImportError(f"{__name__} should not be used as a module.")

default_destination = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(
        description="Extract relevant mbedTLS sources from an upstream release into the Godot source tree"
    )
    parser.add_argument("sources", help="Path to the directory containing the mbedTLS sources")
    parser.add_argument(
        "--destination",
        help="Destination directory where to extract the mbedTLS sources (default: {})".format(default_destination),
        default=default_destination,
    )
    args = parser.parse_args()

    src = Path(args.sources)
    dst = Path(args.destination)

    headers_dirs = [
        "include/mbedtls/",
        "tf-psa-crypto/include/",
        "tf-psa-crypto/drivers/builtin/include",
    ]
    library_dirs = [
        "library/",
        "tf-psa-crypto/core",
        "tf-psa-crypto/dispatch",
        "tf-psa-crypto/extras",
        "tf-psa-crypto/platform",
        "tf-psa-crypto/utilities",
        "tf-psa-crypto/drivers/builtin/src",
    ]
    files = []
    for d in headers_dirs:
        files.extend((src / d).rglob("*.h"))

    for d in library_dirs:
        files.extend((src / d).rglob("*.h"))
        files.extend((src / d).rglob("*.c"))

    for f in files:
        fdst = dst / f.relative_to(src)
        print("Copying '{}' to '{}'".format(f, fdst))
        os.makedirs(fdst.parent, exist_ok=True)
        shutil.copyfile(f, fdst)


try:
    raise SystemExit(main())
except KeyboardInterrupt:
    import signal

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    os.kill(os.getpid(), signal.SIGINT)
