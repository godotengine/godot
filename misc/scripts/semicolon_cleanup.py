#!/usr/bin/env python3

import argparse
import re
import sys
from typing import NoReturn

RE_FIND = re.compile(rb"\)( const| override| final)? ?{};?")
RE_REPLACE = rb")\1 {}"


def parse_file(filename: str) -> int:
    try:
        with open(filename, "rb") as file:
            old = file.read()
    except OSError as err:
        print(err)
        return 1

    new = re.sub(RE_FIND, RE_REPLACE, old)

    if new == old:
        return 0

    try:
        with open(filename, "wb") as file:
            file.write(new)
    except OSError as err:
        print(err)

    return 1


def main() -> NoReturn:
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs="+", help="Files to check.")
    args = parser.parse_args()
    ret = 0

    for filename in args.filenames:
        ret |= parse_file(filename)

    sys.exit(ret)


if __name__ == "__main__":
    main()
