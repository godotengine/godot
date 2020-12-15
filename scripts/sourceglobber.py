#!/usr/bin/env python3

import sys, os
from glob import glob


def glob_pattern(what):
    return glob(what)


def glob_and_print(what):
    files = glob_pattern(what)
    if not files:
        return
    for f in files[:-1]:
        print(f)
    print(files[-1], end="")


if __name__ == "__main__":
    glob_and_print(sys.argv[1])
