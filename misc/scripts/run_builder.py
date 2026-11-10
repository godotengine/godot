#!/usr/bin/env python3
"""Run a builder function from a generator script, without SCons. See `methods.Run()`."""

if __name__ != "__main__":
    raise ImportError(f"{__name__} should not be used as a module.")

import argparse
import importlib.util
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))

import methods

parser = argparse.ArgumentParser(description=__doc__, fromfile_prefix_chars="@")
parser.add_argument("script", help="Path of the generator script")
parser.add_argument("method", help="Name of the builder function, called as `method(target, source, env)`")
parser.add_argument("--target", nargs="*", default=[], type=methods.decode_value, help="Target file(s)")
parser.add_argument("--source", nargs="*", default=[], type=methods.decode_value, help="Source file(s) or values")
parser.add_argument("--production", action="store_true")
parser.add_argument("--dev-build", action="store_true")
args = parser.parse_args()

spec = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(args.script))[0], args.script)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
getattr(module, args.method)(args.target, args.source, {"production": args.production, "dev_build": args.dev_build})
