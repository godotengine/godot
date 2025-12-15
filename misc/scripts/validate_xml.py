#!/usr/bin/env python3

if __name__ != "__main__":
    raise SystemExit(f'Utility script "{__file__}" should not be used as a module!')

import argparse
import sys

import xmlschema  # Third-party module. Automatically installed in associated pre-commit hook.

sys.path.insert(0, "./")

try:
    from methods import print_error
except ImportError:
    raise SystemExit(f"Utility script {__file__} must be run from repository root!")


def main():
    parser = argparse.ArgumentParser(description="Validate XML documents against `doc/class.xsd`")
    parser.add_argument("files", nargs="+", help="A list of XML files to parse")
    args = parser.parse_args()

    SCHEMA = xmlschema.XMLSchema("doc/class.xsd")
    ret = 0

    for file in args.files:
        try:
            SCHEMA.validate(file)
        except xmlschema.validators.exceptions.XMLSchemaValidationError as err:
            print_error(f'Validation failed for "{file}"!\n\n{err}')
            ret += 1

    return ret


try:
    raise SystemExit(main())
except KeyboardInterrupt:
    import os
    import signal

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    os.kill(os.getpid(), signal.SIGINT)
