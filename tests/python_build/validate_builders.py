#!/usr/bin/env python3

from __future__ import annotations

if __name__ != "__main__":
    raise ImportError(f"{__name__} should not be used as a module.")

import os
import sys
from typing import Any, Callable

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))

from gles3_builders import gles3_glsl
from glsl_builders import glsl_header, rd_glsl

FUNC_PATH_KWARGS: list[tuple[Callable[..., None], str, dict[str, Any]]] = [
    (
        gles3_glsl,
        "tests/python_build/fixtures/gles3/vertex_fragment.out",
        {"shader": "tests/python_build/fixtures/gles3/vertex_fragment.glsl"},
    ),
    (
        glsl_header,
        "tests/python_build/fixtures/glsl/compute.out",
        {"shader": "tests/python_build/fixtures/glsl/compute.glsl"},
    ),
    (
        glsl_header,
        "tests/python_build/fixtures/glsl/vertex_fragment.out",
        {"shader": "tests/python_build/fixtures/glsl/vertex_fragment.glsl"},
    ),
    (
        rd_glsl,
        "tests/python_build/fixtures/rd_glsl/compute.out",
        {"shader": "tests/python_build/fixtures/rd_glsl/compute.glsl"},
    ),
    (
        rd_glsl,
        "tests/python_build/fixtures/rd_glsl/vertex_fragment.out",
        {"shader": "tests/python_build/fixtures/rd_glsl/vertex_fragment.glsl"},
    ),
]


def main() -> int:
    ret = 0

    for func, path, kwargs in FUNC_PATH_KWARGS:
        if os.path.exists(out_path := os.path.abspath(path)):
            with open(out_path, "rb") as file:
                raw = file.read()
            func(path, **kwargs)
            with open(out_path, "rb") as file:
                if raw != file.read():
                    ret += 1
        else:
            func(path, **kwargs)
            ret += 1

    return ret


sys.exit(main())
