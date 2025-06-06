#!/usr/bin/env python
import os
import subprocess

import methods

is_able_to_use_module_cache = None


def try_use_cxx20_module(env):
    """
    Use C++20's `module` if current compiler supports that.
    Referencing https://en.cppreference.com/w/cpp/compiler_support/20.
    """
    global is_able_to_use_module_cache

    if is_able_to_use_module_cache is not None:
        return is_able_to_use_module_cache
    else:
        is_able_to_use_module_cache = True

        # Apple clang supports module.
        if methods.using_clang(env) and is_clang_support_module():
            env.Append(CCFLAGS=["-fmodules", "-fcxx-modules"])
        else:
            is_able_to_use_module_cache = False


def is_clang_support_module() -> bool:
    """
    Check if the clang is Apple clang.
    """
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    version_text: str = subprocess.check_output(["clang", "--version"], env=env, encoding="utf-8")
    return "Apple clang version" in version_text
