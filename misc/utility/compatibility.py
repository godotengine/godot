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

        # According to the reference, currently (06/03/2025, mm/dd/yyyy),
        #  only msvc has full support of module.
        if env.msvc and is_msvc_support_module(env):
            env.Append(CCFLAGS=["/experimental:module"])
        # Apple clang also supports module.
        elif methods.using_clang(env) and is_clang_support_module():
            env.Append(CCFLAGS=["-fmodules", "-fcxx-modules"])
        else:
            is_able_to_use_module_cache = False


def is_msvc_support_module(env) -> bool:
    """
    MSVC supports C++ modules after 19.2.8.
    """
    # Only "major", "minor", and "patch" has int type.
    version = methods.get_compiler_version(env)
    major: int = version["major"]
    minor: int = version["minor"]
    patch: int = version["patch"]

    return major > 19 or (version["major"] == 19 and minor >= 2) or (major == 19 and minor == 2 and patch >= 8)


def is_clang_support_module() -> bool:
    """
    Check if the clang is Apple clang.
    """
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    version_text: str = subprocess.check_output(["clang", "--version"], env=env, encoding="utf-8")
    return "Apple clang version" in version_text
