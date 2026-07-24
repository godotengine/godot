#!/usr/bin/env python

from methods import is_apple_clang

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
        if is_apple_clang(env):
            env.Append(CCFLAGS=["-fmodules", "-fcxx-modules"])
        else:
            is_able_to_use_module_cache = False
