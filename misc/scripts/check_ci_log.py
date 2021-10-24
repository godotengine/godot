#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

if len(sys.argv) < 2:
    print("ERROR: You must run program with file name as argument.")
    sys.exit(50)

fname = sys.argv[1]

fileread = open(fname.strip(), "r")
file_contents = fileread.read()

# If find "ERROR: AddressSanitizer:", then happens invalid read or write
# This is critical bug, so we need to fix this as fast as possible
# Example - https://github.com/godotengine/godot/issues/48114#issue-865502805

if file_contents.find("ERROR: AddressSanitizer:") != -1:
    print("FATAL ERROR: An incorrectly used memory was found.")
    sys.exit(51)

# There is also possible, that program crashed with or without backtrace.
# Example - https://github.com/godotengine/godot/issues/51620#issue-970308653

if (
    file_contents.find("Program crashed with signal") != -1
    or file_contents.find("Dumping the backtrace") != -1
    or file_contents.find("Segmentation fault (core dumped)") != -1
):
    print("FATAL ERROR: Godot has been crashed.")
    sys.exit(52)

# Finding memory leaks in Godot is quite difficult, because we need to take into
# account leaks also in external libraries. They are usually provided without
# debugging symbols, so the leak report from it usually has only 2/3 lines,
# so searching for 5 element - "#4 0x" - should correctly detect the vast
# majority of memory leaks
# Example - https://github.com/godotengine/godot/issues/34495#issue-541154242

if file_contents.find("ERROR: LeakSanitizer:") != -1:
    if file_contents.find("#4 0x") != -1:
        print("ERROR: Memory leak was found")
        sys.exit(53)

# It may happen that Godot detects leaking nodes/resources and removes them, so
# this possibility should also be handled as a potential error, even if
# LeakSanitizer doesn't report anything
# Example - https://github.com/godotengine/godot/issues/49438#issue-915348285

if file_contents.find("ObjectDB instances leaked at exit") != -1:
    print('ERROR: Memory leak was found (search for "ObjectDB instances leaked at exit" in CI log)')
    sys.exit(54)

# In test project may be put several assert functions which will control if
# project is executed with right parameters etc. which normally will not stop
# execution of project
# Example - https://github.com/godotengine/godot/issues/37980#issue-602414919

if file_contents.find("Assertion failed") != -1:
    print(
        'ERROR: Assertion failed in project, check execution log for more info (search for "Assertion failed" in CI log)'
    )
    sys.exit(55)

# Sometimes pointers points at objects of different types, which may cause
# to show errors like "runtime error: member call on address 0x1 which does not point to an object of type"
# Example - https://github.com/godotengine/godot/issues/51351#issue-963170661

# Waiting for https://github.com/godotengine/godot/issues/51888

if file_contents.find("vptr for") != -1:
    print('WARNING: Found pointer which not point at valid object (search for "vptr for" in CI log)')
    # sys.exit(56)

# By default overflow or underflow of signed values are in C++ just
# undefined behavior(most of the time value is wrapped)
# Example - https://github.com/godotengine/godot/issues/33644#issue-523623547

if file_contents.find("cannot be represented in type") != -1 or file_contents.find("is outside the range") != -1:
    print(
        'ERROR: Found pointer which not point at valid object (search for "cannot be represented in type" or "is outside the range" in CI log)'
    )
    sys.exit(57)

# Some functions like memcpy doesn't expect that its argument is null pointer.
# This may later be cause of bugs or crashes.
# Example - https://github.com/godotengine/godot/issues/48215#issue-867854743

if file_contents.find("null pointer passed as argument") != -1:
    print(
        'ERROR: Found null pointer passed as argument to function which not expect it (search for "null pointer passed as argument" in CI log)'
    )
    sys.exit(58)

# Casting or pointer moving caused that code trying to violate alignement rules
# Example - https://github.com/godotengine/godot/issues/31203#issue-478487290

if file_contents.find("misaligned address") != -1:
    print('ERROR: Found usage of misaligned pointer (search for "misaligned address" in CI log)')
    sys.exit(59)

# For now Godot leaks a lot of rendering stuff so for now we just show info
# about it and this needs to be re-enabled after fixing this memory leaks.
# Example - https://github.com/godotengine/godot/issues/47941#issue-859356605

# Blocked by https://github.com/godotengine/godot/issues/46833

if file_contents.find("were leaked") != -1 or file_contents.find("were never freed") != -1:
    print('WARNING: Memory leak was found (search for "were leaked" or "were never freed" in CI log)')

# Usually error about trying to free invalid ID is caused by removing wrong object
# Example - https://github.com/godotengine/godot/issues/49623#issue-921610423

# Blocked by Swiftshader bugs in CI

if file_contents.find("Attempted to free invalid ID") != -1:
    print('WARNING: Trying to free invalid object (search for "Attempted to free invalid ID" in CI log)')

sys.exit(0)
