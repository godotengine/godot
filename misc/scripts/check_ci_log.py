#!/usr/bin/env python3

import sys

if len(sys.argv) < 2:
    print("ERROR: You must run program with file name as argument.")
    sys.exit(50)

fname = sys.argv[1]

with open(fname.strip(), "r", encoding="utf-8") as fileread:
    file_contents = fileread.read()

# If find "ERROR: AddressSanitizer:", then happens invalid read or write
# This is critical bug, so we need to fix this as fast as possible

if file_contents.find("ERROR: AddressSanitizer:") != -1:
    print("FATAL ERROR: An incorrectly used memory was found.")
    sys.exit(51)

# There is also possible, that program crashed with or without backtrace.

if (
    file_contents.find("Program crashed with signal") != -1
    or file_contents.find("Dumping the backtrace") != -1
    or file_contents.find("Segmentation fault (core dumped)") != -1
    or file_contents.find("Aborted (core dumped)") != -1
    or file_contents.find("terminate called without an active exception") != -1
    or file_contents.find("execution reached the end of a value-returning function without returning a value") != -1
):
    print("FATAL ERROR: Godot has been crashed.")
    sys.exit(52)

# Finding memory leaks in Godot is quite difficult, because we need to take into
# account leaks also in external libraries. They are usually provided without
# debugging symbols, so the leak report from it usually has only 2/3 lines,
# so searching for 5 element - "#4 0x" - should correctly detect the vast
# majority of memory leaks

if file_contents.find("ERROR: LeakSanitizer:") != -1:
    if file_contents.find("#4 0x") != -1:
        print("ERROR: Memory leak was found")
        sys.exit(53)

# It may happen that Godot detects leaking nodes/resources and removes them, so
# this possibility should also be handled as a potential error, even if
# LeakSanitizer doesn't report anything

if file_contents.find("ObjectDB instances leaked at exit") != -1:
    print("ERROR: Memory leak was found")
    sys.exit(54)

# In test project may be put several assert functions which will control if
# project is executed with right parameters etc. which normally will not stop
# execution of project

if file_contents.find("Assertion failed") != -1:
    print("ERROR: Assertion failed in project, check execution log for more info")
    sys.exit(55)

# For now Godot leaks a lot of rendering stuff so for now we just show info
# about it and this needs to be re-enabled after fixing this memory leaks.

if file_contents.find("were leaked") != -1 or file_contents.find("were never freed") != -1:
    print("WARNING: Memory leak was found")

sys.exit(0)
