#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

if len(sys.argv) < 2:
    print("ERROR: You must run program with file name as argument.")
    sys.exit(1)

fname = sys.argv[1]

fileread = open(fname.strip(), "r")
file_contents = fileread.read()

# If find "ERROR: AddressSanitizer:", then happens invalid read or write
# This is critical bug, so we need to fix this as fast as possible

if file_contents.find("ERROR: AddressSanitizer:") != -1:
    print("FATAL ERROR: An incorrectly used memory was found.")
    sys.exit(1)

# There is also possible, that program crashed with or without backtrace.

if (
    file_contents.find("Program crashed with signal") != -1
    or file_contents.find("Dumping the backtrace") != -1
    or file_contents.find("Segmentation fault (core dumped)") != -1
):
    print("FATAL ERROR: Godot has been crashed.")
    sys.exit(1)

# Finding memory leaks in Godot is quite difficult, because we need to take into
# account leaks also in external libraries. They are usually provided without
# debugging symbols, so the leak report from it usually has only 2/3 lines,
# so searching for 5 element - "#4 0x" - should correctly detect the vast
# majority of memory leaks

if file_contents.find("ERROR: LeakSanitizer:") != -1:
    if file_contents.find("#4 0x") != -1:
        print("ERROR: Memory leak was found")
        sys.exit(1)

# It may happen that Godot detects leaking nodes/resources and removes them, so
# this possibility should also be handled as a potential error, even if
# LeakSanitizer doesn't report anything

if file_contents.find("ObjectDB instances leaked at exit") != -1:
    print("ERROR: Memory leak was found")
    sys.exit(1)

sys.exit(0)
