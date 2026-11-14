#!/usr/bin/env python3
##  Copyright (c) 2012 The WebM project authors. All Rights Reserved.
##
##  Use of this source code is governed by a BSD-style license
##  that can be found in the LICENSE file in the root of the source
##  tree. An additional intellectual property rights grant can be found
##  in the file PATENTS.  All contributing project authors may
##  be found in the AUTHORS file in the root of the source tree.
##
"""Classes for representing diff pieces."""

__author__ = "jkoleszar@google.com"

import re


class DiffLines(object):
    """A container for one half of a diff."""

    def __init__(self, filename, offset, length):
        self.filename = filename
        self.offset = offset
        self.length = length
        self.lines = []
        self.delta_line_nums = []

    def Append(self, line):
        l = len(self.lines)
        if line[0] != " ":
            self.delta_line_nums.append(self.offset + l)
        self.lines.append(line[1:])
        assert l+1 <= self.length

    def Complete(self):
        return len(self.lines) == self.length

    def __contains__(self, item):
        return item >= self.offset and item <= self.offset + self.length - 1


class DiffHunk(object):
    """A container for one diff hunk, consisting of two DiffLines."""

    def __init__(self, header, file_a, file_b, start_a, len_a, start_b, len_b):
        self.header = header
        self.left = DiffLines(file_a, start_a, len_a)
        self.right = DiffLines(file_b, start_b, len_b)
        self.lines = []

    def Append(self, line):
        """Adds a line to the DiffHunk and its DiffLines children."""
        if line[0] == "-":
            self.left.Append(line)
        elif line[0] == "+":
            self.right.Append(line)
        elif line[0] == " ":
            self.left.Append(line)
            self.right.Append(line)
        elif line[0] == "\\":
            # Ignore newline messages from git diff.
            pass
        else:
            assert False, ("Unrecognized character at start of diff line "
                           "%r" % line[0])
        self.lines.append(line)

    def Complete(self):
        return self.left.Complete() and self.right.Complete()

    def __repr__(self):
        return "DiffHunk(%s, %s, len %d)" % (
            self.left.filename, self.right.filename,
            max(self.left.length, self.right.length))


def ParseDiffHunks(stream):
    """Walk a file-like object, yielding DiffHunks as they're parsed."""

    file_regex = re.compile(r"(\+\+\+|---) (\S+)")
    range_regex = re.compile(r"@@ -(\d+)(,(\d+))? \+(\d+)(,(\d+))?")
    hunk = None
    while True:
        line = stream.readline()
        if not line:
            break

        if hunk is None:
            # Parse file names
            diff_file = file_regex.match(line)
            if diff_file:
              if line.startswith("---"):
                  a_line = line
                  a = diff_file.group(2)
                  continue
              if line.startswith("+++"):
                  b_line = line
                  b = diff_file.group(2)
                  continue

            # Parse offset/lengths
            diffrange = range_regex.match(line)
            if diffrange:
                if diffrange.group(2):
                    start_a = int(diffrange.group(1))
                    len_a = int(diffrange.group(3))
                else:
                    start_a = 1
                    len_a = int(diffrange.group(1))

                if diffrange.group(5):
                    start_b = int(diffrange.group(4))
                    len_b = int(diffrange.group(6))
                else:
                    start_b = 1
                    len_b = int(diffrange.group(4))

                header = [a_line, b_line, line]
                hunk = DiffHunk(header, a, b, start_a, len_a, start_b, len_b)
        else:
            # Add the current line to the hunk
            hunk.Append(line)

            # See if the whole hunk has been parsed. If so, yield it and prepare
            # for the next hunk.
            if hunk.Complete():
                yield hunk
                hunk = None

    # Partial hunks are a parse error
    assert hunk is None
