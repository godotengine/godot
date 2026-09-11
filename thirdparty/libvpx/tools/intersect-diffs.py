#!/usr/bin/env python3
##  Copyright (c) 2012 The WebM project authors. All Rights Reserved.
##
##  Use of this source code is governed by a BSD-style license
##  that can be found in the LICENSE file in the root of the source
##  tree. An additional intellectual property rights grant can be found
##  in the file PATENTS.  All contributing project authors may
##  be found in the AUTHORS file in the root of the source tree.
##
"""Calculates the "intersection" of two unified diffs.

Given two diffs, A and B, it finds all hunks in B that had non-context lines
in A and prints them to stdout. This is useful to determine the hunks in B that
are relevant to A. The resulting file can be applied with patch(1) on top of A.
"""

__author__ = "jkoleszar@google.com"

import sys

import diff


def FormatDiffHunks(hunks):
    """Re-serialize a list of DiffHunks."""
    r = []
    last_header = None
    for hunk in hunks:
        this_header = hunk.header[0:2]
        if last_header != this_header:
            r.extend(hunk.header)
            last_header = this_header
        else:
            r.extend(hunk.header[2])
        r.extend(hunk.lines)
        r.append("\n")
    return "".join(r)


def ZipHunks(rhs_hunks, lhs_hunks):
    """Join two hunk lists on filename."""
    for rhs_hunk in rhs_hunks:
        rhs_file = rhs_hunk.right.filename.split("/")[1:]

        for lhs_hunk in lhs_hunks:
            lhs_file = lhs_hunk.left.filename.split("/")[1:]
            if lhs_file != rhs_file:
                continue
            yield (rhs_hunk, lhs_hunk)


def main():
    old_hunks = [x for x in diff.ParseDiffHunks(open(sys.argv[1], "r"))]
    new_hunks = [x for x in diff.ParseDiffHunks(open(sys.argv[2], "r"))]
    out_hunks = []

    # Join the right hand side of the older diff with the left hand side of the
    # newer diff.
    for old_hunk, new_hunk in ZipHunks(old_hunks, new_hunks):
        if new_hunk in out_hunks:
            continue
        old_lines = old_hunk.right
        new_lines = new_hunk.left

        # Determine if this hunk overlaps any non-context line from the other
        for i in old_lines.delta_line_nums:
            if i in new_lines:
                out_hunks.append(new_hunk)
                break

    if out_hunks:
        print(FormatDiffHunks(out_hunks))
        sys.exit(1)

if __name__ == "__main__":
    main()
