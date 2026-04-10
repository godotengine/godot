#!/usr/bin/env python3

import io
import os
import sys
import typing

if typing.TYPE_CHECKING:
    pass


HEADER = """\
/**************************************************************************/
/*  $filename                                                             */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/
"""


def process_file(filename):  # type: (str) -> None
    filename = filename.strip()
    basename = os.path.basename(filename)

    new_buffer = None  # type: typing.Union[io.StringIO, None]
    with open(filename, "r", encoding="utf-8") as file:
        new_buffer = process_file_buffer(basename, file)
    with open(filename, "w", encoding="utf-8") as file:
        CHUNK_SIZE = 1024
        while True:
            chunk = new_buffer.read(CHUNK_SIZE)
            if not chunk:
                break
            file.write(chunk)


def process_file_buffer(filename, text_buffer):  # type: (str, io.TextIOWrapper) -> io.StringIO
    output = io.StringIO()

    # Handle replacing $filename with actual filename and keep alignment.
    token = "$filename"
    padded_filename = filename
    padded_token = token
    filename_length = len(filename)
    token_length = len(token)

    # Pad with spaces to keep alignment.
    if filename_length < token_length:
        padded_filename += " " * (token_length - filename_length)
    elif token_length < filename_length:
        padded_token += " " * (filename_length - token_length)

    if HEADER.find(token) != -1:
        output.write(HEADER.replace(padded_token, padded_filename))
    else:
        output.write(HEADER.replace(token, filename))
    output.write("\n")

    # We now have the proper header, so we want to ignore the one in the original file
    # and potentially empty lines and badly formatted lines, while keeping comments that
    # come after the header, and then keep everything non-header unchanged.
    # To do so, we skip empty lines that may be at the top in a first pass.
    # In a second pass, we skip all consecutive comment lines starting with "/*",
    # then we can append the rest (step 2).

    line = text_buffer.readline()
    header_done = False

    while line.strip() == "" and line != "":  # Skip empty lines at the top
        line = text_buffer.readline()

    if line.find("/**********") == -1:  # Godot header starts this way
        # Maybe starting with a non-Godot comment, abort header magic
        header_done = True

    while not header_done:  # Handle header now
        if line.find("/*") != 0:  # No more starting with a comment
            header_done = True
            if line.strip() != "":
                output.write(line)
        line = text_buffer.readline()

    while line != "":  # Dump everything until EOF
        output.write(line)
        line = text_buffer.readline()

    output.seek(0)
    return output


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Invalid usage of copyright_headers.py, it should be called with a path to one or multiple files.")
        sys.exit(1)

    for f in sys.argv[1:]:
        process_file(f)
