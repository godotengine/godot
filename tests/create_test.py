#!/usr/bin/env python3

import argparse
import os
import re
import sys


def main():
    # Change to the directory where the script is located,
    # so that the script can be run from any location.
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    parser = argparse.ArgumentParser(description="Creates a new unit test file.")
    parser.add_argument("name", type=str, help="the unit test name in PascalCase notation")
    args = parser.parse_args()

    snake_case_regex = re.compile(r"(?<!^)(?=[A-Z])")
    name_snake_case = snake_case_regex.sub("_", args.name).lower()
    file_name = f"test_{name_snake_case}.h"

    if os.path.isfile(file_name):
        print(f'ERROR: The file "{file_name}" already exists.')
        sys.exit(1)
    with open(file_name, "w") as file:
        file.write(
            """/*************************************************************************/
/*  test_{name_snake_case}.h {padding} */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef TEST_{name_upper_snake_case}_H
#define TEST_{name_upper_snake_case}_H

#include "tests/test_macros.h"

namespace Test{name_pascal_case} {{

TEST_CASE("[{name_pascal_case}] Example test case") {{
    // TODO: Remove this comment and write your test code here.
}}

}} // namespace Test{name_pascal_case}

#endif // TEST_{name_upper_snake_case}_H
""".format(
                name_snake_case=name_snake_case,
                # Capitalize the first letter but keep capitalization for the rest of the string.
                # This is done in case the user passes a camelCase string instead of PascalCase.
                name_pascal_case=args.name[0].upper() + args.name[1:],
                name_upper_snake_case=name_snake_case.upper(),
                # The padding length depends on the test name length.
                padding=" " * (60 - len(name_snake_case)),
            )
        )

    # Print an absolute path so it can be Ctrl + clicked in some IDEs and terminal emulators.
    print("Test header file created:")
    print(os.path.abspath(file_name))
    print("\nRemember to #include the new test header in this file (following alphabetical order):")
    print(os.path.abspath("test_main.cpp"))


if __name__ == "__main__":
    main()
