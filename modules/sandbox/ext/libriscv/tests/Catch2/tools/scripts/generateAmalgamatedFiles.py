#!/usr/bin/env python3
#              Copyright Catch2 Authors
# Distributed under the Boost Software License, Version 1.0.
#   (See accompanying file LICENSE.txt or copy at
#        https://www.boost.org/LICENSE_1_0.txt)
# SPDX-License-Identifier: BSL-1.0

import os
import re
import datetime

from scriptCommon import catchPath
from releaseCommon import Version

root_path = os.path.join(catchPath, 'src')
starting_header = os.path.join(root_path, 'catch2', 'catch_all.hpp')
output_header = os.path.join(catchPath, 'extras', 'catch_amalgamated.hpp')
output_cpp = os.path.join(catchPath, 'extras', 'catch_amalgamated.cpp')

# REUSE-IgnoreStart

# These are the copyright comments in each file, we want to ignore them
copyright_lines = [
'//              Copyright Catch2 Authors\n',
'// Distributed under the Boost Software License, Version 1.0.\n',
'//   (See accompanying file LICENSE.txt or copy at\n',
'//        https://www.boost.org/LICENSE_1_0.txt)\n',
'// SPDX-License-Identifier: BSL-1.0\n',
]

# The header of the amalgamated file: copyright information + explanation
# what this file is.
file_header = '''\

//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

//  Catch v{version_string}
//  Generated: {generation_time}
//  ----------------------------------------------------------
//  This file is an amalgamation of multiple different files.
//  You probably shouldn't edit it directly.
//  ----------------------------------------------------------
'''

# REUSE-IgnoreEnd

# Returns file header with proper version string and generation time
def formatted_file_header(version):
    return file_header.format(version_string=version.getVersionString(),
                              generation_time=datetime.datetime.now())

# Which headers were already concatenated (and thus should not be
# processed again)
concatenated_headers = set()

internal_include_parser = re.compile(r'\s*#include <(catch2/.*)>.*')

def concatenate_file(out, filename: str, expand_headers: bool) -> int:
    # Gathers statistics on how many headers were expanded
    concatenated = 1
    with open(filename, mode='r', encoding='utf-8') as input:
        for line in input:
            if line in copyright_lines:
                continue
            m = internal_include_parser.match(line)
            # anything that isn't a Catch2 header can just be copied to
            # the resulting file
            if not m:
                out.write(line)
                continue

            # TBD: We can also strip out include guards from our own
            # headers, but it wasn't worth the time at the time of writing
            # this script.

            # We do not want to expand headers for the cpp file
            # amalgamation but neither do we want to copy them to output
            if not expand_headers:
                continue

            next_header = m.group(1)
            # We have to avoid re-expanding the same header over and
            # over again, or the header will end up with couple
            # hundred thousands lines (~300k as of preview3 :-) )
            if next_header in concatenated_headers:
                continue

            # Skip including the auto-generated user config file,
            # because it has not been generated yet at this point.
            # The code around it should be written so that just not including
            # it is equivalent with all-default user configuration.
            if next_header == 'catch2/catch_user_config.hpp':
                concatenated_headers.add(next_header)
                continue

            concatenated_headers.add(next_header)
            concatenated += concatenate_file(out, os.path.join(root_path, next_header), expand_headers)

    return concatenated


def generate_header():
    with open(output_header, mode='w', encoding='utf-8') as header:
        header.write(formatted_file_header(Version()))
        header.write('#ifndef CATCH_AMALGAMATED_HPP_INCLUDED\n')
        header.write('#define CATCH_AMALGAMATED_HPP_INCLUDED\n')
        print('Concatenated {} headers'.format(concatenate_file(header, starting_header, True)))
        header.write('#endif // CATCH_AMALGAMATED_HPP_INCLUDED\n')

def generate_cpp():
    from glob import glob
    cpp_files = sorted(glob(os.path.join(root_path, 'catch2', '**/*.cpp'), recursive=True))
    with open(output_cpp, mode='w', encoding='utf-8') as cpp:
        cpp.write(formatted_file_header(Version()))
        cpp.write('\n#include "catch_amalgamated.hpp"\n')
        concatenate_file(cpp, os.path.join(root_path, 'catch2/internal/catch_windows_h_proxy.hpp'), False)
        for file in cpp_files:
            concatenate_file(cpp, file, False)
    print('Concatenated {} cpp files'.format(len(cpp_files)))

if __name__ == "__main__":
    generate_header()
    generate_cpp()


# Notes:
# * For .cpp files, internal includes have to be stripped and rewritten
# * for .hpp files, internal includes have to be resolved and included
# * The .cpp file needs to start with `#include "catch_amalgamated.hpp"
# * include guards can be left/stripped, doesn't matter
# * *.cpp files should be included sorted, to minimize diffs between versions
# * *.hpp files should also be somehow sorted -> use catch_all.hpp as the
# *       entrypoint
# * allow disabling main in the .cpp amalgamation
