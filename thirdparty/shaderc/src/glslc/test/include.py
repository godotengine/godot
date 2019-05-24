# Copyright 2015 The Shaderc Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import expect
from glslc_test_framework import inside_glslc_testsuite
from placeholder import SpecializedString
from environment import File, Directory


@inside_glslc_testsuite('Include')
class VerifyIncludeOneSibling(expect.StdoutMatch):
    """Tests #including a sibling file."""

    environment = Directory('.', [
        File('a.vert', '#version 140\ncontent a\n#include "b"\n'),
        File('b', 'content b\n')])

    glslc_args = ['-E', 'a.vert']

    expected_stdout = \
"""#version 140
#extension GL_GOOGLE_include_directive : enable
#line 0 "a.vert"

content a
#line 0 "b"
 content b
#line 3 "a.vert"

"""

@inside_glslc_testsuite('Include')
class VerifyIncludeNotFound(expect.ErrorMessage):
    """Tests #including a not existing sibling file."""

    environment = Directory('.', [
        File('a.vert', '#version 140\ncontent a\n#include "b"\n')])

    glslc_args = ['-E', 'a.vert']
    expected_error = [
        "a.vert:3: error: '#include' : Cannot find or open include file. for header name: b\n",
        '1 error generated.\n'
    ]

@inside_glslc_testsuite('Include')
class VerifyCompileIncludeOneSibling(expect.ValidObjectFile):
    """Tests #including a sibling file via full compilation."""

    environment = Directory('.', [
        File('a.vert', '#version 140\nvoid foo(){}\n#include "b"\n'),
        File('b', 'void main(){foo();}\n')])

    glslc_args = ['a.vert']

@inside_glslc_testsuite('Include')
class VerifyIncludeWithoutNewline(expect.ErrorMessageSubstr):
    """Tests a #include without a newline."""

    environment = Directory('.', [
        File('a.vert', '#version 140\n#include "b"'),
        File('b', 'content b\n')])

    glslc_args = ['-E', 'a.vert']

    expected_error_substr = 'expected newline after header name: b'


@inside_glslc_testsuite('Include')
class VerifyCompileIncludeWithoutNewline(expect.ValidObjectFile):
    """Tests a #include without a newline via full compilation."""

    environment = Directory('.', [
        File('a.vert',
             """#version 140
             void main
             #include "b"
             """),
        File('b',
             """#define PAR ()
             PAR{}
             """)])

    glslc_args = ['a.vert']

@inside_glslc_testsuite('Include')
class VerifyIncludeTwoSiblings(expect.StdoutMatch):
    """Tests #including two sibling files."""

    environment = Directory('.', [
        File('b.vert', '#version 140\n#include "a"\ncontent b\n#include "c"\n'),
        File('a', 'content a\n'),
        File('c', 'content c\n')])

    glslc_args = ['-E', 'b.vert']

    expected_stdout = \
"""#version 140
#extension GL_GOOGLE_include_directive : enable
#line 0 "b.vert"

#line 0 "a"
content a
#line 2 "b.vert"
 content b
#line 0 "c"
 content c
#line 4 "b.vert"

"""

@inside_glslc_testsuite('Include')
class VerifyCompileIncludeTwoSiblings(expect.ValidObjectFile):
    """Tests #including two sibling files via full compilation."""

    environment = Directory('.', [
        File('b.vert',
             """#version 140
             #include "a"
             void bfun(){afun();}
             #include "c"
             """),
        File('a',
             """void afun(){}
             #define BODY {}
             """),
        File('c', 'void main() BODY\n')])

    glslc_args = ['b.vert']

@inside_glslc_testsuite('Include')
class VerifyNestedIncludeAmongSiblings(expect.StdoutMatch):
    """Tests #include inside #included sibling files."""

    environment = Directory('.', [
        File('a.vert', '#version 140\n#include "b"\ncontent a\n'),
        File('b', 'content b\n#include "c"\n'),
        File('c', 'content c\n')])

    glslc_args = ['-E', 'a.vert']

    expected_stdout = \
"""#version 140
#extension GL_GOOGLE_include_directive : enable
#line 0 "a.vert"

#line 0 "b"
content b
#line 0 "c"
 content c
#line 2 "b"
#line 2 "a.vert"
 content a
"""

@inside_glslc_testsuite('Include')
class VerifyCompileNestedIncludeAmongSiblings(expect.ValidObjectFile):
    """Tests #include inside #included sibling files via full compilation."""

    environment = Directory('.', [
        File('a.vert',
             """#version 140
             #define BODY {}
             #include "b"
             void main(){cfun();}
             """),
        File('b',
             """void bfun() BODY
             #include "c"
             """),
        File('c',
             """#define BF bfun()
             void cfun(){BF;}
             """)])

    glslc_args = ['a.vert']

@inside_glslc_testsuite('Include')
class VerifyIncludeSubdir(expect.StdoutMatch):
    """Tests #including a file from a subdirectory."""

    environment = Directory('.', [
        File('a.vert', '#version 140\ncontent a1\n#include "subdir/a"\ncontent a2\n'),
        Directory('subdir', [File('a', 'content suba\n')])])

    glslc_args = ['-E', 'a.vert']

    expected_stdout = \
"""#version 140
#extension GL_GOOGLE_include_directive : enable
#line 0 "a.vert"

content a1
#line 0 "subdir/a"
 content suba
#line 3 "a.vert"
 content a2
"""

@inside_glslc_testsuite('Include')
class VerifyCompileIncludeSubdir(expect.ValidObjectFile):
    """Tests #including a file from a subdirectory via full compilation."""

    environment = Directory('.', [
        File('a.vert',
             """#version 140
             #define BODY {}
             #include "subdir/a"
             void afun()BODY
             """),
        Directory('subdir', [File('a', 'void main() BODY\n')])])

    glslc_args = ['a.vert']

@inside_glslc_testsuite('Include')
class VerifyIncludeDeepSubdir(expect.StdoutMatch):
    """Tests #including a file from a subdirectory nested a few levels down."""

    environment = Directory('.', [
        File('a.vert',
             '#version 140\ncontent a1\n#include "dir/subdir/subsubdir/a"\ncontent a2\n'),
        Directory('dir', [
            Directory('subdir', [
                Directory('subsubdir', [File('a', 'content incl\n')])])])])

    glslc_args = ['-E', 'a.vert']

    expected_stdout = \
"""#version 140
#extension GL_GOOGLE_include_directive : enable
#line 0 "a.vert"

content a1
#line 0 "dir/subdir/subsubdir/a"
 content incl
#line 3 "a.vert"
 content a2
"""

@inside_glslc_testsuite('Include')
class VerifyCompileIncludeDeepSubdir(expect.ValidObjectFile):
    """Tests #including a file from a subdirectory nested a few levels down
       via full compilation."""

    environment = Directory('.', [
        File('a.vert',
             """#version 140
             #define BODY {}
             #include "dir/subdir/subsubdir/a"
             void afun()BODY
             """),
        Directory('dir', [
            Directory('subdir', [
                Directory('subsubdir', [File('a', 'void main() BODY\n')])])])])

    glslc_args = ['a.vert']


@inside_glslc_testsuite('Include')
class TestWrongPoundVersionInIncludingFile(expect.ValidObjectFileWithWarning):
    """Tests that warning message for #version directive in the including file
    has the correct filename."""

    environment = Directory('.', [
        File('a.vert', '#version 100000000\n#include "b.glsl"\n'),
        File('b.glsl', 'void main() {}\n')])
    glslc_args = ['-c', '-std=400', 'a.vert']

    expected_warning = [
        'a.vert: warning: (version, profile) forced to be (400, none),'
        ' while in source code it is (100000000, none)\n'
        '1 warning generated.\n'
    ]


# TODO(antiagainst): now #version in included files results in an error.
# Fix this after #version in included files are supported.
# TODO(dneto): I'm not sure what the expected result should be.
@inside_glslc_testsuite('Include')
class TestWrongPoundVersionInIncludedFile(expect.ErrorMessage):
    """Tests that warning message for #version directive in the included file
    has the correct filename."""

    environment = Directory('.', [
        File('a.vert', '#version 140\n#include "b.glsl"\nvoid main() {}'),
        File('b.glsl', '#version 10000000\n')])
    glslc_args = ['-E', 'a.vert']

    expected_error = [
        "b.glsl:1: error: '#version' : must occur first in shader\n",
        '1 error generated.\n'
    ]


@inside_glslc_testsuite('Include')
class VerifyRelativeInclude(expect.StdoutMatch):
    """Tests #including a relative sibling."""

    environment = Directory('.', [
        File('a.vert', '#version 140\ncontent a\n#include "foo/b.glsl"\n'),
        Directory('foo', [
            File('b.glsl', '#include "c.glsl"\ncontent b\n'),
            File('c.glsl', 'content c\n')
        ])])

    glslc_args = ['-E', 'a.vert']

    expected_stdout = \
"""#version 140
#extension GL_GOOGLE_include_directive : enable
#line 0 "a.vert"

content a
#line 0 "foo/b.glsl"
#line 0 "foo/c.glsl"
 content c
#line 1 "foo/b.glsl"
 content b
#line 3 "a.vert"

"""

@inside_glslc_testsuite('Include')
class VerifyNestedRelativeInclude(expect.StdoutMatch):
    """Tests #including a relative child file."""

    environment = Directory('.', [
        File('a.vert', '#version 140\ncontent a\n#include "foo/b.glsl"\n'),
        Directory('foo', [
            File('b.glsl', '#include "bar/c.glsl"\ncontent b\n'),
            Directory('bar', [
                File('c.glsl', 'content c\n')
            ])
        ])
    ])

    glslc_args = ['-E', 'a.vert']

    expected_stdout = \
"""#version 140
#extension GL_GOOGLE_include_directive : enable
#line 0 "a.vert"

content a
#line 0 "foo/b.glsl"
#line 0 "foo/bar/c.glsl"
 content c
#line 1 "foo/b.glsl"
 content b
#line 3 "a.vert"

"""


@inside_glslc_testsuite('Include')
class VerifyRelativeIncludeWithDashI(expect.StdoutMatch):
    """Tests #including a relative file from a -I directory."""

    environment = Directory('.', [
        File('a.vert', '#version 140\ncontent a\n#include "bar/b.glsl"\n'),
        Directory('foo', [
            Directory('bar', [
                File('b.glsl', '#include "c.glsl"\ncontent b\n'),
            ]),
            File('c.glsl', 'content c\n')
        ])
    ])

    glslc_args = ['-E', 'a.vert', '-Ifoo']

    expected_stdout = \
"""#version 140
#extension GL_GOOGLE_include_directive : enable
#line 0 "a.vert"

content a
#line 0 "foo/bar/b.glsl"
#line 0 "foo/c.glsl"
 content c
#line 1 "foo/bar/b.glsl"
 content b
#line 3 "a.vert"

"""


@inside_glslc_testsuite('Include')
class VerifyRelativeOverridesDashI(expect.StdoutMatch):
    """Tests that relative includes override -I parameters."""

    environment = Directory('.', [
        File('a.vert', '#version 140\ncontent a\n#include "b.glsl"\n'),
        File('b.glsl', 'content base_b\n'),
        Directory('foo', [
            File('b.glsl', '#include "c.glsl"\ncontent b\n'),
            File('c.glsl', 'content c\n')
        ])
    ])

    glslc_args = ['-E', 'a.vert', '-Ifoo']

    expected_stdout = \
"""#version 140
#extension GL_GOOGLE_include_directive : enable
#line 0 "a.vert"

content a
#line 0 "b.glsl"
 content base_b
#line 3 "a.vert"

"""


@inside_glslc_testsuite('Include')
class VerifyRelativeParent(expect.StdoutMatch):
    """Tests #including a parent file."""

    environment = Directory('.', [
        File('a.vert', '#version 140\ncontent a\n#include "b.glsl"\n'),
        File('c.glsl', 'content c\n'),
        Directory('foo', [
            File('b.glsl', '#include "../c.glsl"\ncontent b\n')
        ])
    ])

    glslc_args = ['-E', 'a.vert', '-Ifoo']

    expected_stdout = \
"""#version 140
#extension GL_GOOGLE_include_directive : enable
#line 0 "a.vert"

content a
#line 0 "foo/b.glsl"
#line 0 "foo/../c.glsl"
 content c
#line 1 "foo/b.glsl"
 content b
#line 3 "a.vert"

"""


@inside_glslc_testsuite('Include')
class VerifyRelativeNeighbourDirectory(expect.StdoutMatch):
    """Tests #including a relative file in a neighbour directory."""

    environment = Directory('.', [
        File('a.vert', '#version 140\ncontent a\n#include "foo/b.glsl"\n'),
        Directory('foo', [
            File('b.glsl', '#include "../bar/c.glsl"\ncontent b\n')
        ]),
        Directory('bar', [
            File('c.glsl', 'content c\n')
        ])
    ])

    glslc_args = ['-E', 'a.vert']

    expected_stdout = \
"""#version 140
#extension GL_GOOGLE_include_directive : enable
#line 0 "a.vert"

content a
#line 0 "foo/b.glsl"
#line 0 "foo/../bar/c.glsl"
 content c
#line 1 "foo/b.glsl"
 content b
#line 3 "a.vert"

"""


@inside_glslc_testsuite('Include')
class VerifyRelativeOnlyToSelf(expect.ErrorMessage):
    """Tests that a relative includes are only relative to yourself."""

    environment = Directory('.', [
        File('a.vert', '#version 140\ncontent a\n#include "foo/b.glsl"\n'),
        File('c.glsl', 'content c\n'),
        Directory('foo', [
            File('b.glsl', '#include "c.glsl"\ncontent b\n')
        ]),
    ])

    glslc_args = ['-E', 'a.vert']

    expected_error = [
        "foo/b.glsl:1: error: '#include' : "
        'Cannot find or open include file. for header name: c.glsl\n',
        '1 error generated.\n'
    ]


@inside_glslc_testsuite('Include')
class VerifyRelativeFromAbsolutePath(expect.StdoutMatch):
    """Tests that absolute files can relatively include."""

    environment = Directory('.', [
        File('a.vert', '#version 140\ncontent a\n#include "b.glsl"\n'),
        File('b.glsl', 'content b\n')
    ])

    glslc_args = ['-E', SpecializedString('$directory/a.vert')]

    expected_stdout = SpecializedString(
"""#version 140
#extension GL_GOOGLE_include_directive : enable
#line 0 "$directory/a.vert"

content a
#line 0 "$directory/b.glsl"
 content b
#line 3 "$directory/a.vert"

""")


@inside_glslc_testsuite('Include')
class VerifyDashIAbsolutePath(expect.StdoutMatch):
    """Tests that -I parameters can be absolute paths."""

    environment = Directory('.', [
        File('a.vert', '#version 140\ncontent a\n#include "b.glsl"\n'),
        Directory('foo', {
            File('b.glsl', 'content b\n')
        })
    ])

    glslc_args = ['-E', 'a.vert', '-I', SpecializedString('$directory/foo')]

    expected_stdout = SpecializedString(
"""#version 140
#extension GL_GOOGLE_include_directive : enable
#line 0 "a.vert"

content a
#line 0 "$directory/foo/b.glsl"
 content b
#line 3 "a.vert"

""")
