# SPIR-V Headers

This repository contains machine-readable files for the
[SPIR-V Registry](https://www.khronos.org/registry/spir-v/).
This includes:

* Header files for various languages.
* JSON files describing the grammar for the SPIR-V core instruction set
  and the extended instruction sets.
* The XML registry file.
* A tool to build the headers from the JSON grammar.

Headers are provided in the [include](include) directory, with up-to-date
headers in the `unified1` subdirectory. Older headers are provided according to
their version.

In contrast, the XML registry file has a linear history, so it is
not tied to SPIR-V specification versions.

## How is this repository updated?

When a new version or revision of the SPIR-V specification is published,
the SPIR-V Working Group will push new commits onto master, updating
the files under [include](include).

The SPIR-V XML registry file is updated by Khronos whenever a new enum range is allocated.

Pull requests can be made to 
- request allocation of new enum ranges in the XML registry file
- reserve specific tokens in the JSON grammar

## How to install the headers

```
mkdir build
cd build
cmake ..
cmake --build . --target install
```

Then, for example, you will have `/usr/local/include/spirv/unified1/spirv.h`

If you want to install them somewhere else, then use
`-DCMAKE_INSTALL_PREFIX=/other/path` on the first `cmake` command.

## Using the headers without installing

A CMake-based project can use the headers without installing, as follows:

1. Add an `add_subdirectory` directive to include this source tree.
2. Use `${SPIRV-Headers_SOURCE_DIR}/include}` in a `target_include_directories`
   directive.
3. In your C or C++ source code use `#include` directives that explicitly mention
   the `spirv` path component.
```
#include "spirv/unified1/GLSL.std.450.h"
#include "spirv/unified1/OpenCL.std.h"
#include "spirv/unified1/spirv.hpp"
```

See also the [example](example/) subdirectory.  But since that example is
*inside* this repostory, it doesn't use and `add_subdirectory` directive.

## Generating the headers from the JSON grammar

This will generally be done by Khronos, for a change to the JSON grammar.
However, the project for the tool to do this is included in this repository,
and can be used to test a PR, or even to include the results in the PR.
This is not required though.

The header-generation project is under the `tools/buildHeaders` directory.
Use CMake to build the project, in a `build` subdirectory (under `tools/buildHeaders`).
There is then a bash script at `bin/makeHeaders` that shows how to use the built
header-generator binary to generate the headers from the JSON grammar.
(Execute `bin/makeHeaders` from the `tools/buildHeaders` directory.)

Notes:
- this generator is used in a broader context within Khronos to generate the specification,
  and that influences the languages used, for legacy reasons
- the C++ structures built may similarly include more than strictly necessary, for the same reason

## FAQ

* *How are different versions published?*

  The multiple versions of the headers have been simplified into a
  single `unified1` view. The JSON grammar has a "version" field saying
  what version things first showed up in.

* *How do you handle the evolution of extended instruction sets?*

  Extended instruction sets evolve asynchronously from the core spec.
  Right now there is only a single version of both the GLSL and OpenCL
  headers.  So we don't yet have a problematic example to resolve.

## License
<a name="license"></a>
```
Copyright (c) 2015-2018 The Khronos Group Inc.

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and/or associated documentation files (the
"Materials"), to deal in the Materials without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Materials, and to
permit persons to whom the Materials are furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Materials.

MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS
KHRONOS STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS
SPECIFICATIONS AND HEADER INFORMATION ARE LOCATED AT
   https://www.khronos.org/registry/

THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
```
