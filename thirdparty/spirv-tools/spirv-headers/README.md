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

[The SPIR-V XML registry file](include/spirv/spir-v.xml)
is updated by Khronos whenever a new enum range is allocated.

Pull requests can be made to
- request allocation of new enum ranges in the XML registry file
- register a new magic number for a SPIR-V generator
- reserve specific tokens in the JSON grammar

### Registering a SPIR-V Generator Magic Number

Tools that generate SPIR-V should use a magic number in the SPIR-V to help identify the
generator.

Care should be taken to follow existing precedent in populating the details of reserved tokens.
This includes:
- keeping generator numbers in numeric order
- filling out all the existing fields

### Reserving tokens in the JSON grammar

Care should be taken to follow existing precedent in populating the details of reserved tokens.
This includes:
- pointing to what extension has more information, when possible
- keeping enumerants in numeric order
- when there are aliases, listing the preferred spelling first
- adding the statement `"version" : "None"`

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

### Using CMake
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

### Using Bazel
A Bazel-based project can use the headers without installing, as follows:

1. Add SPIRV-Headers as a submodule of your project, and add a
`local_repository` to your `WORKSPACE` file. For example, if you place
SPIRV-Headers under `external/spirv-headers`, then add the following to your
`WORKSPACE` file:

```
local_repository(
    name = "spirv_headers",
    path = "external/spirv-headers",
)
```

2. Add one of the following to the `deps` attribute of your build target based
on your needs:
```
@spirv_headers//:spirv_c_headers
@spirv_headers//:spirv_cpp_headers
@spirv_headers//:spirv_cpp11_headers
```

For example:

```
cc_library(
  name = "project",
  srcs = [
    # Path to project sources
  ],
  hdrs = [
    # Path to project headers
  ],
  deps = [
    "@spirv_tools//:spirv_c_headers",
    # Other dependencies,
  ],
)
```

3. In your C or C++ source code use `#include` directives that explicitly mention
   the `spirv` path component.
```
#include "spirv/unified1/GLSL.std.450.h"
#include "spirv/unified1/OpenCL.std.h"
#include "spirv/unified1/spirv.hpp"
```

## Generating headers from the JSON grammar for the SPIR-V core instruction set

This will generally be done by Khronos, for a change to the JSON grammar.
However, the project for the tool to do this is included in this repository,
and can be used to test a PR, or even to include the results in the PR.
This is not required though.

The header-generation project is under the `tools/buildHeaders` directory.
Use CMake to build and install the project, in a `build` subdirectory (under `tools/buildHeaders`).
There is then a bash script at `bin/makeHeaders` that shows how to use the built
header-generator binary to generate the headers from the JSON grammar.
(Execute `bin/makeHeaders` from the `tools/buildHeaders` directory.)
Here's a complete example:

```
cd tools/buildHeaders
mkdir build
cd build
cmake ..
cmake --build . --target install
cd ..
./bin/makeHeaders
```

Notes:
- this generator is used in a broader context within Khronos to generate the specification,
  and that influences the languages used, for legacy reasons
- the C++ structures built may similarly include more than strictly necessary, for the same reason

## Generating C headers for extended instruction sets

The [GLSL.std.450.h](include/spirv/unified1/GLSL.std.450.h)
and [OpenCL.std.h](include/spirv/unified1/OpenCL.std.h) extended instruction set headers
are maintained manually.

The C/C++ header for each of the other extended instruction sets
is generated from the corresponding JSON grammar file.  For example, the
[OpenCLDebugInfo100.h](include/spirv/unified1/OpenCLDebugInfo100.h) header
is generated from the
[extinst.opencl.debuginfo.100.grammar.json](include/spirv/unified1/extinst.opencl.debuginfo.100.grammar.json)
grammar file.

To generate these C/C++ headers, first make sure `python3` is in your PATH, then
invoke the build script as follows:
```
cd tools/buildHeaders
python3 bin/makeExtinstHeaders.py
```

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
Copyright (c) 2015-2024 The Khronos Group Inc.

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
