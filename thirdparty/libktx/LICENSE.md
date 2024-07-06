LICENSE file for the KhronosGroup/KTX-Software project    {#license}
======================================================

<!--
 Can't put at start. Doxygen requires page title on first line.
 Copyright 2013-2020 Mark Callow 
 SPDX-License-Identifier: Apache-2.0
-->

Files unique to this repository generally fall under the Apache 2.0 license
with copyright holders including Mark Callow, the KTX-Software author; The
Khronos Group Inc., which has supported KTX development; and other
contributors to the KTX project.

Because KTX-Software incorporates material and contributions from many other
projects, which often have their own licenses, there are many other licenses
in use in this repository. While there are many licenses in this repository,
with rare exceptions all are open source licenses that we believe to be
mutually compatible.

The complete text of each of the licenses used in this repository is found
in LICENSES/*.txt . Additionally, we have updated the repository to pass the
REUSE compliance checker tool (see https://reuse.software/). REUSE verifies
that every file in a git repository either incorporates a license, or that
the license is present in auxiliary files such as .reuse/dep5 . To obtain a
bill of materials for the repository identifying the license for each file,
install the REUSE tool and run

    reuse spdx

inside the repository.

## Special Cases

The file lib/etcdec.cxx is not open source. It is made available under the
terms of an Ericsson license, found in the file itself.
