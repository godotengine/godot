/*  GRAPHITE2 LICENSING

    Copyright 2015, SIL International
    All rights reserved.

    This library is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published
    by the Free Software Foundation; either version 2.1 of License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should also have received a copy of the GNU Lesser General Public
    License along with this library in the file named "LICENSE".
    If not, write to the Free Software Foundation, 51 Franklin Street,
    Suite 500, Boston, MA 02110-1335, USA or visit their web page on the
    internet at http://www.fsf.org/licenses/lgpl.html.

Alternatively, the contents of this file may be used under the terms of the
Mozilla Public License (http://mozilla.org/MPL) or the GNU General Public
License, as published by the Free Software Foundation, either version 2
of the License or (at your option) any later version.
*/

#pragma once

#include <cstddef>

namespace lz4
{

// decompress an LZ4 block
// Parameters:
//      @in         -   Input buffer containing an LZ4 block.
//      @in_size    -   Size of the input LZ4 block in bytes.
//      @out        -   Output buffer to hold decompressed results.
//      @out_size   -   The size of the buffer pointed to by @out.
// Invariants:
//      @in         -   This buffer must be at least 1 machine word in length,
//                      regardless of the actual LZ4 block size.
//      @in_size    -   This must be at least 4 and must also be <= to the
//                      allocated buffer @in.
//      @out        -   This must be bigger than the input buffer and at least
//                      13 bytes.
//      @out_size   -   Must always be big enough to hold the expected size.
// Return:
//      -1          -  Decompression failed.
//      size        -  Actual number of bytes decompressed.
int decompress(void const *in, size_t in_size, void *out, size_t out_size);

} // end of namespace shrinker
