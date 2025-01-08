// SPDX-License-Identifier: MIT OR MPL-2.0 OR LGPL-2.1-or-later OR GPL-2.0-or-later
// Copyright 2015, SIL International, All rights reserved.


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
