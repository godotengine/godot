
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
//By Paul Dreik 2020

#include <catch2/internal/catch_xmlwriter.hpp>

#include "NullOStream.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {

    std::string buf(Data,Data+Size);
    NullOStream nul;
    Catch::XmlEncode encode(buf);
    encode.encodeTo(nul);
    return 0;
}

