
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
//By Paul Dreik 2020

#include <catch2/internal/catch_test_spec_parser.hpp>
#include <catch2/internal/catch_tag_alias_registry.hpp>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {

    Catch::TagAliasRegistry tar;
    Catch::TestSpecParser tsp(tar);

    std::string buf(Data,Data+Size);
    tsp.parse(buf);

    return 0;
}
