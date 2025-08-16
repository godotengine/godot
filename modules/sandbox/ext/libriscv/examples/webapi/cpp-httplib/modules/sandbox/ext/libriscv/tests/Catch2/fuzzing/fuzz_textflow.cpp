
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
//By Paul Dreik 2020

#include <catch2/internal/catch_textflow.hpp>

#include "NullOStream.h"

#include <string>
#include <string_view>


template<class Callback>
void split(const char *Data, size_t Size, Callback callback) {

    using namespace std::literals;
    constexpr auto sep="\n~~~\n"sv;

    std::string_view remainder(Data,Size);
    for (;;) {
        auto pos=remainder.find(sep);
        if(pos==std::string_view::npos) {
            //not found. use the remainder and exit
            callback(remainder);
            return;
        } else {
            //found. invoke callback on the first part, then proceed with the rest.
            callback(remainder.substr(0,pos));
            remainder=remainder.substr(pos+sep.size());
        }
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {

    Catch::TextFlow::Columns columns;

    // break the input on separator
    split((const char*)Data,Size,[&](std::string_view word) {
        columns+=Catch::TextFlow::Column(std::string(word));
    });

    NullOStream nul;
    nul << columns;

    return 0;
}

