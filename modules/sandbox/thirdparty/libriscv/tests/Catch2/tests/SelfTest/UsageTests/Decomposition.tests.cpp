
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <iostream>
#include <cstdio>

namespace {

struct truthy {
    truthy(bool b):m_value(b){}
    operator bool() const {
        return false;
    }
    bool m_value;
};

std::ostream& operator<<(std::ostream& o, truthy) {
    o << "Hey, its truthy!";
    return o;
}

} // end anonymous namespace

#include <catch2/catch_test_macros.hpp>

TEST_CASE( "Reconstruction should be based on stringification: #914" , "[Decomposition][failing][.]") {
    CHECK(truthy(false));
}

TEST_CASE("#1005: Comparing pointer to int and long (NULL can be either on various systems)", "[Decomposition][approvals]") {
    FILE* fptr = nullptr;
    REQUIRE( fptr == 0 );
    REQUIRE_FALSE( fptr != 0 );
    REQUIRE( fptr == 0l );
    REQUIRE_FALSE( fptr != 0l );
}
