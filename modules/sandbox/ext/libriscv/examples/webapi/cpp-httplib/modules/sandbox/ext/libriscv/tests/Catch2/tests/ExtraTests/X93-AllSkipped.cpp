
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/catch_test_macros.hpp>

TEST_CASE( "this test case is being skipped" ) { SKIP(); }

TEST_CASE( "all sections in this test case are being skipped" ) {
    SECTION( "A" ) { SKIP(); }
    SECTION( "B" ) { SKIP(); }
}
