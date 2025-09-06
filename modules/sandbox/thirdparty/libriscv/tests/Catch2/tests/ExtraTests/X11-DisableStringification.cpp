
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

/**\file
 * Test that stringification of original expression can be disabled.
 *
 * This is a workaround for VS 2017, 2019 issue with Raw String literals
 * and preprocessor token pasting.
 */


#include <catch2/catch_test_macros.hpp>

namespace {
    struct Hidden {};

    bool operator==(Hidden, Hidden) { return true; }
}

TEST_CASE("DisableStringification") {
    REQUIRE( Hidden{} == Hidden{} );
}
