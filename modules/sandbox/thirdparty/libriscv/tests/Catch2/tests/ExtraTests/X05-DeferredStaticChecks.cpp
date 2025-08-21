
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

/**\file
 * Checks that when `STATIC_CHECK` is deferred to runtime and fails, it
 * does not abort the test case.
 */

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Deferred static checks") {
    STATIC_CHECK(1 == 2);
    STATIC_CHECK_FALSE(1 != 2);
    // This last assertion must be executed too
    CHECK(1 == 2);
}
