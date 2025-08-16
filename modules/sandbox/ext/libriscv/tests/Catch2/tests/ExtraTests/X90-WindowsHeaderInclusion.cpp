
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

/**\file
 * Test that the Catch2 header compiles even after including windows.h
 * without defining NOMINMAX first.
 *
 * As an FYI, if you do that, you are wrong.
 */

#include <windows.h>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("Catch2 did survive compilation with windows.h", "[compile-test]") {
    SUCCEED();
}
