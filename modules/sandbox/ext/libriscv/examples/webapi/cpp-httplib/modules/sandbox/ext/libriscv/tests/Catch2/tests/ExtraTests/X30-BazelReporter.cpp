
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

/**\file
 * Test the Bazel report functionality with a simple set
 * of dummy test cases.
 */

#include <catch2/catch_test_macros.hpp>

TEST_CASE( "Passing test case" ) { REQUIRE( 1 == 1 ); }
TEST_CASE( "Failing test case" ) { REQUIRE( 2 == 1 ); }
