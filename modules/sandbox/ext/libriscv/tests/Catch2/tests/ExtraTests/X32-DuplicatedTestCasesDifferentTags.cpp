
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

/**\file
 * Checks that test cases with identical name but different tags are
 * not reported as an error.
 */

#include <catch2/catch_test_macros.hpp>

TEST_CASE("A test case with duplicated name but different tags", "[tag1]") {}
TEST_CASE("A test case with duplicated name but different tags", "[tag2]") {}
