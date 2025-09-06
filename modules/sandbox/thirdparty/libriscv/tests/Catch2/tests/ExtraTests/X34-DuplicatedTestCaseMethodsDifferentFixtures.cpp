
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

/**\file
 * Checks that test case methods with different class, but same name and
 * tags name and tags are not reported as error.
 */

#include <catch2/catch_test_macros.hpp>

class TestCaseFixture1 {
public:
    int m_a;
};

class TestCaseFixture2 {
public:
    int m_a;
};

TEST_CASE_METHOD(TestCaseFixture1, "A test case with duplicated name and tags", "[tag1]") {}
TEST_CASE_METHOD(TestCaseFixture2, "A test case with duplicated name and tags", "[tag1]") {}
