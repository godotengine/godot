
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/catch_test_macros.hpp>

TEST_CASE("@Script[C:\\EPM1A]=x;\"SCALA_ZERO:\"", "[script regressions]"){}
TEST_CASE("Some test") {}
TEST_CASE( "Let's have a test case with a long name. Longer. No, even longer. "
           "Really looooooooooooong. Even longer than that. Multiple lines "
           "worth of test name. Yep, like this." ) {}
TEST_CASE( "And now a test case with weird tags.", "[tl;dr][tl;dw][foo,bar]" ) {}
// Also check that we handle tests on class, which have name in output as 'class-name', not 'name'.
class TestCaseFixture {
public:
    int m_a;
};

TEST_CASE_METHOD(TestCaseFixture, "A test case as method", "[tagstagstags]") {}
