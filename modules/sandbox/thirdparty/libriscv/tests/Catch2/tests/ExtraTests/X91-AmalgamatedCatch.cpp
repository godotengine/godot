
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

/**\file
 * This file serves as a simple compilation test against the amalgamated
 * version of Catch2.
 */

#include "catch_amalgamated.hpp"

TEST_CASE("Just a dummy test") {
    auto i = GENERATE(1, 2, 3);
    SECTION("a") {
        REQUIRE(i != 4);
    }
    SECTION("b") {
        CHECK(i != 5);
    }
    REQUIRE_THAT(1,
                 Catch::Matchers::Predicate<int>([](int i) {
                     return i % 2 == 1;
                 }));
}

TEMPLATE_TEST_CASE("Trivial template test case", "", char, short) {
    STATIC_REQUIRE(sizeof(TestType) <= sizeof(int));
}

TEST_CASE("Benchmark test", "[!benchmark]") {
    BENCHMARK("Allocation benchmark") {
        return std::vector<int>(10);
    };
}
