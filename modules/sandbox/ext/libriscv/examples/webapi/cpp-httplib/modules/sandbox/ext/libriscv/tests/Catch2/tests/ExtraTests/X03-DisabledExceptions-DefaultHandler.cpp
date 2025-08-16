
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_predicate.hpp>

TEST_CASE("Tests that run") {
    // All of these should be run and be reported
    CHECK(1 == 2);
    CHECK(1 == 1);
    CHECK(1 != 3);
    CHECK(1 == 4);
}



TEST_CASE("Tests that abort") {
    // Avoid abort and other exceptional exits -- there is no way
    // to tell CMake that abort is the desired outcome of a test.
    std::set_terminate([](){exit(1);});
    REQUIRE(1 == 1);
    REQUIRE(1 != 2);
    REQUIRE(1 == 3);
    // We should not get here, because the test above aborts
    REQUIRE(1 != 4);
}

TEST_CASE( "Misc. macros to check that they compile without exceptions" ) {
    BENCHMARK( "simple benchmark" ) { return 1 * 2 + 3; };
    REQUIRE_THAT( 1,
                  Catch::Matchers::Predicate<int>( []( int i ) { return i == 1; } ) );
}
