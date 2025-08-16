
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

/**\file
 * Test that CATCH_CONFIG_DISABLE turns off TEST_CASE autoregistration
 * and expressions in assertion macros are not run.
 */

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_predicate.hpp>

#include <iostream>

struct foo {
    foo() { REQUIRE_NOTHROW( print() ); }
    void print() const { std::cout << "This should not happen\n"; }
};

#if defined( __clang__ )
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wglobal-constructors"
#endif
// Construct foo, but `foo::print` should not be run
static foo f;

#if defined( __clang__ )
// The test is unused since the registration is disabled
#    pragma clang diagnostic ignored "-Wunused-function"
#endif

// This test should not be run, because it won't be registered
TEST_CASE( "Disabled Macros" ) {
    CHECK( 1 == 2 );
    REQUIRE( 1 == 2 );
    std::cout << "This should not happen\n";
    FAIL();

    // Test that static assertions don't fire when macros are disabled
    STATIC_CHECK( 0 == 1 );
    STATIC_REQUIRE( !true );

    CAPTURE( 1 );
    CAPTURE( 1, "captured" );

    REQUIRE_THAT( 1,
                  Catch::Matchers::Predicate( []( int ) { return false; } ) );
    BENCHMARK( "Disabled benchmark" ) { REQUIRE( 1 == 2 ); };
}

struct DisabledFixture {};

TEST_CASE_PERSISTENT_FIXTURE( DisabledFixture, "Disabled Persistent Fixture" ) {
    CHECK( 1 == 2 );
    REQUIRE( 1 == 2 );
    std::cout << "This should not happen\n";
    FAIL();

    // Test that static assertions don't fire when macros are disabled
    STATIC_CHECK( 0 == 1 );
    STATIC_REQUIRE( !true );

    CAPTURE( 1 );
    CAPTURE( 1, "captured" );

    REQUIRE_THAT( 1,
                  Catch::Matchers::Predicate( []( int ) { return false; } ) );
    BENCHMARK( "Disabled benchmark" ) { REQUIRE( 1 == 2 ); };
}

#if defined( __clang__ )
#    pragma clang diagnostic pop
#endif
