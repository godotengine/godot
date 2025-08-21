
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

/**\file
 * Test that Catch's prefixed macros compile and run properly.
 *
 * We intentionally do not provide full coverage of all macros, but we
 * test a smattering and can add other ones as they have regressions.
 */


#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_predicate.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <type_traits>
#include <stdexcept>

namespace {
    [[noreturn]]
    void this_throws() {
        throw std::runtime_error("Some msg");
    }
    void this_doesnt_throw() {}
}

CATCH_TEST_CASE("PrefixedMacros") {
    using namespace Catch::Matchers;

    CATCH_REQUIRE( 1 == 1 );
    CATCH_REQUIRE_FALSE( 1 != 1 );

    CATCH_REQUIRE_THROWS(this_throws());
    CATCH_REQUIRE_THROWS_AS(this_throws(), std::runtime_error);
    CATCH_REQUIRE_THROWS_WITH(this_throws(), "Some msg");
    CATCH_REQUIRE_THROWS_MATCHES(this_throws(), std::runtime_error, Predicate<std::runtime_error>([](std::runtime_error const&) { return true; }));
    CATCH_REQUIRE_NOTHROW(this_doesnt_throw());

    CATCH_CHECK( 1 == 1 );
    CATCH_CHECK_FALSE( 1 != 1 );
    CATCH_CHECKED_IF( 1 == 1 ) {
        CATCH_SUCCEED("don't care");
    } CATCH_CHECKED_ELSE ( 1 == 1 ) {
        CATCH_SUCCEED("don't care");
    }

    CATCH_CHECK_NOFAIL(1 == 2);

    CATCH_CHECK_THROWS(this_throws());
    CATCH_CHECK_THROWS_AS(this_throws(), std::runtime_error);
    CATCH_CHECK_THROWS_WITH(this_throws(), "Some msg");
    CATCH_CHECK_THROWS_MATCHES(this_throws(), std::runtime_error, Predicate<std::runtime_error>([](std::runtime_error const&) { return true; }));
    CATCH_CHECK_NOTHROW(this_doesnt_throw());

    CATCH_REQUIRE_THAT("abcd", Equals("abcd"));
    CATCH_CHECK_THAT("bdef", Equals("bdef"));

    CATCH_INFO( "some info" );
    CATCH_UNSCOPED_INFO( "some info" );
    CATCH_WARN( "some warn" );
    CATCH_SECTION("some section") {
        int i = 1;
        CATCH_CAPTURE( i );
        CATCH_CAPTURE( i, i + 1 );
        CATCH_DYNAMIC_SECTION("Dynamic section: " << i) {
            CATCH_FAIL_CHECK( "failure" );
        }
    }

    CATCH_STATIC_REQUIRE( std::is_void<void>::value );
    CATCH_STATIC_REQUIRE_FALSE( std::is_void<int>::value );
    CATCH_STATIC_CHECK( std::is_void<void>::value );
    CATCH_STATIC_CHECK_FALSE( std::is_void<int>::value );
    CATCH_FAIL("");
}

// Missing:

//
// #define CATCH_TEST_CASE_METHOD( className, ... ) INTERNAL_CATCH_TEST_CASE_METHOD( className, __VA_ARGS__ )
// #define CATCH_METHOD_AS_TEST_CASE( method, ... ) INTERNAL_CATCH_METHOD_AS_TEST_CASE( method, __VA_ARGS__ )
// #define CATCH_REGISTER_TEST_CASE( Function, ... ) INTERNAL_CATCH_REGISTER_TESTCASE( Function, __VA_ARGS__ )
//
// // "BDD-style" convenience wrappers
// #define CATCH_SCENARIO( ... ) CATCH_TEST_CASE( "Scenario: " __VA_ARGS__ )
// #define CATCH_SCENARIO_METHOD( className, ... ) INTERNAL_CATCH_TEST_CASE_METHOD( className, "Scenario: " __VA_ARGS__ )
// #define CATCH_GIVEN( desc )    INTERNAL_CATCH_DYNAMIC_SECTION( "   Given: " << desc )
// #define CATCH_WHEN( desc )     INTERNAL_CATCH_DYNAMIC_SECTION( "    When: " << desc )
// #define CATCH_AND_WHEN( desc ) INTERNAL_CATCH_DYNAMIC_SECTION( "And when: " << desc )
// #define CATCH_THEN( desc )     INTERNAL_CATCH_DYNAMIC_SECTION( "    Then: " << desc )
// #define CATCH_AND_THEN( desc ) INTERNAL_CATCH_DYNAMIC_SECTION( "     And: " << desc )
//
