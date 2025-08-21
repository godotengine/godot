
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/catch_test_macros.hpp>

TEST_CASE( "Incomplete AssertionHandler", "[assertion-handler][!shouldfail]" ) {
    Catch::AssertionHandler catchAssertionHandler(
        "REQUIRE"_catch_sr,
        CATCH_INTERNAL_LINEINFO,
        "Dummy",
        Catch::ResultDisposition::Normal );
}


static int foo( int i ) {
    REQUIRE( i > 10 );
    return 42;
}

TEST_CASE( "Assertions can be nested - CHECK", "[assertions][!shouldfail]" ) {
    CHECK( foo( 2 ) == 2 );
    // We should hit this, because CHECK continues on failure
    CHECK( true );
}

TEST_CASE( "Assertions can be nested - REQUIRE", "[assertions][!shouldfail]" ) {
    REQUIRE( foo( 2 ) == 2 );
    // We should not hit this, because REQUIRE does not continue on failure
    CHECK( true );
}

static void do_fail() { FAIL( "Throw a Catch::TestFailureException" ); }

TEST_CASE( "FAIL can be nested in assertion", "[assertions][!shouldfail]" ) {
    // Fails, but the error message makes sense.
    CHECK_NOTHROW( do_fail() );
}
