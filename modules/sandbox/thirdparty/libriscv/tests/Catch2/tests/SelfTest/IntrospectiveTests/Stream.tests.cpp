
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/catch_test_macros.hpp>

#include <catch2/internal/catch_istream.hpp>

TEST_CASE( "Cout stream properly declares it writes to stdout", "[streams]" ) {
    REQUIRE( Catch::makeStream( "-" )->isConsole() );
}

TEST_CASE( "Empty stream name opens cout stream", "[streams]" ) {
    REQUIRE( Catch::makeStream( "" )->isConsole() );
}

TEST_CASE( "stdout and stderr streams have %-starting name", "[streams]" ) {
    REQUIRE( Catch::makeStream( "%stderr" )->isConsole() );
    REQUIRE( Catch::makeStream( "%stdout" )->isConsole() );
}

TEST_CASE( "request an unknown %-starting stream fails", "[streams]" ) {
    REQUIRE_THROWS( Catch::makeStream( "%somestream" ) );
}

TEST_CASE( "makeStream recognizes %debug stream name", "[streams]" ) {
    REQUIRE_NOTHROW( Catch::makeStream( "%debug" ) );
}
