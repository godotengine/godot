
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/catch_test_macros.hpp>

#include <catch2/internal/catch_parse_numbers.hpp>

TEST_CASE("Parse uints", "[parse-numbers]") {
	using Catch::parseUInt;
	using Catch::Optional;

	SECTION("proper inputs") {
        REQUIRE( parseUInt( "0" ) == Optional<unsigned int>{ 0 } );
        REQUIRE( parseUInt( "100" ) == Optional<unsigned int>{ 100 } );
        REQUIRE( parseUInt( "4294967295" ) ==
                 Optional<unsigned int>{ 4294967295 } );
        REQUIRE( parseUInt( "0xFF", 16 ) == Optional<unsigned int>{ 255 } );
    }
    SECTION( "Bad inputs" ) {
        // empty
        REQUIRE_FALSE( parseUInt( "" ) );
        // random noise
        REQUIRE_FALSE( parseUInt( "!!KJHF*#" ) );
        // negative
        REQUIRE_FALSE( parseUInt( "-1" ) );
        // too large
        REQUIRE_FALSE( parseUInt( "4294967296" ) );
        REQUIRE_FALSE( parseUInt( "42949672964294967296429496729642949672964294967296" ) );
        REQUIRE_FALSE( parseUInt( "2 4" ) );
        // hex with base 10
        REQUIRE_FALSE( parseUInt( "0xFF", 10 ) );
    }
}
