
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

// 030-Asn-Require-Check.cpp

// Catch has two natural expression assertion macro's:
// - REQUIRE() stops at first failure.
// - CHECK() continues after failure.

// There are two variants to support decomposing negated expressions:
// - REQUIRE_FALSE() stops at first failure.
// - CHECK_FALSE() continues after failure.

// main() provided by linkage to Catch2WithMain

#include <catch2/catch_test_macros.hpp>

static std::string one() {
    return "1";
}

TEST_CASE( "Assert that something is true (pass)", "[require]" ) {
    REQUIRE( one() == "1" );
}

TEST_CASE( "Assert that something is true (fail)", "[require]" ) {
    REQUIRE( one() == "x" );
}

TEST_CASE( "Assert that something is true (stop at first failure)", "[require]" ) {
    WARN( "REQUIRE stops at first failure:" );

    REQUIRE( one() == "x" );
    REQUIRE( one() == "1" );
}

TEST_CASE( "Assert that something is true (continue after failure)", "[check]" ) {
    WARN( "CHECK continues after failure:" );

    CHECK(   one() == "x" );
    REQUIRE( one() == "1" );
}

TEST_CASE( "Assert that something is false (stops at first failure)", "[require-false]" ) {
    WARN( "REQUIRE_FALSE stops at first failure:" );

    REQUIRE_FALSE( one() == "1" );
    REQUIRE_FALSE( one() != "1" );
}

TEST_CASE( "Assert that something is false (continue after failure)", "[check-false]" ) {
    WARN( "CHECK_FALSE continues after failure:" );

    CHECK_FALSE(   one() == "1" );
    REQUIRE_FALSE( one() != "1" );
}

// Compile & run:
// - g++ -std=c++14 -Wall -I$(CATCH_SINGLE_INCLUDE) -o 030-Asn-Require-Check 030-Asn-Require-Check.cpp && 030-Asn-Require-Check --success
// - cl -EHsc -I%CATCH_SINGLE_INCLUDE% 030-Asn-Require-Check.cpp && 030-Asn-Require-Check --success

// Expected compact output (all assertions):
//
// prompt> 030-Asn-Require-Check.exe --reporter compact --success
// 030-Asn-Require-Check.cpp:20: passed: one() == "1" for: "1" == "1"
// 030-Asn-Require-Check.cpp:24: failed: one() == "x" for: "1" == "x"
// 030-Asn-Require-Check.cpp:28: warning: 'REQUIRE stops at first failure:'
// 030-Asn-Require-Check.cpp:30: failed: one() == "x" for: "1" == "x"
// 030-Asn-Require-Check.cpp:35: warning: 'CHECK continues after failure:'
// 030-Asn-Require-Check.cpp:37: failed: one() == "x" for: "1" == "x"
// 030-Asn-Require-Check.cpp:38: passed: one() == "1" for: "1" == "1"
// 030-Asn-Require-Check.cpp:42: warning: 'REQUIRE_FALSE stops at first failure:'
// 030-Asn-Require-Check.cpp:44: failed: !(one() == "1") for: !("1" == "1")
// 030-Asn-Require-Check.cpp:49: warning: 'CHECK_FALSE continues after failure:'
// 030-Asn-Require-Check.cpp:51: failed: !(one() == "1") for: !("1" == "1")
// 030-Asn-Require-Check.cpp:52: passed: !(one() != "1") for: !("1" != "1")
// Failed 5 test cases, failed 5 assertions.
