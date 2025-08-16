
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

// 020-TestCase-2.cpp

// main() provided by Catch in file 020-TestCase-1.cpp.

#include <catch2/catch_test_macros.hpp>

static int Factorial( int number ) {
   return number <= 1 ? number : Factorial( number - 1 ) * number;  // fail
// return number <= 1 ? 1      : Factorial( number - 1 ) * number;  // pass
}

TEST_CASE( "2: Factorial of 0 is 1 (fail)", "[multi-file:2]" ) {
    REQUIRE( Factorial(0) == 1 );
}

TEST_CASE( "2: Factorials of 1 and higher are computed (pass)", "[multi-file:2]" ) {
    REQUIRE( Factorial(1) == 1 );
    REQUIRE( Factorial(2) == 2 );
    REQUIRE( Factorial(3) == 6 );
    REQUIRE( Factorial(10) == 3628800 );
}

// Compile: see 020-TestCase-1.cpp

// Expected compact output (all assertions):
//
// prompt> 020-TestCase --reporter compact --success
// 020-TestCase-2.cpp:13: failed: Factorial(0) == 1 for: 0 == 1
// 020-TestCase-2.cpp:17: passed: Factorial(1) == 1 for: 1 == 1
// 020-TestCase-2.cpp:18: passed: Factorial(2) == 2 for: 2 == 2
// 020-TestCase-2.cpp:19: passed: Factorial(3) == 6 for: 6 == 6
// 020-TestCase-2.cpp:20: passed: Factorial(10) == 3628800 for: 3628800 (0x375f00) == 3628800 (0x375f00)
// Failed 1 test case, failed 1 assertion.
