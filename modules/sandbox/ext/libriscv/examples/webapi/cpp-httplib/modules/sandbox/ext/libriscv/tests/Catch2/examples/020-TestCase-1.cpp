
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

// 020-TestCase-1.cpp

#include <catch2/catch_test_macros.hpp>

TEST_CASE( "1: All test cases reside in other .cpp files (empty)", "[multi-file:1]" ) {
}

// ^^^
// Normally no TEST_CASEs in this file.
// Here just to show there are two source files via option --list-tests.

// Compile & run:
// - g++ -std=c++14 -Wall -I$(CATCH_SINGLE_INCLUDE) -c 020-TestCase-1.cpp
// - g++ -std=c++14 -Wall -I$(CATCH_SINGLE_INCLUDE) -o 020-TestCase TestCase-1.o 020-TestCase-2.cpp && 020-TestCase --success
//
// - cl -EHsc -I%CATCH_SINGLE_INCLUDE% -c 020-TestCase-1.cpp
// - cl -EHsc -I%CATCH_SINGLE_INCLUDE% -Fe020-TestCase.exe 020-TestCase-1.obj 020-TestCase-2.cpp && 020-TestCase --success

// Expected test case listing:
//
// prompt> 020-TestCase --list-tests *
// Matching test cases:
//   1: All test cases reside in other .cpp files (empty)
//       [multi-file:1]
//   2: Factorial of 0 is computed (fail)
//       [multi-file:2]
//   2: Factorials of 1 and higher are computed (pass)
//       [multi-file:2]
// 3 matching test cases
