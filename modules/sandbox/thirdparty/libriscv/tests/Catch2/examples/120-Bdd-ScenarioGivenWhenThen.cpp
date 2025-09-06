
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

// 120-Bdd-ScenarioGivenWhenThen.cpp

// main() provided by linkage with Catch2WithMain

#include <catch2/catch_test_macros.hpp>

SCENARIO( "vectors can be sized and resized", "[vector]" ) {

    GIVEN( "A vector with some items" ) {
        std::vector<int> v( 5 );

        REQUIRE( v.size() == 5 );
        REQUIRE( v.capacity() >= 5 );

        WHEN( "the size is increased" ) {
            v.resize( 10 );

            THEN( "the size and capacity change" ) {
                REQUIRE( v.size() == 10 );
                REQUIRE( v.capacity() >= 10 );
            }
        }
        WHEN( "the size is reduced" ) {
            v.resize( 0 );

            THEN( "the size changes but not capacity" ) {
                REQUIRE( v.size() == 0 );
                REQUIRE( v.capacity() >= 5 );
            }
        }
        WHEN( "more capacity is reserved" ) {
            v.reserve( 10 );

            THEN( "the capacity changes but not the size" ) {
                REQUIRE( v.size() == 5 );
                REQUIRE( v.capacity() >= 10 );
            }
        }
        WHEN( "less capacity is reserved" ) {
            v.reserve( 0 );

            THEN( "neither size nor capacity are changed" ) {
                REQUIRE( v.size() == 5 );
                REQUIRE( v.capacity() >= 5 );
            }
        }
    }
}

// Compile & run:
// - g++ -std=c++14 -Wall -I$(CATCH_SINGLE_INCLUDE) -o 120-Bdd-ScenarioGivenWhenThen 120-Bdd-ScenarioGivenWhenThen.cpp && 120-Bdd-ScenarioGivenWhenThen --success
// - cl -EHsc -I%CATCH_SINGLE_INCLUDE% 120-Bdd-ScenarioGivenWhenThen.cpp && 120-Bdd-ScenarioGivenWhenThen --success

// Expected compact output (all assertions):
//
// prompt> 120-Bdd-ScenarioGivenWhenThen.exe --reporter compact --success
// 120-Bdd-ScenarioGivenWhenThen.cpp:12: passed: v.size() == 5 for: 5 == 5
// 120-Bdd-ScenarioGivenWhenThen.cpp:13: passed: v.capacity() >= 5 for: 5 >= 5
// 120-Bdd-ScenarioGivenWhenThen.cpp:19: passed: v.size() == 10 for: 10 == 10
// 120-Bdd-ScenarioGivenWhenThen.cpp:20: passed: v.capacity() >= 10 for: 10 >= 10
// 120-Bdd-ScenarioGivenWhenThen.cpp:12: passed: v.size() == 5 for: 5 == 5
// 120-Bdd-ScenarioGivenWhenThen.cpp:13: passed: v.capacity() >= 5 for: 5 >= 5
// 120-Bdd-ScenarioGivenWhenThen.cpp:27: passed: v.size() == 0 for: 0 == 0
// 120-Bdd-ScenarioGivenWhenThen.cpp:28: passed: v.capacity() >= 5 for: 5 >= 5
// 120-Bdd-ScenarioGivenWhenThen.cpp:12: passed: v.size() == 5 for: 5 == 5
// 120-Bdd-ScenarioGivenWhenThen.cpp:13: passed: v.capacity() >= 5 for: 5 >= 5
// 120-Bdd-ScenarioGivenWhenThen.cpp:35: passed: v.size() == 5 for: 5 == 5
// 120-Bdd-ScenarioGivenWhenThen.cpp:36: passed: v.capacity() >= 10 for: 10 >= 10
// 120-Bdd-ScenarioGivenWhenThen.cpp:12: passed: v.size() == 5 for: 5 == 5
// 120-Bdd-ScenarioGivenWhenThen.cpp:13: passed: v.capacity() >= 5 for: 5 >= 5
// 120-Bdd-ScenarioGivenWhenThen.cpp:43: passed: v.size() == 5 for: 5 == 5
// 120-Bdd-ScenarioGivenWhenThen.cpp:44: passed: v.capacity() >= 5 for: 5 >= 5
// Passed 1 test case with 16 assertions.
