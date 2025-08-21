
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/catch_test_macros.hpp>

namespace {

    static bool itDoesThis() { return true; }

    static bool itDoesThat() { return true; }

    // a trivial fixture example to support SCENARIO_METHOD tests
    struct Fixture {
        Fixture(): d_counter( 0 ) {}

        int counter() { return d_counter++; }

        int d_counter;
    };

}


SCENARIO("Do that thing with the thing", "[Tags]") {
    GIVEN("This stuff exists") {
        // make stuff exist
        AND_GIVEN("And some assumption") {
            // Validate assumption
            WHEN("I do this") {
                // do this
                THEN("it should do this") {
                    REQUIRE(itDoesThis());
                    AND_THEN("do that") {
                        REQUIRE(itDoesThat());
                    }
                }
            }
        }
    }
}

SCENARIO( "Vector resizing affects size and capacity",
          "[vector][bdd][size][capacity]" ) {
    GIVEN( "an empty vector" ) {
        std::vector<int> v;
        REQUIRE( v.size() == 0 );

        WHEN( "it is made larger" ) {
            v.resize( 10 );
            THEN( "the size and capacity go up" ) {
                REQUIRE( v.size() == 10 );
                REQUIRE( v.capacity() >= 10 );

                AND_WHEN( "it is made smaller again" ) {
                    v.resize( 5 );
                    THEN(
                        "the size goes down but the capacity stays the same" ) {
                        REQUIRE( v.size() == 5 );
                        REQUIRE( v.capacity() >= 10 );
                    }
                }
            }
        }

        WHEN( "we reserve more space" ) {
            v.reserve( 10 );
            THEN( "The capacity is increased but the size remains the same" ) {
                REQUIRE( v.capacity() >= 10 );
                REQUIRE( v.size() == 0 );
            }
        }
    }
}

SCENARIO("This is a really long scenario name to see how the list command deals with wrapping",
         "[very long tags][lots][long][tags][verbose]"
                 "[one very long tag name that should cause line wrapping writing out using the list command]"
                 "[anotherReallyLongTagNameButThisOneHasNoObviousWrapPointsSoShouldSplitWithinAWordUsingADashCharacter]") {
    GIVEN("A section name that is so long that it cannot fit in a single console width") {
        WHEN("The test headers are printed as part of the normal running of the scenario") {
            THEN("The, deliberately very long and overly verbose (you see what I did there?) section names must wrap, along with an indent") {
                SUCCEED("boo!");
            }
        }
    }
}

SCENARIO_METHOD(Fixture,
                "BDD tests requiring Fixtures to provide commonly-accessed data or methods",
                "[bdd][fixtures]") {
    const int before(counter());
    GIVEN("No operations precede me") {
        REQUIRE(before == 0);
        WHEN("We get the count") {
            const int after(counter());
            THEN("Subsequently values are higher") {
                REQUIRE(after > before);
            }
        }
    }
}
