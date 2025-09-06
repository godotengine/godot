
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/internal/catch_test_case_tracker.hpp>


using namespace Catch;

namespace {
Catch::TestCaseTracking::NameAndLocationRef makeNAL( StringRef name ) {
    return Catch::TestCaseTracking::NameAndLocationRef( name, Catch::SourceLineInfo("",0) );
}
}

TEST_CASE( "Tracker" ) {

    TrackerContext ctx;
    ctx.startRun();
    ctx.startCycle();


    ITracker& testCase = SectionTracker::acquire( ctx, makeNAL( "Testcase" ) );
    REQUIRE( testCase.isOpen() );

    ITracker& s1 = SectionTracker::acquire( ctx, makeNAL( "S1" ) );
    REQUIRE( s1.isOpen() );

    SECTION( "successfully close one section" ) {
        s1.close();
        REQUIRE( s1.isSuccessfullyCompleted() );
        REQUIRE( testCase.isComplete() == false );

        testCase.close();
        REQUIRE( ctx.completedCycle() );
        REQUIRE( testCase.isSuccessfullyCompleted() );
    }

    SECTION( "fail one section" ) {
        s1.fail();
        REQUIRE( s1.isComplete() );
        REQUIRE( s1.isSuccessfullyCompleted() == false );
        REQUIRE( testCase.isComplete() == false );

        testCase.close();
        REQUIRE( ctx.completedCycle() );
        REQUIRE( testCase.isSuccessfullyCompleted() == false );

        SECTION( "re-enter after failed section" ) {
            ctx.startCycle();
            ITracker& testCase2 = SectionTracker::acquire( ctx, makeNAL( "Testcase" ) );
            REQUIRE( testCase2.isOpen() );

            ITracker& s1b = SectionTracker::acquire( ctx, makeNAL( "S1" ) );
            REQUIRE( s1b.isOpen() == false );

            testCase2.close();
            REQUIRE( ctx.completedCycle() );
            REQUIRE( testCase.isComplete() );
            REQUIRE( testCase.isSuccessfullyCompleted() );
        }
        SECTION( "re-enter after failed section and find next section" ) {
            ctx.startCycle();
            ITracker& testCase2 = SectionTracker::acquire( ctx, makeNAL( "Testcase" ) );
            REQUIRE( testCase2.isOpen() );

            ITracker& s1b = SectionTracker::acquire( ctx, makeNAL( "S1" ) );
            REQUIRE( s1b.isOpen() == false );

            ITracker& s2 = SectionTracker::acquire( ctx, makeNAL( "S2" ) );
            REQUIRE( s2.isOpen() );

            s2.close();
            REQUIRE( ctx.completedCycle() );

            testCase2.close();
            REQUIRE( testCase.isComplete() );
            REQUIRE( testCase.isSuccessfullyCompleted() );
        }
    }

    SECTION( "successfully close one section, then find another" ) {
        s1.close();

        ITracker& s2 = SectionTracker::acquire( ctx, makeNAL( "S2" ) );
        REQUIRE( s2.isOpen() == false );

        testCase.close();
        REQUIRE( testCase.isComplete() == false );

        SECTION( "Re-enter - skips S1 and enters S2" ) {
            ctx.startCycle();
            ITracker& testCase2 = SectionTracker::acquire( ctx, makeNAL( "Testcase" ) );
            REQUIRE( testCase2.isOpen() );

            ITracker& s1b = SectionTracker::acquire( ctx, makeNAL( "S1" ) );
            REQUIRE( s1b.isOpen() == false );

            ITracker& s2b = SectionTracker::acquire( ctx, makeNAL( "S2" ) );
            REQUIRE( s2b.isOpen() );

            REQUIRE( ctx.completedCycle() == false );

            SECTION ("Successfully close S2") {
                s2b.close();
                REQUIRE( ctx.completedCycle() );

                REQUIRE( s2b.isSuccessfullyCompleted() );
                REQUIRE( testCase2.isComplete() == false );

                testCase2.close();
                REQUIRE( testCase2.isSuccessfullyCompleted() );
            }
            SECTION ("fail S2") {
                s2b.fail();
                REQUIRE( ctx.completedCycle() );

                REQUIRE( s2b.isComplete() );
                REQUIRE( s2b.isSuccessfullyCompleted() == false );

                testCase2.close();
                REQUIRE( testCase2.isSuccessfullyCompleted() == false );

                // Need a final cycle
                ctx.startCycle();
                ITracker& testCase3 = SectionTracker::acquire( ctx, makeNAL( "Testcase" ) );
                REQUIRE( testCase3.isOpen() );

                ITracker& s1c = SectionTracker::acquire( ctx, makeNAL( "S1" ) );
                REQUIRE( s1c.isOpen() == false );

                ITracker& s2c = SectionTracker::acquire( ctx, makeNAL( "S2" ) );
                REQUIRE( s2c.isOpen() == false );

                testCase3.close();
                REQUIRE( testCase3.isSuccessfullyCompleted() );
            }
        }
    }

    SECTION( "open a nested section" ) {
        ITracker& s2 = SectionTracker::acquire( ctx, makeNAL( "S2" ) );
        REQUIRE( s2.isOpen() );

        s2.close();
        REQUIRE( s2.isComplete() );
        REQUIRE( s1.isComplete() == false );

        s1.close();
        REQUIRE( s1.isComplete() );
        REQUIRE( testCase.isComplete() == false );

        testCase.close();
        REQUIRE( testCase.isComplete() );
    }
}

static bool previouslyRun = false;
static bool previouslyRunNested = false;

TEST_CASE( "#1394", "[.][approvals][tracker]" ) {
    // -- Don't re-run after specified section is done
    REQUIRE(previouslyRun == false);

    SECTION( "RunSection" ) {
        previouslyRun = true;
    }
    SECTION( "SkipSection" ) {
        // cause an error if this section is called because it shouldn't be
        REQUIRE(1 == 0);
    }
}

TEST_CASE( "#1394 nested", "[.][approvals][tracker]" ) {
    REQUIRE(previouslyRunNested == false);

    SECTION( "NestedRunSection" ) {
        SECTION( "s1" ) {
            previouslyRunNested = true;
        }
    }
    SECTION( "NestedSkipSection" ) {
        // cause an error if this section is called because it shouldn't be
        REQUIRE(1 == 0);
    }
}

// Selecting a "not last" section inside a test case via -c "section" would
// previously only run the first subsection, instead of running all of them.
// This allows us to check that `"#1670 regression check" -c A` leads to
// 2 successful assertions.
TEST_CASE("#1670 regression check", "[.approvals][tracker]") {
    SECTION("A") {
        SECTION("1") SUCCEED();
        SECTION("2") SUCCEED();
    }
    SECTION("B") {
        SECTION("1") SUCCEED();
        SECTION("2") SUCCEED();
    }
}

// #1938 required a rework on how generator tracking works, so that `GENERATE`
// supports being sandwiched between two `SECTION`s. The following tests check
// various other scenarios through checking output in approval tests.
TEST_CASE("#1938 - GENERATE after a section", "[.][regression][generators]") {
    SECTION("A") {
        SUCCEED("A");
    }
    auto m = GENERATE(1, 2, 3);
    SECTION("B") {
        REQUIRE(m);
    }
}

TEST_CASE("#1938 - flat generate", "[.][regression][generators]") {
    auto m = GENERATE(1, 2, 3);
    REQUIRE(m);
}

TEST_CASE("#1938 - nested generate", "[.][regression][generators]") {
    auto m = GENERATE(1, 2, 3);
    auto n = GENERATE(1, 2, 3);
    REQUIRE(m);
    REQUIRE(n);
}

TEST_CASE("#1938 - mixed sections and generates", "[.][regression][generators]") {
    auto i = GENERATE(1, 2);
    SECTION("A") {
        SUCCEED("A");
    }
    auto j = GENERATE(3, 4);
    SECTION("B") {
        SUCCEED("B");
    }
    auto k = GENERATE(5, 6);
    CAPTURE(i, j, k);
    SUCCEED();
}

TEST_CASE("#1938 - Section followed by flat generate", "[.][regression][generators]") {
    SECTION("A") {
        REQUIRE(1);
    }
    auto m = GENERATE(2, 3);
    REQUIRE(m);
}
