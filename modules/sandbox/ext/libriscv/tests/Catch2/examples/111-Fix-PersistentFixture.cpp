
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

// Fixture.cpp

// Catch2 has three ways to express fixtures:
// - Sections
// - Traditional class-based fixtures that are created and destroyed on every
// partial run
// - Traditional class-based fixtures that are created at the start of a test
// case and destroyed at the end of a test case (this file)

// main() provided by linkage to Catch2WithMain

#include <catch2/catch_test_macros.hpp>

#include <thread>

class ClassWithExpensiveSetup {
public:
    ClassWithExpensiveSetup() {
        // Imagine some really expensive set up here.
        // e.g.
        // setting up a D3D12/Vulkan Device,
        // connecting to a database,
        // loading a file
        // etc etc etc
        std::this_thread::sleep_for( std::chrono::seconds( 2 ) );
    }

    ~ClassWithExpensiveSetup() noexcept {
        // We can do any clean up of the expensive class in the destructor
        // e.g.
        // destroy D3D12/Vulkan Device,
        // disconnecting from a database,
        // release file handle
        // etc etc etc
        std::this_thread::sleep_for( std::chrono::seconds( 1 ) );
    }

    int getInt() const { return 42; }
};

struct MyFixture {

    // The test case member function is const.
    // Therefore we need to mark any member of the fixture
    // that needs to mutate as mutable.
    mutable int myInt = 0;
    ClassWithExpensiveSetup expensive;
};

// Only one object of type MyFixture will be instantiated for the run
// of this test case even though there are two leaf sections.
// This is useful if your test case requires an object that is
// expensive to create and could be reused for each partial run of the
// test case.
TEST_CASE_PERSISTENT_FIXTURE( MyFixture, "Tests with MyFixture" ) {

    const int val = myInt++;

    SECTION( "First partial run" ) {
        const auto otherValue = expensive.getInt();
        REQUIRE( val == 0 );
        REQUIRE( otherValue == 42 );
    }

    SECTION( "Second partial run" ) { REQUIRE( val == 1 ); }
}