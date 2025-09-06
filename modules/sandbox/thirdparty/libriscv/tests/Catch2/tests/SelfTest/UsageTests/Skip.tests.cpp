
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include <iostream>

TEST_CASE( "tests can be skipped dynamically at runtime", "[skipping]" ) {
    SKIP();
    FAIL( "this is not reached" );
}

TEST_CASE( "skipped tests can optionally provide a reason", "[skipping]" ) {
    const int answer = 43;
    SKIP( "skipping because answer = " << answer );
    FAIL( "this is not reached" );
}

TEST_CASE( "sections can be skipped dynamically at runtime", "[skipping]" ) {
    SECTION( "not skipped" ) { SUCCEED(); }
    SECTION( "skipped" ) { SKIP(); }
    SECTION( "also not skipped" ) { SUCCEED(); }
}

TEST_CASE( "nested sections can be skipped dynamically at runtime",
           "[skipping]" ) {
    SECTION( "A" ) { std::cout << "a"; }
    SECTION( "B" ) {
        SECTION( "B1" ) { std::cout << "b1"; }
        SECTION( "B2" ) { SKIP(); }
    }
    std::cout << "!\n";
}

TEST_CASE( "dynamic skipping works with generators", "[skipping]" ) {
    const int answer = GENERATE( 41, 42, 43 );
    if ( answer != 42 ) { SKIP( "skipping because answer = " << answer ); }
    SUCCEED();
}

TEST_CASE( "failed assertions before SKIP cause test case to fail",
           "[skipping][!shouldfail]" ) {
    CHECK( 3 == 4 );
    SKIP();
}

TEST_CASE( "a succeeding test can still be skipped",
           "[skipping][!shouldfail]" ) {
    SUCCEED();
    SKIP();
}

TEST_CASE( "failing in some unskipped sections causes entire test case to fail",
           "[skipping][!shouldfail]" ) {
    SECTION( "skipped" ) { SKIP(); }
    SECTION( "not skipped" ) { FAIL(); }
}

TEST_CASE( "failing for some generator values causes entire test case to fail",
           "[skipping][!shouldfail]" ) {
    int i = GENERATE( 1, 2, 3, 4 );
    if ( i % 2 == 0 ) {
        SKIP();
    } else {
        FAIL();
    }
}

namespace {
    class test_skip_generator : public Catch::Generators::IGenerator<int> {
    public:
        explicit test_skip_generator() { SKIP( "This generator is empty" ); }

        auto get() const -> int const& override {
            static constexpr int value = 1;
            return value;
        }

        auto next() -> bool override { return false; }
    };

    static auto make_test_skip_generator()
        -> Catch::Generators::GeneratorWrapper<int> {
        return { new test_skip_generator() };
    }

} // namespace

TEST_CASE( "Empty generators can SKIP in constructor", "[skipping]" ) {
    // The generator signals emptiness with `SKIP`
    auto sample = GENERATE( make_test_skip_generator() );
    // This assertion would fail, but shouldn't trigger
    REQUIRE( sample == 0 );
}
