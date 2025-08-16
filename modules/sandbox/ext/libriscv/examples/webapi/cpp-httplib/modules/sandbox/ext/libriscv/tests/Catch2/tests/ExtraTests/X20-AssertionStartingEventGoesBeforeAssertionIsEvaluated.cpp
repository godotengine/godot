
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

/**\file
 * Registers an event listener to increments counter of assertionStarting events.
 *
 * Different assertion macros then check that the counter is at expected
 * value when they are evaluated.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/reporters/catch_reporter_event_listener.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>
#include <catch2/matchers/catch_matchers_predicate.hpp>

namespace {

    static size_t assertion_starting_events_seen = 0;

    class AssertionStartingListener : public Catch::EventListenerBase {
    public:
        AssertionStartingListener( Catch::IConfig const* config ):
            EventListenerBase( config ) {}

        void assertionStarting( Catch::AssertionInfo const& ) override {
            ++assertion_starting_events_seen;
        }
    };

    static bool f1() {
        return assertion_starting_events_seen == 1;
    }

    static void f2() {
        if ( assertion_starting_events_seen != 2 ) { throw 1; }
    }

    static void f3() {
        if ( assertion_starting_events_seen == 3 ) { throw 1; }
    }

    static bool f4() { return assertion_starting_events_seen == 4; }

    static void f5() { throw assertion_starting_events_seen; }

} // anonymous namespace

CATCH_REGISTER_LISTENER( AssertionStartingListener )

TEST_CASE() {
    // **IMPORTANT**
    // The order of assertions below matters.
    REQUIRE( f1() );
    REQUIRE_NOTHROW( f2() );
    REQUIRE_THROWS( f3() );
    REQUIRE_THAT( f4(),
                  Catch::Matchers::Predicate<bool>( []( bool b ) { return b; } ) );
    REQUIRE_THROWS_MATCHES(
        f5(), size_t, Catch::Matchers::Predicate<size_t>( []( size_t i ) {
            return i == 5;
        } ) );

    CAPTURE( assertion_starting_events_seen ); // **not** an assertion
    INFO( "some info msg" );                   // **not** an assertion
    WARN( "warning! warning!" );               // assertion-like message
    SUCCEED();                                 // assertion-like message

    // We skip FAIL/SKIP and so on, which fail the test.

    // This require will also increment the count once
    REQUIRE( assertion_starting_events_seen == 8 );
}
