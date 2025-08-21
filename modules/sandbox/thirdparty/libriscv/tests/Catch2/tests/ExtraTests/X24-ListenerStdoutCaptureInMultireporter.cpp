
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

/**\file
 * Registers custom listener that does not ask for captured stdout/err
 *
 * Running the binary with this listener, and asking for multiple _capturing_
 * reporters (one would be sufficient, but that would also mean depending on
 * implementation details inside Catch2's handling of listeners), we check that
 * nothing is written to stdout, because listeners should not be considered in
 * whether the stdout should be passed-through or not.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/reporters/catch_reporter_event_listener.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>

#include <iostream>

namespace {
	class NonCapturingListener : public Catch::EventListenerBase {
    public:
        NonCapturingListener( Catch::IConfig const* config ):
            EventListenerBase( config ) {
            m_preferences.shouldRedirectStdOut = false;
            std::cerr << "X24 - NonCapturingListener initialized.\n";
		}
	};
}

CATCH_REGISTER_LISTENER( NonCapturingListener )

TEST_CASE( "Writes to stdout" ) {
	std::cout << "X24 - FooBarBaz\n";
}
