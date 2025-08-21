
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

/**\file
 * Registers custom listener that asks for captured stdout/err
 *
 * Running the binary with this listener, and asking for multiple _noncapturing_
 * reporters (one would be sufficient, but that would also mean depending on
 * implementation details inside Catch2's handling of listeners), we check that
 * the listener gets redirected stdout, even though the reporters didn't ask for
 * it.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/reporters/catch_reporter_event_listener.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>

#include <iostream>

namespace {
	class CapturingListener : public Catch::EventListenerBase {
    public:
        CapturingListener( Catch::IConfig const* config ):
            EventListenerBase( config ) {
            m_preferences.shouldRedirectStdOut = true;
            std::cerr << "CapturingListener initialized\n";
		}

		void
        testCaseEnded( Catch::TestCaseStats const& testCaseStats ) override {
            if ( testCaseStats.stdOut.empty() ) {
                std::cerr << "X25 - ERROR: empty stdout\n";
            }
        }
	};
}

CATCH_REGISTER_LISTENER( CapturingListener )

TEST_CASE( "Writes to stdout" ) {
	std::cout << "X25 - FooBarBaz\n";
}
