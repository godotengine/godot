
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

/**\file
 * Checks that signals/SEH within open section does not hard crash JUnit
 * (or similar reporter) while we are trying to report fatal error.
 */

#include <catch2/catch_test_macros.hpp>

#include <csignal>

// On Windows we need to send SEH and not signal to test the
// RunContext::handleFatalErrorCondition code path
#if defined( _MSC_VER )
#    include <windows.h>
#endif

TEST_CASE( "raises signal" ) {
    SECTION( "section" ) {
#if defined( _MSC_VER )
        RaiseException( 0xC0000005, 0, 0, NULL );
#else
        std::raise( SIGILL );
#endif
    }
}
