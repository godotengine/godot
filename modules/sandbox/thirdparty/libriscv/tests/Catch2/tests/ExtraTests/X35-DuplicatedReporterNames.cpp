
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

/**\file
 * Checks that reporter registration errors are caught and handled as
 * startup errors, by causing a registration error by registering multiple
 * reporters with the same name.
 */

#include <catch2/catch_test_macros.hpp>

#include <catch2/reporters/catch_reporter_registrars.hpp>
#include <catch2/reporters/catch_reporter_streaming_base.hpp>

namespace {
    //! Trivial custom reporter for registration
    class TestReporter : public Catch::StreamingReporterBase {
    public:
        using StreamingReporterBase::StreamingReporterBase;

        static std::string getDescription() { return "X35 test reporter"; }
    };
}

CATCH_REGISTER_REPORTER( "test-reporter", TestReporter )
CATCH_REGISTER_REPORTER( "test-reporter", TestReporter )
