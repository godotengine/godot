
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

/**\file
 * Test that the different events are sent to listeners before they are
 * sent to the reporters.
 *
 * We only do this for a subset of the events, as doing all of them would
 * be annoying, and we can assume that their implementation is roughly
 * the same, and thus if few work, all work.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/reporters/catch_reporter_event_listener.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>
#include <catch2/reporters/catch_reporter_streaming_base.hpp>

#include <iostream>
#include <utility>

namespace {

    static bool testRunStartingReceivedByListener = false;
    static bool testRunEndedReceivedByListener = false;
    static bool assertionStartingReceivedByListener = false;
    static bool assertionEndedReceivedByListener = false;

    class TestListener : public Catch::EventListenerBase {
    public:
        TestListener( Catch::IConfig const* config ):
            EventListenerBase( config ) {
            std::cout << "X28 - TestListener constructed.\n";
        }

        void testRunStarting( Catch::TestRunInfo const& ) override {
            testRunStartingReceivedByListener = true;
        }

        void testRunEnded( Catch::TestRunStats const& ) override {
            testRunEndedReceivedByListener = true;
        }

        void assertionStarting( Catch::AssertionInfo const& ) override {
            assertionStartingReceivedByListener = true;
        }

        void assertionEnded( Catch::AssertionStats const& ) override {
            assertionEndedReceivedByListener = true;
        }
    };

    class TestReporter : public Catch::StreamingReporterBase {
    public:
        TestReporter( Catch::ReporterConfig&& _config ):
            StreamingReporterBase( std::move(_config) ) {
            std::cout << "X28 - TestReporter constructed\n";
        }

        void testRunStarting( Catch::TestRunInfo const& ) override {
            if ( !testRunStartingReceivedByListener ) {
                std::cout << "X28 - ERROR\n";
            }
        }

        void testRunEnded( Catch::TestRunStats const& ) override {
            if ( !testRunEndedReceivedByListener ) {
                std::cout << "X28 - ERROR\n";
            }
        }

        void assertionStarting( Catch::AssertionInfo const& ) override {
            if ( !assertionStartingReceivedByListener ) {
                std::cout << "X28 - ERROR\n";
            }
        }

        void assertionEnded( Catch::AssertionStats const& ) override {
            if ( !assertionEndedReceivedByListener ) {
                std::cout << "X28 - ERROR\n";
            }
        }

        static std::string getDescription() { return "X28 test reporter"; }
        ~TestReporter() override;
    };

    TestReporter::~TestReporter() = default;

} // end unnamed namespace

CATCH_REGISTER_REPORTER( "test-reporter", TestReporter )
CATCH_REGISTER_LISTENER( TestListener )

TEST_CASE( "Dummy test case" ) { REQUIRE( 1 == 1 ); }
