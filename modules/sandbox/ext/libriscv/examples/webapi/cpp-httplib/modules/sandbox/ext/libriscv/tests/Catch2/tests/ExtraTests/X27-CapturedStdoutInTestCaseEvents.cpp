
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

/**\file
 * Test that the captured stdout/err in (partial) testCaseEnded events
 * is correct (e.g. that the partial test case event does not get accumulated
 * output).
 *
 * This is done by having a single test case that is entered multiple
 * times through generator, and a custom capturing reporter that knows
 * what it should expect captured from the test case.
 */


#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/reporters/catch_reporter_streaming_base.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>


#include <iostream>
#include <string>
#include <utility>

class TestReporter : public Catch::StreamingReporterBase {
    std::string stdOutString( uint64_t iter ){
        return "stdout " + std::to_string( iter ) + '\n';
    }
    std::string stdErrString(uint64_t iter) {
        return "stderr " + std::to_string( iter ) + '\n';
    }

public:
    TestReporter( Catch::ReporterConfig&& _config ):
        StreamingReporterBase( std::move(_config) ) {
        m_preferences.shouldRedirectStdOut = true;
        std::cout << "X27 - TestReporter constructed\n";
    }

    static std::string getDescription() {
        return "X27 test reporter";
    }

    void testCasePartialEnded( Catch::TestCaseStats const& stats,
                               uint64_t iter ) override {
        if ( stats.stdOut != stdOutString( iter ) ) {
            std::cerr << "X27 ERROR in partial stdout\n" << stats.stdOut;
        }
        if ( stats.stdErr != stdErrString( iter ) ) {
            std::cerr << "X27 ERROR in partial stderr\n" << stats.stdErr;
        }
    }

    void testCaseEnded( Catch::TestCaseStats const& stats ) override {
        if ( stats.stdOut != "stdout 0\nstdout 1\nstdout 2\nstdout 3\nstdout 4\nstdout 5\n" ) {
            std::cerr << "X27 ERROR in full stdout\n" << stats.stdOut;
        }
        if ( stats.stdErr != "stderr 0\nstderr 1\nstderr 2\nstderr 3\nstderr 4\nstderr 5\n" ) {
            std::cerr << "X27 ERROR in full stderr\n" << stats.stdErr;
        }
    }

    ~TestReporter() override;
};

TestReporter::~TestReporter() = default;

CATCH_REGISTER_REPORTER( "test-reporter", TestReporter )

TEST_CASE( "repeatedly entered test case" ) {
    auto i = GENERATE( range(0, 6) );
    std::cout << "stdout " << i << '\n';
    // Switch between writing to std::cerr and std::clog just to make sure
    // both are properly captured and redirected.
    ( ( i % 2 == 0 ) ? std::cerr : std::clog ) << "stderr " << i << '\n';
}
