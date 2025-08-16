
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

/**\file
 * Test that custom options are properly passed down to the reporter.
 *
 * We print out the arguments sorted by key, to have a stable expected
 * output.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/reporters/catch_reporter_streaming_base.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>

#include <algorithm>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

class TestReporter : public Catch::StreamingReporterBase {
public:
    TestReporter( Catch::ReporterConfig&& _config ):
        StreamingReporterBase( std::move(_config) ) {
        std::cout << "X29 - TestReporter constructed\n";
    }

    static std::string getDescription() {
        return "X29 test reporter";
    }

    void testRunStarting( Catch::TestRunInfo const& ) override {
        std::vector<std::pair<std::string, std::string>> options;
        options.reserve( m_customOptions.size() );
        for ( auto const& kv : m_customOptions ) {
            options.push_back( kv );
        }
        std::sort( options.begin(), options.end() );
        bool first = true;
        for ( auto const& kv : options ) {
            if ( !first ) { std::cout << "::"; }
            std::cout << kv.first << "=" << kv.second;
            first = false;
        }
        std::cout << '\n';
    }

    ~TestReporter() override;
};

TestReporter::~TestReporter() = default;

CATCH_REGISTER_REPORTER( "test-reporter", TestReporter )

TEST_CASE( "Just a test case to run things" ) {}
