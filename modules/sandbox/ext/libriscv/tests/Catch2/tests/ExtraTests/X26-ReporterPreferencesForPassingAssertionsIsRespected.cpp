
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

/**\file
 * Test that reporter is not passed passing assertions when it
 * doesn't ask for it.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/reporters/catch_reporter_streaming_base.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>

#include <iostream>
#include <utility>

namespace {

  class TestReporter : public Catch::StreamingReporterBase {
  public:
      TestReporter(Catch::ReporterConfig&& _config):
          StreamingReporterBase(std::move(_config)) {
          m_preferences.shouldReportAllAssertions = false;
          std::cout << "X26 - TestReporter constructed\n";
      }

      static std::string getDescription() {
          return "X26 - test reporter that opts out of passing assertions";
      }

      void
      assertionEnded( Catch::AssertionStats const& ) override {
          std::cerr << "X26 - assertionEnded\n";
      }

      ~TestReporter() override;
  };

  TestReporter::~TestReporter() = default;

}

CATCH_REGISTER_REPORTER("test-reporter", TestReporter)

TEST_CASE( "Test with only passing assertions" ) {
    REQUIRE( 1 == 1 );
    REQUIRE( 2 == 2 );
}
