
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_REPORTER_TAP_HPP_INCLUDED
#define CATCH_REPORTER_TAP_HPP_INCLUDED

#include <catch2/reporters/catch_reporter_streaming_base.hpp>
#include <catch2/internal/catch_move_and_forward.hpp>

namespace Catch {

    class TAPReporter final : public StreamingReporterBase {
    public:
        TAPReporter( ReporterConfig&& config ):
            StreamingReporterBase( CATCH_MOVE(config) ) {
            m_preferences.shouldReportAllAssertions = true;
            m_preferences.shouldReportAllAssertionStarts = false;
        }

        static std::string getDescription() {
            using namespace std::string_literals;
            return "Reports test results in TAP format, suitable for test harnesses"s;
        }

        void testRunStarting( TestRunInfo const& testInfo ) override;

        void noMatchingTestCases( StringRef unmatchedSpec ) override;

        void assertionEnded(AssertionStats const& _assertionStats) override;

        void testRunEnded(TestRunStats const& _testRunStats) override;

    private:
        std::size_t counter = 0;
    };

} // end namespace Catch

#endif // CATCH_REPORTER_TAP_HPP_INCLUDED
