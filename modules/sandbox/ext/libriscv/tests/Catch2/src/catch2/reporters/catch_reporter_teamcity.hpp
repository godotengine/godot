
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_REPORTER_TEAMCITY_HPP_INCLUDED
#define CATCH_REPORTER_TEAMCITY_HPP_INCLUDED

#include <catch2/reporters/catch_reporter_streaming_base.hpp>
#include <catch2/catch_timer.hpp>

#include <cstring>

#ifdef __clang__
#   pragma clang diagnostic push
#   pragma clang diagnostic ignored "-Wpadded"
#endif

namespace Catch {

    class TeamCityReporter final : public StreamingReporterBase {
    public:
        TeamCityReporter( ReporterConfig&& _config )
        :   StreamingReporterBase( CATCH_MOVE(_config) )
        {
            m_preferences.shouldRedirectStdOut = true;
            m_preferences.shouldReportAllAssertionStarts = false;
        }

        ~TeamCityReporter() override;

        static std::string getDescription() {
            using namespace std::string_literals;
            return "Reports test results as TeamCity service messages"s;
        }

        void testRunStarting( TestRunInfo const& runInfo ) override;
        void testRunEnded( TestRunStats const& runStats ) override;


        void assertionEnded(AssertionStats const& assertionStats) override;

        void sectionStarting(SectionInfo const& sectionInfo) override {
            m_headerPrintedForThisSection = false;
            StreamingReporterBase::sectionStarting( sectionInfo );
        }

        void testCaseStarting(TestCaseInfo const& testInfo) override;

        void testCaseEnded(TestCaseStats const& testCaseStats) override;

    private:
        void printSectionHeader(std::ostream& os);

        bool m_headerPrintedForThisSection = false;
        Timer m_testTimer;
    };

} // end namespace Catch

#ifdef __clang__
#   pragma clang diagnostic pop
#endif

#endif // CATCH_REPORTER_TEAMCITY_HPP_INCLUDED
