
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_REPORTER_SONARQUBE_HPP_INCLUDED
#define CATCH_REPORTER_SONARQUBE_HPP_INCLUDED

#include <catch2/reporters/catch_reporter_cumulative_base.hpp>

#include <catch2/internal/catch_xmlwriter.hpp>
#include <catch2/internal/catch_move_and_forward.hpp>

namespace Catch {

    class SonarQubeReporter final : public CumulativeReporterBase {
    public:
        SonarQubeReporter(ReporterConfig&& config)
        : CumulativeReporterBase(CATCH_MOVE(config))
        , xml(m_stream) {
            m_preferences.shouldRedirectStdOut = true;
            m_preferences.shouldReportAllAssertions = false;
            m_preferences.shouldReportAllAssertionStarts = false;
            m_shouldStoreSuccesfulAssertions = false;
        }

        static std::string getDescription() {
            using namespace std::string_literals;
            return "Reports test results in the Generic Test Data SonarQube XML format"s;
        }

        void testRunStarting( TestRunInfo const& testRunInfo ) override;

        void testRunEndedCumulative() override {
            writeRun( *m_testRun );
            xml.endElement();
        }

        void writeRun( TestRunNode const& runNode );

        void writeTestFile(StringRef filename, std::vector<TestCaseNode const*> const& testCaseNodes);

        void writeTestCase(TestCaseNode const& testCaseNode);

        void writeSection(std::string const& rootName, SectionNode const& sectionNode, bool okToFail);

        void writeAssertions(SectionNode const& sectionNode, bool okToFail);

        void writeAssertion(AssertionStats const& stats, bool okToFail);

    private:
        XmlWriter xml;
    };


} // end namespace Catch

#endif // CATCH_REPORTER_SONARQUBE_HPP_INCLUDED
