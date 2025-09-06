
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_REPORTER_JUNIT_HPP_INCLUDED
#define CATCH_REPORTER_JUNIT_HPP_INCLUDED


#include <catch2/reporters/catch_reporter_cumulative_base.hpp>
#include <catch2/internal/catch_xmlwriter.hpp>
#include <catch2/catch_timer.hpp>

namespace Catch {

    class JunitReporter final : public CumulativeReporterBase {
    public:
        JunitReporter(ReporterConfig&& _config);

        static std::string getDescription();

        void testRunStarting(TestRunInfo const& runInfo) override;

        void testCaseStarting(TestCaseInfo const& testCaseInfo) override;
        void assertionEnded(AssertionStats const& assertionStats) override;

        void testCaseEnded(TestCaseStats const& testCaseStats) override;

        void testRunEndedCumulative() override;

    private:
        void writeRun(TestRunNode const& testRunNode, double suiteTime);

        void writeTestCase(TestCaseNode const& testCaseNode);

        void writeSection( std::string const& className,
                           std::string const& rootName,
                           SectionNode const& sectionNode,
                           bool testOkToFail );

        void writeAssertions(SectionNode const& sectionNode);
        void writeAssertion(AssertionStats const& stats);

        XmlWriter xml;
        Timer suiteTimer;
        std::string stdOutForSuite;
        std::string stdErrForSuite;
        unsigned int unexpectedExceptions = 0;
        bool m_okToFail = false;
    };

} // end namespace Catch

#endif // CATCH_REPORTER_JUNIT_HPP_INCLUDED
