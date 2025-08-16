
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_REPORTER_CONSOLE_HPP_INCLUDED
#define CATCH_REPORTER_CONSOLE_HPP_INCLUDED

#include <catch2/reporters/catch_reporter_streaming_base.hpp>
#include <catch2/internal/catch_unique_ptr.hpp>

namespace Catch {
    // Fwd decls
    class TablePrinter;

    class ConsoleReporter final : public StreamingReporterBase {
        Detail::unique_ptr<TablePrinter> m_tablePrinter;

    public:
        ConsoleReporter(ReporterConfig&& config);
        ~ConsoleReporter() override;
        static std::string getDescription();

        void noMatchingTestCases( StringRef unmatchedSpec ) override;
        void reportInvalidTestSpec( StringRef arg ) override;

        void assertionEnded(AssertionStats const& _assertionStats) override;

        void sectionStarting(SectionInfo const& _sectionInfo) override;
        void sectionEnded(SectionStats const& _sectionStats) override;

        void benchmarkPreparing( StringRef name ) override;
        void benchmarkStarting(BenchmarkInfo const& info) override;
        void benchmarkEnded(BenchmarkStats<> const& stats) override;
        void benchmarkFailed( StringRef error ) override;

        void testCaseEnded(TestCaseStats const& _testCaseStats) override;
        void testRunEnded(TestRunStats const& _testRunStats) override;
        void testRunStarting(TestRunInfo const& _testRunInfo) override;

    private:
        void lazyPrint();

        void lazyPrintWithoutClosingBenchmarkTable();
        void lazyPrintRunInfo();
        void printTestCaseAndSectionHeader();

        void printClosedHeader(std::string const& _name);
        void printOpenHeader(std::string const& _name);

        // if string has a : in first line will set indent to follow it on
        // subsequent lines
        void printHeaderString(std::string const& _string, std::size_t indent = 0);

        void printTotalsDivider(Totals const& totals);

        bool m_headerPrinted = false;
        bool m_testRunInfoPrinted = false;
    };

} // end namespace Catch

#endif // CATCH_REPORTER_CONSOLE_HPP_INCLUDED
