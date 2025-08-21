
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_REPORTER_XML_HPP_INCLUDED
#define CATCH_REPORTER_XML_HPP_INCLUDED

#include <catch2/reporters/catch_reporter_streaming_base.hpp>

#include <catch2/internal/catch_xmlwriter.hpp>
#include <catch2/catch_timer.hpp>


namespace Catch {
    class XmlReporter : public StreamingReporterBase {
    public:
        XmlReporter(ReporterConfig&& _config);

        ~XmlReporter() override;

        static std::string getDescription();

        virtual std::string getStylesheetRef() const;

        void writeSourceInfo(SourceLineInfo const& sourceInfo);

    public: // StreamingReporterBase

        void testRunStarting(TestRunInfo const& testInfo) override;

        void testCaseStarting(TestCaseInfo const& testInfo) override;

        void sectionStarting(SectionInfo const& sectionInfo) override;

        void assertionEnded(AssertionStats const& assertionStats) override;

        void sectionEnded(SectionStats const& sectionStats) override;

        void testCaseEnded(TestCaseStats const& testCaseStats) override;

        void testRunEnded(TestRunStats const& testRunStats) override;

        void benchmarkPreparing( StringRef name ) override;
        void benchmarkStarting(BenchmarkInfo const&) override;
        void benchmarkEnded(BenchmarkStats<> const&) override;
        void benchmarkFailed( StringRef error ) override;

        void listReporters(std::vector<ReporterDescription> const& descriptions) override;
        void listListeners(std::vector<ListenerDescription> const& descriptions) override;
        void listTests(std::vector<TestCaseHandle> const& tests) override;
        void listTags(std::vector<TagInfo> const& tags) override;

    private:
        Timer m_testCaseTimer;
        XmlWriter m_xml;
        int m_sectionDepth = 0;
    };

} // end namespace Catch

#endif // CATCH_REPORTER_XML_HPP_INCLUDED
