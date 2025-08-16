
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_REPORTER_MULTI_HPP_INCLUDED
#define CATCH_REPORTER_MULTI_HPP_INCLUDED

#include <catch2/interfaces/catch_interfaces_reporter.hpp>

namespace Catch {

    class MultiReporter final : public IEventListener {
        /*
         * Stores all added reporters and listeners
         *
         * All Listeners are stored before all reporters, and individual
         * listeners/reporters are stored in order of insertion.
         */
        std::vector<IEventListenerPtr> m_reporterLikes;
        bool m_haveNoncapturingReporters = false;

        // Keep track of how many listeners we have already inserted,
        // so that we can insert them into the main vector at the right place
        size_t m_insertedListeners = 0;

        void updatePreferences(IEventListener const& reporterish);

    public:
        MultiReporter( IConfig const* config ):
            IEventListener( config ) {
            m_preferences.shouldReportAllAssertionStarts = false;
        }

        using IEventListener::IEventListener;

        void addListener( IEventListenerPtr&& listener );
        void addReporter( IEventListenerPtr&& reporter );

    public: // IEventListener

        void noMatchingTestCases( StringRef unmatchedSpec ) override;
        void fatalErrorEncountered( StringRef error ) override;
        void reportInvalidTestSpec( StringRef arg ) override;

        void benchmarkPreparing( StringRef name ) override;
        void benchmarkStarting( BenchmarkInfo const& benchmarkInfo ) override;
        void benchmarkEnded( BenchmarkStats<> const& benchmarkStats ) override;
        void benchmarkFailed( StringRef error ) override;

        void testRunStarting( TestRunInfo const& testRunInfo ) override;
        void testCaseStarting( TestCaseInfo const& testInfo ) override;
        void testCasePartialStarting(TestCaseInfo const& testInfo, uint64_t partNumber) override;
        void sectionStarting( SectionInfo const& sectionInfo ) override;
        void assertionStarting( AssertionInfo const& assertionInfo ) override;

        void assertionEnded( AssertionStats const& assertionStats ) override;
        void sectionEnded( SectionStats const& sectionStats ) override;
        void testCasePartialEnded(TestCaseStats const& testStats, uint64_t partNumber) override;
        void testCaseEnded( TestCaseStats const& testCaseStats ) override;
        void testRunEnded( TestRunStats const& testRunStats ) override;

        void skipTest( TestCaseInfo const& testInfo ) override;

        void listReporters(std::vector<ReporterDescription> const& descriptions) override;
        void listListeners(std::vector<ListenerDescription> const& descriptions) override;
        void listTests(std::vector<TestCaseHandle> const& tests) override;
        void listTags(std::vector<TagInfo> const& tags) override;


    };

} // end namespace Catch

#endif // CATCH_REPORTER_MULTI_HPP_INCLUDED
