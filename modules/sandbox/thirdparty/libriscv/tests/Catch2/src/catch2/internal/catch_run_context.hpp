
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_RUN_CONTEXT_HPP_INCLUDED
#define CATCH_RUN_CONTEXT_HPP_INCLUDED

#include <catch2/interfaces/catch_interfaces_capture.hpp>
#include <catch2/internal/catch_test_registry.hpp>
#include <catch2/catch_test_run_info.hpp>
#include <catch2/internal/catch_fatal_condition_handler.hpp>
#include <catch2/catch_test_case_info.hpp>
#include <catch2/catch_message.hpp>
#include <catch2/catch_totals.hpp>
#include <catch2/internal/catch_test_case_tracker.hpp>
#include <catch2/catch_assertion_info.hpp>
#include <catch2/catch_assertion_result.hpp>
#include <catch2/internal/catch_optional.hpp>
#include <catch2/internal/catch_move_and_forward.hpp>
#include <catch2/internal/catch_thread_support.hpp>

#include <string>

namespace Catch {

    class IGeneratorTracker;
    class IConfig;
    class IEventListener;
    using IEventListenerPtr = Detail::unique_ptr<IEventListener>;
    class OutputRedirect;

    ///////////////////////////////////////////////////////////////////////////

    class RunContext final : public IResultCapture {

    public:
        RunContext( RunContext const& ) = delete;
        RunContext& operator =( RunContext const& ) = delete;

        explicit RunContext( IConfig const* _config, IEventListenerPtr&& reporter );

        ~RunContext() override;

        Totals runTest(TestCaseHandle const& testCase);

    public: // IResultCapture

        // Assertion handlers
        void handleExpr
                (   AssertionInfo const& info,
                    ITransientExpression const& expr,
                    AssertionReaction& reaction ) override;
        void handleMessage
                (   AssertionInfo const& info,
                    ResultWas::OfType resultType,
                    std::string&& message,
                    AssertionReaction& reaction ) override;
        void handleUnexpectedExceptionNotThrown
                (   AssertionInfo const& info,
                    AssertionReaction& reaction ) override;
        void handleUnexpectedInflightException
                (   AssertionInfo const& info,
                    std::string&& message,
                    AssertionReaction& reaction ) override;
        void handleIncomplete
                (   AssertionInfo const& info ) override;
        void handleNonExpr
                (   AssertionInfo const &info,
                    ResultWas::OfType resultType,
                    AssertionReaction &reaction ) override;

        void notifyAssertionStarted( AssertionInfo const& info ) override;
        bool sectionStarted( StringRef sectionName,
                             SourceLineInfo const& sectionLineInfo,
                             Counts& assertions ) override;

        void sectionEnded( SectionEndInfo&& endInfo ) override;
        void sectionEndedEarly( SectionEndInfo&& endInfo ) override;

        IGeneratorTracker*
        acquireGeneratorTracker( StringRef generatorName,
                                 SourceLineInfo const& lineInfo ) override;
        IGeneratorTracker* createGeneratorTracker(
            StringRef generatorName,
            SourceLineInfo lineInfo,
            Generators::GeneratorBasePtr&& generator ) override;


        void benchmarkPreparing( StringRef name ) override;
        void benchmarkStarting( BenchmarkInfo const& info ) override;
        void benchmarkEnded( BenchmarkStats<> const& stats ) override;
        void benchmarkFailed( StringRef error ) override;

        void pushScopedMessage( MessageInfo const& message ) override;
        void popScopedMessage( MessageInfo const& message ) override;

        void emplaceUnscopedMessage( MessageBuilder&& builder ) override;

        std::string getCurrentTestName() const override;

        const AssertionResult* getLastResult() const override;

        void exceptionEarlyReported() override;

        void handleFatalErrorCondition( StringRef message ) override;

        bool lastAssertionPassed() override;

    public:
        // !TBD We need to do this another way!
        bool aborting() const;

    private:
        void assertionPassedFastPath( SourceLineInfo lineInfo );
        // Update the non-thread-safe m_totals from the atomic assertion counts.
        void updateTotalsFromAtomics();

        void runCurrentTest();
        void invokeActiveTestCase();

        bool testForMissingAssertions( Counts& assertions );

        void assertionEnded( AssertionResult&& result );
        void reportExpr
                (   AssertionInfo const &info,
                    ResultWas::OfType resultType,
                    ITransientExpression const *expr,
                    bool negated );

        void populateReaction( AssertionReaction& reaction, bool has_normal_disposition );

        // Creates dummy info for unexpected exceptions/fatal errors,
        // where we do not have the access to one, but we still need
        // to send one to the reporters.
        AssertionInfo makeDummyAssertionInfo();

    private:

        void handleUnfinishedSections();
        mutable Detail::Mutex m_assertionMutex;
        TestRunInfo m_runInfo;
        TestCaseHandle const* m_activeTestCase = nullptr;
        ITracker* m_testCaseTracker = nullptr;
        Optional<AssertionResult> m_lastResult;
        IConfig const* m_config;
        Totals m_totals;
        Detail::AtomicCounts m_atomicAssertionCount;
        IEventListenerPtr m_reporter;
        std::vector<SectionEndInfo> m_unfinishedSections;
        std::vector<ITracker*> m_activeSections;
        TrackerContext m_trackerContext;
        Detail::unique_ptr<OutputRedirect> m_outputRedirect;
        FatalConditionHandler m_fatalConditionhandler;
        // Caches m_config->abortAfter() to avoid vptr calls/allow inlining
        size_t m_abortAfterXFailedAssertions;
        bool m_shouldReportUnexpected = true;
        // Caches whether `assertionStarting` events should be sent to the reporter.
        bool m_reportAssertionStarting;
        // Caches whether `assertionEnded` events for successful assertions should be sent to the reporter
        bool m_includeSuccessfulResults;
        // Caches m_config->shouldDebugBreak() to avoid vptr calls/allow inlining
        bool m_shouldDebugBreak;
    };

    void seedRng(IConfig const& config);
    unsigned int rngSeed();
} // end namespace Catch

#endif // CATCH_RUN_CONTEXT_HPP_INCLUDED
