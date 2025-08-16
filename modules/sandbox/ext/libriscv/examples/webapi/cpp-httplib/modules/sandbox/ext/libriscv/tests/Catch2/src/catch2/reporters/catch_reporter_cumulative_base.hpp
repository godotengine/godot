
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_REPORTER_CUMULATIVE_BASE_HPP_INCLUDED
#define CATCH_REPORTER_CUMULATIVE_BASE_HPP_INCLUDED

#include <catch2/reporters/catch_reporter_common_base.hpp>
#include <catch2/internal/catch_move_and_forward.hpp>
#include <catch2/internal/catch_unique_ptr.hpp>
#include <catch2/internal/catch_optional.hpp>

#include <string>
#include <vector>

namespace Catch {

    namespace Detail {

        //! Represents either an assertion or a benchmark result to be handled by cumulative reporter later
        class AssertionOrBenchmarkResult {
            // This should really be a variant, but this is much faster
            // to write and the data layout here is already terrible
            // enough that we do not have to care about the object size.
            Optional<AssertionStats> m_assertion;
            Optional<BenchmarkStats<>> m_benchmark;
        public:
            AssertionOrBenchmarkResult(AssertionStats const& assertion);
            AssertionOrBenchmarkResult(BenchmarkStats<> const& benchmark);

            bool isAssertion() const;
            bool isBenchmark() const;

            AssertionStats const& asAssertion() const;
            BenchmarkStats<> const& asBenchmark() const;
        };
    }

    /**
     * Utility base for reporters that need to handle all results at once
     *
     * It stores tree of all test cases, sections and assertions, and after the
     * test run is finished, calls into `testRunEndedCumulative` to pass the
     * control to the deriving class.
     *
     * If you are deriving from this class and override any testing related
     * member functions, you should first call into the base's implementation to
     * avoid breaking the tree construction.
     *
     * Due to the way this base functions, it has to expand assertions up-front,
     * even if they are later unused (e.g. because the deriving reporter does
     * not report successful assertions, or because the deriving reporter does
     * not use assertion expansion at all). Derived classes can use two
     * customization points, `m_shouldStoreSuccesfulAssertions` and
     * `m_shouldStoreFailedAssertions`, to disable the expansion and gain extra
     * performance. **Accessing the assertion expansions if it wasn't stored is
     * UB.**
     */
    class CumulativeReporterBase : public ReporterBase {
    public:
        template<typename T, typename ChildNodeT>
        struct Node {
            explicit Node( T const& _value ) : value( _value ) {}

            using ChildNodes = std::vector<Detail::unique_ptr<ChildNodeT>>;
            T value;
            ChildNodes children;
        };
        struct SectionNode {
            explicit SectionNode(SectionStats const& _stats) : stats(_stats) {}

            bool operator == (SectionNode const& other) const {
                return stats.sectionInfo.lineInfo == other.stats.sectionInfo.lineInfo;
            }

            bool hasAnyAssertions() const;

            SectionStats stats;
            std::vector<Detail::unique_ptr<SectionNode>> childSections;
            std::vector<Detail::AssertionOrBenchmarkResult> assertionsAndBenchmarks;
            std::string stdOut;
            std::string stdErr;
        };


        using TestCaseNode = Node<TestCaseStats, SectionNode>;
        using TestRunNode = Node<TestRunStats, TestCaseNode>;

        // GCC5 compat: we cannot use inherited constructor, because it
        //              doesn't implement backport of P0136
        CumulativeReporterBase(ReporterConfig&& _config):
            ReporterBase(CATCH_MOVE(_config))
        {}
        ~CumulativeReporterBase() override;

        void benchmarkPreparing( StringRef ) override {}
        void benchmarkStarting( BenchmarkInfo const& ) override {}
        void benchmarkEnded( BenchmarkStats<> const& benchmarkStats ) override;
        void benchmarkFailed( StringRef ) override {}

        void noMatchingTestCases( StringRef ) override {}
        void reportInvalidTestSpec( StringRef ) override {}
        void fatalErrorEncountered( StringRef /*error*/ ) override {}

        void testRunStarting( TestRunInfo const& ) override {}

        void testCaseStarting( TestCaseInfo const& ) override {}
        void testCasePartialStarting( TestCaseInfo const&, uint64_t ) override {}
        void sectionStarting( SectionInfo const& sectionInfo ) override;

        void assertionStarting( AssertionInfo const& ) override {}

        void assertionEnded( AssertionStats const& assertionStats ) override;
        void sectionEnded( SectionStats const& sectionStats ) override;
        void testCasePartialEnded( TestCaseStats const&, uint64_t ) override {}
        void testCaseEnded( TestCaseStats const& testCaseStats ) override;
        void testRunEnded( TestRunStats const& testRunStats ) override;
        //! Customization point: called after last test finishes (testRunEnded has been handled)
        virtual void testRunEndedCumulative() = 0;

        void skipTest(TestCaseInfo const&) override {}

    protected:
        //! Should the cumulative base store the assertion expansion for successful assertions?
        bool m_shouldStoreSuccesfulAssertions = true;
        //! Should the cumulative base store the assertion expansion for failed assertions?
        bool m_shouldStoreFailedAssertions = true;

        // We need lazy construction here. We should probably refactor it
        // later, after the events are redone.
        //! The root node of the test run tree.
        Detail::unique_ptr<TestRunNode> m_testRun;

    private:
        // Note: We rely on pointer identity being stable, which is why
        //       we store pointers to the nodes rather than the values.
        std::vector<Detail::unique_ptr<TestCaseNode>> m_testCases;
        // Root section of the _current_ test case
        Detail::unique_ptr<SectionNode> m_rootSection;
        // Deepest section of the _current_ test case
        SectionNode* m_deepestSection = nullptr;
        // Stack of _active_ sections in the _current_ test case
        std::vector<SectionNode*> m_sectionStack;
    };

} // end namespace Catch

#endif // CATCH_REPORTER_CUMULATIVE_BASE_HPP_INCLUDED
