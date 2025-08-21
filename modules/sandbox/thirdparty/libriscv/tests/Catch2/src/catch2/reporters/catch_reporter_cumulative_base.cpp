
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#include <catch2/reporters/catch_reporter_cumulative_base.hpp>

#include <catch2/internal/catch_move_and_forward.hpp>

#include <algorithm>
#include <cassert>

namespace Catch {
    namespace {
        struct BySectionInfo {
            BySectionInfo( SectionInfo const& other ): m_other( other ) {}
            BySectionInfo( BySectionInfo const& other ) = default;
            bool operator()(
                Detail::unique_ptr<CumulativeReporterBase::SectionNode> const&
                    node ) const {
                return (
                    ( node->stats.sectionInfo.name == m_other.name ) &&
                    ( node->stats.sectionInfo.lineInfo == m_other.lineInfo ) );
            }
            void operator=( BySectionInfo const& ) = delete;

        private:
            SectionInfo const& m_other;
        };

    } // namespace

    namespace Detail {
        AssertionOrBenchmarkResult::AssertionOrBenchmarkResult(
            AssertionStats const& assertion ):
            m_assertion( assertion ) {}

        AssertionOrBenchmarkResult::AssertionOrBenchmarkResult(
            BenchmarkStats<> const& benchmark ):
            m_benchmark( benchmark ) {}

        bool AssertionOrBenchmarkResult::isAssertion() const {
            return m_assertion.some();
        }
        bool AssertionOrBenchmarkResult::isBenchmark() const {
            return m_benchmark.some();
        }

        AssertionStats const& AssertionOrBenchmarkResult::asAssertion() const {
            assert(m_assertion.some());

            return *m_assertion;
        }
        BenchmarkStats<> const& AssertionOrBenchmarkResult::asBenchmark() const {
            assert(m_benchmark.some());

            return *m_benchmark;
        }

    }

    CumulativeReporterBase::~CumulativeReporterBase() = default;

    void CumulativeReporterBase::benchmarkEnded(BenchmarkStats<> const& benchmarkStats) {
        m_sectionStack.back()->assertionsAndBenchmarks.emplace_back(benchmarkStats);
    }

    void
    CumulativeReporterBase::sectionStarting( SectionInfo const& sectionInfo ) {
        // We need a copy, because SectionStats expect to take ownership
        SectionStats incompleteStats( SectionInfo(sectionInfo), Counts(), 0, false );
        SectionNode* node;
        if ( m_sectionStack.empty() ) {
            if ( !m_rootSection ) {
                m_rootSection =
                    Detail::make_unique<SectionNode>( incompleteStats );
            }
            node = m_rootSection.get();
        } else {
            SectionNode& parentNode = *m_sectionStack.back();
            auto it = std::find_if( parentNode.childSections.begin(),
                                    parentNode.childSections.end(),
                                    BySectionInfo( sectionInfo ) );
            if ( it == parentNode.childSections.end() ) {
                auto newNode =
                    Detail::make_unique<SectionNode>( incompleteStats );
                node = newNode.get();
                parentNode.childSections.push_back( CATCH_MOVE( newNode ) );
            } else {
                node = it->get();
            }
        }

        m_deepestSection = node;
        m_sectionStack.push_back( node );
    }

    void CumulativeReporterBase::assertionEnded(
        AssertionStats const& assertionStats ) {
        assert( !m_sectionStack.empty() );
        // AssertionResult holds a pointer to a temporary DecomposedExpression,
        // which getExpandedExpression() calls to build the expression string.
        // Our section stack copy of the assertionResult will likely outlive the
        // temporary, so it must be expanded or discarded now to avoid calling
        // a destroyed object later.
        if ( m_shouldStoreFailedAssertions &&
             !assertionStats.assertionResult.isOk() ) {
            static_cast<void>(
                assertionStats.assertionResult.getExpandedExpression() );
        }
        if ( m_shouldStoreSuccesfulAssertions &&
             assertionStats.assertionResult.isOk() ) {
            static_cast<void>(
                assertionStats.assertionResult.getExpandedExpression() );
        }
        SectionNode& sectionNode = *m_sectionStack.back();
        sectionNode.assertionsAndBenchmarks.emplace_back( assertionStats );
    }

    void CumulativeReporterBase::sectionEnded( SectionStats const& sectionStats ) {
        assert( !m_sectionStack.empty() );
        SectionNode& node = *m_sectionStack.back();
        node.stats = sectionStats;
        m_sectionStack.pop_back();
    }

    void CumulativeReporterBase::testCaseEnded(
        TestCaseStats const& testCaseStats ) {
        auto node = Detail::make_unique<TestCaseNode>( testCaseStats );
        assert( m_sectionStack.size() == 0 );
        node->children.push_back( CATCH_MOVE(m_rootSection) );
        m_testCases.push_back( CATCH_MOVE(node) );

        assert( m_deepestSection );
        m_deepestSection->stdOut = testCaseStats.stdOut;
        m_deepestSection->stdErr = testCaseStats.stdErr;
    }


    void CumulativeReporterBase::testRunEnded( TestRunStats const& testRunStats ) {
        assert(!m_testRun && "CumulativeReporterBase assumes there can only be one test run");
        m_testRun = Detail::make_unique<TestRunNode>( testRunStats );
        m_testRun->children.swap( m_testCases );
        testRunEndedCumulative();
    }

    bool CumulativeReporterBase::SectionNode::hasAnyAssertions() const {
        return std::any_of(
            assertionsAndBenchmarks.begin(),
            assertionsAndBenchmarks.end(),
            []( Detail::AssertionOrBenchmarkResult const& res ) {
                return res.isAssertion();
            } );
    }

} // end namespace Catch
