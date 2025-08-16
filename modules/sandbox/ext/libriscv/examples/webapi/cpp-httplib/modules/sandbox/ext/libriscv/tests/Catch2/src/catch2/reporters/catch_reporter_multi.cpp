
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#include <catch2/reporters/catch_reporter_multi.hpp>

#include <catch2/catch_config.hpp>
#include <catch2/internal/catch_move_and_forward.hpp>
#include <catch2/internal/catch_stdstreams.hpp>

#include <ostream>

namespace Catch {
    void MultiReporter::updatePreferences(IEventListener const& reporterish) {
        m_preferences.shouldRedirectStdOut |=
            reporterish.getPreferences().shouldRedirectStdOut;
        m_preferences.shouldReportAllAssertions |=
            reporterish.getPreferences().shouldReportAllAssertions;
        m_preferences.shouldReportAllAssertionStarts |=
            reporterish.getPreferences().shouldReportAllAssertionStarts;
    }

    void MultiReporter::addListener( IEventListenerPtr&& listener ) {
        updatePreferences(*listener);
        m_reporterLikes.insert(m_reporterLikes.begin() + m_insertedListeners, CATCH_MOVE(listener) );
        ++m_insertedListeners;
    }

    void MultiReporter::addReporter( IEventListenerPtr&& reporter ) {
        updatePreferences(*reporter);

        // We will need to output the captured stdout if there are reporters
        // that do not want it captured.
        // We do not consider listeners, because it is generally assumed that
        // listeners are output-transparent, even though they can ask for stdout
        // capture to do something with it.
        m_haveNoncapturingReporters |= !reporter->getPreferences().shouldRedirectStdOut;

        // Reporters can always be placed to the back without breaking the
        // reporting order
        m_reporterLikes.push_back( CATCH_MOVE( reporter ) );
    }

    void MultiReporter::noMatchingTestCases( StringRef unmatchedSpec ) {
        for ( auto& reporterish : m_reporterLikes ) {
            reporterish->noMatchingTestCases( unmatchedSpec );
        }
    }

    void MultiReporter::fatalErrorEncountered( StringRef error ) {
        for ( auto& reporterish : m_reporterLikes ) {
            reporterish->fatalErrorEncountered( error );
        }
    }

    void MultiReporter::reportInvalidTestSpec( StringRef arg ) {
        for ( auto& reporterish : m_reporterLikes ) {
            reporterish->reportInvalidTestSpec( arg );
        }
    }

    void MultiReporter::benchmarkPreparing( StringRef name ) {
        for (auto& reporterish : m_reporterLikes) {
            reporterish->benchmarkPreparing(name);
        }
    }
    void MultiReporter::benchmarkStarting( BenchmarkInfo const& benchmarkInfo ) {
        for ( auto& reporterish : m_reporterLikes ) {
            reporterish->benchmarkStarting( benchmarkInfo );
        }
    }
    void MultiReporter::benchmarkEnded( BenchmarkStats<> const& benchmarkStats ) {
        for ( auto& reporterish : m_reporterLikes ) {
            reporterish->benchmarkEnded( benchmarkStats );
        }
    }

    void MultiReporter::benchmarkFailed( StringRef error ) {
        for (auto& reporterish : m_reporterLikes) {
            reporterish->benchmarkFailed(error);
        }
    }

    void MultiReporter::testRunStarting( TestRunInfo const& testRunInfo ) {
        for ( auto& reporterish : m_reporterLikes ) {
            reporterish->testRunStarting( testRunInfo );
        }
    }

    void MultiReporter::testCaseStarting( TestCaseInfo const& testInfo ) {
        for ( auto& reporterish : m_reporterLikes ) {
            reporterish->testCaseStarting( testInfo );
        }
    }

    void
    MultiReporter::testCasePartialStarting( TestCaseInfo const& testInfo,
                                                uint64_t partNumber ) {
        for ( auto& reporterish : m_reporterLikes ) {
            reporterish->testCasePartialStarting( testInfo, partNumber );
        }
    }

    void MultiReporter::sectionStarting( SectionInfo const& sectionInfo ) {
        for ( auto& reporterish : m_reporterLikes ) {
            reporterish->sectionStarting( sectionInfo );
        }
    }

    void MultiReporter::assertionStarting( AssertionInfo const& assertionInfo ) {
        for ( auto& reporterish : m_reporterLikes ) {
            reporterish->assertionStarting( assertionInfo );
        }
    }

    void MultiReporter::assertionEnded( AssertionStats const& assertionStats ) {
        const bool reportByDefault =
            assertionStats.assertionResult.getResultType() != ResultWas::Ok ||
            m_config->includeSuccessfulResults();

        for ( auto & reporterish : m_reporterLikes ) {
            if ( reportByDefault ||
                 reporterish->getPreferences().shouldReportAllAssertions ) {
                    reporterish->assertionEnded( assertionStats );
            }
        }
    }

    void MultiReporter::sectionEnded( SectionStats const& sectionStats ) {
        for ( auto& reporterish : m_reporterLikes ) {
            reporterish->sectionEnded( sectionStats );
        }
    }

    void MultiReporter::testCasePartialEnded( TestCaseStats const& testStats,
                                                  uint64_t partNumber ) {
        if ( m_preferences.shouldRedirectStdOut &&
             m_haveNoncapturingReporters ) {
            if ( !testStats.stdOut.empty() ) {
                Catch::cout() << testStats.stdOut << std::flush;
            }
            if ( !testStats.stdErr.empty() ) {
                Catch::cerr() << testStats.stdErr << std::flush;
            }
        }

        for ( auto& reporterish : m_reporterLikes ) {
            reporterish->testCasePartialEnded( testStats, partNumber );
        }
    }

    void MultiReporter::testCaseEnded( TestCaseStats const& testCaseStats ) {
        for ( auto& reporterish : m_reporterLikes ) {
            reporterish->testCaseEnded( testCaseStats );
        }
    }

    void MultiReporter::testRunEnded( TestRunStats const& testRunStats ) {
        for ( auto& reporterish : m_reporterLikes ) {
            reporterish->testRunEnded( testRunStats );
        }
    }


    void MultiReporter::skipTest( TestCaseInfo const& testInfo ) {
        for ( auto& reporterish : m_reporterLikes ) {
            reporterish->skipTest( testInfo );
        }
    }

    void MultiReporter::listReporters(std::vector<ReporterDescription> const& descriptions) {
        for (auto& reporterish : m_reporterLikes) {
            reporterish->listReporters(descriptions);
        }
    }

    void MultiReporter::listListeners(
        std::vector<ListenerDescription> const& descriptions ) {
        for ( auto& reporterish : m_reporterLikes ) {
            reporterish->listListeners( descriptions );
        }
    }

    void MultiReporter::listTests(std::vector<TestCaseHandle> const& tests) {
        for (auto& reporterish : m_reporterLikes) {
            reporterish->listTests(tests);
        }
    }

    void MultiReporter::listTags(std::vector<TagInfo> const& tags) {
        for (auto& reporterish : m_reporterLikes) {
            reporterish->listTags(tags);
        }
    }

} // end namespace Catch
