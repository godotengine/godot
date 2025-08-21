
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/reporters/catch_reporter_event_listener.hpp>

namespace Catch {

    void EventListenerBase::fatalErrorEncountered( StringRef ) {}

    void EventListenerBase::benchmarkPreparing( StringRef ) {}
    void EventListenerBase::benchmarkStarting( BenchmarkInfo const& ) {}
    void EventListenerBase::benchmarkEnded( BenchmarkStats<> const& ) {}
    void EventListenerBase::benchmarkFailed( StringRef ) {}

    void EventListenerBase::assertionStarting( AssertionInfo const& ) {}

    void EventListenerBase::assertionEnded( AssertionStats const& ) {}
    void EventListenerBase::listReporters(
        std::vector<ReporterDescription> const& ) {}
    void EventListenerBase::listListeners(
        std::vector<ListenerDescription> const& ) {}
    void EventListenerBase::listTests( std::vector<TestCaseHandle> const& ) {}
    void EventListenerBase::listTags( std::vector<TagInfo> const& ) {}
    void EventListenerBase::noMatchingTestCases( StringRef ) {}
    void EventListenerBase::reportInvalidTestSpec( StringRef ) {}
    void EventListenerBase::testRunStarting( TestRunInfo const& ) {}
    void EventListenerBase::testCaseStarting( TestCaseInfo const& ) {}
    void EventListenerBase::testCasePartialStarting(TestCaseInfo const&, uint64_t) {}
    void EventListenerBase::sectionStarting( SectionInfo const& ) {}
    void EventListenerBase::sectionEnded( SectionStats const& ) {}
    void EventListenerBase::testCasePartialEnded(TestCaseStats const&, uint64_t) {}
    void EventListenerBase::testCaseEnded( TestCaseStats const& ) {}
    void EventListenerBase::testRunEnded( TestRunStats const& ) {}
    void EventListenerBase::skipTest( TestCaseInfo const& ) {}
} // namespace Catch
