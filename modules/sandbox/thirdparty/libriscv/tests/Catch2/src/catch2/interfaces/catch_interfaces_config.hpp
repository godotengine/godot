
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_INTERFACES_CONFIG_HPP_INCLUDED
#define CATCH_INTERFACES_CONFIG_HPP_INCLUDED

#include <catch2/internal/catch_noncopyable.hpp>
#include <catch2/internal/catch_stringref.hpp>

#include <chrono>
#include <string>
#include <vector>

namespace Catch {

    enum class Verbosity {
        Quiet = 0,
        Normal,
        High
    };

    struct WarnAbout { enum What {
        Nothing = 0x00,
        //! A test case or leaf section did not run any assertions
        NoAssertions = 0x01,
        //! A command line test spec matched no test cases
        UnmatchedTestSpec = 0x02,
    }; };

    enum class ShowDurations {
        DefaultForReporter,
        Always,
        Never
    };
    enum class TestRunOrder {
        Declared,
        LexicographicallySorted,
        Randomized
    };
    enum class ColourMode : std::uint8_t {
        //! Let Catch2 pick implementation based on platform detection
        PlatformDefault,
        //! Use ANSI colour code escapes
        ANSI,
        //! Use Win32 console colour API
        Win32,
        //! Don't use any colour
        None
    };
    struct WaitForKeypress { enum When {
        Never,
        BeforeStart = 1,
        BeforeExit = 2,
        BeforeStartAndExit = BeforeStart | BeforeExit
    }; };

    class TestSpec;
    class IStream;

    class IConfig : public Detail::NonCopyable {
    public:
        virtual ~IConfig();

        virtual bool allowThrows() const = 0;
        virtual StringRef name() const = 0;
        virtual bool includeSuccessfulResults() const = 0;
        virtual bool shouldDebugBreak() const = 0;
        virtual bool warnAboutMissingAssertions() const = 0;
        virtual bool warnAboutUnmatchedTestSpecs() const = 0;
        virtual bool zeroTestsCountAsSuccess() const = 0;
        virtual int abortAfter() const = 0;
        virtual bool showInvisibles() const = 0;
        virtual ShowDurations showDurations() const = 0;
        virtual double minDuration() const = 0;
        virtual TestSpec const& testSpec() const = 0;
        virtual bool hasTestFilters() const = 0;
        virtual std::vector<std::string> const& getTestsOrTags() const = 0;
        virtual TestRunOrder runOrder() const = 0;
        virtual uint32_t rngSeed() const = 0;
        virtual unsigned int shardCount() const = 0;
        virtual unsigned int shardIndex() const = 0;
        virtual ColourMode defaultColourMode() const = 0;
        virtual std::vector<std::string> const& getSectionsToRun() const = 0;
        virtual Verbosity verbosity() const = 0;

        virtual bool skipBenchmarks() const = 0;
        virtual bool benchmarkNoAnalysis() const = 0;
        virtual unsigned int benchmarkSamples() const = 0;
        virtual double benchmarkConfidenceInterval() const = 0;
        virtual unsigned int benchmarkResamples() const = 0;
        virtual std::chrono::milliseconds benchmarkWarmupTime() const = 0;
    };
}

#endif // CATCH_INTERFACES_CONFIG_HPP_INCLUDED
