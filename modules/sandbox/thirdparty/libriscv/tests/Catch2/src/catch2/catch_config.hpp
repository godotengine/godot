
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_CONFIG_HPP_INCLUDED
#define CATCH_CONFIG_HPP_INCLUDED

#include <catch2/catch_test_spec.hpp>
#include <catch2/interfaces/catch_interfaces_config.hpp>
#include <catch2/internal/catch_unique_ptr.hpp>
#include <catch2/internal/catch_optional.hpp>
#include <catch2/internal/catch_stringref.hpp>
#include <catch2/internal/catch_random_seed_generation.hpp>
#include <catch2/internal/catch_reporter_spec_parser.hpp>

#include <chrono>
#include <map>
#include <string>
#include <vector>

namespace Catch {

    class IStream;

    /**
     * `ReporterSpec` but with the defaults filled in.
     *
     * Like `ReporterSpec`, the semantics are unchecked.
     */
    struct ProcessedReporterSpec {
        std::string name;
        std::string outputFilename;
        ColourMode colourMode;
        std::map<std::string, std::string> customOptions;
        friend bool operator==( ProcessedReporterSpec const& lhs,
                                ProcessedReporterSpec const& rhs );
        friend bool operator!=( ProcessedReporterSpec const& lhs,
                                ProcessedReporterSpec const& rhs ) {
            return !( lhs == rhs );
        }
    };

    struct ConfigData {

        bool listTests = false;
        bool listTags = false;
        bool listReporters = false;
        bool listListeners = false;

        bool showSuccessfulTests = false;
        bool shouldDebugBreak = false;
        bool noThrow = false;
        bool showHelp = false;
        bool showInvisibles = false;
        bool filenamesAsTags = false;
        bool libIdentify = false;
        bool allowZeroTests = false;

        int abortAfter = -1;
        uint32_t rngSeed = generateRandomSeed(GenerateFrom::Default);

        unsigned int shardCount = 1;
        unsigned int shardIndex = 0;

        bool skipBenchmarks = false;
        bool benchmarkNoAnalysis = false;
        unsigned int benchmarkSamples = 100;
        double benchmarkConfidenceInterval = 0.95;
        unsigned int benchmarkResamples = 100'000;
        std::chrono::milliseconds::rep benchmarkWarmupTime = 100;

        Verbosity verbosity = Verbosity::Normal;
        WarnAbout::What warnings = WarnAbout::Nothing;
        ShowDurations showDurations = ShowDurations::DefaultForReporter;
        double minDuration = -1;
        TestRunOrder runOrder = TestRunOrder::Randomized;
        ColourMode defaultColourMode = ColourMode::PlatformDefault;
        WaitForKeypress::When waitForKeypress = WaitForKeypress::Never;

        std::string defaultOutputFilename;
        std::string name;
        std::string processName;
        std::vector<ReporterSpec> reporterSpecifications;

        std::vector<std::string> testsOrTags;
        std::vector<std::string> sectionsToRun;
    };


    class Config : public IConfig {
    public:

        Config() = default;
        Config( ConfigData const& data );
        ~Config() override; // = default in the cpp file

        bool listTests() const;
        bool listTags() const;
        bool listReporters() const;
        bool listListeners() const;

        std::vector<ReporterSpec> const& getReporterSpecs() const;
        std::vector<ProcessedReporterSpec> const&
        getProcessedReporterSpecs() const;

        std::vector<std::string> const& getTestsOrTags() const override;
        std::vector<std::string> const& getSectionsToRun() const override;

        TestSpec const& testSpec() const override;
        bool hasTestFilters() const override;

        bool showHelp() const;

        // IConfig interface
        bool allowThrows() const override;
        StringRef name() const override;
        bool includeSuccessfulResults() const override;
        bool warnAboutMissingAssertions() const override;
        bool warnAboutUnmatchedTestSpecs() const override;
        bool zeroTestsCountAsSuccess() const override;
        ShowDurations showDurations() const override;
        double minDuration() const override;
        TestRunOrder runOrder() const override;
        uint32_t rngSeed() const override;
        unsigned int shardCount() const override;
        unsigned int shardIndex() const override;
        ColourMode defaultColourMode() const override;
        bool shouldDebugBreak() const override;
        int abortAfter() const override;
        bool showInvisibles() const override;
        Verbosity verbosity() const override;
        bool skipBenchmarks() const override;
        bool benchmarkNoAnalysis() const override;
        unsigned int benchmarkSamples() const override;
        double benchmarkConfidenceInterval() const override;
        unsigned int benchmarkResamples() const override;
        std::chrono::milliseconds benchmarkWarmupTime() const override;

    private:
        // Reads Bazel env vars and applies them to the config
        void readBazelEnvVars();

        ConfigData m_data;
        std::vector<ProcessedReporterSpec> m_processedReporterSpecs;
        TestSpec m_testSpec;
        bool m_hasTestFilters = false;
    };
} // end namespace Catch

#endif // CATCH_CONFIG_HPP_INCLUDED
