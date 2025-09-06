
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#include <catch2/catch_config.hpp>
#include <catch2/catch_user_config.hpp>
#include <catch2/internal/catch_enforce.hpp>
#include <catch2/internal/catch_parse_numbers.hpp>
#include <catch2/internal/catch_stdstreams.hpp>
#include <catch2/internal/catch_stringref.hpp>
#include <catch2/internal/catch_string_manip.hpp>
#include <catch2/internal/catch_test_spec_parser.hpp>
#include <catch2/interfaces/catch_interfaces_tag_alias_registry.hpp>
#include <catch2/internal/catch_getenv.hpp>

#include <fstream>

namespace Catch {

    namespace {
        static bool enableBazelEnvSupport() {
#if defined( CATCH_CONFIG_BAZEL_SUPPORT )
            return true;
#else
            return Detail::getEnv( "BAZEL_TEST" ) != nullptr;
#endif
        }

        struct bazelShardingOptions {
            unsigned int shardIndex, shardCount;
            std::string shardFilePath;
        };

        static Optional<bazelShardingOptions> readBazelShardingOptions() {
            const auto bazelShardIndex = Detail::getEnv( "TEST_SHARD_INDEX" );
            const auto bazelShardTotal = Detail::getEnv( "TEST_TOTAL_SHARDS" );
            const auto bazelShardInfoFile = Detail::getEnv( "TEST_SHARD_STATUS_FILE" );


            const bool has_all =
                bazelShardIndex && bazelShardTotal && bazelShardInfoFile;
            if ( !has_all ) {
                // We provide nice warning message if the input is
                // misconfigured.
                auto warn = []( const char* env_var ) {
                    Catch::cerr()
                        << "Warning: Bazel shard configuration is missing '"
                        << env_var << "'. Shard configuration is skipped.\n";
                };
                if ( !bazelShardIndex ) {
                    warn( "TEST_SHARD_INDEX" );
                }
                if ( !bazelShardTotal ) {
                    warn( "TEST_TOTAL_SHARDS" );
                }
                if ( !bazelShardInfoFile ) {
                    warn( "TEST_SHARD_STATUS_FILE" );
                }
                return {};
            }

            auto shardIndex = parseUInt( bazelShardIndex );
            if ( !shardIndex ) {
                Catch::cerr()
                    << "Warning: could not parse 'TEST_SHARD_INDEX' ('" << bazelShardIndex
                    << "') as unsigned int.\n";
                return {};
            }
            auto shardTotal = parseUInt( bazelShardTotal );
            if ( !shardTotal ) {
                Catch::cerr()
                    << "Warning: could not parse 'TEST_TOTAL_SHARD' ('"
                    << bazelShardTotal << "') as unsigned int.\n";
                return {};
            }

            return bazelShardingOptions{
                *shardIndex, *shardTotal, bazelShardInfoFile };

        }
    } // end namespace


    bool operator==( ProcessedReporterSpec const& lhs,
                     ProcessedReporterSpec const& rhs ) {
        return lhs.name == rhs.name &&
               lhs.outputFilename == rhs.outputFilename &&
               lhs.colourMode == rhs.colourMode &&
               lhs.customOptions == rhs.customOptions;
    }

    Config::Config( ConfigData const& data ):
        m_data( data ) {
        // We need to trim filter specs to avoid trouble with superfluous
        // whitespace (esp. important for bdd macros, as those are manually
        // aligned with whitespace).

        for (auto& elem : m_data.testsOrTags) {
            elem = trim(elem);
        }
        for (auto& elem : m_data.sectionsToRun) {
            elem = trim(elem);
        }

        // Insert the default reporter if user hasn't asked for a specific one
        if ( m_data.reporterSpecifications.empty() ) {
#if defined( CATCH_CONFIG_DEFAULT_REPORTER )
            const auto default_spec = CATCH_CONFIG_DEFAULT_REPORTER;
#else
            const auto default_spec = "console";
#endif
            auto parsed = parseReporterSpec(default_spec);
            CATCH_ENFORCE( parsed,
                           "Cannot parse the provided default reporter spec: '"
                               << default_spec << '\'' );
            m_data.reporterSpecifications.push_back( std::move( *parsed ) );
        }

        if ( enableBazelEnvSupport() ) {
            readBazelEnvVars();
        }

        // Bazel support can modify the test specs, so parsing has to happen
        // after reading Bazel env vars.
        TestSpecParser parser( ITagAliasRegistry::get() );
        if ( !m_data.testsOrTags.empty() ) {
            m_hasTestFilters = true;
            for ( auto const& testOrTags : m_data.testsOrTags ) {
                parser.parse( testOrTags );
            }
        }
        m_testSpec = parser.testSpec();


        // We now fixup the reporter specs to handle default output spec,
        // default colour spec, etc
        bool defaultOutputUsed = false;
        for ( auto const& reporterSpec : m_data.reporterSpecifications ) {
            // We do the default-output check separately, while always
            // using the default output below to make the code simpler
            // and avoid superfluous copies.
            if ( reporterSpec.outputFile().none() ) {
                CATCH_ENFORCE( !defaultOutputUsed,
                               "Internal error: cannot use default output for "
                               "multiple reporters" );
                defaultOutputUsed = true;
            }

            m_processedReporterSpecs.push_back( ProcessedReporterSpec{
                reporterSpec.name(),
                reporterSpec.outputFile() ? *reporterSpec.outputFile()
                                          : data.defaultOutputFilename,
                reporterSpec.colourMode().valueOr( data.defaultColourMode ),
                reporterSpec.customOptions() } );
        }
    }

    Config::~Config() = default;


    bool Config::listTests() const          { return m_data.listTests; }
    bool Config::listTags() const           { return m_data.listTags; }
    bool Config::listReporters() const      { return m_data.listReporters; }
    bool Config::listListeners() const      { return m_data.listListeners; }

    std::vector<std::string> const& Config::getTestsOrTags() const { return m_data.testsOrTags; }
    std::vector<std::string> const& Config::getSectionsToRun() const { return m_data.sectionsToRun; }

    std::vector<ReporterSpec> const& Config::getReporterSpecs() const {
        return m_data.reporterSpecifications;
    }

    std::vector<ProcessedReporterSpec> const&
    Config::getProcessedReporterSpecs() const {
        return m_processedReporterSpecs;
    }

    TestSpec const& Config::testSpec() const { return m_testSpec; }
    bool Config::hasTestFilters() const { return m_hasTestFilters; }

    bool Config::showHelp() const { return m_data.showHelp; }

    // IConfig interface
    bool Config::allowThrows() const                   { return !m_data.noThrow; }
    StringRef Config::name() const { return m_data.name.empty() ? m_data.processName : m_data.name; }
    bool Config::includeSuccessfulResults() const      { return m_data.showSuccessfulTests; }
    bool Config::warnAboutMissingAssertions() const {
        return !!( m_data.warnings & WarnAbout::NoAssertions );
    }
    bool Config::warnAboutUnmatchedTestSpecs() const {
        return !!( m_data.warnings & WarnAbout::UnmatchedTestSpec );
    }
    bool Config::zeroTestsCountAsSuccess() const       { return m_data.allowZeroTests; }
    ShowDurations Config::showDurations() const        { return m_data.showDurations; }
    double Config::minDuration() const                 { return m_data.minDuration; }
    TestRunOrder Config::runOrder() const              { return m_data.runOrder; }
    uint32_t Config::rngSeed() const                   { return m_data.rngSeed; }
    unsigned int Config::shardCount() const            { return m_data.shardCount; }
    unsigned int Config::shardIndex() const            { return m_data.shardIndex; }
    ColourMode Config::defaultColourMode() const       { return m_data.defaultColourMode; }
    bool Config::shouldDebugBreak() const              { return m_data.shouldDebugBreak; }
    int Config::abortAfter() const                     { return m_data.abortAfter; }
    bool Config::showInvisibles() const                { return m_data.showInvisibles; }
    Verbosity Config::verbosity() const                { return m_data.verbosity; }

    bool Config::skipBenchmarks() const                           { return m_data.skipBenchmarks; }
    bool Config::benchmarkNoAnalysis() const                      { return m_data.benchmarkNoAnalysis; }
    unsigned int Config::benchmarkSamples() const                 { return m_data.benchmarkSamples; }
    double Config::benchmarkConfidenceInterval() const            { return m_data.benchmarkConfidenceInterval; }
    unsigned int Config::benchmarkResamples() const               { return m_data.benchmarkResamples; }
    std::chrono::milliseconds Config::benchmarkWarmupTime() const { return std::chrono::milliseconds(m_data.benchmarkWarmupTime); }

    void Config::readBazelEnvVars() {
        // Register a JUnit reporter for Bazel. Bazel sets an environment
        // variable with the path to XML output. If this file is written to
        // during test, Bazel will not generate a default XML output.
        // This allows the XML output file to contain higher level of detail
        // than what is possible otherwise.
        const auto bazelOutputFile = Detail::getEnv( "XML_OUTPUT_FILE" );

        if ( bazelOutputFile ) {
            m_data.reporterSpecifications.push_back(
                { "junit", std::string( bazelOutputFile ), {}, {} } );
        }

        const auto bazelTestSpec = Detail::getEnv( "TESTBRIDGE_TEST_ONLY" );
        if ( bazelTestSpec ) {
            // Presumably the test spec from environment should overwrite
            // the one we got from CLI (if we got any)
            m_data.testsOrTags.clear();
            m_data.testsOrTags.push_back( bazelTestSpec );
        }

        const auto bazelShardOptions = readBazelShardingOptions();
        if ( bazelShardOptions ) {
            std::ofstream f( bazelShardOptions->shardFilePath,
                             std::ios_base::out | std::ios_base::trunc );
            if ( f.is_open() ) {
                f << "";
                m_data.shardIndex = bazelShardOptions->shardIndex;
                m_data.shardCount = bazelShardOptions->shardCount;
            }
        }
    }

} // end namespace Catch
