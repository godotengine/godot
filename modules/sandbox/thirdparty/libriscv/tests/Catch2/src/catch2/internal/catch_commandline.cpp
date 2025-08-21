
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#include <catch2/internal/catch_commandline.hpp>

#include <catch2/catch_config.hpp>
#include <catch2/internal/catch_string_manip.hpp>
#include <catch2/interfaces/catch_interfaces_config.hpp>
#include <catch2/interfaces/catch_interfaces_registry_hub.hpp>
#include <catch2/internal/catch_reporter_registry.hpp>
#include <catch2/internal/catch_console_colour.hpp>
#include <catch2/internal/catch_parse_numbers.hpp>
#include <catch2/internal/catch_reporter_spec_parser.hpp>

#include <fstream>
#include <string>

namespace Catch {

    Clara::Parser makeCommandLineParser( ConfigData& config ) {

        using namespace Clara;

        auto const setWarning = [&]( std::string const& warning ) {
            if ( warning == "NoAssertions" ) {
                config.warnings = static_cast<WarnAbout::What>(config.warnings | WarnAbout::NoAssertions);
                return ParserResult::ok( ParseResultType::Matched );
            } else if ( warning == "UnmatchedTestSpec" ) {
                config.warnings = static_cast<WarnAbout::What>(config.warnings | WarnAbout::UnmatchedTestSpec);
                return ParserResult::ok( ParseResultType::Matched );
            }

            return ParserResult ::runtimeError(
                "Unrecognised warning option: '" + warning + '\'' );
        };
        auto const loadTestNamesFromFile = [&]( std::string const& filename ) {
                std::ifstream f( filename.c_str() );
                if( !f.is_open() )
                    return ParserResult::runtimeError( "Unable to load input file: '" + filename + '\'' );

                std::string line;
                while( std::getline( f, line ) ) {
                    line = trim(line);
                    if( !line.empty() && !startsWith( line, '#' ) ) {
                        if( !startsWith( line, '"' ) )
                            line = '"' + CATCH_MOVE(line) + '"';
                        config.testsOrTags.push_back( line );
                        config.testsOrTags.emplace_back( "," );
                    }
                }
                //Remove comma in the end
                if(!config.testsOrTags.empty())
                    config.testsOrTags.erase( config.testsOrTags.end()-1 );

                return ParserResult::ok( ParseResultType::Matched );
            };
        auto const setTestOrder = [&]( std::string const& order ) {
                if( startsWith( "declared", order ) )
                    config.runOrder = TestRunOrder::Declared;
                else if( startsWith( "lexical", order ) )
                    config.runOrder = TestRunOrder::LexicographicallySorted;
                else if( startsWith( "random", order ) )
                    config.runOrder = TestRunOrder::Randomized;
                else
                    return ParserResult::runtimeError( "Unrecognised ordering: '" + order + '\'' );
                return ParserResult::ok( ParseResultType::Matched );
            };
        auto const setRngSeed = [&]( std::string const& seed ) {
                if( seed == "time" ) {
                    config.rngSeed = generateRandomSeed(GenerateFrom::Time);
                    return ParserResult::ok(ParseResultType::Matched);
                } else if (seed == "random-device") {
                    config.rngSeed = generateRandomSeed(GenerateFrom::RandomDevice);
                    return ParserResult::ok(ParseResultType::Matched);
                }

                // TODO: ideally we should be parsing uint32_t directly
                //       fix this later when we add new parse overload
                auto parsedSeed = parseUInt( seed, 0 );
                if ( !parsedSeed ) {
                    return ParserResult::runtimeError( "Could not parse '" + seed + "' as seed" );
                }
                config.rngSeed = *parsedSeed;
                return ParserResult::ok( ParseResultType::Matched );
            };
        auto const setDefaultColourMode = [&]( std::string const& colourMode ) {
            Optional<ColourMode> maybeMode = Catch::Detail::stringToColourMode(toLower( colourMode ));
            if ( !maybeMode ) {
                return ParserResult::runtimeError(
                    "colour mode must be one of: default, ansi, win32, "
                    "or none. '" +
                    colourMode + "' is not recognised" );
            }
            auto mode = *maybeMode;
            if ( !isColourImplAvailable( mode ) ) {
                return ParserResult::runtimeError(
                    "colour mode '" + colourMode +
                    "' is not supported in this binary" );
            }
            config.defaultColourMode = mode;
            return ParserResult::ok( ParseResultType::Matched );
        };
        auto const setWaitForKeypress = [&]( std::string const& keypress ) {
                auto keypressLc = toLower( keypress );
                if (keypressLc == "never")
                    config.waitForKeypress = WaitForKeypress::Never;
                else if( keypressLc == "start" )
                    config.waitForKeypress = WaitForKeypress::BeforeStart;
                else if( keypressLc == "exit" )
                    config.waitForKeypress = WaitForKeypress::BeforeExit;
                else if( keypressLc == "both" )
                    config.waitForKeypress = WaitForKeypress::BeforeStartAndExit;
                else
                    return ParserResult::runtimeError( "keypress argument must be one of: never, start, exit or both. '" + keypress + "' not recognised" );
            return ParserResult::ok( ParseResultType::Matched );
            };
        auto const setVerbosity = [&]( std::string const& verbosity ) {
            auto lcVerbosity = toLower( verbosity );
            if( lcVerbosity == "quiet" )
                config.verbosity = Verbosity::Quiet;
            else if( lcVerbosity == "normal" )
                config.verbosity = Verbosity::Normal;
            else if( lcVerbosity == "high" )
                config.verbosity = Verbosity::High;
            else
                return ParserResult::runtimeError( "Unrecognised verbosity, '" + verbosity + '\'' );
            return ParserResult::ok( ParseResultType::Matched );
        };
        auto const setReporter = [&]( std::string const& userReporterSpec ) {
            if ( userReporterSpec.empty() ) {
                return ParserResult::runtimeError( "Received empty reporter spec." );
            }

            Optional<ReporterSpec> parsed =
                parseReporterSpec( userReporterSpec );
            if ( !parsed ) {
                return ParserResult::runtimeError(
                    "Could not parse reporter spec '" + userReporterSpec +
                    "'" );
            }

            auto const& reporterSpec = *parsed;

            auto const& factories =
                getRegistryHub().getReporterRegistry().getFactories();
            auto result = factories.find( reporterSpec.name() );

            if ( result == factories.end() ) {
                return ParserResult::runtimeError(
                    "Unrecognized reporter, '" + reporterSpec.name() +
                    "'. Check available with --list-reporters" );
            }


            const bool hadOutputFile = reporterSpec.outputFile().some();
            config.reporterSpecifications.push_back( CATCH_MOVE( *parsed ) );
            // It would be enough to check this only once at the very end, but
            // there is  not a place where we could call this check, so do it
            // every time it could fail. For valid inputs, this is still called
            // at most once.
            if (!hadOutputFile) {
                int n_reporters_without_file = 0;
                for (auto const& spec : config.reporterSpecifications) {
                    if (spec.outputFile().none()) {
                        n_reporters_without_file++;
                    }
                }
                if (n_reporters_without_file > 1) {
                    return ParserResult::runtimeError( "Only one reporter may have unspecified output file." );
                }
            }

            return ParserResult::ok( ParseResultType::Matched );
        };
        auto const setShardCount = [&]( std::string const& shardCount ) {
            auto parsedCount = parseUInt( shardCount );
            if ( !parsedCount ) {
                return ParserResult::runtimeError(
                    "Could not parse '" + shardCount + "' as shard count" );
            }
            if ( *parsedCount == 0 ) {
                return ParserResult::runtimeError(
                    "Shard count must be positive" );
            }
            config.shardCount = *parsedCount;
            return ParserResult::ok( ParseResultType::Matched );
        };

        auto const setShardIndex = [&](std::string const& shardIndex) {
            auto parsedIndex = parseUInt( shardIndex );
            if ( !parsedIndex ) {
                return ParserResult::runtimeError(
                    "Could not parse '" + shardIndex + "' as shard index" );
            }
            config.shardIndex = *parsedIndex;
            return ParserResult::ok( ParseResultType::Matched );
        };

        auto cli
            = ExeName( config.processName )
            | Help( config.showHelp )
            | Opt( config.showSuccessfulTests )
                ["-s"]["--success"]
                ( "include successful tests in output" )
            | Opt( config.shouldDebugBreak )
                ["-b"]["--break"]
                ( "break into debugger on failure" )
            | Opt( config.noThrow )
                ["-e"]["--nothrow"]
                ( "skip exception tests" )
            | Opt( config.showInvisibles )
                ["-i"]["--invisibles"]
                ( "show invisibles (tabs, newlines)" )
            | Opt( config.defaultOutputFilename, "filename" )
                ["-o"]["--out"]
                ( "default output filename" )
            | Opt( accept_many, setReporter, "name[::key=value]*" )
                ["-r"]["--reporter"]
                ( "reporter to use (defaults to console)" )
            | Opt( config.name, "name" )
                ["-n"]["--name"]
                ( "suite name" )
            | Opt( [&]( bool ){ config.abortAfter = 1; } )
                ["-a"]["--abort"]
                ( "abort at first failure" )
            | Opt( [&]( int x ){ config.abortAfter = x; }, "no. failures" )
                ["-x"]["--abortx"]
                ( "abort after x failures" )
            | Opt( accept_many, setWarning, "warning name" )
                ["-w"]["--warn"]
                ( "enable warnings" )
            | Opt( [&]( bool flag ) { config.showDurations = flag ? ShowDurations::Always : ShowDurations::Never; }, "yes|no" )
                ["-d"]["--durations"]
                ( "show test durations" )
            | Opt( config.minDuration, "seconds" )
                ["-D"]["--min-duration"]
                ( "show test durations for tests taking at least the given number of seconds" )
            | Opt( loadTestNamesFromFile, "filename" )
                ["-f"]["--input-file"]
                ( "load test names to run from a file" )
            | Opt( config.filenamesAsTags )
                ["-#"]["--filenames-as-tags"]
                ( "adds a tag for the filename" )
            | Opt( config.sectionsToRun, "section name" )
                ["-c"]["--section"]
                ( "specify section to run" )
            | Opt( setVerbosity, "quiet|normal|high" )
                ["-v"]["--verbosity"]
                ( "set output verbosity" )
            | Opt( config.listTests )
                ["--list-tests"]
                ( "list all/matching test cases" )
            | Opt( config.listTags )
                ["--list-tags"]
                ( "list all/matching tags" )
            | Opt( config.listReporters )
                ["--list-reporters"]
                ( "list all available reporters" )
            | Opt( config.listListeners )
                ["--list-listeners"]
                ( "list all listeners" )
            | Opt( setTestOrder, "decl|lex|rand" )
                ["--order"]
                ( "test case order (defaults to decl)" )
            | Opt( setRngSeed, "'time'|'random-device'|number" )
                ["--rng-seed"]
                ( "set a specific seed for random numbers" )
            | Opt( setDefaultColourMode, "ansi|win32|none|default" )
                ["--colour-mode"]
                ( "what color mode should be used as default" )
            | Opt( config.libIdentify )
                ["--libidentify"]
                ( "report name and version according to libidentify standard" )
            | Opt( setWaitForKeypress, "never|start|exit|both" )
                ["--wait-for-keypress"]
                ( "waits for a keypress before exiting" )
            | Opt( config.skipBenchmarks)
                ["--skip-benchmarks"]
                ( "disable running benchmarks")
            | Opt( config.benchmarkSamples, "samples" )
                ["--benchmark-samples"]
                ( "number of samples to collect (default: 100)" )
            | Opt( config.benchmarkResamples, "resamples" )
                ["--benchmark-resamples"]
                ( "number of resamples for the bootstrap (default: 100000)" )
            | Opt( config.benchmarkConfidenceInterval, "confidence interval" )
                ["--benchmark-confidence-interval"]
                ( "confidence interval for the bootstrap (between 0 and 1, default: 0.95)" )
            | Opt( config.benchmarkNoAnalysis )
                ["--benchmark-no-analysis"]
                ( "perform only measurements; do not perform any analysis" )
            | Opt( config.benchmarkWarmupTime, "benchmarkWarmupTime" )
                ["--benchmark-warmup-time"]
                ( "amount of time in milliseconds spent on warming up each test (default: 100)" )
            | Opt( setShardCount, "shard count" )
                ["--shard-count"]
                ( "split the tests to execute into this many groups" )
            | Opt( setShardIndex, "shard index" )
                ["--shard-index"]
                ( "index of the group of tests to execute (see --shard-count)" )
            | Opt( config.allowZeroTests )
                ["--allow-running-no-tests"]
                ( "Treat 'No tests run' as a success" )
            | Arg( config.testsOrTags, "test name|pattern|tags" )
                ( "which test or tests to use" );

        return cli;
    }

} // end namespace Catch
