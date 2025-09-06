
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#include <catch2/catch_session.hpp>
#include <catch2/internal/catch_console_colour.hpp>
#include <catch2/internal/catch_enforce.hpp>
#include <catch2/internal/catch_list.hpp>
#include <catch2/internal/catch_context.hpp>
#include <catch2/internal/catch_run_context.hpp>
#include <catch2/catch_test_spec.hpp>
#include <catch2/catch_version.hpp>
#include <catch2/internal/catch_startup_exception_registry.hpp>
#include <catch2/internal/catch_sharding.hpp>
#include <catch2/internal/catch_test_case_registry_impl.hpp>
#include <catch2/internal/catch_textflow.hpp>
#include <catch2/internal/catch_windows_h_proxy.hpp>
#include <catch2/reporters/catch_reporter_multi.hpp>
#include <catch2/internal/catch_reporter_registry.hpp>
#include <catch2/interfaces/catch_interfaces_reporter_factory.hpp>
#include <catch2/internal/catch_move_and_forward.hpp>
#include <catch2/internal/catch_stdstreams.hpp>
#include <catch2/internal/catch_istream.hpp>

#include <cassert>
#include <exception>
#include <iomanip>
#include <set>

namespace Catch {

    namespace {
        IEventListenerPtr createReporter(std::string const& reporterName, ReporterConfig&& config) {
            auto reporter = Catch::getRegistryHub().getReporterRegistry().create(reporterName, CATCH_MOVE(config));
            CATCH_ENFORCE(reporter, "No reporter registered with name: '" << reporterName << '\'');

            return reporter;
        }

        IEventListenerPtr prepareReporters(Config const* config) {
            if (Catch::getRegistryHub().getReporterRegistry().getListeners().empty()
                    && config->getProcessedReporterSpecs().size() == 1) {
                auto const& spec = config->getProcessedReporterSpecs()[0];
                return createReporter(
                    spec.name,
                    ReporterConfig( config,
                                    makeStream( spec.outputFilename ),
                                    spec.colourMode,
                                    spec.customOptions ) );
            }

            auto multi = Detail::make_unique<MultiReporter>(config);

            auto const& listeners = Catch::getRegistryHub().getReporterRegistry().getListeners();
            for (auto const& listener : listeners) {
                multi->addListener(listener->create(config));
            }

            for ( auto const& reporterSpec : config->getProcessedReporterSpecs() ) {
                multi->addReporter( createReporter(
                    reporterSpec.name,
                    ReporterConfig( config,
                                    makeStream( reporterSpec.outputFilename ),
                                    reporterSpec.colourMode,
                                    reporterSpec.customOptions ) ) );
            }

            return multi;
        }

        class TestGroup {
        public:
            explicit TestGroup(IEventListenerPtr&& reporter, Config const* config):
                m_reporter(reporter.get()),
                m_config{config},
                m_context{config, CATCH_MOVE(reporter)} {

                assert( m_config->testSpec().getInvalidSpecs().empty() &&
                        "Invalid test specs should be handled before running tests" );

                auto const& allTestCases = getAllTestCasesSorted(*m_config);
                auto const& testSpec = m_config->testSpec();
                if ( !testSpec.hasFilters() ) {
                    for ( auto const& test : allTestCases ) {
                        if ( !test.getTestCaseInfo().isHidden() ) {
                            m_tests.emplace( &test );
                        }
                    }
                } else {
                    m_matches =
                        testSpec.matchesByFilter( allTestCases, *m_config );
                    for ( auto const& match : m_matches ) {
                        m_tests.insert( match.tests.begin(),
                                        match.tests.end() );
                    }
                }

                m_tests = createShard(m_tests, m_config->shardCount(), m_config->shardIndex());
            }

            Totals execute() {
                Totals totals;
                for (auto const& testCase : m_tests) {
                    if (!m_context.aborting())
                        totals += m_context.runTest(*testCase);
                    else
                        m_reporter->skipTest(testCase->getTestCaseInfo());
                }

                for (auto const& match : m_matches) {
                    if (match.tests.empty()) {
                        m_unmatchedTestSpecs = true;
                        m_reporter->noMatchingTestCases( match.name );
                    }
                }

                return totals;
            }

            bool hadUnmatchedTestSpecs() const {
                return m_unmatchedTestSpecs;
            }


        private:
            IEventListener* m_reporter;
            Config const* m_config;
            RunContext m_context;
            std::set<TestCaseHandle const*> m_tests;
            TestSpec::Matches m_matches;
            bool m_unmatchedTestSpecs = false;
        };

        void applyFilenamesAsTags() {
            for (auto const& testInfo : getRegistryHub().getTestCaseRegistry().getAllInfos()) {
                testInfo->addFilenameTag();
            }
        }

    } // anon namespace

    Session::Session() {
        static bool alreadyInstantiated = false;
        if( alreadyInstantiated ) {
            CATCH_TRY { CATCH_INTERNAL_ERROR( "Only one instance of Catch::Session can ever be used" ); }
            CATCH_CATCH_ALL { getMutableRegistryHub().registerStartupException(); }
        }

        // There cannot be exceptions at startup in no-exception mode.
#if !defined(CATCH_CONFIG_DISABLE_EXCEPTIONS)
        const auto& exceptions = getRegistryHub().getStartupExceptionRegistry().getExceptions();
        if ( !exceptions.empty() ) {
            config();
            getCurrentMutableContext().setConfig(m_config.get());

            m_startupExceptions = true;
            auto errStream = makeStream( "%stderr" );
            auto colourImpl = makeColourImpl(
                ColourMode::PlatformDefault, errStream.get() );
            auto guard = colourImpl->guardColour( Colour::Red );
            errStream->stream() << "Errors occurred during startup!" << '\n';
            // iterate over all exceptions and notify user
            for ( const auto& ex_ptr : exceptions ) {
                try {
                    std::rethrow_exception(ex_ptr);
                } catch ( std::exception const& ex ) {
                    errStream->stream() << TextFlow::Column( ex.what() ).indent(2) << '\n';
                }
            }
        }
#endif

        alreadyInstantiated = true;
        m_cli = makeCommandLineParser( m_configData );
    }
    Session::~Session() {
        Catch::cleanUp();
    }

    void Session::showHelp() const {
        Catch::cout()
                << "\nCatch2 v" << libraryVersion() << '\n'
                << m_cli << '\n'
                << "For more detailed usage please see the project docs\n\n" << std::flush;
    }
    void Session::libIdentify() {
        Catch::cout()
                << std::left << std::setw(16) << "description: " << "A Catch2 test executable\n"
                << std::left << std::setw(16) << "category: " << "testframework\n"
                << std::left << std::setw(16) << "framework: " << "Catch2\n"
                << std::left << std::setw(16) << "version: " << libraryVersion() << '\n' << std::flush;
    }

    int Session::applyCommandLine( int argc, char const * const * argv ) {
        if ( m_startupExceptions ) { return UnspecifiedErrorExitCode; }

        auto result = m_cli.parse( Clara::Args( argc, argv ) );

        if( !result ) {
            config();
            getCurrentMutableContext().setConfig(m_config.get());
            auto errStream = makeStream( "%stderr" );
            auto colour = makeColourImpl( ColourMode::PlatformDefault, errStream.get() );

            errStream->stream()
                << colour->guardColour( Colour::Red )
                << "\nError(s) in input:\n"
                << TextFlow::Column( result.errorMessage() ).indent( 2 )
                << "\n\n";
            errStream->stream() << "Run with -? for usage\n\n" << std::flush;
            return UnspecifiedErrorExitCode;
        }

        if( m_configData.showHelp )
            showHelp();
        if( m_configData.libIdentify )
            libIdentify();

        m_config.reset();
        return 0;
    }

#if defined(CATCH_CONFIG_WCHAR) && defined(_WIN32) && defined(UNICODE)
    int Session::applyCommandLine( int argc, wchar_t const * const * argv ) {

        char **utf8Argv = new char *[ argc ];

        for ( int i = 0; i < argc; ++i ) {
            int bufSize = WideCharToMultiByte( CP_UTF8, 0, argv[i], -1, nullptr, 0, nullptr, nullptr );

            utf8Argv[ i ] = new char[ bufSize ];

            WideCharToMultiByte( CP_UTF8, 0, argv[i], -1, utf8Argv[i], bufSize, nullptr, nullptr );
        }

        int returnCode = applyCommandLine( argc, utf8Argv );

        for ( int i = 0; i < argc; ++i )
            delete [] utf8Argv[ i ];

        delete [] utf8Argv;

        return returnCode;
    }
#endif

    void Session::useConfigData( ConfigData const& configData ) {
        m_configData = configData;
        m_config.reset();
    }

    int Session::run() {
        if( ( m_configData.waitForKeypress & WaitForKeypress::BeforeStart ) != 0 ) {
            Catch::cout() << "...waiting for enter/ return before starting\n" << std::flush;
            static_cast<void>(std::getchar());
        }
        int exitCode = runInternal();
        if( ( m_configData.waitForKeypress & WaitForKeypress::BeforeExit ) != 0 ) {
            Catch::cout() << "...waiting for enter/ return before exiting, with code: " << exitCode << '\n' << std::flush;
            static_cast<void>(std::getchar());
        }
        return exitCode;
    }

    Clara::Parser const& Session::cli() const {
        return m_cli;
    }
    void Session::cli( Clara::Parser const& newParser ) {
        m_cli = newParser;
    }
    ConfigData& Session::configData() {
        return m_configData;
    }
    Config& Session::config() {
        if( !m_config )
            m_config = Detail::make_unique<Config>( m_configData );
        return *m_config;
    }

    int Session::runInternal() {
        if ( m_startupExceptions ) { return UnspecifiedErrorExitCode; }

        if (m_configData.showHelp || m_configData.libIdentify) {
            return 0;
        }

        if ( m_configData.shardIndex >= m_configData.shardCount ) {
            Catch::cerr() << "The shard count (" << m_configData.shardCount
                          << ") must be greater than the shard index ("
                          << m_configData.shardIndex << ")\n"
                          << std::flush;
            return UnspecifiedErrorExitCode;
        }

        CATCH_TRY {
            config(); // Force config to be constructed

            seedRng( *m_config );

            if (m_configData.filenamesAsTags) {
                applyFilenamesAsTags();
            }

            // Set up global config instance before we start calling into other functions
            getCurrentMutableContext().setConfig(m_config.get());

            // Create reporter(s) so we can route listings through them
            auto reporter = prepareReporters(m_config.get());

            auto const& invalidSpecs = m_config->testSpec().getInvalidSpecs();
            if ( !invalidSpecs.empty() ) {
                for ( auto const& spec : invalidSpecs ) {
                    reporter->reportInvalidTestSpec( spec );
                }
                return InvalidTestSpecExitCode;
            }


            // Handle list request
            if (list(*reporter, *m_config)) {
                return 0;
            }

            TestGroup tests { CATCH_MOVE(reporter), m_config.get() };
            auto const totals = tests.execute();

            if ( tests.hadUnmatchedTestSpecs()
                && m_config->warnAboutUnmatchedTestSpecs() ) {
                // UnmatchedTestSpecExitCode
                return UnmatchedTestSpecExitCode;
            }

            if ( totals.testCases.total() == 0
                && !m_config->zeroTestsCountAsSuccess() ) {
                return NoTestsRunExitCode;
            }

            if ( totals.testCases.total() > 0 &&
                 totals.testCases.total() == totals.testCases.skipped
                && !m_config->zeroTestsCountAsSuccess() ) {
                return AllTestsSkippedExitCode;
            }

            if ( totals.assertions.failed ) { return TestFailureExitCode; }
            return 0;

        }
#if !defined(CATCH_CONFIG_DISABLE_EXCEPTIONS)
        catch( std::exception& ex ) {
            Catch::cerr() << ex.what() << '\n' << std::flush;
            return UnspecifiedErrorExitCode;
        }
#endif
    }

} // end namespace Catch
