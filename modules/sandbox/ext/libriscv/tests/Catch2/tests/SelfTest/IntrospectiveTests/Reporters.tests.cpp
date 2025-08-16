
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/catch_test_macros.hpp>

#include <catch2/catch_test_case_info.hpp>
#include <catch2/catch_config.hpp>
#include <catch2/interfaces/catch_interfaces_reporter.hpp>
#include <catch2/interfaces/catch_interfaces_reporter_factory.hpp>
#include <catch2/internal/catch_console_colour.hpp>
#include <catch2/internal/catch_enforce.hpp>
#include <catch2/internal/catch_list.hpp>
#include <catch2/internal/catch_reporter_registry.hpp>
#include <catch2/internal/catch_istream.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <catch2/reporters/catch_reporter_helpers.hpp>
#include <catch2/reporters/catch_reporter_event_listener.hpp>
#include <catch2/reporters/catch_reporter_streaming_base.hpp>
#include <catch2/reporters/catch_reporter_multi.hpp>
#include <catch2/internal/catch_move_and_forward.hpp>

#include <sstream>

namespace {
    class StringIStream : public Catch::IStream {
    public:
        std::ostream& stream() override { return sstr; }
        std::string str() const { return sstr.str(); }
    private:
        std::stringstream sstr;
    };

    //! config must outlive the function
    Catch::ReporterConfig makeDummyRepConfig( Catch::Config const& config ) {
        return Catch::ReporterConfig{
            &config,
            Catch::Detail::make_unique<StringIStream>(),
            Catch::ColourMode::None,
            {} };
    }
}

TEST_CASE( "The default listing implementation write to provided stream",
           "[reporters][reporter-helpers]" ) {
    using Catch::Matchers::ContainsSubstring;
    using namespace std::string_literals;

    StringIStream sstream;
    SECTION( "Listing tags" ) {
        std::vector<Catch::TagInfo> tags(1);
        tags[0].add("fakeTag"_catch_sr);
        Catch::defaultListTags(sstream.stream(), tags, false);

        auto listingString = sstream.str();
        REQUIRE_THAT(listingString, ContainsSubstring("[fakeTag]"s));
    }
    SECTION( "Listing reporters" ) {
        std::vector<Catch::ReporterDescription> reporters(
            { { "fake reporter", "fake description" } } );
        Catch::defaultListReporters(sstream.stream(), reporters, Catch::Verbosity::Normal);

        auto listingString = sstream.str();
        REQUIRE_THAT( listingString,
                      ContainsSubstring( "fake reporter"s ) &&
                          ContainsSubstring( "fake description"s ) );
    }
    SECTION( "Listing tests" ) {
        Catch::TestCaseInfo fakeInfo{
            ""s,
            { "fake test name"_catch_sr, "[fakeTestTag]"_catch_sr },
            { "fake-file.cpp", 123456789 } };
        std::vector<Catch::TestCaseHandle> tests({ {&fakeInfo, nullptr} });
        auto colour = Catch::makeColourImpl( Catch::ColourMode::None, &sstream);
        Catch::defaultListTests(sstream.stream(), colour.get(), tests, false, Catch::Verbosity::Normal);

        auto listingString = sstream.str();
        REQUIRE_THAT( listingString,
                      ContainsSubstring( "fake test name"s ) &&
                          ContainsSubstring( "fakeTestTag"s ) );
    }
    SECTION( "Listing listeners" ) {
        std::vector<Catch::ListenerDescription> listeners(
            { { "fakeListener"_catch_sr, "fake description" } } );

        Catch::defaultListListeners( sstream.stream(), listeners );
        auto listingString = sstream.str();
        REQUIRE_THAT( listingString,
                      ContainsSubstring( "fakeListener"s ) &&
                          ContainsSubstring( "fake description"s ) );
    }
}

TEST_CASE( "Reporter's write listings to provided stream", "[reporters]" ) {
    using Catch::Matchers::ContainsSubstring;
    using namespace std::string_literals;

    auto const& factories = Catch::getRegistryHub().getReporterRegistry().getFactories();
    // If there are no reporters, the test would pass falsely
    // while there is something obviously broken
    REQUIRE_FALSE(factories.empty());

    for (auto const& factory : factories) {
        INFO("Tested reporter: " << factory.first);
        auto sstream = Catch::Detail::make_unique<StringIStream>();
        auto& sstreamRef = *sstream;

        Catch::ConfigData cfg_data;
        cfg_data.rngSeed = 1234;
        Catch::Config config( cfg_data );
        auto reporter = factory.second->create( Catch::ReporterConfig{
            &config, CATCH_MOVE( sstream ), Catch::ColourMode::None, {} } );

        DYNAMIC_SECTION( factory.first << " reporter lists tags" ) {
            std::vector<Catch::TagInfo> tags(1);
            tags[0].add("fakeTag"_catch_sr);
            reporter->listTags(tags);

            auto listingString = sstreamRef.str();
            REQUIRE_THAT(listingString, ContainsSubstring("fakeTag"s));
        }

        DYNAMIC_SECTION( factory.first << " reporter lists reporters" ) {
            std::vector<Catch::ReporterDescription> reporters(
                { { "fake reporter", "fake description" } } );
            reporter->listReporters(reporters);

            auto listingString = sstreamRef.str();
            REQUIRE_THAT(listingString, ContainsSubstring("fake reporter"s));
        }

        DYNAMIC_SECTION( factory.first << " reporter lists tests" ) {
            Catch::TestCaseInfo fakeInfo{
                ""s,
                { "fake test name"_catch_sr, "[fakeTestTag]"_catch_sr },
                { "fake-file.cpp", 123456789 } };
            std::vector<Catch::TestCaseHandle> tests({ {&fakeInfo, nullptr} });
            reporter->listTests(tests);

            auto listingString = sstreamRef.str();
            REQUIRE_THAT( listingString,
                          ContainsSubstring( "fake test name"s ) &&
                              ContainsSubstring( "fakeTestTag"s ) );
        }
    }
}


TEST_CASE("Reproducer for #2309 - a very long description past 80 chars (default console width) with a late colon : blablabla", "[console-reporter]") {
    SUCCEED();
}

namespace {
    // A listener that writes provided string into destination,
    // to record order of testRunStarting invocation.
    class MockListener : public Catch::EventListenerBase {
        std::string m_witness;
        std::vector<std::string>& m_recorder;
    public:
        MockListener( std::string witness,
                      std::vector<std::string>& recorder,
                      Catch::IConfig const* config ):
            EventListenerBase( config ),
            m_witness( CATCH_MOVE(witness) ),
            m_recorder( recorder )
        {}

        void testRunStarting( Catch::TestRunInfo const& ) override {
            m_recorder.push_back( m_witness );
        }
    };
    // A reporter that writes provided string into destination,
    // to record order of testRunStarting invocation.
    class MockReporter : public Catch::StreamingReporterBase {
        std::string m_witness;
        std::vector<std::string>& m_recorder;
    public:
        MockReporter( std::string witness,
                      std::vector<std::string>& recorder,
                      Catch::ReporterConfig&& config ):
            StreamingReporterBase( CATCH_MOVE(config) ),
            m_witness( CATCH_MOVE(witness) ),
            m_recorder( recorder )
        {}

        void testRunStarting( Catch::TestRunInfo const& ) override {
            m_recorder.push_back( m_witness );
        }
    };
} // namespace

TEST_CASE("Multireporter calls reporters and listeners in correct order",
          "[reporters][multi-reporter]") {
    Catch::Config config( Catch::ConfigData{} );

    // We add reporters before listeners, to check that internally they
    // get sorted properly, and listeners are called first anyway.
    Catch::MultiReporter multiReporter( &config );
    std::vector<std::string> records;
    multiReporter.addReporter( Catch::Detail::make_unique<MockReporter>(
        "Goodbye", records, makeDummyRepConfig(config) ) );
    multiReporter.addListener(
        Catch::Detail::make_unique<MockListener>( "Hello", records, &config ) );
    multiReporter.addListener(
        Catch::Detail::make_unique<MockListener>( "world", records, &config ) );
    multiReporter.addReporter( Catch::Detail::make_unique<MockReporter>(
        "world", records, makeDummyRepConfig(config) ) );
    multiReporter.testRunStarting( { "" } );

    std::vector<std::string> expected( { "Hello", "world", "Goodbye", "world" } );
    REQUIRE( records == expected );
}

namespace {
    // A listener that sets it preferences to test that multireporter,
    // properly sets up its own preferences
    class PreferenceListener : public Catch::EventListenerBase {
    public:
        PreferenceListener( bool redirectStdout,
                            bool reportAllAssertions,
                            Catch::IConfig const* config ):
            EventListenerBase( config ) {
            m_preferences.shouldRedirectStdOut = redirectStdout;
            m_preferences.shouldReportAllAssertions = reportAllAssertions;
        }
    };
    // A reporter that sets it preferences to test that multireporter,
    // properly sets up its own preferences
    class PreferenceReporter : public Catch::StreamingReporterBase {
    public:
        PreferenceReporter( bool redirectStdout,
                            bool reportAllAssertions,
                            Catch::ReporterConfig&& config ):
            StreamingReporterBase( CATCH_MOVE(config) ) {
            m_preferences.shouldRedirectStdOut = redirectStdout;
            m_preferences.shouldReportAllAssertions = reportAllAssertions;
        }
    };
} // namespace

TEST_CASE("Multireporter updates ReporterPreferences properly",
          "[reporters][multi-reporter]") {

    Catch::Config config( Catch::ConfigData{} );
    Catch::MultiReporter multiReporter( &config );

    // Post init defaults
    REQUIRE( multiReporter.getPreferences().shouldRedirectStdOut == false );
    REQUIRE( multiReporter.getPreferences().shouldReportAllAssertions == false );

    SECTION( "Adding listeners" ) {
        multiReporter.addListener(
            Catch::Detail::make_unique<PreferenceListener>(
                true, false, &config ) );
        REQUIRE( multiReporter.getPreferences().shouldRedirectStdOut == true );
        REQUIRE( multiReporter.getPreferences().shouldReportAllAssertions == false );

        multiReporter.addListener(
            Catch::Detail::make_unique<PreferenceListener>(
                false, true, &config ) );
        REQUIRE( multiReporter.getPreferences().shouldRedirectStdOut == true );
        REQUIRE( multiReporter.getPreferences().shouldReportAllAssertions == true);

        multiReporter.addListener(
            Catch::Detail::make_unique<PreferenceListener>(
                false, false, &config ) );
        REQUIRE( multiReporter.getPreferences().shouldRedirectStdOut == true );
        REQUIRE( multiReporter.getPreferences().shouldReportAllAssertions == true );
    }
    SECTION( "Adding reporters" ) {
        multiReporter.addReporter(
            Catch::Detail::make_unique<PreferenceReporter>(
                true, false, makeDummyRepConfig(config) ) );
        REQUIRE( multiReporter.getPreferences().shouldRedirectStdOut == true );
        REQUIRE( multiReporter.getPreferences().shouldReportAllAssertions == false );

        multiReporter.addReporter(
            Catch::Detail::make_unique<PreferenceReporter>(
                false, true, makeDummyRepConfig( config ) ) );
        REQUIRE( multiReporter.getPreferences().shouldRedirectStdOut == true );
        REQUIRE( multiReporter.getPreferences().shouldReportAllAssertions == true );

        multiReporter.addReporter(
            Catch::Detail::make_unique<PreferenceReporter>(
                false, false, makeDummyRepConfig( config ) ) );
        REQUIRE( multiReporter.getPreferences().shouldRedirectStdOut == true );
        REQUIRE( multiReporter.getPreferences().shouldReportAllAssertions == true );
    }
}

namespace {
    class TestReporterFactory : public Catch::IReporterFactory {
        Catch::IEventListenerPtr create( Catch::ReporterConfig&& ) const override {
            CATCH_INTERNAL_ERROR(
                "This factory should never create a reporter" );
        }
        std::string getDescription() const override {
            return "Fake test factory";
        }
    };
}

TEST_CASE("Registering reporter with '::' in name fails",
          "[reporters][registration]") {
    Catch::ReporterRegistry registry;

    REQUIRE_THROWS_WITH( registry.registerReporter(
        "with::doublecolons",
        Catch::Detail::make_unique<TestReporterFactory>() ),
        "'::' is not allowed in reporter name: 'with::doublecolons'" );
}

TEST_CASE("Registering multiple reporters with the same name fails",
          "[reporters][registration][approvals]") {
    Catch::ReporterRegistry registry;

    registry.registerReporter(
        "some-reporter-name",
        Catch::Detail::make_unique<TestReporterFactory>() );

    REQUIRE_THROWS_WITH(
        registry.registerReporter(
            "some-reporter-name",
            Catch::Detail::make_unique<TestReporterFactory>() ),
        "reporter using 'some-reporter-name' as name was already registered" );
}
