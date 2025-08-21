
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/catch_test_macros.hpp>
#include <catch2/internal/catch_reporter_spec_parser.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include <catch2/interfaces/catch_interfaces_config.hpp>

TEST_CASE("Reporter spec splitting", "[reporter-spec][cli][approvals]") {
	using Catch::Detail::splitReporterSpec;
	using Catch::Matchers::Equals;
	using namespace std::string_literals;

    SECTION("Various edge cases") {
        REQUIRE_THAT( splitReporterSpec( "" ),
                      Equals( std::vector<std::string>{ ""s } ) );
        REQUIRE_THAT( splitReporterSpec( "::" ),
                      Equals( std::vector<std::string>{ "", "" } ) );
        REQUIRE_THAT( splitReporterSpec( "::rep" ),
                      Equals( std::vector<std::string>{ "", "rep" } ) );
        REQUIRE_THAT( splitReporterSpec( "rep::" ),
                      Equals( std::vector<std::string>{ "rep", "" } ) );

    }

    SECTION("Validish specs") {
        REQUIRE_THAT( splitReporterSpec( "newReporter" ),
                      Equals( std::vector<std::string>{ "newReporter"s } ) );
        REQUIRE_THAT(
            splitReporterSpec( "foo-reporter::key1=value1::key2=value with "
                               "space::key with space=some-value" ),
            Equals(
                std::vector<std::string>{ "foo-reporter"s,
                                          "key1=value1"s,
                                          "key2=value with space"s,
                                          "key with space=some-value"s } ) );
        REQUIRE_THAT(
            splitReporterSpec( "spaced reporter name::key:key=value:value" ),
            Equals( std::vector<std::string>{ "spaced reporter name"s,
                                              "key:key=value:value"s } ) );
    }
}

TEST_CASE( "Parsing colour mode", "[cli][colour][approvals]" ) {
    using Catch::Detail::stringToColourMode;
    using Catch::ColourMode;
    SECTION("Valid strings") {
        REQUIRE( stringToColourMode( "none" ) == ColourMode::None );
        REQUIRE( stringToColourMode( "ansi" ) == ColourMode::ANSI );
        REQUIRE( stringToColourMode( "win32" ) == ColourMode::Win32 );
        REQUIRE( stringToColourMode( "default" ) ==
                 ColourMode::PlatformDefault );
    }
    SECTION("Wrong strings") {
        REQUIRE_FALSE( stringToColourMode( "NONE" ) );
        REQUIRE_FALSE( stringToColourMode( "-" ) );
        REQUIRE_FALSE( stringToColourMode( "asdbjsdb kasbd" ) );
    }
}


TEST_CASE("Parsing reporter specs", "[cli][reporter-spec][approvals]") {
    using Catch::parseReporterSpec;
    using Catch::ReporterSpec;
    using namespace std::string_literals;

    SECTION( "Correct specs" ) {
        REQUIRE( parseReporterSpec( "someReporter" ) ==
                 ReporterSpec( "someReporter"s, {}, {}, {} ) );
        REQUIRE( parseReporterSpec( "otherReporter::Xk=v::out=c:\\blah" ) ==
                 ReporterSpec(
                     "otherReporter"s, "c:\\blah"s, {}, { { "Xk"s, "v"s } } ) );
        REQUIRE( parseReporterSpec( "diffReporter::Xk1=v1::Xk2==v2" ) ==
                 ReporterSpec( "diffReporter",
                               {},
                               {},
                               { { "Xk1"s, "v1"s }, { "Xk2"s, "=v2"s } } ) );
        REQUIRE( parseReporterSpec(
                     "Foo:bar:reporter::colour-mode=ansi::Xk 1=v 1::Xk2=v:3" ) ==
                 ReporterSpec( "Foo:bar:reporter",
                               {},
                               Catch::ColourMode::ANSI,
                               { { "Xk 1"s, "v 1"s }, { "Xk2"s, "v:3"s } } ) );
    }

    SECTION( "Bad specs" ) {
        REQUIRE_FALSE( parseReporterSpec( "::" ) );
        // Unknown Catch2 arg (should be "out")
        REQUIRE_FALSE( parseReporterSpec( "reporter::output=filename" ) );
        // Wrong colour spec
        REQUIRE_FALSE( parseReporterSpec( "reporter::colour-mode=custom" ) );
        // Duplicated colour spec
        REQUIRE_FALSE( parseReporterSpec( "reporter::colour-mode=ansi::colour-mode=ansi" ) );
        // Duplicated out arg
        REQUIRE_FALSE( parseReporterSpec( "reporter::out=f.txt::out=z.txt" ) );
        // Duplicated custom arg
        REQUIRE_FALSE( parseReporterSpec( "reporter::Xa=foo::Xa=bar" ) );
        // Empty key
        REQUIRE_FALSE( parseReporterSpec( "reporter::X=foo" ) );
        REQUIRE_FALSE( parseReporterSpec( "reporter::=foo" ) );
        // Empty value
        REQUIRE_FALSE( parseReporterSpec( "reporter::Xa=" ) );
        // non-key value later field
        REQUIRE_FALSE( parseReporterSpec( "reporter::Xab" ) );
    }
}
