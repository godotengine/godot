
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/catch_config.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <catch2/internal/catch_test_spec_parser.hpp>
#include <catch2/catch_user_config.hpp>
#include <catch2/catch_test_case_info.hpp>
#include <catch2/internal/catch_commandline.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/internal/catch_compiler_capabilities.hpp>

#include <helpers/parse_test_spec.hpp>

namespace {
    auto fakeTestCase(const char* name, const char* desc = "") { return Catch::makeTestCaseInfo("", { name, desc }, CATCH_INTERNAL_LINEINFO); }
}

TEST_CASE( "Parse test names and tags", "[command-line][test-spec][approvals]" ) {
    using Catch::parseTestSpec;
    using Catch::TestSpec;

    auto tcA = fakeTestCase( "a" );
    auto tcB = fakeTestCase( "b", "[one][x]" );
    auto tcC = fakeTestCase( "longer name with spaces", "[two][three][.][x]" );
    auto tcD = fakeTestCase( "zlonger name with spacesz" );

    SECTION( "Empty test spec should have no filters" ) {
        TestSpec spec;
        CHECK( spec.hasFilters() == false );
        CHECK( spec.matches( *tcA ) == false );
        CHECK( spec.matches( *tcB ) == false );
    }

    SECTION( "Test spec from empty string should have no filters" ) {
        TestSpec spec = parseTestSpec( "" );
        CHECK( spec.hasFilters() == false );
        CHECK( spec.matches( *tcA ) == false );
        CHECK( spec.matches( *tcB ) == false );
    }

    SECTION( "Test spec from just a comma should have no filters" ) {
        TestSpec spec = parseTestSpec( "," );
        CHECK( spec.hasFilters() == false );
        CHECK( spec.matches( *tcA ) == false );
        CHECK( spec.matches( *tcB ) == false );
    }

    SECTION( "Test spec from name should have one filter" ) {
        TestSpec spec = parseTestSpec( "b" );
        CHECK( spec.hasFilters() == true );
        CHECK( spec.matches( *tcA ) == false );
        CHECK( spec.matches( *tcB ) == true );
    }

    SECTION( "Test spec from quoted name should have one filter" ) {
        TestSpec spec = parseTestSpec( "\"b\"" );
        CHECK( spec.hasFilters() == true );
        CHECK( spec.matches( *tcA ) == false );
        CHECK( spec.matches( *tcB ) == true );
    }

    SECTION( "Test spec from name should have one filter" ) {
        TestSpec spec = parseTestSpec( "b" );
        CHECK( spec.hasFilters() == true );
        CHECK( spec.matches( *tcA ) == false );
        CHECK( spec.matches( *tcB ) == true );
        CHECK( spec.matches( *tcC ) == false );
    }

    SECTION( "Wildcard at the start" ) {
        TestSpec spec = parseTestSpec( "*spaces" );
        CHECK( spec.hasFilters() == true );
        CHECK( spec.matches( *tcA ) == false );
        CHECK( spec.matches( *tcB ) == false );
        CHECK( spec.matches( *tcC ) == true );
        CHECK( spec.matches( *tcD ) == false );
        CHECK( parseTestSpec( "*a" ).matches( *tcA ) == true );
    }
    SECTION( "Wildcard at the end" ) {
        TestSpec spec = parseTestSpec( "long*" );
        CHECK( spec.hasFilters() == true );
        CHECK( spec.matches( *tcA ) == false );
        CHECK( spec.matches( *tcB ) == false );
        CHECK( spec.matches( *tcC ) == true );
        CHECK( spec.matches( *tcD ) == false );
        CHECK( parseTestSpec( "a*" ).matches( *tcA ) == true );
    }
    SECTION( "Wildcard at both ends" ) {
        TestSpec spec = parseTestSpec( "*name*" );
        CHECK( spec.hasFilters() == true );
        CHECK( spec.matches( *tcA ) == false );
        CHECK( spec.matches( *tcB ) == false );
        CHECK( spec.matches( *tcC ) == true );
        CHECK( spec.matches( *tcD ) == true );
        CHECK( parseTestSpec( "*a*" ).matches( *tcA ) == true );
    }
    SECTION( "Redundant wildcard at the start" ) {
        TestSpec spec = parseTestSpec( "*a" );
        CHECK( spec.hasFilters() == true );
        CHECK( spec.matches( *tcA ) == true );
        CHECK( spec.matches( *tcB ) == false );
    }
    SECTION( "Redundant wildcard at the end" ) {
        TestSpec spec = parseTestSpec( "a*" );
        CHECK( spec.hasFilters() == true );
        CHECK( spec.matches( *tcA ) == true );
        CHECK( spec.matches( *tcB ) == false );
    }
    SECTION( "Redundant wildcard at both ends" ) {
        TestSpec spec = parseTestSpec( "*a*" );
        CHECK( spec.hasFilters() == true );
        CHECK( spec.matches( *tcA ) == true );
        CHECK( spec.matches( *tcB ) == false );
    }
    SECTION( "Wildcard at both ends, redundant at start" ) {
        TestSpec spec = parseTestSpec( "*longer*" );
        CHECK( spec.hasFilters() == true );
        CHECK( spec.matches( *tcA ) == false );
        CHECK( spec.matches( *tcB ) == false );
        CHECK( spec.matches( *tcC ) == true );
        CHECK( spec.matches( *tcD ) == true );
    }
    SECTION( "Just wildcard" ) {
        TestSpec spec = parseTestSpec( "*" );
        CHECK( spec.hasFilters() == true );
        CHECK( spec.matches( *tcA ) == true );
        CHECK( spec.matches( *tcB ) == true );
        CHECK( spec.matches( *tcC ) == true );
        CHECK( spec.matches( *tcD ) == true );
    }

    SECTION( "Single tag" ) {
        TestSpec spec = parseTestSpec( "[one]" );
        CHECK( spec.hasFilters() == true );
        CHECK( spec.matches( *tcA ) == false );
        CHECK( spec.matches( *tcB ) == true );
        CHECK( spec.matches( *tcC ) == false );
    }
    SECTION( "Single tag, two matches" ) {
        TestSpec spec = parseTestSpec( "[x]" );
        CHECK( spec.hasFilters() == true );
        CHECK( spec.matches( *tcA ) == false );
        CHECK( spec.matches( *tcB ) == true );
        CHECK( spec.matches( *tcC ) == true );
    }
    SECTION( "Two tags" ) {
        TestSpec spec = parseTestSpec( "[two][x]" );
        CHECK( spec.hasFilters() == true );
        CHECK( spec.matches( *tcA ) == false );
        CHECK( spec.matches( *tcB ) == false );
        CHECK( spec.matches( *tcC ) == true );
    }
    SECTION( "Two tags, spare separated" ) {
        TestSpec spec = parseTestSpec( "[two] [x]" );
        CHECK( spec.hasFilters() == true );
        CHECK( spec.matches( *tcA ) == false );
        CHECK( spec.matches( *tcB ) == false );
        CHECK( spec.matches( *tcC ) == true );
    }
    SECTION( "Wildcarded name and tag" ) {
        TestSpec spec = parseTestSpec( "*name*[x]" );
        CHECK( spec.hasFilters() == true );
        CHECK( spec.matches( *tcA ) == false );
        CHECK( spec.matches( *tcB ) == false );
        CHECK( spec.matches( *tcC ) == true );
        CHECK( spec.matches( *tcD ) == false );
    }
    SECTION( "Single tag exclusion" ) {
        TestSpec spec = parseTestSpec( "~[one]" );
        CHECK( spec.hasFilters() == true );
        CHECK( spec.matches( *tcA ) == true );
        CHECK( spec.matches( *tcB ) == false );
        CHECK( spec.matches( *tcC ) == false );
    }
    SECTION( "One tag exclusion and one tag inclusion" ) {
        TestSpec spec = parseTestSpec( "~[two][x]" );
        CHECK( spec.hasFilters() == true );
        CHECK( spec.matches( *tcA ) == false );
        CHECK( spec.matches( *tcB ) == true );
        CHECK( spec.matches( *tcC ) == false );
    }
    SECTION( "One tag exclusion and one wldcarded name inclusion" ) {
        TestSpec spec = parseTestSpec( "~[two]*name*" );
        CHECK( spec.hasFilters() == true );
        CHECK( spec.matches( *tcA ) == false );
        CHECK( spec.matches( *tcB ) == false );
        CHECK( spec.matches( *tcC ) == false );
        CHECK( spec.matches( *tcD ) == true );
    }
    SECTION( "One tag exclusion, using exclude:, and one wldcarded name inclusion" ) {
        TestSpec spec = parseTestSpec( "exclude:[two]*name*" );
        CHECK( spec.hasFilters() == true );
        CHECK( spec.matches( *tcA ) == false );
        CHECK( spec.matches( *tcB ) == false );
        CHECK( spec.matches( *tcC ) == false );
        CHECK( spec.matches( *tcD ) == true );
    }
    SECTION( "name exclusion" ) {
        TestSpec spec = parseTestSpec( "~b" );
        CHECK( spec.hasFilters() == true );
        CHECK( spec.matches( *tcA ) == true );
        CHECK( spec.matches( *tcB ) == false );
        CHECK( spec.matches( *tcC ) == false );
        CHECK( spec.matches( *tcD ) == true );
    }
    SECTION( "wildcarded name exclusion" ) {
        TestSpec spec = parseTestSpec( "~*name*" );
        CHECK( spec.hasFilters() == true );
        CHECK( spec.matches( *tcA ) == true );
        CHECK( spec.matches( *tcB ) == true );
        CHECK( spec.matches( *tcC ) == false );
        CHECK( spec.matches( *tcD ) == false );
    }
    SECTION( "wildcarded name exclusion with tag inclusion" ) {
        TestSpec spec = parseTestSpec( "~*name*,[three]" );
        CHECK( spec.hasFilters() == true );
        CHECK( spec.matches( *tcA ) == true );
        CHECK( spec.matches( *tcB ) == true );
        CHECK( spec.matches( *tcC ) == true );
        CHECK( spec.matches( *tcD ) == false );
    }
    SECTION( "wildcarded name exclusion, using exclude:, with tag inclusion" ) {
        TestSpec spec = parseTestSpec( "exclude:*name*,[three]" );
        CHECK( spec.hasFilters() == true );
        CHECK( spec.matches( *tcA ) == true );
        CHECK( spec.matches( *tcB ) == true );
        CHECK( spec.matches( *tcC ) == true );
        CHECK( spec.matches( *tcD ) == false );
    }
    SECTION( "two wildcarded names" ) {
        TestSpec spec = parseTestSpec( R"("longer*""*spaces")" );
        CHECK( spec.hasFilters() == true );
        CHECK( spec.matches( *tcA ) == false );
        CHECK( spec.matches( *tcB ) == false );
        CHECK( spec.matches( *tcC ) == true );
        CHECK( spec.matches( *tcD ) == false );
    }
    SECTION( "empty tag" ) {
        TestSpec spec = parseTestSpec( "[]" );
        CHECK( spec.hasFilters() == false );
        CHECK( spec.matches( *tcA ) == false );
        CHECK( spec.matches( *tcB ) == false );
        CHECK( spec.matches( *tcC ) == false );
        CHECK( spec.matches( *tcD ) == false );
    }
    SECTION( "empty quoted name" ) {
        TestSpec spec = parseTestSpec( "\"\"" );
        CHECK( spec.hasFilters() == false );
        CHECK( spec.matches( *tcA ) == false );
        CHECK( spec.matches( *tcB ) == false );
        CHECK( spec.matches( *tcC ) == false );
        CHECK( spec.matches( *tcD ) == false );
    }
    SECTION( "quoted string followed by tag exclusion" ) {
        TestSpec spec = parseTestSpec( "\"*name*\"~[.]" );
        CHECK( spec.hasFilters() == true );
        CHECK( spec.matches( *tcA ) == false );
        CHECK( spec.matches( *tcB ) == false );
        CHECK( spec.matches( *tcC ) == false );
        CHECK( spec.matches( *tcD ) == true );
    }
    SECTION( "Leading and trailing spaces in test spec" ) {
        TestSpec spec = parseTestSpec( "\"  aardvark \"" );
        CHECK( spec.matches( *fakeTestCase( "  aardvark " ) ) );
        CHECK( spec.matches( *fakeTestCase( "  aardvark" ) ) );
        CHECK( spec.matches( *fakeTestCase( " aardvark " ) ) );
        CHECK( spec.matches( *fakeTestCase( "aardvark " ) ) );
        CHECK( spec.matches( *fakeTestCase( "aardvark" ) ) );

    }
    SECTION( "Leading and trailing spaces in test name" ) {
        TestSpec spec = parseTestSpec( "aardvark" );
        CHECK( spec.matches( *fakeTestCase( "  aardvark " ) ) );
        CHECK( spec.matches( *fakeTestCase( "  aardvark" ) ) );
        CHECK( spec.matches( *fakeTestCase( " aardvark " ) ) );
        CHECK( spec.matches( *fakeTestCase( "aardvark " ) ) );
        CHECK( spec.matches( *fakeTestCase( "aardvark" ) ) );
    }
    SECTION("Shortened hide tags are split apart when parsing") {
        TestSpec spec = parseTestSpec("[.foo]");
        CHECK(spec.matches(*fakeTestCase("hidden and foo", "[.][foo]")));
        CHECK_FALSE(spec.matches(*fakeTestCase("only foo", "[foo]")));
    }
    SECTION("Shortened hide tags also properly handle exclusion") {
        TestSpec spec = parseTestSpec("~[.foo]");
        CHECK_FALSE(spec.matches(*fakeTestCase("hidden and foo", "[.][foo]")));
        CHECK_FALSE(spec.matches(*fakeTestCase("only foo", "[foo]")));
        CHECK_FALSE(spec.matches(*fakeTestCase("only hidden", "[.]")));
        CHECK(spec.matches(*fakeTestCase("neither foo nor hidden", "[bar]")));
    }
}

TEST_CASE("#1905 -- test spec parser properly clears internal state between compound tests", "[command-line][test-spec]") {
    using Catch::parseTestSpec;
    using Catch::TestSpec;
    // We ask for one of 2 different tests and the latter one of them has a , in name that needs escaping
    TestSpec spec = parseTestSpec(R"("spec . char","spec \, char")");

    REQUIRE(spec.matches(*fakeTestCase("spec . char")));
    REQUIRE(spec.matches(*fakeTestCase("spec , char")));
    REQUIRE_FALSE(spec.matches(*fakeTestCase(R"(spec \, char)")));
}

TEST_CASE("#1912 -- test spec parser handles escaping", "[command-line][test-spec]") {
    using Catch::parseTestSpec;
    using Catch::TestSpec;

    SECTION("Various parentheses") {
        TestSpec spec = parseTestSpec(R"(spec {a} char,spec \[a] char)");

        REQUIRE(spec.matches(*fakeTestCase(R"(spec {a} char)")));
        REQUIRE(spec.matches(*fakeTestCase(R"(spec [a] char)")));
        REQUIRE_FALSE(spec.matches(*fakeTestCase("differs but has similar tag", "[a]")));
    }
    SECTION("backslash in test name") {
        TestSpec spec = parseTestSpec(R"(spec \\ char)");

        REQUIRE(spec.matches(*fakeTestCase(R"(spec \ char)")));
    }
}

TEST_CASE("Test spec serialization is round-trippable", "[test-spec][serialization][approvals]") {
    using Catch::parseTestSpec;
    using Catch::TestSpec;

    auto serializedTestSpec = []( std::string const& spec ) {
        Catch::ReusableStringStream sstr;
        sstr << parseTestSpec( spec );
        return sstr.str();
    };

    SECTION("Spaces are normalized") {
        CHECK( serializedTestSpec( "[abc][def]" ) == "[abc] [def]" );
        CHECK( serializedTestSpec( "[def]    [abc]" ) == "[def] [abc]" );
        CHECK( serializedTestSpec( "[def] [abc]" ) == "[def] [abc]" );
    }
    SECTION("Output is order dependent") {
        CHECK( serializedTestSpec( "[abc][def]" ) == "[abc] [def]" );
        CHECK( serializedTestSpec( "[def][abc]" ) == "[def] [abc]" );
    }
    SECTION("Multiple disjunct filters") {
        CHECK( serializedTestSpec( "[abc],[def]" ) == "[abc],[def]" );
        CHECK( serializedTestSpec( "[def],[abc],[idkfa]" ) == "[def],[abc],[idkfa]" );
    }
    SECTION("Test names are enclosed in string") {
        CHECK( serializedTestSpec( "Some test" ) == "\"Some test\"" );
        CHECK( serializedTestSpec( "*Some test" ) == "\"*Some test\"" );
        CHECK( serializedTestSpec( "* Some test" ) == "\"* Some test\"" );
        CHECK( serializedTestSpec( "* Some test *" ) == "\"* Some test *\"" );
    }
    SECTION( "Mixing test names and tags" ) {
        CHECK( serializedTestSpec( "some test[abcd]" ) ==
               "\"some test\" [abcd]" );
        CHECK( serializedTestSpec( "[ab]some test[cd]" ) ==
               "[ab] \"some test\" [cd]" );
    }
}
