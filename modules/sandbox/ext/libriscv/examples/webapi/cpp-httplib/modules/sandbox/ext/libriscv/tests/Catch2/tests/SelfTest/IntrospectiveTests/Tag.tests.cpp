
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/matchers/catch_matchers_string.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include <catch2/internal/catch_tag_alias_registry.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_test_case_info.hpp>

TEST_CASE( "Tag alias can be registered against tag patterns" ) {

    Catch::TagAliasRegistry registry;

    registry.add( "[@zzz]", "[one][two]", Catch::SourceLineInfo( "file", 2 ) );

    SECTION( "The same tag alias can only be registered once" ) {

        try {
            registry.add( "[@zzz]", "[one][two]", Catch::SourceLineInfo( "file", 10 ) );
            FAIL( "expected exception" );
        }
        catch( std::exception& ex ) {
            std::string what = ex.what();
            using namespace Catch::Matchers;
            CHECK_THAT( what, ContainsSubstring( "[@zzz]" ) );
            CHECK_THAT( what, ContainsSubstring( "file" ) );
            CHECK_THAT( what, ContainsSubstring( "2" ) );
            CHECK_THAT( what, ContainsSubstring( "10" ) );
        }
    }

    SECTION( "Tag aliases must be of the form [@name]" ) {
        CHECK_THROWS( registry.add( "[no ampersat]", "", Catch::SourceLineInfo( "file", 3 ) ) );
        CHECK_THROWS( registry.add( "[the @ is not at the start]", "", Catch::SourceLineInfo( "file", 3 ) ) );
        CHECK_THROWS( registry.add( "@no square bracket at start]", "", Catch::SourceLineInfo( "file", 3 ) ) );
        CHECK_THROWS( registry.add( "[@no square bracket at end", "", Catch::SourceLineInfo( "file", 3 ) ) );
    }
}

// Dummy line info for creating dummy test cases below
static constexpr Catch::SourceLineInfo dummySourceLineInfo = CATCH_INTERNAL_LINEINFO;

TEST_CASE("shortened hide tags are split apart", "[tags]") {
    using Catch::StringRef;
    using Catch::Tag;
    using Catch::Matchers::VectorContains;

    Catch::TestCaseInfo testcase("", {"fake test name", "[.magic-tag]"}, dummySourceLineInfo);
    REQUIRE_THAT( testcase.tags, VectorContains( Tag( "magic-tag" ) )
                              && VectorContains( Tag( "."_catch_sr ) ) );
}

TEST_CASE("tags with dots in later positions are not parsed as hidden", "[tags]") {
    using Catch::StringRef;
    using Catch::Matchers::VectorContains;
    Catch::TestCaseInfo testcase("", { "fake test name", "[magic.tag]" }, dummySourceLineInfo);

    REQUIRE(testcase.tags.size() == 1);
    REQUIRE(testcase.tags[0].original == "magic.tag"_catch_sr);
}

TEST_CASE( "empty tags are not allowed", "[tags]" ) {
    REQUIRE_THROWS(
        Catch::TestCaseInfo("", { "test with an empty tag", "[]" }, dummySourceLineInfo)
    );
}

TEST_CASE( "Tags with spaces and non-alphanumerical characters are accepted",
           "[tags]" ) {
    using Catch::Tag;
    using Catch::Matchers::VectorContains;

    Catch::TestCaseInfo testCase(
        "",
        { "fake test name", "[tag with spaces][I said \"good day\" sir!]" },
        dummySourceLineInfo );

    REQUIRE( testCase.tags.size() == 2 );
    REQUIRE_THAT( testCase.tags,
                  VectorContains( Tag( "tag with spaces" ) ) &&
                  VectorContains( Tag( "I said \"good day\" sir!"_catch_sr ) ) );
}

TEST_CASE( "Test case with identical tags keeps just one", "[tags]" ) {
    using Catch::Tag;

    Catch::TestCaseInfo testCase(
        "",
        { "fake test name", "[TaG1][tAg1][TAG1][tag1]" },
        dummySourceLineInfo );

    REQUIRE( testCase.tags.size() == 1 );
    REQUIRE( testCase.tags[0] == Tag( "tag1" ) );
}

TEST_CASE("Mismatched square brackets in tags are caught and reported",
          "[tags][approvals]") {
    using Catch::TestCaseInfo;
    using Catch::Matchers::ContainsSubstring;
            REQUIRE_THROWS_WITH( TestCaseInfo( "",
                                       { "test with unclosed tag", "[abc" },
                                       dummySourceLineInfo ),
                         ContainsSubstring("registering test case 'test with unclosed tag'") );
    REQUIRE_THROWS_WITH( TestCaseInfo( "",
                      { "test with nested tags", "[abc[def]]" },
                      dummySourceLineInfo ),
        ContainsSubstring("registering test case 'test with nested tags'") );
    REQUIRE_THROWS_WITH( TestCaseInfo( "",
                      { "test with superfluous close tags", "[abc][def]]" },
                      dummySourceLineInfo ),
        ContainsSubstring("registering test case 'test with superfluous close tags'") );
}
