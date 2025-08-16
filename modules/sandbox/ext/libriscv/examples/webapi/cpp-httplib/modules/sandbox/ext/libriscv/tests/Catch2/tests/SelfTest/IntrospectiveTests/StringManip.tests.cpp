
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include <catch2/internal/catch_string_manip.hpp>

static const char * const no_whitespace = "There is no extra whitespace here";
static const char * const leading_whitespace = " \r \t\n There is no extra whitespace here";
static const char * const trailing_whitespace = "There is no extra whitespace here \t \n \r ";
static const char * const whitespace_at_both_ends = " \r\n \t There is no extra whitespace here  \t\t\t \n";

TEST_CASE("Trim strings", "[string-manip]") {
    using Catch::trim; using Catch::StringRef;
    static_assert(std::is_same<std::string, decltype(trim(std::string{}))>::value, "Trimming std::string should return std::string");
    static_assert(std::is_same<StringRef, decltype(trim(StringRef{}))>::value, "Trimming StringRef should return StringRef");

    REQUIRE(trim(std::string(no_whitespace)) == no_whitespace);
    REQUIRE(trim(std::string(leading_whitespace)) == no_whitespace);
    REQUIRE(trim(std::string(trailing_whitespace)) == no_whitespace);
    REQUIRE(trim(std::string(whitespace_at_both_ends)) == no_whitespace);

    REQUIRE(trim(StringRef(no_whitespace)) == StringRef(no_whitespace));
    REQUIRE(trim(StringRef(leading_whitespace)) == StringRef(no_whitespace));
    REQUIRE(trim(StringRef(trailing_whitespace)) == StringRef(no_whitespace));
    REQUIRE(trim(StringRef(whitespace_at_both_ends)) == StringRef(no_whitespace));
}

TEST_CASE("replaceInPlace", "[string-manip]") {
    std::string letters = "abcdefcg";
    SECTION("replace single char") {
        CHECK(Catch::replaceInPlace(letters, "b", "z"));
        CHECK(letters == "azcdefcg");
    }
    SECTION("replace two chars") {
        CHECK(Catch::replaceInPlace(letters, "c", "z"));
        CHECK(letters == "abzdefzg");
    }
    SECTION("replace first char") {
        CHECK(Catch::replaceInPlace(letters, "a", "z"));
        CHECK(letters == "zbcdefcg");
    }
    SECTION("replace last char") {
        CHECK(Catch::replaceInPlace(letters, "g", "z"));
        CHECK(letters == "abcdefcz");
    }
    SECTION("replace all chars") {
        CHECK(Catch::replaceInPlace(letters, letters, "replaced"));
        CHECK(letters == "replaced");
    }
    SECTION("replace no chars") {
        CHECK_FALSE(Catch::replaceInPlace(letters, "x", "z"));
        CHECK(letters == letters);
    }
    SECTION("no replace in already-replaced string") {
        SECTION("lengthening") {
            CHECK(Catch::replaceInPlace(letters, "c", "cc"));
            CHECK(letters == "abccdefccg");
        }
        SECTION("shortening") {
            std::string s = "----";
            CHECK(Catch::replaceInPlace(s, "--", "-"));
            CHECK(s == "--");
        }
    }
    SECTION("escape '") {
        std::string s = "didn't";
        CHECK(Catch::replaceInPlace(s, "'", "|'"));
        CHECK(s == "didn|'t");
    }
}

TEST_CASE("splitString", "[string-manip]") {
    using namespace Catch::Matchers;
    using Catch::splitStringRef;
    using Catch::StringRef;

    CHECK_THAT(splitStringRef("", ','), Equals(std::vector<StringRef>()));
    CHECK_THAT(splitStringRef("abc", ','), Equals(std::vector<StringRef>{"abc"}));
    CHECK_THAT(splitStringRef("abc,def", ','), Equals(std::vector<StringRef>{"abc", "def"}));
}

TEST_CASE("startsWith", "[string-manip]") {
    using Catch::startsWith;

    CHECK_FALSE(startsWith("", 'c'));
    CHECK(startsWith(std::string("abc"), 'a'));
    CHECK(startsWith("def"_catch_sr, 'd'));
}
