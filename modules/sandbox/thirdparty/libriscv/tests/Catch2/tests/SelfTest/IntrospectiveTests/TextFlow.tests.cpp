
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/catch_test_macros.hpp>
#include <catch2/internal/catch_textflow.hpp>

#include <sstream>

using Catch::TextFlow::Column;
using Catch::TextFlow::AnsiSkippingString;

namespace {
    static std::string as_written(Column const& c) {
        std::stringstream sstr;
        sstr << c;
        return sstr.str();
    }
}

TEST_CASE( "TextFlow::Column one simple line",
           "[TextFlow][column][approvals]" ) {
    Column col( "simple short line" );

    REQUIRE(as_written(col) == "simple short line");
}

TEST_CASE( "TextFlow::Column respects already present newlines",
           "[TextFlow][column][approvals]" ) {
    Column col( "abc\ndef" );
    REQUIRE( as_written( col ) == "abc\ndef" );
}

TEST_CASE( "TextFlow::Column respects width setting",
           "[TextFlow][column][approvals]" ) {
    Column col( "The quick brown fox jumped over the lazy dog" );

    SECTION( "width=20" ) {
        col.width( 20 );
        REQUIRE( as_written( col ) == "The quick brown fox\n"
                                      "jumped over the lazy\n"
                                      "dog" );
    }
    SECTION("width=10") {
        col.width( 10 );
        REQUIRE( as_written( col ) == "The quick\n"
                                      "brown fox\n"
                                      "jumped\n"
                                      "over the\n"
                                      "lazy dog" );
    }
    SECTION("width=5") {
        // This is so small some words will have to be split with hyphen
        col.width(5);
        REQUIRE( as_written( col ) == "The\n"
                                      "quick\n"
                                      "brown\n"
                                      "fox\n"
                                      "jump-\n"
                                      "ed\n"
                                      "over\n"
                                      "the\n"
                                      "lazy\n"
                                      "dog" );
    }
}

TEST_CASE( "TextFlow::Column respects indentation setting",
           "[TextFlow][column][approvals]" ) {
    Column col( "First line\nSecond line\nThird line" );

    SECTION("Default: no indentation at all") {
        REQUIRE(as_written(col) == "First line\nSecond line\nThird line");
    }
    SECTION("Indentation on first line only") {
        col.initialIndent(3);
        REQUIRE(as_written(col) == "   First line\nSecond line\nThird line");
    }
    SECTION("Indentation on all lines") {
        col.indent(3);
        REQUIRE(as_written(col) == "   First line\n   Second line\n   Third line");
    }
    SECTION("Indentation on later lines only") {
        col.indent(5).initialIndent(0);
        REQUIRE(as_written(col) == "First line\n     Second line\n     Third line");
    }
    SECTION("Different indentation on first and later lines") {
        col.initialIndent(1).indent(2);
        REQUIRE(as_written(col) == " First line\n  Second line\n  Third line");
    }
}

TEST_CASE("TextFlow::Column indentation respects whitespace", "[TextFlow][column][approvals]") {
    Column col(" text with whitespace\n after newlines");

    SECTION("No extra indentation") {
        col.initialIndent(0).indent(0);
        REQUIRE(as_written(col) == " text with whitespace\n after newlines");
    }
    SECTION("Different indentation on first and later lines") {
        col.initialIndent(1).indent(2);
        REQUIRE(as_written(col) == "  text with whitespace\n   after newlines");
    }
}

TEST_CASE( "TextFlow::Column linebreaking prefers boundary characters",
           "[TextFlow][column][approvals]" ) {
    SECTION("parentheses") {
        Column col("(Hello)aaa(World)");
        SECTION("width=20") {
            col.width(20);
            REQUIRE(as_written(col) == "(Hello)aaa(World)");
        }
        SECTION("width=15") {
            col.width(15);
            REQUIRE(as_written(col) == "(Hello)aaa\n(World)");
        }
        SECTION("width=8") {
            col.width(8);
            REQUIRE(as_written(col) == "(Hello)\naaa\n(World)");
        }
    }
    SECTION("commas") {
        Column col("Hello, world");
        col.width(8);

        REQUIRE(as_written(col) == "Hello,\nworld");
    }
}


TEST_CASE( "TextFlow::Column respects indentation for empty lines",
           "[TextFlow][column][approvals][!shouldfail]" ) {
    // This is currently bugged and does not do what it should
    Column col("\n\nthird line");
    col.indent(2);

    //auto b = col.begin();
    //auto e = col.end();

    //auto b1 = *b;
    //++b;
    //auto b2 = *b;
    //++b;
    //auto b3 = *b;
    //++b;

    //REQUIRE(b == e);

    std::string written = as_written(col);

    REQUIRE(written == "  \n  \n  third line");
}

TEST_CASE( "TextFlow::Column leading/trailing whitespace",
           "[TextFlow][column][approvals]" ) {
    SECTION("Trailing whitespace") {
        Column col("some trailing whitespace: \t");
        REQUIRE(as_written(col) == "some trailing whitespace: \t");
    }
    SECTION("Some leading whitespace") {
        Column col("\t \t whitespace wooo");
        REQUIRE(as_written(col) == "\t \t whitespace wooo");
    }
    SECTION("both") {
        Column col(" abc ");
        REQUIRE(as_written(col) == " abc ");
    }
    SECTION("whitespace only") {
        Column col("\t \t");
        REQUIRE(as_written(col) == "\t \t");
    }
}

TEST_CASE( "TextFlow::Column can handle empty string",
           "[TextFlow][column][approvals]" ) {
    Column col("");
    REQUIRE(as_written(col) == "");
}

TEST_CASE( "#1400 - TextFlow::Column wrapping would sometimes duplicate words",
           "[TextFlow][column][regression][approvals]" ) {
    const auto long_string = std::string(
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Pellentesque nisl \n"
        "massa, luctus ut ligula vitae, suscipit tempus velit. Vivamus sodales, quam in \n"
        "convallis posuere, libero nisi ultricies orci, nec lobortis.\n");

    auto col = Column(long_string)
        .width(79)
        .indent(2);

    REQUIRE(as_written(col) ==
            "  Lorem ipsum dolor sit amet, consectetur adipiscing elit. Pellentesque nisl \n"
            "  massa, luctus ut ligula vitae, suscipit tempus velit. Vivamus sodales, quam\n"
            "  in \n"
            "  convallis posuere, libero nisi ultricies orci, nec lobortis.");
}

TEST_CASE( "TextFlow::AnsiSkippingString skips ansi sequences",
           "[TextFlow][ansiskippingstring][approvals]" ) {

    SECTION("basic string") {
        std::string text = "a\033[38;2;98;174;239mb\033[38mc\033[0md\033[me";
        AnsiSkippingString str(text);

        SECTION( "iterates forward" ) {
            auto it = str.begin();
            CHECK(*it == 'a');
            ++it;
            CHECK(*it == 'b');
            ++it;
            CHECK(*it == 'c');
            ++it;
            CHECK(*it == 'd');
            ++it;
            CHECK(*it == 'e');
            ++it;
            CHECK(it == str.end());
        }
        SECTION( "iterates backwards" ) {
            auto it = str.end();
            --it;
            CHECK(*it == 'e');
            --it;
            CHECK(*it == 'd');
            --it;
            CHECK(*it == 'c');
            --it;
            CHECK(*it == 'b');
            --it;
            CHECK(*it == 'a');
            CHECK(it == str.begin());
        }
    }

    SECTION( "ansi escape sequences at the start" ) {
        std::string text = "\033[38;2;98;174;239ma\033[38;2;98;174;239mb\033[38mc\033[0md\033[me";
        AnsiSkippingString str(text);
        auto it = str.begin();
        CHECK(*it == 'a');
        ++it;
        CHECK(*it == 'b');
        ++it;
        CHECK(*it == 'c');
        ++it;
        CHECK(*it == 'd');
        ++it;
        CHECK(*it == 'e');
        ++it;
        CHECK(it == str.end());
        --it;
        CHECK(*it == 'e');
        --it;
        CHECK(*it == 'd');
        --it;
        CHECK(*it == 'c');
        --it;
        CHECK(*it == 'b');
        --it;
        CHECK(*it == 'a');
        CHECK(it == str.begin());
    }

    SECTION( "ansi escape sequences at the end" ) {
        std::string text = "a\033[38;2;98;174;239mb\033[38mc\033[0md\033[me\033[38;2;98;174;239m";
        AnsiSkippingString str(text);
        auto it = str.begin();
        CHECK(*it == 'a');
        ++it;
        CHECK(*it == 'b');
        ++it;
        CHECK(*it == 'c');
        ++it;
        CHECK(*it == 'd');
        ++it;
        CHECK(*it == 'e');
        ++it;
        CHECK(it == str.end());
        --it;
        CHECK(*it == 'e');
        --it;
        CHECK(*it == 'd');
        --it;
        CHECK(*it == 'c');
        --it;
        CHECK(*it == 'b');
        --it;
        CHECK(*it == 'a');
        CHECK(it == str.begin());
    }

    SECTION( "skips consecutive escapes" ) {
        std::string text = "\033[38;2;98;174;239m\033[38;2;98;174;239ma\033[38;2;98;174;239mb\033[38m\033[38m\033[38mc\033[0md\033[me";
        AnsiSkippingString str(text);
        auto it = str.begin();
        CHECK(*it == 'a');
        ++it;
        CHECK(*it == 'b');
        ++it;
        CHECK(*it == 'c');
        ++it;
        CHECK(*it == 'd');
        ++it;
        CHECK(*it == 'e');
        ++it;
        CHECK(it == str.end());
        --it;
        CHECK(*it == 'e');
        --it;
        CHECK(*it == 'd');
        --it;
        CHECK(*it == 'c');
        --it;
        CHECK(*it == 'b');
        --it;
        CHECK(*it == 'a');
        CHECK(it == str.begin());
    }

    SECTION( "handles incomplete ansi sequences" ) {
        std::string text = "a\033[b\033[30c\033[30;d\033[30;2e";
        AnsiSkippingString str(text);
        CHECK(std::string(str.begin(), str.end()) == text);
    }
}

TEST_CASE( "TextFlow::AnsiSkippingString computes the size properly",
           "[TextFlow][ansiskippingstring][approvals]" ) {
    std::string text = "\033[38;2;98;174;239m\033[38;2;98;174;239ma\033[38;2;98;174;239mb\033[38m\033[38m\033[38mc\033[0md\033[me";
    AnsiSkippingString str(text);
    CHECK(str.size() == 5);
}

TEST_CASE( "TextFlow::AnsiSkippingString substrings properly",
           "[TextFlow][ansiskippingstring][approvals]" ) {
    SECTION("basic test") {
        std::string text = "a\033[38;2;98;174;239mb\033[38mc\033[0md\033[me";
        AnsiSkippingString str(text);
        auto a = str.begin();
        auto b = str.begin();
        ++b;
        ++b;
        CHECK(str.substring(a, b) == "a\033[38;2;98;174;239mb\033[38m");
        ++a;
        ++b;
        CHECK(str.substring(a, b) == "b\033[38mc\033[0m");
        CHECK(str.substring(a, str.end()) == "b\033[38mc\033[0md\033[me");
        CHECK(str.substring(str.begin(), str.end()) == text);
    }
    SECTION("escapes at the start") {
        std::string text = "\033[38;2;98;174;239m\033[38;2;98;174;239ma\033[38;2;98;174;239mb\033[38m\033[38m\033[38mc\033[0md\033[me";
        AnsiSkippingString str(text);
        auto a = str.begin();
        auto b = str.begin();
        ++b;
        ++b;
        CHECK(str.substring(a, b) == "\033[38;2;98;174;239m\033[38;2;98;174;239ma\033[38;2;98;174;239mb\033[38m\033[38m\033[38m");
        ++a;
        ++b;
        CHECK(str.substring(a, b) == "b\033[38m\033[38m\033[38mc\033[0m");
        CHECK(str.substring(a, str.end()) == "b\033[38m\033[38m\033[38mc\033[0md\033[me");
        CHECK(str.substring(str.begin(), str.end()) == text);
    }
    SECTION("escapes at the end") {
        std::string text = "a\033[38;2;98;174;239mb\033[38mc\033[0md\033[me\033[38m";
        AnsiSkippingString str(text);
        auto a = str.begin();
        auto b = str.begin();
        ++b;
        ++b;
        CHECK(str.substring(a, b) == "a\033[38;2;98;174;239mb\033[38m");
        ++a;
        ++b;
        CHECK(str.substring(a, b) == "b\033[38mc\033[0m");
        CHECK(str.substring(a, str.end()) == "b\033[38mc\033[0md\033[me\033[38m");
        CHECK(str.substring(str.begin(), str.end()) == text);
    }
}

TEST_CASE( "TextFlow::Column skips ansi escape sequences",
           "[TextFlow][column][approvals]" ) {
    std::string text = "\033[38;2;98;174;239m\033[38;2;198;120;221mThe quick brown \033[38;2;198;120;221mfox jumped over the lazy dog\033[0m";
    Column col(text);

    SECTION( "width=20" ) {
        col.width( 20 );
        REQUIRE( as_written( col ) == "\033[38;2;98;174;239m\033[38;2;198;120;221mThe quick brown \033[38;2;198;120;221mfox\n"
                                      "jumped over the lazy\n"
                                      "dog\033[0m" );
    }

    SECTION( "width=80" ) {
        col.width( 80 );
        REQUIRE( as_written( col ) == text );
    }
}
