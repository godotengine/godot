
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#define CATCH_CONFIG_ENABLE_PAIR_STRINGMAKER
#include <catch2/catch_test_macros.hpp>

#include <map>
#include <set>

TEST_CASE( "Character pretty printing" ){
    SECTION("Specifically escaped"){
        CHECK(::Catch::Detail::stringify('\t') == "'\\t'");
        CHECK(::Catch::Detail::stringify('\n') == "'\\n'");
        CHECK(::Catch::Detail::stringify('\r') == "'\\r'");
        CHECK(::Catch::Detail::stringify('\f') == "'\\f'");
    }
    SECTION("General chars"){
        CHECK(::Catch::Detail::stringify( ' ' ) == "' '" );
        CHECK(::Catch::Detail::stringify( 'A' ) == "'A'" );
        CHECK(::Catch::Detail::stringify( 'z' ) == "'z'" );
    }
    SECTION("Low ASCII"){
        CHECK(::Catch::Detail::stringify( '\0' ) == "0" );
        CHECK(::Catch::Detail::stringify( static_cast<char>(2) ) == "2" );
        CHECK(::Catch::Detail::stringify( static_cast<char>(5) ) == "5" );
    }
}


TEST_CASE( "Capture and info messages" ) {
    SECTION("Capture should stringify like assertions") {
        int i = 2;
        CAPTURE(i);
        REQUIRE(true);
    }
    SECTION("Info should NOT stringify the way assertions do") {
        int i = 3;
        INFO(i);
        REQUIRE(true);
    }
}

TEST_CASE( "std::map is convertible string", "[toString]" ) {

    SECTION( "empty" ) {
        std::map<std::string, int> emptyMap;

        REQUIRE( Catch::Detail::stringify( emptyMap ) == "{  }" );
    }

    SECTION( "single item" ) {
        std::map<std::string, int> map = { { "one", 1 } };

        REQUIRE( Catch::Detail::stringify( map ) == "{ { \"one\", 1 } }" );
    }

    SECTION( "several items" ) {
        std::map<std::string, int> map = {
                { "abc", 1 },
                { "def", 2 },
                { "ghi", 3 }
            };

        REQUIRE( Catch::Detail::stringify( map ) == "{ { \"abc\", 1 }, { \"def\", 2 }, { \"ghi\", 3 } }" );
    }
}

TEST_CASE( "std::set is convertible string", "[toString]" ) {

    SECTION( "empty" ) {
        std::set<std::string> emptySet;

        REQUIRE( Catch::Detail::stringify( emptySet ) == "{  }" );
    }

    SECTION( "single item" ) {
        std::set<std::string> set = { "one" };

        REQUIRE( Catch::Detail::stringify( set ) == "{ \"one\" }" );
    }

    SECTION( "several items" ) {
        std::set<std::string> set = { "abc", "def", "ghi" };

        REQUIRE( Catch::Detail::stringify( set ) == "{ \"abc\", \"def\", \"ghi\" }" );
    }
}

TEST_CASE("Static arrays are convertible to string", "[toString]") {
    SECTION("Single item") {
        int singular[1] = { 1 };
        REQUIRE(Catch::Detail::stringify(singular) == "{ 1 }");
    }
    SECTION("Multiple") {
        int arr[3] = { 3, 2, 1 };
        REQUIRE(Catch::Detail::stringify(arr) == "{ 3, 2, 1 }");
    }
    SECTION("Non-trivial inner items") {
        std::vector<std::string> arr[2] = { {"1:1", "1:2", "1:3"}, {"2:1", "2:2"} };
        REQUIRE(Catch::Detail::stringify(arr) == R"({ { "1:1", "1:2", "1:3" }, { "2:1", "2:2" } })");
    }
}

#ifdef CATCH_CONFIG_CPP17_STRING_VIEW

TEST_CASE("String views are stringified like other strings", "[toString][approvals]") {
    std::string_view view{"abc"};
    CHECK(Catch::Detail::stringify(view) == R"("abc")");

    std::string_view arr[] { view };
    CHECK(Catch::Detail::stringify(arr) == R"({ "abc" })");
}

#endif

TEST_CASE("Precision of floating point stringification can be set", "[toString][floatingPoint]") {
    SECTION("Floats") {
        using sm = Catch::StringMaker<float>;
        const auto oldPrecision = sm::precision;

        const float testFloat = 1.12345678901234567899f;
        sm::precision = 5;
        auto str1 = sm::convert( testFloat );
        // "1." prefix = 2 chars, f suffix is another char
        CHECK(str1.size() == 3 + 5);

        sm::precision = 10;
        auto str2 = sm::convert(testFloat);
        REQUIRE(str2.size() == 3 + 10);
        sm::precision = oldPrecision;
    }
    SECTION("Double") {
        using sm = Catch::StringMaker<double>;
        const auto oldPrecision = sm::precision;

        const double testDouble = 1.123456789012345678901234567899;
        sm::precision = 5;
        auto str1 = sm::convert(testDouble);
        // "1." prefix = 2 chars
        CHECK(str1.size() == 2 + 5);

        sm::precision = 15;
        auto str2 = sm::convert(testDouble);
        REQUIRE(str2.size() == 2 + 15);

        sm::precision = oldPrecision;
    }
}

namespace {

struct WhatException : std::exception {
    char const* what() const noexcept override {
        return "This exception has overridden what() method";
    }
    ~WhatException() override;
};

struct OperatorException : std::exception {
    ~OperatorException() override;
};

std::ostream& operator<<(std::ostream& out, OperatorException const&) {
    out << "OperatorException";
    return out;
}

struct StringMakerException : std::exception {
    ~StringMakerException() override;
};

} // end anonymous namespace

namespace Catch {
template <>
struct StringMaker<StringMakerException> {
    static std::string convert(StringMakerException const&) {
        return "StringMakerException";
    }
};
}

// Avoid -Wweak-tables
WhatException::~WhatException() = default;
OperatorException::~OperatorException() = default;
StringMakerException::~StringMakerException() = default;




TEST_CASE("Exception as a value (e.g. in REQUIRE_THROWS_MATCHES) can be stringified", "[toString][exception]") {
    REQUIRE(::Catch::Detail::stringify(WhatException{}) == "This exception has overridden what() method");
    REQUIRE(::Catch::Detail::stringify(OperatorException{}) == "OperatorException");
    REQUIRE(::Catch::Detail::stringify(StringMakerException{}) == "StringMakerException");
}
