
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#define CATCH_CONFIG_ENABLE_VARIANT_STRINGMAKER
#include <catch2/catch_test_macros.hpp>

#if defined(CATCH_CONFIG_CPP17_VARIANT)

#include <string>
#include <variant>

// We need 2 types with non-trivial copies/moves
struct MyType1 {
    MyType1() = default;
    [[noreturn]] MyType1(MyType1 const&) { throw 1; }
    MyType1& operator=(MyType1 const&) { throw 3; }
};
struct MyType2 {
    MyType2() = default;
    [[noreturn]] MyType2(MyType2 const&) { throw 2; }
    MyType2& operator=(MyType2 const&) { throw 4; }
};

TEST_CASE( "variant<std::monostate>", "[toString][variant][approvals]")
{
    using type = std::variant<std::monostate>;
    CHECK( "{ }" == ::Catch::Detail::stringify(type{}) );
    type value {};
    CHECK( "{ }" == ::Catch::Detail::stringify(value) );
    CHECK( "{ }" == ::Catch::Detail::stringify(std::get<0>(value)) );
}

TEST_CASE( "variant<int>", "[toString][variant][approvals]")
{
    using type = std::variant<int>;
    CHECK( "0" == ::Catch::Detail::stringify(type{0}) );
}

TEST_CASE( "variant<float, int>", "[toString][variant][approvals]")
{
    using type = std::variant<float, int>;
    CHECK( "0.5f" == ::Catch::Detail::stringify(type{0.5f}) );
    CHECK( "0" == ::Catch::Detail::stringify(type{0}) );
}

TEST_CASE( "variant -- valueless-by-exception", "[toString][variant][approvals]" ) {
    using type = std::variant<MyType1, MyType2>;

    type value;
    REQUIRE_THROWS_AS(value.emplace<MyType2>(MyType2{}), int);
    REQUIRE(value.valueless_by_exception());
    CHECK("{valueless variant}" == ::Catch::Detail::stringify(value));
}


TEST_CASE( "variant<string, int>", "[toString][variant][approvals]")
{
    using type = std::variant<std::string, int>;
    CHECK( "\"foo\"" == ::Catch::Detail::stringify(type{"foo"}) );
    CHECK( "0" == ::Catch::Detail::stringify(type{0}) );
}

TEST_CASE( "variant<variant<float, int>, string>", "[toString][variant][approvals]")
{
    using inner = std::variant<MyType1, float, int>;
    using type = std::variant<inner, std::string>;
    CHECK( "0.5f" == ::Catch::Detail::stringify(type{0.5f}) );
    CHECK( "0" == ::Catch::Detail::stringify(type{0}) );
    CHECK( "\"foo\"" == ::Catch::Detail::stringify(type{"foo"}) );

    SECTION("valueless nested variant") {
        type value = inner{0.5f};
        REQUIRE( std::holds_alternative<inner>(value) );
        REQUIRE( std::holds_alternative<float>(std::get<inner>(value)) );

        REQUIRE_THROWS_AS( std::get<0>(value).emplace<MyType1>(MyType1{}), int );

        // outer variant is still valid and contains inner
        REQUIRE( std::holds_alternative<inner>(value) );
        // inner variant is valueless
        REQUIRE( std::get<inner>(value).valueless_by_exception() );
        CHECK( "{valueless variant}" == ::Catch::Detail::stringify(value) );
    }
}

TEST_CASE( "variant<nullptr,int,const char *>", "[toString][variant][approvals]" )
{
    using type = std::variant<std::nullptr_t,int,const char *>;
    CHECK( "nullptr" == ::Catch::Detail::stringify(type{nullptr}) );
    CHECK( "42" == ::Catch::Detail::stringify(type{42}) );
    CHECK( "\"Catch me\"" == ::Catch::Detail::stringify(type{"Catch me"}) );
}

#endif // CATCH_INTERNAL_CONFIG_CPP17_VARIANT
