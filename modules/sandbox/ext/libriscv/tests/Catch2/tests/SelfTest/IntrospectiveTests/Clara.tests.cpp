
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/catch_test_macros.hpp>
#include <catch2/internal/catch_clara.hpp>


#include <string>

TEST_CASE("is_unary_function", "[clara][compilation]") {
    auto unary1 = [](int) {};
    auto unary2 = [](std::string const&) {};
    auto const unary3 = [](std::string const&) {};
    auto unary4 = [](int) { return 42; };
    void unary5(char);
    double unary6(long);

    double binary1(long, int);
    auto binary2 = [](int, char) {};
    auto nullary1 = []() {};
    auto nullary2 = []() {return 42;};

    STATIC_REQUIRE(Catch::Clara::Detail::is_unary_function_v<decltype(unary1)>);
    STATIC_REQUIRE(Catch::Clara::Detail::is_unary_function_v<decltype(unary2)>);
    STATIC_REQUIRE(Catch::Clara::Detail::is_unary_function_v<decltype(unary3)>);
    STATIC_REQUIRE(Catch::Clara::Detail::is_unary_function_v<decltype(unary4)>);
    STATIC_REQUIRE(Catch::Clara::Detail::is_unary_function_v<decltype(unary5)>);
    STATIC_REQUIRE(Catch::Clara::Detail::is_unary_function_v<decltype(unary6)>);

    STATIC_REQUIRE_FALSE(Catch::Clara::Detail::is_unary_function_v<decltype(binary1)>);
    STATIC_REQUIRE_FALSE(Catch::Clara::Detail::is_unary_function_v<decltype(binary2)>);
    STATIC_REQUIRE_FALSE(Catch::Clara::Detail::is_unary_function_v<decltype(nullary1)>);
    STATIC_REQUIRE_FALSE(Catch::Clara::Detail::is_unary_function_v<decltype(nullary2)>);
    STATIC_REQUIRE_FALSE(Catch::Clara::Detail::is_unary_function_v<int>);
    STATIC_REQUIRE_FALSE(Catch::Clara::Detail::is_unary_function_v<std::string const&>);
}


TEST_CASE("Clara::Arg supports single-arg parse the way Opt does", "[clara][arg][compilation]") {
    std::string name;
    auto p = Catch::Clara::Arg(name, "just one arg");

    CHECK(name.empty());

    p.parse( Catch::Clara::Args{ "UnitTest", "foo" } );
    REQUIRE(name == "foo");
}

TEST_CASE("Clara::Arg does not crash on incomplete input", "[clara][arg][compilation]") {
    std::string name;
    auto p = Catch::Clara::Arg(name, "-");

    CHECK(name.empty());

    auto result = p.parse( Catch::Clara::Args{ "UnitTest", "-" } );
    CHECK( result );
    CHECK( result.type() == Catch::Clara::Detail::ResultType::Ok );
    const auto& parsed = result.value();
    CHECK( parsed.type() == Catch::Clara::ParseResultType::NoMatch );
    CHECK( parsed.remainingTokens().count() == 2 );
    CHECK( name.empty() );
}

TEST_CASE("Clara::Opt supports accept-many lambdas", "[clara][opt]") {
    using namespace Catch::Clara;
    std::vector<std::string> res;
    const auto push_to_res = [&](std::string const& s) {
        res.push_back(s);
        return ParserResult::ok( ParseResultType::Matched );
    };

    SECTION("Parsing fails on multiple options without accept_many") {
        auto p = Parser() | Opt(push_to_res, "value")["-o"];
        auto parse_result = p.parse( Args{ "UnitTest", "-o", "aaa", "-o", "bbb" } );
        CHECK_FALSE(parse_result);
    }
    SECTION("Parsing succeeds on multiple options with accept_many") {
        auto p = Parser() | Opt(accept_many, push_to_res, "value")["-o"];
        auto parse_result = p.parse( Args{ "UnitTest", "-o", "aaa", "-o", "bbb" } );
        CHECK(parse_result);
        CHECK(res == std::vector<std::string>{ "aaa", "bbb" });
    }
}
