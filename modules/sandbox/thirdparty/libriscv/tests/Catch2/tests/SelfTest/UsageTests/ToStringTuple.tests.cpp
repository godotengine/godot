
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#define CATCH_CONFIG_ENABLE_TUPLE_STRINGMAKER
#include <catch2/catch_test_macros.hpp>

#include <tuple>

TEST_CASE( "tuple<>", "[toString][tuple]" )
{
    typedef std::tuple<> type;
    CHECK( "{ }" == ::Catch::Detail::stringify(type{}) );
    type value {};
    CHECK( "{ }" == ::Catch::Detail::stringify(value) );
}

TEST_CASE( "tuple<int>", "[toString][tuple]" )
{
    typedef std::tuple<int> type;
    CHECK( "{ 0 }" == ::Catch::Detail::stringify(type{0}) );
}


TEST_CASE( "tuple<float,int>", "[toString][tuple]" )
{
    typedef std::tuple<float,int> type;
    CHECK( "1.5f" == ::Catch::Detail::stringify(float(1.5)) );
    CHECK( "{ 1.5f, 0 }" == ::Catch::Detail::stringify(type{1.5f,0}) );
}

TEST_CASE( "tuple<string,string>", "[toString][tuple]" )
{
    typedef std::tuple<std::string,std::string> type;
    CHECK( "{ \"hello\", \"world\" }" == ::Catch::Detail::stringify(type{"hello","world"}) );
}

TEST_CASE( "tuple<tuple<int>,tuple<>,float>", "[toString][tuple]" )
{
    typedef std::tuple<std::tuple<int>,std::tuple<>,float> type;
    type value { std::tuple<int>{42}, {}, 1.5f };
    CHECK( "{ { 42 }, { }, 1.5f }" == ::Catch::Detail::stringify(value) );
}

TEST_CASE( "tuple<nullptr,int,const char *>", "[approvals][toString][tuple]" ) {
    typedef std::tuple<std::nullptr_t,int,const char *> type;
    type value { nullptr, 42, "Catch me" };
    CHECK( "{ nullptr, 42, \"Catch me\" }" == ::Catch::Detail::stringify(value) );
}

