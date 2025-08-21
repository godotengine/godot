
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#define CATCH_CONFIG_ENABLE_OPTIONAL_STRINGMAKER
#include <catch2/catch_test_macros.hpp>

#if defined(CATCH_CONFIG_CPP17_OPTIONAL)

TEST_CASE( "std::optional<int> -> toString", "[toString][optional][approvals]" ) {
    using type = std::optional<int>;
    REQUIRE( "{ }" == ::Catch::Detail::stringify( type{} ) );
    REQUIRE( "0" == ::Catch::Detail::stringify( type{ 0 } ) );
}

TEST_CASE( "std::optional<std::string> -> toString", "[toString][optional][approvals]" ) {
    using type = std::optional<std::string>;
    REQUIRE( "{ }" == ::Catch::Detail::stringify( type{} ) );
    REQUIRE( "\"abc\"" == ::Catch::Detail::stringify( type{ "abc" } ) );
}

TEST_CASE( "std::vector<std::optional<int> > -> toString", "[toString][optional][approvals]" ) {
    using type = std::vector<std::optional<int> >;
    REQUIRE( "{ 0, { }, 2 }" == ::Catch::Detail::stringify( type{ 0, {}, 2 } ) );
}

TEST_CASE( "std::nullopt -> toString", "[toString][optional][approvals]" ) {
    REQUIRE( "{ }" == ::Catch::Detail::stringify( std::nullopt ) );
}

#endif // CATCH_INTERNAL_CONFIG_CPP17_OPTIONAL
