
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/catch_test_macros.hpp>

#if defined(CATCH_CONFIG_CPP17_BYTE)

TEST_CASE( "std::byte -> toString", "[toString][byte][approvals]" ) {
    using type = std::byte;
    REQUIRE( "0" == ::Catch::Detail::stringify( type{ 0 } ) );
}

TEST_CASE( "std::vector<std::byte> -> toString", "[toString][byte][approvals]" ) {
    using type = std::vector<std::byte>;
    REQUIRE( "{ 0, 1, 2 }" == ::Catch::Detail::stringify( type{ std::byte{0}, std::byte{1}, std::byte{2} } ) );
}

#endif // CATCH_INTERNAL_CONFIG_CPP17_BYTE
