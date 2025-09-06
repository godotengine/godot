
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/catch_test_macros.hpp>


TEST_CASE()
{
    SUCCEED( "anonymous test case" );
}

TEST_CASE( "Test case with one argument" )
{
    SUCCEED( "no assertions" );
}

TEST_CASE( "Variadic macros", "[variadic][sections]" )
{
    SECTION( "Section with one argument" )
    {
        SUCCEED( "no assertions" );
    }
}

