
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <thread>

TEST_CASE( "sleep_for_100ms", "[.min_duration_test][approvals]" )
{
  std::this_thread::sleep_for( std::chrono::milliseconds( 100 ) );
  CHECK( true );
}

TEST_CASE( "sleep_for_1000ms", "[.min_duration_test][approvals]" )
{
  std::this_thread::sleep_for( std::chrono::milliseconds( 1'000 ) );
  CHECK( true );
}
