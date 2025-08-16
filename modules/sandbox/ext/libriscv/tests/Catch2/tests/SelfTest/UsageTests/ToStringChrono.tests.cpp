
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cstdint>

TEST_CASE("Stringifying std::chrono::duration helpers", "[toString][chrono]") {
    // No literals because we still support c++11
    auto hour = std::chrono::hours(1);
    auto minute = std::chrono::minutes(1);
    auto seconds = std::chrono::seconds(60);
    auto micro = std::chrono::microseconds(1);
    auto milli = std::chrono::milliseconds(1);
    auto nano = std::chrono::nanoseconds(1);
    REQUIRE(minute == seconds);
    REQUIRE(hour != seconds);
    REQUIRE(micro != milli);
    REQUIRE(nano != micro);
}

TEST_CASE("Stringifying std::chrono::duration with weird ratios", "[toString][chrono]") {
    std::chrono::duration<int64_t, std::ratio<30>> half_minute(1);
    std::chrono::duration<int64_t, std::ratio<1, 1000000000000>> pico_second(1);
    std::chrono::duration<int64_t, std::ratio<1, 1000000000000000>> femto_second(1);
    std::chrono::duration<int64_t, std::ratio<1, 1000000000000000000>> atto_second(1);
    REQUIRE(half_minute != femto_second);
    REQUIRE(pico_second != atto_second);
}

TEST_CASE("Stringifying std::chrono::time_point<system_clock>", "[toString][chrono]") {
    auto now = std::chrono::system_clock::now();
    auto later = now + std::chrono::minutes(2);
    REQUIRE(now != later);
}

TEST_CASE("Stringifying std::chrono::time_point<Clock>", "[toString][chrono][!nonportable]") {
    auto now = std::chrono::high_resolution_clock::now();
    auto later = now + std::chrono::minutes(2);
    REQUIRE(now != later);

    auto now2 = std::chrono::steady_clock::now();
    auto later2 = now2 + std::chrono::minutes(2);
    REQUIRE(now2 != later2);
}

TEST_CASE( "system_clock timepoint with non-default duration", "[toString][chrono]" ) {
    std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds>
        tp1, tp2;
    CHECK( tp1 == tp2 );
}
