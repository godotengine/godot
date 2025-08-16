
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/internal/catch_floating_point_helpers.hpp>
#include <catch2/internal/catch_random_floating_point_helpers.hpp>

#include <limits>

TEST_CASE("convertToBits", "[floating-point][conversion]") {
    using Catch::Detail::convertToBits;

    CHECK( convertToBits( 0.f ) == 0 );
    CHECK( convertToBits( -0.f ) == ( 1ULL << 31 ) );
    CHECK( convertToBits( 0. ) == 0 );
    CHECK( convertToBits( -0. ) == ( 1ULL << 63 ) );
    CHECK( convertToBits( std::numeric_limits<float>::denorm_min() ) == 1 );
    CHECK( convertToBits( std::numeric_limits<double>::denorm_min() ) == 1 );
}

TEMPLATE_TEST_CASE("type-shared ulpDistance tests", "[floating-point][ulp][approvals]", float, double) {
    using FP = TestType;
    using Catch::ulpDistance;

    // Distance between zeros is zero
    CHECK( ulpDistance( FP{}, FP{} ) == 0 );
    CHECK( ulpDistance( FP{}, -FP{} ) == 0 );
    CHECK( ulpDistance( -FP{}, -FP{} ) == 0 );

    // Distance between same-sign infinities is zero
    static constexpr FP infinity = std::numeric_limits<FP>::infinity();
    CHECK( ulpDistance( infinity, infinity ) == 0 );
    CHECK( ulpDistance( -infinity, -infinity ) == 0 );

    // Distance between max-finite-val and same sign infinity is 1
    static constexpr FP max_finite = std::numeric_limits<FP>::max();
    CHECK( ulpDistance( max_finite, infinity ) == 1 );
    CHECK( ulpDistance( -max_finite, -infinity ) == 1 );

    // Distance between X and 0 is half of distance between X and -X
    CHECK( ulpDistance( -infinity, infinity ) ==
           2 * ulpDistance( infinity, FP{} ) );
    CHECK( 2 * ulpDistance( FP{ -2. }, FP{} ) ==
           ulpDistance( FP{ -2. }, FP{ 2. } ) );
    CHECK( 2 * ulpDistance( FP{ 2. }, FP{} ) ==
           ulpDistance( FP{ -2. }, FP{ 2. } ) );

    // Denorms are supported
    CHECK( ulpDistance( std::numeric_limits<FP>::denorm_min(), FP{} ) == 1 );
    CHECK( ulpDistance( std::numeric_limits<FP>::denorm_min(), -FP{} ) == 1 );
    CHECK( ulpDistance( -std::numeric_limits<FP>::denorm_min(), FP{} ) == 1 );
    CHECK( ulpDistance( -std::numeric_limits<FP>::denorm_min(), -FP{} ) == 1 );
    CHECK( ulpDistance( std::numeric_limits<FP>::denorm_min(),
                        -std::numeric_limits<FP>::denorm_min() ) == 2 );

    // Machine epsilon
    CHECK( ulpDistance( FP{ 1. },
                        FP{ 1. } + std::numeric_limits<FP>::epsilon() ) == 1 );
    CHECK( ulpDistance( -FP{ 1. },
                        -FP{ 1. } - std::numeric_limits<FP>::epsilon() ) == 1 );
}

TEST_CASE("UlpDistance", "[floating-point][ulp][approvals]") {
    using Catch::ulpDistance;

    CHECK( ulpDistance( 1., 2. ) == 0x10'00'00'00'00'00'00 );
    CHECK( ulpDistance( -2., 2. ) == 0x80'00'00'00'00'00'00'00 );
    CHECK( ulpDistance( 1.f, 2.f ) == 0x80'00'00 );
    CHECK( ulpDistance( -2.f, 2.f ) == 0x80'00'00'00 );
}



TEMPLATE_TEST_CASE("gamma", "[approvals][floating-point][ulp][gamma]", float, double) {
    using Catch::Detail::gamma;
    using Catch::Detail::directCompare;

    // We need to butcher the equal tests with the directCompare helper,
    // because the Wfloat-equal triggers in decomposer rather than here,
    // so we cannot locally disable it. Goddamn GCC.
    CHECK( directCompare( gamma( TestType( -1. ), TestType( 1. ) ),
                          gamma( TestType( 0.2332 ), TestType( 1.0 ) ) ) );
    CHECK( directCompare( gamma( TestType( -2. ), TestType( 0 ) ),
                          gamma( TestType( 1. ), TestType( 1.5 ) ) ) );
    CHECK( gamma( TestType( 0. ), TestType( 1.0 ) ) <
           gamma( TestType( 1.0 ), TestType( 1.5 ) ) );
    CHECK( gamma( TestType( 0 ), TestType( 1. ) ) <
           std::numeric_limits<TestType>::epsilon() );
    CHECK( gamma( TestType( -1. ), TestType( -0. ) ) <
           std::numeric_limits<TestType>::epsilon() );
    CHECK( directCompare( gamma( TestType( 1. ), TestType( 2. ) ),
                          std::numeric_limits<TestType>::epsilon() ) );
    CHECK( directCompare( gamma( TestType( -2. ), TestType( -1. ) ),
                          std::numeric_limits<TestType>::epsilon() ) );
}

TEMPLATE_TEST_CASE("count_equidistant_floats",
                   "[approvals][floating-point][distance]",
                   float,
                   double) {
    using Catch::Detail::count_equidistant_floats;
    auto count_steps = []( TestType a, TestType b ) {
        return count_equidistant_floats( a, b, Catch::Detail::gamma( a, b ) );
    };

    CHECK( count_steps( TestType( -1. ), TestType( 1. ) ) ==
           2 * count_steps( TestType( 0. ), TestType( 1. ) ) );
}

TEST_CASE( "count_equidistant_floats",
           "[approvals][floating-point][distance]" ) {
    using Catch::Detail::count_equidistant_floats;
    auto count_floats_with_scaled_ulp = []( auto a, auto b ) {
        return count_equidistant_floats( a, b, Catch::Detail::gamma( a, b ) );
    };

    CHECK( count_floats_with_scaled_ulp( 1., 1.5 ) == 1ull << 51 );
    CHECK( count_floats_with_scaled_ulp( 1.25, 1.5 ) == 1ull << 50 );
    CHECK( count_floats_with_scaled_ulp( 1.f, 1.5f ) == 1 << 22 );
    CHECK( count_floats_with_scaled_ulp( -std::numeric_limits<float>::max(),
                                         std::numeric_limits<float>::max() ) ==
           33554430 ); // (1 << 25) - 2 due to not including infinities
    CHECK( count_floats_with_scaled_ulp( -std::numeric_limits<double>::max(),
                                         std::numeric_limits<double>::max() ) ==
           18014398509481982 ); // (1 << 54) - 2 due to not including infinities

    STATIC_REQUIRE( std::is_same<std::uint64_t,
                                 decltype( count_floats_with_scaled_ulp(
                                     0., 1. ) )>::value );
    STATIC_REQUIRE( std::is_same<std::uint32_t,
                                 decltype( count_floats_with_scaled_ulp(
                                     0.f, 1.f ) )>::value );
}
