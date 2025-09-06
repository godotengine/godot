
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/catch_test_macros.hpp>
#include <catch2/internal/catch_random_integer_helpers.hpp>
#include <random>

namespace {
    template <typename Int>
    static void
    CommutativeMultCheck( Int a, Int b, Int upper_result, Int lower_result ) {
        using Catch::Detail::extendedMult;
        using Catch::Detail::ExtendedMultResult;
        CHECK( extendedMult( a, b ) ==
               ExtendedMultResult<Int>{ upper_result, lower_result } );
        CHECK( extendedMult( b, a ) ==
               ExtendedMultResult<Int>{ upper_result, lower_result } );
    }

    // Simple (and slow) implementation of extended multiplication for tests
    constexpr Catch::Detail::ExtendedMultResult<std::uint64_t>
    extendedMultNaive( std::uint64_t lhs, std::uint64_t rhs ) {
        // This is a simple long multiplication, where we split lhs and rhs
        // into two 32-bit "digits", so that we can do ops with carry in 64-bits.
        //
        //            32b    32b    32b    32b
        //     lhs                  L1     L2
        //   * rhs                  R1     R2
        //            ------------------------
        //                       |  R2 * L2  |
        //                 |  R2 * L1  |
        //                 |  R1 * L2  |
        //           |  R1 * L1  |
        //           -------------------------
        //           |  a  |  b  |  c  |  d  |

#define CarryBits( x ) ( x >> 32 )
#define Digits( x ) ( x & 0xFF'FF'FF'FF )

        auto r2l2 = Digits( rhs ) * Digits( lhs );
        auto r2l1 = Digits( rhs ) * CarryBits( lhs );
        auto r1l2 = CarryBits( rhs ) * Digits( lhs );
        auto r1l1 = CarryBits( rhs ) * CarryBits( lhs );

        // Sum to columns first
        auto d = Digits( r2l2 );
        auto c = CarryBits( r2l2 ) + Digits( r2l1 ) + Digits( r1l2 );
        auto b = CarryBits( r2l1 ) + CarryBits( r1l2 ) + Digits( r1l1 );
        auto a = CarryBits( r1l1 );

        // Propagate carries between columns
        c += CarryBits( d );
        b += CarryBits( c );
        a += CarryBits( b );

        // Remove the used carries
        c = Digits( c );
        b = Digits( b );
        a = Digits( a );

#undef CarryBits
#undef Digits

        return {
            a << 32 | b, // upper 64 bits
            c << 32 | d  // lower 64 bits
        };
    }


} // namespace

TEST_CASE( "extendedMult 64x64", "[Integer][approvals]" ) {
    // a x 0 == 0
    CommutativeMultCheck<uint64_t>( 0x1234'5678'9ABC'DEFF, 0, 0, 0 );

    // bit carried from low half to upper half
    CommutativeMultCheck<uint64_t>( uint64_t( 1 ) << 63, 2, 1, 0 );

    // bits in upper half on one side, bits in lower half on other side
    CommutativeMultCheck<uint64_t>( 0xcdcd'dcdc'0000'0000,
                                    0x0000'0000'aeae'aeae,
                                    0x0000'0000'8c6e'5a77,
                                    0x7391'a588'0000'0000 );

    // Some input numbers without interesting patterns
    CommutativeMultCheck<uint64_t>( 0xaaaa'aaaa'aaaa'aaaa,
                                    0xbbbb'bbbb'bbbb'bbbb,
                                    0x7d27'd27d'27d2'7d26,
                                    0xd82d'82d8'2d82'd82e );

    CommutativeMultCheck<uint64_t>( 0x7d27'd27d'27d2'7d26,
                                    0xd82d'82d8'2d82'd82e,
                                    0x69af'd991'8256'b953,
                                    0x8724'8909'fcb6'8cd4 );

    CommutativeMultCheck<uint64_t>( 0xdead'beef'dead'beef,
                                    0xfeed'feed'feed'feef,
                                    0xddbf'680b'2b0c'b558,
                                    0x7a36'b06f'2ce9'6321 );

    CommutativeMultCheck<uint64_t>( 0xddbf'680b'2b0c'b558,
                                    0x7a36'b06f'2ce9'6321,
                                    0x69dc'96c9'294b'fc7f,
                                    0xd038'39fa'a3dc'6858 );

    CommutativeMultCheck<uint64_t>( 0x61c8'8646'80b5'83eb,
                                    0x61c8'8646'80b5'83eb,
                                    0x2559'92d3'8220'8bbe,
                                    0xdf44'2d22'ce48'59b9 );
}

TEST_CASE("extendedMult 64x64 - all implementations", "[integer][approvals]") {
    using Catch::Detail::extendedMult;
    using Catch::Detail::extendedMultPortable;
    using Catch::Detail::fillBitsFrom;

    std::random_device rng;
    for (size_t i = 0; i < 100; ++i) {
        auto a = fillBitsFrom<std::uint64_t>( rng );
        auto b = fillBitsFrom<std::uint64_t>( rng );
        CAPTURE( a, b );

        auto naive_ab = extendedMultNaive( a, b );

        REQUIRE( naive_ab == extendedMultNaive( b, a ) );
        REQUIRE( naive_ab == extendedMultPortable( a, b ) );
        REQUIRE( naive_ab == extendedMultPortable( b, a ) );
        REQUIRE( naive_ab == extendedMult( a, b ) );
        REQUIRE( naive_ab == extendedMult( b, a ) );
    }
}

TEST_CASE( "SizedUnsignedType helpers", "[integer][approvals]" ) {
    using Catch::Detail::SizedUnsignedType_t;
    using Catch::Detail::DoubleWidthUnsignedType_t;

    STATIC_REQUIRE( sizeof( SizedUnsignedType_t<1> ) == 1 );
    STATIC_REQUIRE( sizeof( SizedUnsignedType_t<2> ) == 2 );
    STATIC_REQUIRE( sizeof( SizedUnsignedType_t<4> ) == 4 );
    STATIC_REQUIRE( sizeof( SizedUnsignedType_t<8> ) == 8 );

    STATIC_REQUIRE( sizeof( DoubleWidthUnsignedType_t<std::uint8_t> ) == 2 );
    STATIC_REQUIRE( std::is_unsigned<DoubleWidthUnsignedType_t<std::uint8_t>>::value );
    STATIC_REQUIRE( sizeof( DoubleWidthUnsignedType_t<std::uint16_t> ) == 4 );
    STATIC_REQUIRE( std::is_unsigned<DoubleWidthUnsignedType_t<std::uint16_t>>::value );
    STATIC_REQUIRE( sizeof( DoubleWidthUnsignedType_t<std::uint32_t> ) == 8 );
    STATIC_REQUIRE( std::is_unsigned<DoubleWidthUnsignedType_t<std::uint32_t>>::value );
}

TEST_CASE( "extendedMult 32x32", "[integer][approvals]" ) {
    // a x 0 == 0
    CommutativeMultCheck<uint32_t>( 0x1234'5678, 0, 0, 0 );

    // bit carried from low half to upper half
    CommutativeMultCheck<uint32_t>( uint32_t(1) << 31, 2, 1, 0 );

    // bits in upper half on one side, bits in lower half on other side
    CommutativeMultCheck<uint32_t>( 0xdcdc'0000, 0x0000'aabb, 0x0000'934b, 0x6cb4'0000 );

    // Some input numbers without interesting patterns
    CommutativeMultCheck<uint32_t>(
        0xaaaa'aaaa, 0xbbbb'bbbb, 0x7d27'd27c, 0x2d82'd82e );

    CommutativeMultCheck<uint32_t>(
        0x7d27'd27c, 0x2d82'd82e, 0x163f'f7e8, 0xc5b8'7248 );

    CommutativeMultCheck<uint32_t>(
        0xdead'beef, 0xfeed'feed, 0xddbf'6809, 0x6f8d'e543 );

    CommutativeMultCheck<uint32_t>(
        0xddbf'6809, 0x6f8d'e543, 0x60a0'e71e, 0x751d'475b );
}

TEST_CASE( "extendedMult 8x8", "[integer][approvals]" ) {
    // a x 0 == 0
    CommutativeMultCheck<uint8_t>( 0xcd, 0, 0, 0 );

    // bit carried from low half to upper half
    CommutativeMultCheck<uint8_t>( uint8_t( 1 ) << 7, 2, 1, 0 );

    // bits in upper half on one side, bits in lower half on other side
    CommutativeMultCheck<uint8_t>( 0x80, 0x03, 0x01, 0x80 );

    // Some input numbers without interesting patterns
    CommutativeMultCheck<uint8_t>( 0xaa, 0xbb, 0x7c, 0x2e );
    CommutativeMultCheck<uint8_t>( 0x7c, 0x2e, 0x16, 0x48 );
    CommutativeMultCheck<uint8_t>( 0xdc, 0xcd, 0xb0, 0x2c );
    CommutativeMultCheck<uint8_t>( 0xb0, 0x2c, 0x1e, 0x40 );
}


TEST_CASE( "negative and positive signed integers keep their order after transposeToNaturalOrder",
                    "[integer][approvals]") {
    using Catch::Detail::transposeToNaturalOrder;
    int32_t negative( -1 );
    int32_t positive( 1 );
    uint32_t adjusted_negative =
        transposeToNaturalOrder<int32_t>( static_cast<uint32_t>( negative ) );
    uint32_t adjusted_positive =
        transposeToNaturalOrder<int32_t>( static_cast<uint32_t>( positive ) );
    REQUIRE( adjusted_negative < adjusted_positive );
    REQUIRE( adjusted_positive - adjusted_negative == 2 );

    // Conversion has to be reversible
    REQUIRE( negative == static_cast<int32_t>( transposeToNaturalOrder<int32_t>(
                             adjusted_negative ) ) );
    REQUIRE( positive == static_cast<int32_t>( transposeToNaturalOrder<int32_t>(
                             adjusted_positive ) ) );
}

TEST_CASE( "unsigned integers are unchanged by transposeToNaturalOrder",
           "[integer][approvals]") {
    using Catch::Detail::transposeToNaturalOrder;
    uint32_t max = std::numeric_limits<uint32_t>::max();
    uint32_t zero = 0;
    REQUIRE( max == transposeToNaturalOrder<uint32_t>( max ) );
    REQUIRE( zero == transposeToNaturalOrder<uint32_t>( zero ) );
}
