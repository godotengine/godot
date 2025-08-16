
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_FLOATING_POINT_HELPERS_HPP_INCLUDED
#define CATCH_FLOATING_POINT_HELPERS_HPP_INCLUDED

#include <catch2/internal/catch_polyfills.hpp>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <utility>
#include <limits>

namespace Catch {
    namespace Detail {

        uint32_t convertToBits(float f);
        uint64_t convertToBits(double d);

        // Used when we know we want == comparison of two doubles
        // to centralize warning suppression
        bool directCompare( float lhs, float rhs );
        bool directCompare( double lhs, double rhs );

    } // end namespace Detail



#if defined( __GNUC__ ) || defined( __clang__ )
#    pragma GCC diagnostic push
    // We do a bunch of direct compensations of floating point numbers,
    // because we know what we are doing and actually do want the direct
    // comparison behaviour.
#    pragma GCC diagnostic ignored "-Wfloat-equal"
#endif

    /**
     * Calculates the ULP distance between two floating point numbers
     *
     * The ULP distance of two floating point numbers is the count of
     * valid floating point numbers representable between them.
     *
     * There are some exceptions between how this function counts the
     * distance, and the interpretation of the standard as implemented.
     * by e.g. `nextafter`. For this function it always holds that:
     * * `(x == y) => ulpDistance(x, y) == 0` (so `ulpDistance(-0, 0) == 0`)
     * * `ulpDistance(maxFinite, INF) == 1`
     * * `ulpDistance(x, -x) == 2 * ulpDistance(x, 0)`
     *
     * \pre `!isnan( lhs )`
     * \pre `!isnan( rhs )`
     * \pre floating point numbers are represented in IEEE-754 format
     */
    template <typename FP>
    uint64_t ulpDistance( FP lhs, FP rhs ) {
        assert( std::numeric_limits<FP>::is_iec559 &&
            "ulpDistance assumes IEEE-754 format for floating point types" );
        assert( !Catch::isnan( lhs ) &&
                "Distance between NaN and number is not meaningful" );
        assert( !Catch::isnan( rhs ) &&
                "Distance between NaN and number is not meaningful" );

        // We want X == Y to imply 0 ULP distance even if X and Y aren't
        // bit-equal (-0 and 0), or X - Y != 0 (same sign infinities).
        if ( lhs == rhs ) { return 0; }

        // We need a properly typed positive zero for type inference.
        static constexpr FP positive_zero{};

        // We want to ensure that +/- 0 is always represented as positive zero
        if ( lhs == positive_zero ) { lhs = positive_zero; }
        if ( rhs == positive_zero ) { rhs = positive_zero; }

        // If arguments have different signs, we can handle them by summing
        // how far are they from 0 each.
        if ( std::signbit( lhs ) != std::signbit( rhs ) ) {
            return ulpDistance( std::abs( lhs ), positive_zero ) +
                   ulpDistance( std::abs( rhs ), positive_zero );
        }

        // When both lhs and rhs are of the same sign, we can just
        // read the numbers bitwise as integers, and then subtract them
        // (assuming IEEE).
        uint64_t lc = Detail::convertToBits( lhs );
        uint64_t rc = Detail::convertToBits( rhs );

        // The ulp distance between two numbers is symmetric, so to avoid
        // dealing with overflows we want the bigger converted number on the lhs
        if ( lc < rc ) {
            std::swap( lc, rc );
        }

        return lc - rc;
    }

#if defined( __GNUC__ ) || defined( __clang__ )
#    pragma GCC diagnostic pop
#endif


} // end namespace Catch

#endif // CATCH_FLOATING_POINT_HELPERS_HPP_INCLUDED
