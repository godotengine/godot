
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#ifndef CATCH_RANDOM_FLOATING_POINT_HELPERS_HPP_INCLUDED
#define CATCH_RANDOM_FLOATING_POINT_HELPERS_HPP_INCLUDED

#include <catch2/internal/catch_polyfills.hpp>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace Catch {

    namespace Detail {
        /**
         * Returns the largest magnitude of 1-ULP distance inside the [a, b] range.
         *
         * Assumes `a < b`.
         */
        template <typename FloatType>
        FloatType gamma(FloatType a, FloatType b) {
            static_assert( std::is_floating_point<FloatType>::value,
                           "gamma returns the largest ULP magnitude within "
                           "floating point range [a, b]. This only makes sense "
                           "for floating point types" );
            assert( a <= b );

            const auto gamma_up = Catch::nextafter( a, std::numeric_limits<FloatType>::infinity() ) - a;
            const auto gamma_down = b - Catch::nextafter( b, -std::numeric_limits<FloatType>::infinity() );

            return gamma_up < gamma_down ? gamma_down : gamma_up;
        }

        template <typename FloatingPoint>
        struct DistanceTypePicker;
        template <>
        struct DistanceTypePicker<float> {
            using type = std::uint32_t;
        };
        template <>
        struct DistanceTypePicker<double> {
            using type = std::uint64_t;
        };

        template <typename T>
        using DistanceType = typename DistanceTypePicker<T>::type;

#if defined( __GNUC__ ) || defined( __clang__ )
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wfloat-equal"
#endif
        /**
         * Computes the number of equi-distant floats in [a, b]
         *
         * Since not every range can be split into equidistant floats
         * exactly, we actually compute ceil(b/distance - a/distance),
         * because in those cases we want to overcount.
         *
         * Uses modified Dekker's FastTwoSum algorithm to handle rounding.
         */
        template <typename FloatType>
        DistanceType<FloatType>
        count_equidistant_floats( FloatType a, FloatType b, FloatType distance ) {
            assert( a <= b );
            // We get distance as gamma for our uniform float distribution,
            // so this will round perfectly.
            const auto ag = a / distance;
            const auto bg = b / distance;

            const auto s = bg - ag;
            const auto err = ( std::fabs( a ) <= std::fabs( b ) )
                                 ? -ag - ( s - bg )
                                 : bg - ( s + ag );
            const auto ceil_s = static_cast<DistanceType<FloatType>>( std::ceil( s ) );

            return ( ceil_s != s ) ? ceil_s : ceil_s + ( err > 0 );
        }
#if defined( __GNUC__ ) || defined( __clang__ )
#    pragma GCC diagnostic pop
#endif

    }

} // end namespace Catch

#endif // CATCH_RANDOM_FLOATING_POINT_HELPERS_HPP_INCLUDED
