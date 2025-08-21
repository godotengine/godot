
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#ifndef CATCH_UNIFORM_FLOATING_POINT_DISTRIBUTION_HPP_INCLUDED
#define CATCH_UNIFORM_FLOATING_POINT_DISTRIBUTION_HPP_INCLUDED

#include <catch2/internal/catch_random_floating_point_helpers.hpp>
#include <catch2/internal/catch_uniform_integer_distribution.hpp>

#include <cmath>
#include <type_traits>

namespace Catch {

    namespace Detail {
#if defined( __GNUC__ ) || defined( __clang__ )
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wfloat-equal"
#endif
        // The issue with overflow only happens with maximal ULP and HUGE
        // distance, e.g. when generating numbers in [-inf, inf] for given
        // type. So we only check for the largest possible ULP in the
        // type, and return something that does not overflow to inf in 1 mult.
        constexpr std::uint64_t calculate_max_steps_in_one_go(double gamma) {
            if ( gamma == 1.99584030953472e+292 ) { return 9007199254740991; }
            return static_cast<std::uint64_t>( -1 );
        }
        constexpr std::uint32_t calculate_max_steps_in_one_go(float gamma) {
            if ( gamma == 2.028241e+31f ) { return 16777215; }
            return static_cast<std::uint32_t>( -1 );
        }
#if defined( __GNUC__ ) || defined( __clang__ )
#    pragma GCC diagnostic pop
#endif
    }

/**
 * Implementation of uniform distribution on floating point numbers.
 *
 * Note that we support only `float` and `double` types, because these
 * usually mean the same thing across different platform. `long double`
 * varies wildly by platform and thus we cannot provide reproducible
 * implementation. Also note that we don't implement all parts of
 * distribution per standard: this distribution is not serializable, nor
 * can the range be arbitrarily reset.
 *
 * The implementation also uses different approach than the one taken by
 * `std::uniform_real_distribution`, where instead of generating a number
 * between [0, 1) and then multiplying the range bounds with it, we first
 * split the [a, b] range into a set of equidistributed floating point
 * numbers, and then use uniform int distribution to pick which one to
 * return.
 *
 * This has the advantage of guaranteeing uniformity (the multiplication
 * method loses uniformity due to rounding when multiplying floats), except
 * for small non-uniformity at one side of the interval, where we have
 * to deal with the fact that not every interval is splittable into
 * equidistributed floats.
 *
 * Based on "Drawing random floating-point numbers from an interval" by
 * Frederic Goualard.
 */
template <typename FloatType>
class uniform_floating_point_distribution {
    static_assert(std::is_floating_point<FloatType>::value, "...");
    static_assert(!std::is_same<FloatType, long double>::value,
                  "We do not support long double due to inconsistent behaviour between platforms");

    using WidthType = Detail::DistanceType<FloatType>;

    FloatType m_a, m_b;
    FloatType m_ulp_magnitude;
    WidthType m_floats_in_range;
    uniform_integer_distribution<WidthType> m_int_dist;

    // In specific cases, we can overflow into `inf` when computing the
    // `steps * g` offset. To avoid this, we don't offset by more than this
    // in one multiply + addition.
    WidthType m_max_steps_in_one_go;
    // We don't want to do the magnitude check every call to `operator()`
    bool m_a_has_leq_magnitude;

public:
    using result_type = FloatType;

    uniform_floating_point_distribution( FloatType a, FloatType b ):
        m_a( a ),
        m_b( b ),
        m_ulp_magnitude( Detail::gamma( m_a, m_b ) ),
        m_floats_in_range( Detail::count_equidistant_floats( m_a, m_b, m_ulp_magnitude ) ),
        m_int_dist(0, m_floats_in_range),
        m_max_steps_in_one_go( Detail::calculate_max_steps_in_one_go(m_ulp_magnitude)),
        m_a_has_leq_magnitude(std::fabs(m_a) <= std::fabs(m_b))
    {
        assert( a <= b );
    }

    template <typename Generator>
    result_type operator()( Generator& g ) {
        WidthType steps = m_int_dist( g );
        if ( m_a_has_leq_magnitude ) {
            if ( steps == m_floats_in_range ) { return m_a; }
            auto b = m_b;
            while (steps > m_max_steps_in_one_go) {
                b -= m_max_steps_in_one_go * m_ulp_magnitude;
                steps -= m_max_steps_in_one_go;
            }
            return b - steps * m_ulp_magnitude;
        } else {
            if ( steps == m_floats_in_range ) { return m_b; }
            auto a = m_a;
            while (steps > m_max_steps_in_one_go) {
                a += m_max_steps_in_one_go * m_ulp_magnitude;
                steps -= m_max_steps_in_one_go;
            }
            return a + steps * m_ulp_magnitude;
        }
    }

    result_type a() const { return m_a; }
    result_type b() const { return m_b; }
};

} // end namespace Catch

#endif // CATCH_UNIFORM_FLOATING_POINT_DISTRIBUTION_HPP_INCLUDED
