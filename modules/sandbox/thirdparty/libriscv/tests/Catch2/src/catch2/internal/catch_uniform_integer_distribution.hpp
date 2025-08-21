
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#ifndef CATCH_UNIFORM_INTEGER_DISTRIBUTION_HPP_INCLUDED
#define CATCH_UNIFORM_INTEGER_DISTRIBUTION_HPP_INCLUDED

#include <catch2/internal/catch_random_integer_helpers.hpp>

namespace Catch {

/**
 * Implementation of uniform distribution on integers.
 *
 * Unlike `std::uniform_int_distribution`, this implementation supports
 * various 1 byte integral types, including bool (but you should not
 * actually use it for bools).
 *
 * The underlying algorithm is based on the one described in "Fast Random
 * Integer Generation in an Interval" by Daniel Lemire, but has been
 * optimized under the assumption of reuse of the same distribution object.
 */
template <typename IntegerType>
class uniform_integer_distribution {
    static_assert(std::is_integral<IntegerType>::value, "...");

    using UnsignedIntegerType = Detail::SizedUnsignedType_t<sizeof(IntegerType)>;

    // Only the left bound is stored, and we store it converted to its
    // unsigned image. This avoids having to do the conversions inside
    // the operator(), at the cost of having to do the conversion in
    // the a() getter. The right bound is only needed in the b() getter,
    // so we recompute it there from other stored data.
    UnsignedIntegerType m_a;

    // How many different values are there in [a, b]. a == b => 1, can be 0 for distribution over all values in the type.
    UnsignedIntegerType m_ab_distance;

    // We hoisted this out of the main generation function. Technically,
    // this means that using this distribution will be slower than Lemire's
    // algorithm if this distribution instance will be used only few times,
    // but it will be faster if it is used many times. Since Catch2 uses
    // distributions only to implement random generators, we assume that each
    // distribution will be reused many times and this is an optimization.
    UnsignedIntegerType m_rejection_threshold = 0;

    static constexpr UnsignedIntegerType computeDistance(IntegerType a, IntegerType b) {
        // This overflows and returns 0 if a == 0 and b == TYPE_MAX.
        // We handle that later when generating the number.
        return transposeTo(b) - transposeTo(a) + 1;
    }

    static constexpr UnsignedIntegerType computeRejectionThreshold(UnsignedIntegerType ab_distance) {
        // distance == 0 means that we will return all possible values from
        // the type's range, and that we shouldn't reject anything.
        if ( ab_distance == 0 ) { return 0; }
        return ( ~ab_distance + 1 ) % ab_distance;
    }

    static constexpr UnsignedIntegerType transposeTo(IntegerType in) {
        return Detail::transposeToNaturalOrder<IntegerType>(
            static_cast<UnsignedIntegerType>( in ) );
    }
    static constexpr IntegerType transposeBack(UnsignedIntegerType in) {
        return static_cast<IntegerType>(
            Detail::transposeToNaturalOrder<IntegerType>(in) );
    }

public:
    using result_type = IntegerType;

    constexpr uniform_integer_distribution( IntegerType a, IntegerType b ):
        m_a( transposeTo(a) ),
        m_ab_distance( computeDistance(a, b) ),
        m_rejection_threshold( computeRejectionThreshold(m_ab_distance) ) {
        assert( a <= b );
    }

    template <typename Generator>
    constexpr result_type operator()( Generator& g ) {
        // All possible values of result_type are valid.
        if ( m_ab_distance == 0 ) {
            return transposeBack( Detail::fillBitsFrom<UnsignedIntegerType>( g ) );
        }

        auto random_number = Detail::fillBitsFrom<UnsignedIntegerType>( g );
        auto emul = Detail::extendedMult( random_number, m_ab_distance );
        // Unlike Lemire's algorithm we skip the ab_distance check, since
        // we precomputed the rejection threshold, which is always tighter.
        while (emul.lower < m_rejection_threshold) {
            random_number = Detail::fillBitsFrom<UnsignedIntegerType>( g );
            emul = Detail::extendedMult( random_number, m_ab_distance );
        }

        return transposeBack(m_a + emul.upper);
    }

    constexpr result_type a() const { return transposeBack(m_a); }
    constexpr result_type b() const { return transposeBack(m_ab_distance + m_a - 1); }
};

} // end namespace Catch

#endif // CATCH_UNIFORM_INTEGER_DISTRIBUTION_HPP_INCLUDED
