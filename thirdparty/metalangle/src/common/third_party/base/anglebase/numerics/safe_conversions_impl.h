// Copyright 2014 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef ANGLEBASE_NUMERICS_SAFE_CONVERSIONS_IMPL_H_
#define ANGLEBASE_NUMERICS_SAFE_CONVERSIONS_IMPL_H_

#include <limits.h>
#include <stdint.h>

#include <climits>
#include <limits>

namespace angle
{

namespace base
{
namespace internal
{

// The std library doesn't provide a binary max_exponent for integers, however
// we can compute one by adding one to the number of non-sign bits. This allows
// for accurate range comparisons between floating point and integer types.
template <typename NumericType>
struct MaxExponent
{
    static_assert(std::is_arithmetic<NumericType>::value, "Argument must be numeric.");
    static const int value =
        std::numeric_limits<NumericType>::is_iec559
            ? std::numeric_limits<NumericType>::max_exponent
            : (sizeof(NumericType) * CHAR_BIT + 1 - std::numeric_limits<NumericType>::is_signed);
};

enum IntegerRepresentation
{
    INTEGER_REPRESENTATION_UNSIGNED,
    INTEGER_REPRESENTATION_SIGNED
};

// A range for a given nunmeric Src type is contained for a given numeric Dst
// type if both numeric_limits<Src>::max() <= numeric_limits<Dst>::max() and
// numeric_limits<Src>::min() >= numeric_limits<Dst>::min() are true.
// We implement this as template specializations rather than simple static
// comparisons to ensure type correctness in our comparisons.
enum NumericRangeRepresentation
{
    NUMERIC_RANGE_NOT_CONTAINED,
    NUMERIC_RANGE_CONTAINED
};

// Helper templates to statically determine if our destination type can contain
// maximum and minimum values represented by the source type.

template <typename Dst,
          typename Src,
          IntegerRepresentation DstSign = std::numeric_limits<Dst>::is_signed
                                              ? INTEGER_REPRESENTATION_SIGNED
                                              : INTEGER_REPRESENTATION_UNSIGNED,
          IntegerRepresentation SrcSign = std::numeric_limits<Src>::is_signed
                                              ? INTEGER_REPRESENTATION_SIGNED
                                              : INTEGER_REPRESENTATION_UNSIGNED>
struct StaticDstRangeRelationToSrcRange;

// Same sign: Dst is guaranteed to contain Src only if its range is equal or
// larger.
template <typename Dst, typename Src, IntegerRepresentation Sign>
struct StaticDstRangeRelationToSrcRange<Dst, Src, Sign, Sign>
{
    static const NumericRangeRepresentation value =
        MaxExponent<Dst>::value >= MaxExponent<Src>::value ? NUMERIC_RANGE_CONTAINED
                                                           : NUMERIC_RANGE_NOT_CONTAINED;
};

// Unsigned to signed: Dst is guaranteed to contain source only if its range is
// larger.
template <typename Dst, typename Src>
struct StaticDstRangeRelationToSrcRange<Dst,
                                        Src,
                                        INTEGER_REPRESENTATION_SIGNED,
                                        INTEGER_REPRESENTATION_UNSIGNED>
{
    static const NumericRangeRepresentation value =
        MaxExponent<Dst>::value > MaxExponent<Src>::value ? NUMERIC_RANGE_CONTAINED
                                                          : NUMERIC_RANGE_NOT_CONTAINED;
};

// Signed to unsigned: Dst cannot be statically determined to contain Src.
template <typename Dst, typename Src>
struct StaticDstRangeRelationToSrcRange<Dst,
                                        Src,
                                        INTEGER_REPRESENTATION_UNSIGNED,
                                        INTEGER_REPRESENTATION_SIGNED>
{
    static const NumericRangeRepresentation value = NUMERIC_RANGE_NOT_CONTAINED;
};

enum RangeConstraint : unsigned char
{
    RANGE_VALID     = 0x0,  // Value can be represented by the destination type.
    RANGE_UNDERFLOW = 0x1,  // Value would overflow.
    RANGE_OVERFLOW  = 0x2,  // Value would underflow.
    RANGE_INVALID   = RANGE_UNDERFLOW | RANGE_OVERFLOW  // Invalid (i.e. NaN).
};

// Helper function for coercing an int back to a RangeContraint.
constexpr RangeConstraint GetRangeConstraint(int integer_range_constraint)
{
    // TODO(jschuh): Once we get full C++14 support we want this
    // assert(integer_range_constraint >= RANGE_VALID &&
    //        integer_range_constraint <= RANGE_INVALID)
    return static_cast<RangeConstraint>(integer_range_constraint);
}

// This function creates a RangeConstraint from an upper and lower bound
// check by taking advantage of the fact that only NaN can be out of range in
// both directions at once.
constexpr inline RangeConstraint GetRangeConstraint(bool is_in_upper_bound, bool is_in_lower_bound)
{
    return GetRangeConstraint((is_in_upper_bound ? 0 : RANGE_OVERFLOW) |
                              (is_in_lower_bound ? 0 : RANGE_UNDERFLOW));
}

// The following helper template addresses a corner case in range checks for
// conversion from a floating-point type to an integral type of smaller range
// but larger precision (e.g. float -> unsigned). The problem is as follows:
//   1. Integral maximum is always one less than a power of two, so it must be
//      truncated to fit the mantissa of the floating point. The direction of
//      rounding is implementation defined, but by default it's always IEEE
//      floats, which round to nearest and thus result in a value of larger
//      magnitude than the integral value.
//      Example: float f = UINT_MAX; // f is 4294967296f but UINT_MAX
//                                   // is 4294967295u.
//   2. If the floating point value is equal to the promoted integral maximum
//      value, a range check will erroneously pass.
//      Example: (4294967296f <= 4294967295u) // This is true due to a precision
//                                            // loss in rounding up to float.
//   3. When the floating point value is then converted to an integral, the
//      resulting value is out of range for the target integral type and
//      thus is implementation defined.
//      Example: unsigned u = (float)INT_MAX; // u will typically overflow to 0.
// To fix this bug we manually truncate the maximum value when the destination
// type is an integral of larger precision than the source floating-point type,
// such that the resulting maximum is represented exactly as a floating point.
template <typename Dst, typename Src>
struct NarrowingRange
{
    typedef typename std::numeric_limits<Src> SrcLimits;
    typedef typename std::numeric_limits<Dst> DstLimits;
    // The following logic avoids warnings where the max function is
    // instantiated with invalid values for a bit shift (even though
    // such a function can never be called).
    static const int shift =
        (MaxExponent<Src>::value > MaxExponent<Dst>::value &&
         SrcLimits::digits < DstLimits::digits && SrcLimits::is_iec559 && DstLimits::is_integer)
            ? (DstLimits::digits - SrcLimits::digits)
            : 0;

    static constexpr Dst max()
    {
        // We use UINTMAX_C below to avoid compiler warnings about shifting floating
        // points. Since it's a compile time calculation, it shouldn't have any
        // performance impact.
        return DstLimits::max() - static_cast<Dst>((UINTMAX_C(1) << shift) - 1);
    }

    static constexpr Dst min()
    {
        return std::numeric_limits<Dst>::is_iec559 ? -DstLimits::max() : DstLimits::min();
    }
};

template <typename Dst,
          typename Src,
          IntegerRepresentation DstSign = std::numeric_limits<Dst>::is_signed
                                              ? INTEGER_REPRESENTATION_SIGNED
                                              : INTEGER_REPRESENTATION_UNSIGNED,
          IntegerRepresentation SrcSign = std::numeric_limits<Src>::is_signed
                                              ? INTEGER_REPRESENTATION_SIGNED
                                              : INTEGER_REPRESENTATION_UNSIGNED,
          NumericRangeRepresentation DstRange = StaticDstRangeRelationToSrcRange<Dst, Src>::value>
struct DstRangeRelationToSrcRangeImpl;

// The following templates are for ranges that must be verified at runtime. We
// split it into checks based on signedness to avoid confusing casts and
// compiler warnings on signed an unsigned comparisons.

// Dst range is statically determined to contain Src: Nothing to check.
template <typename Dst, typename Src, IntegerRepresentation DstSign, IntegerRepresentation SrcSign>
struct DstRangeRelationToSrcRangeImpl<Dst, Src, DstSign, SrcSign, NUMERIC_RANGE_CONTAINED>
{
    static constexpr RangeConstraint Check(Src value) { return RANGE_VALID; }
};

// Signed to signed narrowing: Both the upper and lower boundaries may be
// exceeded.
template <typename Dst, typename Src>
struct DstRangeRelationToSrcRangeImpl<Dst,
                                      Src,
                                      INTEGER_REPRESENTATION_SIGNED,
                                      INTEGER_REPRESENTATION_SIGNED,
                                      NUMERIC_RANGE_NOT_CONTAINED>
{
    static constexpr RangeConstraint Check(Src value)
    {
        return GetRangeConstraint((value <= NarrowingRange<Dst, Src>::max()),
                                  (value >= NarrowingRange<Dst, Src>::min()));
    }
};

// Unsigned to unsigned narrowing: Only the upper boundary can be exceeded.
template <typename Dst, typename Src>
struct DstRangeRelationToSrcRangeImpl<Dst,
                                      Src,
                                      INTEGER_REPRESENTATION_UNSIGNED,
                                      INTEGER_REPRESENTATION_UNSIGNED,
                                      NUMERIC_RANGE_NOT_CONTAINED>
{
    static constexpr RangeConstraint Check(Src value)
    {
        return GetRangeConstraint(value <= NarrowingRange<Dst, Src>::max(), true);
    }
};

// Unsigned to signed: The upper boundary may be exceeded.
template <typename Dst, typename Src>
struct DstRangeRelationToSrcRangeImpl<Dst,
                                      Src,
                                      INTEGER_REPRESENTATION_SIGNED,
                                      INTEGER_REPRESENTATION_UNSIGNED,
                                      NUMERIC_RANGE_NOT_CONTAINED>
{
    static constexpr RangeConstraint Check(Src value)
    {
        return sizeof(Dst) > sizeof(Src)
                   ? RANGE_VALID
                   : GetRangeConstraint(value <= static_cast<Src>(NarrowingRange<Dst, Src>::max()),
                                        true);
    }
};

// Signed to unsigned: The upper boundary may be exceeded for a narrower Dst,
// and any negative value exceeds the lower boundary.
template <typename Dst, typename Src>
struct DstRangeRelationToSrcRangeImpl<Dst,
                                      Src,
                                      INTEGER_REPRESENTATION_UNSIGNED,
                                      INTEGER_REPRESENTATION_SIGNED,
                                      NUMERIC_RANGE_NOT_CONTAINED>
{
    static constexpr RangeConstraint Check(Src value)
    {
        return (MaxExponent<Dst>::value >= MaxExponent<Src>::value)
                   ? GetRangeConstraint(true, value >= static_cast<Src>(0))
                   : GetRangeConstraint(value <= static_cast<Src>(NarrowingRange<Dst, Src>::max()),
                                        value >= static_cast<Src>(0));
    }
};

template <typename Dst, typename Src>
constexpr RangeConstraint DstRangeRelationToSrcRange(Src value)
{
    static_assert(std::numeric_limits<Src>::is_specialized, "Argument must be numeric.");
    static_assert(std::numeric_limits<Dst>::is_specialized, "Result must be numeric.");
    return DstRangeRelationToSrcRangeImpl<Dst, Src>::Check(value);
}

}  // namespace internal
}  // namespace base

}  // namespace angle

#endif  // ANGLEBASE_NUMERICS_SAFE_CONVERSIONS_IMPL_H_
