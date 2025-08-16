// This file is part of the FidelityFX SDK.
//
// Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

/// A define for a true value in a boolean expression.
///
/// @ingroup GPU
#define FFX_TRUE (true)

/// A define for a false value in a boolean expression.
///
/// @ingroup GPU
#define FFX_FALSE (false)

/// A define value for positive infinity.
///
/// @ingroup GPU
#define FFX_POSITIVE_INFINITY_FLOAT ffxAsFloat(0x7f800000u)

/// A define value for negative infinity.
///
/// @ingroup GPU
#define FFX_NEGATIVE_INFINITY_FLOAT ffxAsFloat(0xff800000u)

/// A define value for PI.
/// 
/// @ingroup GPU
#define FFX_PI  (3.14159)


/// Compute the reciprocal of <c><i>value</i></c>.
///
/// @param [in] value               The value to compute the reciprocal of.
///
/// @returns
/// The 1 / <c><i>value</i></c>.
///
/// @ingroup GPU
FfxFloat32 ffxReciprocal(FfxFloat32 value)
{
    return rcp(value);
}

/// Compute the reciprocal of <c><i>value</i></c>.
///
/// @param [in] value               The value to compute the reciprocal of.
///
/// @returns
/// The 1 / <c><i>value</i></c>.
///
/// @ingroup GPU
FfxFloat32x2 ffxReciprocal(FfxFloat32x2 value)
{
    return rcp(value);
}

/// Compute the reciprocal of <c><i>value</i></c>.
///
/// @param [in] value               The value to compute the reciprocal of.
///
/// @returns
/// The 1 / <c><i>value</i></c>.
///
/// @ingroup GPU
FfxFloat32x3 ffxReciprocal(FfxFloat32x3 value)
{
    return rcp(value);
}

/// Compute the reciprocal of <c><i>value</i></c>.
///
/// @param [in] value               The value to compute the reciprocal of.
///
/// @returns
/// The 1 / <c><i>value</i></c>.
///
/// @ingroup GPU
FfxFloat32x4 ffxReciprocal(FfxFloat32x4 value)
{
    return rcp(value);
}

/// Compute the min of two values.
///
/// @param [in] x                   The first value to compute the min of.
/// @param [in] y                   The second value to compute the min of.
///
/// @returns
/// The the lowest of two values.
///
/// @ingroup GPU
FfxFloat32 ffxMin(FfxFloat32 x, FfxFloat32 y)
{
    return min(x, y);
}

/// Compute the min of two values.
///
/// @param [in] x                   The first value to compute the min of.
/// @param [in] y                   The second value to compute the min of.
///
/// @returns
/// The the lowest of two values.
///
/// @ingroup GPU
FfxFloat32x2 ffxMin(FfxFloat32x2 x, FfxFloat32x2 y)
{
    return min(x, y);
}

/// Compute the min of two values.
///
/// @param [in] x                   The first value to compute the min of.
/// @param [in] y                   The second value to compute the min of.
///
/// @returns
/// The the lowest of two values.
///
/// @ingroup GPU
FfxFloat32x3 ffxMin(FfxFloat32x3 x, FfxFloat32x3 y)
{
    return min(x, y);
}

/// Compute the min of two values.
///
/// @param [in] x                   The first value to compute the min of.
/// @param [in] y                   The second value to compute the min of.
///
/// @returns
/// The the lowest of two values.
///
/// @ingroup GPU
FfxFloat32x4 ffxMin(FfxFloat32x4 x, FfxFloat32x4 y)
{
    return min(x, y);
}

/// Compute the min of two values.
///
/// @param [in] x                   The first value to compute the min of.
/// @param [in] y                   The second value to compute the min of.
///
/// @returns
/// The the lowest of two values.
///
/// @ingroup GPU
FfxInt32 ffxMin(FfxInt32 x, FfxInt32 y)
{
    return min(x, y);
}

/// Compute the min of two values.
///
/// @param [in] x                   The first value to compute the min of.
/// @param [in] y                   The second value to compute the min of.
///
/// @returns
/// The the lowest of two values.
///
/// @ingroup GPU
FfxInt32x2 ffxMin(FfxInt32x2 x, FfxInt32x2 y)
{
    return min(x, y);
}

/// Compute the min of two values.
///
/// @param [in] x                   The first value to compute the min of.
/// @param [in] y                   The second value to compute the min of.
///
/// @returns
/// The the lowest of two values.
///
/// @ingroup GPU
FfxInt32x3 ffxMin(FfxInt32x3 x, FfxInt32x3 y)
{
    return min(x, y);
}

/// Compute the min of two values.
///
/// @param [in] x                   The first value to compute the min of.
/// @param [in] y                   The second value to compute the min of.
///
/// @returns
/// The the lowest of two values.
///
/// @ingroup GPU
FfxInt32x4 ffxMin(FfxInt32x4 x, FfxInt32x4 y)
{
    return min(x, y);
}

/// Compute the min of two values.
///
/// @param [in] x                   The first value to compute the min of.
/// @param [in] y                   The second value to compute the min of.
///
/// @returns
/// The the lowest of two values.
///
/// @ingroup GPU
FfxUInt32 ffxMin(FfxUInt32 x, FfxUInt32 y)
{
    return min(x, y);
}

/// Compute the min of two values.
///
/// @param [in] x                   The first value to compute the min of.
/// @param [in] y                   The second value to compute the min of.
///
/// @returns
/// The the lowest of two values.
///
/// @ingroup GPU
FfxUInt32x2 ffxMin(FfxUInt32x2 x, FfxUInt32x2 y)
{
    return min(x, y);
}

/// Compute the min of two values.
///
/// @param [in] x                   The first value to compute the min of.
/// @param [in] y                   The second value to compute the min of.
///
/// @returns
/// The the lowest of two values.
///
/// @ingroup GPU
FfxUInt32x3 ffxMin(FfxUInt32x3 x, FfxUInt32x3 y)
{
    return min(x, y);
}

/// Compute the min of two values.
///
/// @param [in] x                   The first value to compute the min of.
/// @param [in] y                   The second value to compute the min of.
///
/// @returns
/// The the lowest of two values.
///
/// @ingroup GPU
FfxUInt32x4 ffxMin(FfxUInt32x4 x, FfxUInt32x4 y)
{
    return min(x, y);
}

/// Compute the max of two values.
///
/// @param [in] x                   The first value to compute the max of.
/// @param [in] y                   The second value to compute the max of.
///
/// @returns
/// The the lowest of two values.
///
/// @ingroup GPU
FfxFloat32 ffxMax(FfxFloat32 x, FfxFloat32 y)
{
    return max(x, y);
}

/// Compute the max of two values.
///
/// @param [in] x                   The first value to compute the max of.
/// @param [in] y                   The second value to compute the max of.
///
/// @returns
/// The the lowest of two values.
///
/// @ingroup GPU
FfxFloat32x2 ffxMax(FfxFloat32x2 x, FfxFloat32x2 y)
{
    return max(x, y);
}

/// Compute the max of two values.
///
/// @param [in] x                   The first value to compute the max of.
/// @param [in] y                   The second value to compute the max of.
///
/// @returns
/// The the lowest of two values.
///
/// @ingroup GPU
FfxFloat32x3 ffxMax(FfxFloat32x3 x, FfxFloat32x3 y)
{
    return max(x, y);
}

/// Compute the max of two values.
///
/// @param [in] x                   The first value to compute the max of.
/// @param [in] y                   The second value to compute the max of.
///
/// @returns
/// The the lowest of two values.
///
/// @ingroup GPU
FfxFloat32x4 ffxMax(FfxFloat32x4 x, FfxFloat32x4 y)
{
    return max(x, y);
}

/// Compute the max of two values.
///
/// @param [in] x                   The first value to compute the max of.
/// @param [in] y                   The second value to compute the max of.
///
/// @returns
/// The the lowest of two values.
///
/// @ingroup GPU
FfxInt32 ffxMax(FfxInt32 x, FfxInt32 y)
{
    return max(x, y);
}

/// Compute the max of two values.
///
/// @param [in] x                   The first value to compute the max of.
/// @param [in] y                   The second value to compute the max of.
///
/// @returns
/// The the lowest of two values.
///
/// @ingroup GPU
FfxInt32x2 ffxMax(FfxInt32x2 x, FfxInt32x2 y)
{
    return max(x, y);
}

/// Compute the max of two values.
///
/// @param [in] x                   The first value to compute the max of.
/// @param [in] y                   The second value to compute the max of.
///
/// @returns
/// The the lowest of two values.
///
/// @ingroup GPU
FfxInt32x3 ffxMax(FfxInt32x3 x, FfxInt32x3 y)
{
    return max(x, y);
}

/// Compute the max of two values.
///
/// @param [in] x                   The first value to compute the max of.
/// @param [in] y                   The second value to compute the max of.
///
/// @returns
/// The the lowest of two values.
///
/// @ingroup GPU
FfxInt32x4 ffxMax(FfxInt32x4 x, FfxInt32x4 y)
{
    return max(x, y);
}

/// Compute the max of two values.
///
/// @param [in] x                   The first value to compute the max of.
/// @param [in] y                   The second value to compute the max of.
///
/// @returns
/// The the lowest of two values.
///
/// @ingroup GPU
FfxUInt32 ffxMax(FfxUInt32 x, FfxUInt32 y)
{
    return max(x, y);
}

/// Compute the max of two values.
///
/// @param [in] x                   The first value to compute the max of.
/// @param [in] y                   The second value to compute the max of.
///
/// @returns
/// The the lowest of two values.
///
/// @ingroup GPU
FfxUInt32x2 ffxMax(FfxUInt32x2 x, FfxUInt32x2 y)
{
    return max(x, y);
}

/// Compute the max of two values.
///
/// @param [in] x                   The first value to compute the max of.
/// @param [in] y                   The second value to compute the max of.
///
/// @returns
/// The the lowest of two values.
///
/// @ingroup GPU
FfxUInt32x3 ffxMax(FfxUInt32x3 x, FfxUInt32x3 y)
{
    return max(x, y);
}

/// Compute the max of two values.
///
/// @param [in] x                   The first value to compute the max of.
/// @param [in] y                   The second value to compute the max of.
///
/// @returns
/// The the lowest of two values.
///
/// @ingroup GPU
FfxUInt32x4 ffxMax(FfxUInt32x4 x, FfxUInt32x4 y)
{
    return max(x, y);
}

/// Compute the value of the first parameter raised to the power of the second.
///
/// @param [in] x                   The value to raise to the power y.
/// @param [in] y                   The power to which to raise x.
///
/// @returns
/// The value of the first parameter raised to the power of the second.
///
/// @ingroup GPU
FfxFloat32 ffxPow(FfxFloat32 x, FfxFloat32 y)
{
    return pow(x, y);
}

/// Compute the value of the first parameter raised to the power of the second.
///
/// @param [in] x                   The value to raise to the power y.
/// @param [in] y                   The power to which to raise x.
///
/// @returns
/// The value of the first parameter raised to the power of the second.
///
/// @ingroup GPU
FfxFloat32x2 ffxPow(FfxFloat32x2 x, FfxFloat32x2 y)
{
    return pow(x, y);
}

/// Compute the value of the first parameter raised to the power of the second.
///
/// @param [in] x                   The value to raise to the power y.
/// @param [in] y                   The power to which to raise x.
///
/// @returns
/// The value of the first parameter raised to the power of the second.
///
/// @ingroup GPU
FfxFloat32x3 ffxPow(FfxFloat32x3 x, FfxFloat32x3 y)
{
    return pow(x, y);
}

/// Compute the value of the first parameter raised to the power of the second.
///
/// @param [in] x                   The value to raise to the power y.
/// @param [in] y                   The power to which to raise x.
///
/// @returns
/// The value of the first parameter raised to the power of the second.
///
/// @ingroup GPU
FfxFloat32x4 ffxPow(FfxFloat32x4 x, FfxFloat32x4 y)
{
    return pow(x, y);
}

/// Compute the square root of a value.
///
/// @param [in] x                   The first value to compute the min of.
///
/// @returns
/// The the square root of <c><i>x</i></c>.
///
/// @ingroup GPU
FfxFloat32 ffxSqrt(FfxFloat32 x)
{
    return sqrt(x);
}

/// Compute the square root of a value.
///
/// @param [in] x                   The first value to compute the min of.
///
/// @returns
/// The the square root of <c><i>x</i></c>.
///
/// @ingroup GPU
FfxFloat32x2 ffxSqrt(FfxFloat32x2 x)
{
    return sqrt(x);
}

/// Compute the square root of a value.
///
/// @param [in] x                   The first value to compute the min of.
///
/// @returns
/// The the square root of <c><i>x</i></c>.
///
/// @ingroup GPU
FfxFloat32x3 ffxSqrt(FfxFloat32x3 x)
{
    return sqrt(x);
}

/// Compute the square root of a value.
///
/// @param [in] x                   The first value to compute the min of.
///
/// @returns
/// The the square root of <c><i>x</i></c>.
///
/// @ingroup GPU
FfxFloat32x4 ffxSqrt(FfxFloat32x4 x)
{
    return sqrt(x);
}

/// Copy the sign bit from 's' to positive 'd'.
///
/// @param [in] d                   The value to copy the sign bit into.
/// @param [in] s                   The value to copy the sign bit from.
/// 
/// @returns
/// The value of <c><i>d</i></c> with the sign bit from <c><i>s</i></c>.
/// 
/// @ingroup GPU
FfxFloat32 ffxCopySignBit(FfxFloat32 d, FfxFloat32 s)
{
    return ffxAsFloat(ffxAsUInt32(d) | (ffxAsUInt32(s) & FfxUInt32(0x80000000u)));
}

/// Copy the sign bit from 's' to positive 'd'.
///
/// @param [in] d                   The value to copy the sign bit into.
/// @param [in] s                   The value to copy the sign bit from.
///
/// @returns
/// The value of <c><i>d</i></c> with the sign bit from <c><i>s</i></c>.
///
/// @ingroup GPU
FfxFloat32x2 ffxCopySignBit(FfxFloat32x2 d, FfxFloat32x2 s)
{
    return ffxAsFloat(ffxAsUInt32(d) | (ffxAsUInt32(s) & ffxBroadcast2(0x80000000u)));
}

/// Copy the sign bit from 's' to positive 'd'.
///
/// @param [in] d                   The value to copy the sign bit into.
/// @param [in] s                   The value to copy the sign bit from.
///
/// @returns
/// The value of <c><i>d</i></c> with the sign bit from <c><i>s</i></c>.
///
/// @ingroup GPU
FfxFloat32x3 ffxCopySignBit(FfxFloat32x3 d, FfxFloat32x3 s)
{
    return ffxAsFloat(ffxAsUInt32(d) | (ffxAsUInt32(s) & ffxBroadcast3(0x80000000u)));
}

/// Copy the sign bit from 's' to positive 'd'.
///
/// @param [in] d                   The value to copy the sign bit into.
/// @param [in] s                   The value to copy the sign bit from.
///
/// @returns
/// The value of <c><i>d</i></c> with the sign bit from <c><i>s</i></c>.
///
/// @ingroup GPU
FfxFloat32x4 ffxCopySignBit(FfxFloat32x4 d, FfxFloat32x4 s)
{
    return ffxAsFloat(ffxAsUInt32(d) | (ffxAsUInt32(s) & ffxBroadcast4(0x80000000u)));
}

/// A single operation to return the following:
///     m = NaN := 0
///     m >= 0  := 0
///     m < 0   := 1
///
/// Uses the following useful floating point logic,
///     saturate(+a*(-INF)==-INF) := 0
///     saturate( 0*(-INF)== NaN) := 0
///     saturate(-a*(-INF)==+INF) := 1
/// 
/// This function is useful when creating masks for branch-free logic.
/// 
/// @param [in] m                       The value to test against 0.
/// 
/// @returns
/// 1.0 when the value is negative, or 0.0 when the value is 0 or position.
/// 
/// @ingroup GPU
FfxFloat32 ffxIsSigned(FfxFloat32 m)
{
    return ffxSaturate(m * FfxFloat32(FFX_NEGATIVE_INFINITY_FLOAT));
}

/// A single operation to return the following:
///     m = NaN := 0
///     m >= 0  := 0
///     m < 0   := 1
///
/// Uses the following useful floating point logic,
///     saturate(+a*(-INF)==-INF) := 0
///     saturate( 0*(-INF)== NaN) := 0
///     saturate(-a*(-INF)==+INF) := 1
///
/// This function is useful when creating masks for branch-free logic.
///
/// @param [in] m                       The value to test against 0.
///
/// @returns
/// 1.0 when the value is negative, or 0.0 when the value is 0 or position.
///
/// @ingroup GPU
FfxFloat32x2 ffxIsSigned(FfxFloat32x2 m)
{
    return ffxSaturate(m * ffxBroadcast2(FFX_NEGATIVE_INFINITY_FLOAT));
}

/// A single operation to return the following:
///     m = NaN := 0
///     m >= 0  := 0
///     m < 0   := 1
///
/// Uses the following useful floating point logic,
///     saturate(+a*(-INF)==-INF) := 0
///     saturate( 0*(-INF)== NaN) := 0
///     saturate(-a*(-INF)==+INF) := 1
///
/// This function is useful when creating masks for branch-free logic.
///
/// @param [in] m                       The value to test against 0.
///
/// @returns
/// 1.0 when the value is negative, or 0.0 when the value is 0 or position.
///
/// @ingroup GPU
FfxFloat32x3 ffxIsSigned(FfxFloat32x3 m)
{
    return ffxSaturate(m * ffxBroadcast3(FFX_NEGATIVE_INFINITY_FLOAT));
}

/// A single operation to return the following:
///     m = NaN := 0
///     m >= 0  := 0
///     m < 0   := 1
///
/// Uses the following useful floating point logic,
///     saturate(+a*(-INF)==-INF) := 0
///     saturate( 0*(-INF)== NaN) := 0
///     saturate(-a*(-INF)==+INF) := 1
///
/// This function is useful when creating masks for branch-free logic.
///
/// @param [in] m                       The value to test against for have the sign set.
///
/// @returns
/// 1.0 when the value is negative, or 0.0 when the value is 0 or positive.
///
/// @ingroup GPU
FfxFloat32x4 ffxIsSigned(FfxFloat32x4 m)
{
    return ffxSaturate(m * ffxBroadcast4(FFX_NEGATIVE_INFINITY_FLOAT));
}

/// A single operation to return the following:
///     m = NaN := 1
///     m > 0   := 0
///     m <= 0  := 1
///
/// This function is useful when creating masks for branch-free logic.
///
/// @param [in] m                       The value to test against zero.
///
/// @returns
/// 1.0 when the value is position, or 0.0 when the value is 0 or negative.
///
/// @ingroup GPU
FfxFloat32 ffxIsGreaterThanZero(FfxFloat32 m)
{
    return ffxSaturate(m * FfxFloat32(FFX_POSITIVE_INFINITY_FLOAT));
}

/// A single operation to return the following:
///     m = NaN := 1
///     m > 0   := 0
///     m <= 0  := 1
///
/// This function is useful when creating masks for branch-free logic.
///
/// @param [in] m                       The value to test against zero.
///
/// @returns
/// 1.0 when the value is position, or 0.0 when the value is 0 or negative.
///
/// @ingroup GPU
FfxFloat32x2 ffxIsGreaterThanZero(FfxFloat32x2 m)
{
    return ffxSaturate(m * ffxBroadcast2(FFX_POSITIVE_INFINITY_FLOAT));
}

/// A single operation to return the following:
///     m = NaN := 1
///     m > 0   := 0
///     m <= 0  := 1
///
/// This function is useful when creating masks for branch-free logic.
///
/// @param [in] m                       The value to test against zero.
///
/// @returns
/// 1.0 when the value is position, or 0.0 when the value is 0 or negative.
///
/// @ingroup GPU
FfxFloat32x3 ffxIsGreaterThanZero(FfxFloat32x3 m)
{
    return ffxSaturate(m * ffxBroadcast3(FFX_POSITIVE_INFINITY_FLOAT));
}

/// A single operation to return the following:
///     m = NaN := 1
///     m > 0   := 0
///     m <= 0  := 1
///
/// This function is useful when creating masks for branch-free logic.
///
/// @param [in] m                       The value to test against zero.
///
/// @returns
/// 1.0 when the value is position, or 0.0 when the value is 0 or negative.
///
/// @ingroup GPU
FfxFloat32x4 ffxIsGreaterThanZero(FfxFloat32x4 m)
{
    return ffxSaturate(m * ffxBroadcast4(FFX_POSITIVE_INFINITY_FLOAT));
}

/// Convert a 32bit floating point value to sortable integer.
/// 
///  - If sign bit=0, flip the sign bit (positives).
///  - If sign bit=1, flip all bits     (negatives).
/// 
/// The function has the side effects that:
///  - Larger integers are more positive values.
///  - Float zero is mapped to center of integers (so clear to integer zero is a nice default for atomic max usage).
/// 
/// @param [in] value                       The floating point value to make sortable.
/// 
/// @returns
/// The sortable integer value.
/// 
/// @ingroup GPU
FfxUInt32 ffxFloatToSortableInteger(FfxUInt32 value)
{
    return value ^ ((AShrSU1(value, FfxUInt32(31))) | FfxUInt32(0x80000000));
}

/// Convert a sortable integer to a 32bit floating point value.
///
/// The function has the side effects that:
///  - If sign bit=1, flip the sign bit (positives).
///  - If sign bit=0, flip all bits     (negatives).
///
/// @param [in] value                       The floating point value to make sortable.
///
/// @returns
/// The sortable integer value.
///
/// @ingroup GPU
FfxUInt32 ffxSortableIntegerToFloat(FfxUInt32 value)
{
    return value ^ ((~AShrSU1(value, FfxUInt32(31))) | FfxUInt32(0x80000000));
}

/// Calculate a low-quality approximation for the square root of a value.
///
/// For additional information on the approximation family of functions, you can refer to Michal Drobot's excellent 
/// presentation materials:
/// 
///  - https://michaldrobot.files.wordpress.com/2014/05/gcn_alu_opt_digitaldragons2014.pdf
///  - https://github.com/michaldrobot/ShaderFastLibs/blob/master/ShaderFastMathLib.h
/// 
/// @param [in] value           The value to calculate an approximate to the square root for.
///
/// @returns
/// An approximation of the square root, estimated to low quality.
///
/// @ingroup GPU
FfxFloat32 ffxApproximateSqrt(FfxFloat32 a)
{
    return ffxAsFloat((ffxAsUInt32(a) >> FfxUInt32(1)) + FfxUInt32(0x1fbc4639));
}

/// Calculate a low-quality approximation for the reciprocal of a value.
///
/// For additional information on the approximation family of functions, you can refer to Michal Drobot's excellent
/// presentation materials:
///
///  - https://michaldrobot.files.wordpress.com/2014/05/gcn_alu_opt_digitaldragons2014.pdf
///  - https://github.com/michaldrobot/ShaderFastLibs/blob/master/ShaderFastMathLib.h
///
/// @param [in] value           The value to calculate an approximate to the reciprocal for.
///
/// @returns
/// An approximation of the reciprocal, estimated to low quality.
///
/// @ingroup GPU
FfxFloat32 ffxApproximateReciprocal(FfxFloat32 a)
{
    return ffxAsFloat(FfxUInt32(0x7ef07ebb) - ffxAsUInt32(a));
}

/// Calculate a medium-quality approximation for the reciprocal of a value.
/// 
/// For additional information on the approximation family of functions, you can refer to Michal Drobot's excellent
/// presentation materials:
///
///  - https://michaldrobot.files.wordpress.com/2014/05/gcn_alu_opt_digitaldragons2014.pdf
///  - https://github.com/michaldrobot/ShaderFastLibs/blob/master/ShaderFastMathLib.h
///
/// @param [in] value           The value to calculate an approximate to the reciprocal for.
///
/// @returns
/// An approximation of the reciprocal, estimated to medium quality.
/// 
/// @ingroup GPU
FfxFloat32 ffxApproximateReciprocalMedium(FfxFloat32 value)
{
    FfxFloat32 b = ffxAsFloat(FfxUInt32(0x7ef19fff) - ffxAsUInt32(value));
    return b * (-b * value + FfxFloat32(2.0));
}

/// Calculate a low-quality approximation for the reciprocal of a value.
///
/// For additional information on the approximation family of functions, you can refer to Michal Drobot's excellent
/// presentation materials:
///
///  - https://michaldrobot.files.wordpress.com/2014/05/gcn_alu_opt_digitaldragons2014.pdf
///  - https://github.com/michaldrobot/ShaderFastLibs/blob/master/ShaderFastMathLib.h
///
/// @param [in] value           The value to calculate an approximate to the reciprocal square root for.
///
/// @returns
/// An approximation of the reciprocal square root, estimated to low quality.
///
/// @ingroup GPU
FfxFloat32 ffxApproximateReciprocalSquareRoot(FfxFloat32 a)
{
    return ffxAsFloat(FfxUInt32(0x5f347d74) - (ffxAsUInt32(a) >> FfxUInt32(1)));
}

/// Calculate a low-quality approximation for the square root of a value.
///
/// For additional information on the approximation family of functions, you can refer to Michal Drobot's excellent
/// presentation materials:
///
///  - https://michaldrobot.files.wordpress.com/2014/05/gcn_alu_opt_digitaldragons2014.pdf
///  - https://github.com/michaldrobot/ShaderFastLibs/blob/master/ShaderFastMathLib.h
///
/// @param [in] value           The value to calculate an approximate to the square root for.
///
/// @returns
/// An approximation of the square root, estimated to low quality.
///
/// @ingroup GPU
FfxFloat32x2 ffxApproximateSqrt(FfxFloat32x2 a)
{
    return ffxAsFloat((ffxAsUInt32(a) >> ffxBroadcast2(1u)) + ffxBroadcast2(0x1fbc4639u));
}

/// Calculate a low-quality approximation for the reciprocal of a value.
///
/// For additional information on the approximation family of functions, you can refer to Michal Drobot's excellent
/// presentation materials:
///
///  - https://michaldrobot.files.wordpress.com/2014/05/gcn_alu_opt_digitaldragons2014.pdf
///  - https://github.com/michaldrobot/ShaderFastLibs/blob/master/ShaderFastMathLib.h
///
/// @param [in] value           The value to calculate an approximate to the reciprocal for.
///
/// @returns
/// An approximation of the reciprocal, estimated to low quality.
///
/// @ingroup GPU
FfxFloat32x2 ffxApproximateReciprocal(FfxFloat32x2 a)
{
    return ffxAsFloat(ffxBroadcast2(0x7ef07ebbu) - ffxAsUInt32(a));
}

/// Calculate a medium-quality approximation for the reciprocal of a value.
///
/// For additional information on the approximation family of functions, you can refer to Michal Drobot's excellent
/// presentation materials:
///
///  - https://michaldrobot.files.wordpress.com/2014/05/gcn_alu_opt_digitaldragons2014.pdf
///  - https://github.com/michaldrobot/ShaderFastLibs/blob/master/ShaderFastMathLib.h
///
/// @param [in] value           The value to calculate an approximate to the reciprocal for.
///
/// @returns
/// An approximation of the reciprocal, estimated to medium quality.
///
/// @ingroup GPU
FfxFloat32x2 ffxApproximateReciprocalMedium(FfxFloat32x2 a)
{
    FfxFloat32x2 b = ffxAsFloat(ffxBroadcast2(0x7ef19fffu) - ffxAsUInt32(a));
    return b * (-b * a + ffxBroadcast2(2.0f));
}

/// Calculate a low-quality approximation for the square root of a value.
///
/// For additional information on the approximation family of functions, you can refer to Michal Drobot's excellent
/// presentation materials:
///
///  - https://michaldrobot.files.wordpress.com/2014/05/gcn_alu_opt_digitaldragons2014.pdf
///  - https://github.com/michaldrobot/ShaderFastLibs/blob/master/ShaderFastMathLib.h
///
/// @param [in] value           The value to calculate an approximate to the square root for.
///
/// @returns
/// An approximation of the square root, estimated to low quality.
///
/// @ingroup GPU
FfxFloat32x2 ffxApproximateReciprocalSquareRoot(FfxFloat32x2 a)
{
    return ffxAsFloat(ffxBroadcast2(0x5f347d74u) - (ffxAsUInt32(a) >> ffxBroadcast2(1u)));
}

/// Calculate a low-quality approximation for the square root of a value.
///
/// For additional information on the approximation family of functions, you can refer to Michal Drobot's excellent
/// presentation materials:
///
///  - https://michaldrobot.files.wordpress.com/2014/05/gcn_alu_opt_digitaldragons2014.pdf
///  - https://github.com/michaldrobot/ShaderFastLibs/blob/master/ShaderFastMathLib.h
///
/// @param [in] value           The value to calculate an approximate to the square root for.
///
/// @returns
/// An approximation of the square root, estimated to low quality.
///
/// @ingroup GPU
FfxFloat32x3 ffxApproximateSqrt(FfxFloat32x3 a)
{
    return ffxAsFloat((ffxAsUInt32(a) >> ffxBroadcast3(1u)) + ffxBroadcast3(0x1fbc4639u));
}

/// Calculate a low-quality approximation for the reciprocal of a value.
///
/// For additional information on the approximation family of functions, you can refer to Michal Drobot's excellent
/// presentation materials:
///
///  - https://michaldrobot.files.wordpress.com/2014/05/gcn_alu_opt_digitaldragons2014.pdf
///  - https://github.com/michaldrobot/ShaderFastLibs/blob/master/ShaderFastMathLib.h
///
/// @param [in] value           The value to calculate an approximate to the reciprocal for.
///
/// @returns
/// An approximation of the reciprocal, estimated to low quality.
///
/// @ingroup GPU
FfxFloat32x3 ffxApproximateReciprocal(FfxFloat32x3 a)
{
    return ffxAsFloat(ffxBroadcast3(0x7ef07ebbu) - ffxAsUInt32(a));
}

/// Calculate a medium-quality approximation for the reciprocal of a value.
///
/// For additional information on the approximation family of functions, you can refer to Michal Drobot's excellent
/// presentation materials:
///
///  - https://michaldrobot.files.wordpress.com/2014/05/gcn_alu_opt_digitaldragons2014.pdf
///  - https://github.com/michaldrobot/ShaderFastLibs/blob/master/ShaderFastMathLib.h
///
/// @param [in] value           The value to calculate an approximate to the reciprocal for.
///
/// @returns
/// An approximation of the reciprocal, estimated to medium quality.
///
/// @ingroup GPU
FfxFloat32x3 ffxApproximateReciprocalMedium(FfxFloat32x3 a)
{
    FfxFloat32x3 b = ffxAsFloat(ffxBroadcast3(0x7ef19fffu) - ffxAsUInt32(a));
    return b * (-b * a + ffxBroadcast3(2.0f));
}

/// Calculate a low-quality approximation for the square root of a value.
///
/// For additional information on the approximation family of functions, you can refer to Michal Drobot's excellent
/// presentation materials:
///
///  - https://michaldrobot.files.wordpress.com/2014/05/gcn_alu_opt_digitaldragons2014.pdf
///  - https://github.com/michaldrobot/ShaderFastLibs/blob/master/ShaderFastMathLib.h
///
/// @param [in] value           The value to calculate an approximate to the square root for.
///
/// @returns
/// An approximation of the square root, estimated to low quality.
///
/// @ingroup GPU
FfxFloat32x3 ffxApproximateReciprocalSquareRoot(FfxFloat32x3 a)
{
    return ffxAsFloat(ffxBroadcast3(0x5f347d74u) - (ffxAsUInt32(a) >> ffxBroadcast3(1u)));
}

/// Calculate a low-quality approximation for the square root of a value.
///
/// For additional information on the approximation family of functions, you can refer to Michal Drobot's excellent
/// presentation materials:
///
///  - https://michaldrobot.files.wordpress.com/2014/05/gcn_alu_opt_digitaldragons2014.pdf
///  - https://github.com/michaldrobot/ShaderFastLibs/blob/master/ShaderFastMathLib.h
///
/// @param [in] value           The value to calculate an approximate to the square root for.
///
/// @returns
/// An approximation of the square root, estimated to low quality.
///
/// @ingroup GPU
FfxFloat32x4 ffxApproximateSqrt(FfxFloat32x4 a)
{
    return ffxAsFloat((ffxAsUInt32(a) >> ffxBroadcast4(1u)) + ffxBroadcast4(0x1fbc4639u));
}

/// Calculate a low-quality approximation for the reciprocal of a value.
///
/// For additional information on the approximation family of functions, you can refer to Michal Drobot's excellent
/// presentation materials:
///
///  - https://michaldrobot.files.wordpress.com/2014/05/gcn_alu_opt_digitaldragons2014.pdf
///  - https://github.com/michaldrobot/ShaderFastLibs/blob/master/ShaderFastMathLib.h
///
/// @param [in] value           The value to calculate an approximate to the reciprocal for.
///
/// @returns
/// An approximation of the reciprocal, estimated to low quality.
///
/// @ingroup GPU
FfxFloat32x4 ffxApproximateReciprocal(FfxFloat32x4 a)
{
    return ffxAsFloat(ffxBroadcast4(0x7ef07ebbu) - ffxAsUInt32(a));
}

/// Calculate a medium-quality approximation for the reciprocal of a value.
///
/// For additional information on the approximation family of functions, you can refer to Michal Drobot's excellent
/// presentation materials:
///
///  - https://michaldrobot.files.wordpress.com/2014/05/gcn_alu_opt_digitaldragons2014.pdf
///  - https://github.com/michaldrobot/ShaderFastLibs/blob/master/ShaderFastMathLib.h
///
/// @param [in] value           The value to calculate an approximate to the reciprocal for.
///
/// @returns
/// An approximation of the reciprocal, estimated to medium quality.
///
/// @ingroup GPU
FfxFloat32x4 ffxApproximateReciprocalMedium(FfxFloat32x4 a)
{
    FfxFloat32x4 b = ffxAsFloat(ffxBroadcast4(0x7ef19fffu) - ffxAsUInt32(a));
    return b * (-b * a + ffxBroadcast4(2.0f));
}

/// Calculate a low-quality approximation for the square root of a value.
///
/// For additional information on the approximation family of functions, you can refer to Michal Drobot's excellent
/// presentation materials:
///
///  - https://michaldrobot.files.wordpress.com/2014/05/gcn_alu_opt_digitaldragons2014.pdf
///  - https://github.com/michaldrobot/ShaderFastLibs/blob/master/ShaderFastMathLib.h
///
/// @param [in] value           The value to calculate an approximate to the square root for.
///
/// @returns
/// An approximation of the square root, estimated to low quality.
///
/// @ingroup GPU
FfxFloat32x4 ffxApproximateReciprocalSquareRoot(FfxFloat32x4 a)
{
    return ffxAsFloat(ffxBroadcast4(0x5f347d74u) - (ffxAsUInt32(a) >> ffxBroadcast4(1u)));
}

/// Calculate dot product of 'a' and 'b'.
///
/// @param [in] a                   First vector input.
/// @param [in] b                   Second vector input.
///
/// @returns
/// The value of <c><i>a</i></c> dot <c><i>b</i></c>.
///
/// @ingroup GPU
FfxFloat32 ffxDot2(FfxFloat32x2 a, FfxFloat32x2 b)
{
    return dot(a, b);
}

/// Calculate dot product of 'a' and 'b'.
///
/// @param [in] a                   First vector input.
/// @param [in] b                   Second vector input.
///
/// @returns
/// The value of <c><i>a</i></c> dot <c><i>b</i></c>.
///
/// @ingroup GPU
FfxFloat32 ffxDot3(FfxFloat32x3 a, FfxFloat32x3 b)
{
    return dot(a, b);
}

/// Calculate dot product of 'a' and 'b'.
///
/// @param [in] a                   First vector input.
/// @param [in] b                   Second vector input.
///
/// @returns
/// The value of <c><i>a</i></c> dot <c><i>b</i></c>.
///
/// @ingroup GPU
FfxFloat32 ffxDot4(FfxFloat32x4 a, FfxFloat32x4 b)
{
    return dot(a, b);
}


/// Compute an approximate conversion from PQ to Gamma2 space.
///
/// PQ is very close to x^(1/8). The functions below Use the fast FfxFloat32 approximation method to do
/// PQ conversions to and from Gamma2 (4th power and fast 4th root), and PQ to and from Linear 
/// (8th power and fast 8th root). The maximum error is approximately 0.2%.
///
/// @param a                    The value to convert between PQ and Gamma2.
///
/// @returns
/// The value <c><i>a</i></c> converted into Gamma2.
///
/// @ingroup GPU
FfxFloat32 ffxApproximatePQToGamma2Medium(FfxFloat32 a)
{
    return a * a * a * a;
}

/// Compute an approximate conversion from PQ to linear space.
///
/// PQ is very close to x^(1/8). The functions below Use the fast FfxFloat32 approximation method to do
/// PQ conversions to and from Gamma2 (4th power and fast 4th root), and PQ to and from Linear
/// (8th power and fast 8th root). The maximum error is approximately 0.2%.
///
/// @param a                    The value to convert between PQ and linear.
///
/// @returns
/// The value <c><i>a</i></c> converted into linear.
///
/// @ingroup GPU
FfxFloat32 ffxApproximatePQToLinear(FfxFloat32 a)
{
    return a * a * a * a * a * a * a * a;
}

/// Compute an approximate conversion from gamma2 to PQ space.
///
/// PQ is very close to x^(1/8). The functions below Use the fast FfxFloat32 approximation method to do
/// PQ conversions to and from Gamma2 (4th power and fast 4th root), and PQ to and from Linear
/// (8th power and fast 8th root). The maximum error is approximately 0.2%.
///
/// @param a                    The value to convert between gamma2 and PQ.
///
/// @returns
/// The value <c><i>a</i></c> converted into PQ.
///
/// @ingroup GPU
FfxFloat32 ffxApproximateGamma2ToPQ(FfxFloat32 a)
{
    return ffxAsFloat((ffxAsUInt32(a) >> FfxUInt32(2)) + FfxUInt32(0x2F9A4E46));
}

/// Compute a more accurate approximate conversion from gamma2 to PQ space.
///
/// PQ is very close to x^(1/8). The functions below Use the fast FfxFloat32 approximation method to do
/// PQ conversions to and from Gamma2 (4th power and fast 4th root), and PQ to and from Linear
/// (8th power and fast 8th root). The maximum error is approximately 0.2%.
///
/// @param a                    The value to convert between gamma2 and PQ.
///
/// @returns
/// The value <c><i>a</i></c> converted into PQ.
///
/// @ingroup GPU
FfxFloat32 ffxApproximateGamma2ToPQMedium(FfxFloat32 a)
{
    FfxFloat32 b  = ffxAsFloat((ffxAsUInt32(a) >> FfxUInt32(2)) + FfxUInt32(0x2F9A4E46));
    FfxFloat32 b4 = b * b * b * b;
    return b - b * (b4 - a) / (FfxFloat32(4.0) * b4);
}

/// Compute a high accuracy approximate conversion from gamma2 to PQ space.
///
/// PQ is very close to x^(1/8). The functions below Use the fast FfxFloat32 approximation method to do
/// PQ conversions to and from Gamma2 (4th power and fast 4th root), and PQ to and from Linear
/// (8th power and fast 8th root). The maximum error is approximately 0.2%.
///
/// @param a                    The value to convert between gamma2 and PQ.
///
/// @returns
/// The value <c><i>a</i></c> converted into PQ.
///
/// @ingroup GPU
FfxFloat32 ffxApproximateGamma2ToPQHigh(FfxFloat32 a)
{
    return ffxSqrt(ffxSqrt(a));
}

/// Compute an approximate conversion from linear to PQ space.
///
/// PQ is very close to x^(1/8). The functions below Use the fast FfxFloat32 approximation method to do
/// PQ conversions to and from Gamma2 (4th power and fast 4th root), and PQ to and from Linear
/// (8th power and fast 8th root). The maximum error is approximately 0.2%.
///
/// @param a                    The value to convert between linear and PQ.
///
/// @returns
/// The value <c><i>a</i></c> converted into PQ.
///
/// @ingroup GPU
FfxFloat32 ffxApproximateLinearToPQ(FfxFloat32 a)
{
    return ffxAsFloat((ffxAsUInt32(a) >> FfxUInt32(3)) + FfxUInt32(0x378D8723));
}

/// Compute a more accurate approximate conversion from linear to PQ space.
///
/// PQ is very close to x^(1/8). The functions below Use the fast FfxFloat32 approximation method to do
/// PQ conversions to and from Gamma2 (4th power and fast 4th root), and PQ to and from Linear
/// (8th power and fast 8th root). The maximum error is approximately 0.2%.
///
/// @param a                    The value to convert between linear and PQ.
///
/// @returns
/// The value <c><i>a</i></c> converted into PQ.
///
/// @ingroup GPU
FfxFloat32 ffxApproximateLinearToPQMedium(FfxFloat32 a)
{
    FfxFloat32 b  = ffxAsFloat((ffxAsUInt32(a) >> FfxUInt32(3)) + FfxUInt32(0x378D8723));
    FfxFloat32 b8 = b * b * b * b * b * b * b * b;
    return b - b * (b8 - a) / (FfxFloat32(8.0) * b8);
}

/// Compute a very accurate approximate conversion from linear to PQ space.
///
/// PQ is very close to x^(1/8). The functions below Use the fast FfxFloat32 approximation method to do
/// PQ conversions to and from Gamma2 (4th power and fast 4th root), and PQ to and from Linear
/// (8th power and fast 8th root). The maximum error is approximately 0.2%.
///
/// @param a                    The value to convert between linear and PQ.
///
/// @returns
/// The value <c><i>a</i></c> converted into PQ.
///
/// @ingroup GPU
FfxFloat32 ffxApproximateLinearToPQHigh(FfxFloat32 a)
{
    return ffxSqrt(ffxSqrt(ffxSqrt(a)));
}

/// Compute an approximate conversion from PQ to Gamma2 space.
///
/// PQ is very close to x^(1/8). The functions below Use the fast FfxFloat32 approximation method to do
/// PQ conversions to and from Gamma2 (4th power and fast 4th root), and PQ to and from Linear
/// (8th power and fast 8th root). The maximum error is approximately 0.2%.
///
/// @param a                    The value to convert between PQ and Gamma2.
///
/// @returns
/// The value <c><i>a</i></c> converted into Gamma2.
///
/// @ingroup GPU
FfxFloat32x2 ffxApproximatePQToGamma2Medium(FfxFloat32x2 a)
{
    return a * a * a * a;
}

/// Compute an approximate conversion from PQ to linear space.
///
/// PQ is very close to x^(1/8). The functions below Use the fast FfxFloat32 approximation method to do
/// PQ conversions to and from Gamma2 (4th power and fast 4th root), and PQ to and from Linear
/// (8th power and fast 8th root). The maximum error is approximately 0.2%.
///
/// @param a                    The value to convert between PQ and linear.
///
/// @returns
/// The value <c><i>a</i></c> converted into linear.
///
/// @ingroup GPU
FfxFloat32x2 ffxApproximatePQToLinear(FfxFloat32x2 a)
{
    return a * a * a * a * a * a * a * a;
}

/// Compute an approximate conversion from gamma2 to PQ space.
///
/// PQ is very close to x^(1/8). The functions below Use the fast FfxFloat32 approximation method to do
/// PQ conversions to and from Gamma2 (4th power and fast 4th root), and PQ to and from Linear
/// (8th power and fast 8th root). The maximum error is approximately 0.2%.
///
/// @param a                    The value to convert between gamma2 and PQ.
///
/// @returns
/// The value <c><i>a</i></c> converted into PQ.
///
/// @ingroup GPU
FfxFloat32x2 ffxApproximateGamma2ToPQ(FfxFloat32x2 a)
{
    return ffxAsFloat((ffxAsUInt32(a) >> ffxBroadcast2(2u)) + ffxBroadcast2(0x2F9A4E46u));
}

/// Compute a more accurate approximate conversion from gamma2 to PQ space.
///
/// PQ is very close to x^(1/8). The functions below Use the fast FfxFloat32 approximation method to do
/// PQ conversions to and from Gamma2 (4th power and fast 4th root), and PQ to and from Linear
/// (8th power and fast 8th root). The maximum error is approximately 0.2%.
///
/// @param a                    The value to convert between gamma2 and PQ.
///
/// @returns
/// The value <c><i>a</i></c> converted into PQ.
///
/// @ingroup GPU
FfxFloat32x2 ffxApproximateGamma2ToPQMedium(FfxFloat32x2 a)
{
    FfxFloat32x2 b  = ffxAsFloat((ffxAsUInt32(a) >> ffxBroadcast2(2u)) + ffxBroadcast2(0x2F9A4E46u));
    FfxFloat32x2 b4 = b * b * b * b;
    return b - b * (b4 - a) / (FfxFloat32(4.0) * b4);
}

/// Compute a high accuracy approximate conversion from gamma2 to PQ space.
///
/// PQ is very close to x^(1/8). The functions below Use the fast FfxFloat32 approximation method to do
/// PQ conversions to and from Gamma2 (4th power and fast 4th root), and PQ to and from Linear
/// (8th power and fast 8th root). The maximum error is approximately 0.2%.
///
/// @param a                    The value to convert between gamma2 and PQ.
///
/// @returns
/// The value <c><i>a</i></c> converted into PQ.
///
/// @ingroup GPU
FfxFloat32x2 ffxApproximateGamma2ToPQHigh(FfxFloat32x2 a)
{
    return ffxSqrt(ffxSqrt(a));
}

/// Compute an approximate conversion from linear to PQ space.
///
/// PQ is very close to x^(1/8). The functions below Use the fast FfxFloat32 approximation method to do
/// PQ conversions to and from Gamma2 (4th power and fast 4th root), and PQ to and from Linear
/// (8th power and fast 8th root). The maximum error is approximately 0.2%.
///
/// @param a                    The value to convert between linear and PQ.
///
/// @returns
/// The value <c><i>a</i></c> converted into PQ.
///
/// @ingroup GPU
FfxFloat32x2 ffxApproximateLinearToPQ(FfxFloat32x2 a)
{
    return ffxAsFloat((ffxAsUInt32(a) >> ffxBroadcast2(3u)) + ffxBroadcast2(0x378D8723u));
}

/// Compute a more accurate approximate conversion from linear to PQ space.
///
/// PQ is very close to x^(1/8). The functions below Use the fast FfxFloat32 approximation method to do
/// PQ conversions to and from Gamma2 (4th power and fast 4th root), and PQ to and from Linear
/// (8th power and fast 8th root). The maximum error is approximately 0.2%.
///
/// @param a                    The value to convert between linear and PQ.
///
/// @returns
/// The value <c><i>a</i></c> converted into PQ.
///
/// @ingroup GPU
FfxFloat32x2 ffxApproximateLinearToPQMedium(FfxFloat32x2 a)
{
    FfxFloat32x2 b  = ffxAsFloat((ffxAsUInt32(a) >> ffxBroadcast2(3u)) + ffxBroadcast2(0x378D8723u));
    FfxFloat32x2 b8 = b * b * b * b * b * b * b * b;
    return b - b * (b8 - a) / (FfxFloat32(8.0) * b8);
}

/// Compute a very accurate approximate conversion from linear to PQ space.
///
/// PQ is very close to x^(1/8). The functions below Use the fast FfxFloat32 approximation method to do
/// PQ conversions to and from Gamma2 (4th power and fast 4th root), and PQ to and from Linear
/// (8th power and fast 8th root). The maximum error is approximately 0.2%.
///
/// @param a                    The value to convert between linear and PQ.
///
/// @returns
/// The value <c><i>a</i></c> converted into PQ.
///
/// @ingroup GPU
FfxFloat32x2 ffxApproximateLinearToPQHigh(FfxFloat32x2 a)
{
    return ffxSqrt(ffxSqrt(ffxSqrt(a)));
}

/// Compute an approximate conversion from PQ to Gamma2 space.
///
/// PQ is very close to x^(1/8). The functions below Use the fast FfxFloat32 approximation method to do
/// PQ conversions to and from Gamma2 (4th power and fast 4th root), and PQ to and from Linear
/// (8th power and fast 8th root). The maximum error is approximately 0.2%.
///
/// @param a                    The value to convert between PQ and Gamma2.
///
/// @returns
/// The value <c><i>a</i></c> converted into Gamma2.
///
/// @ingroup GPU
FfxFloat32x3 ffxApproximatePQToGamma2Medium(FfxFloat32x3 a)
{
    return a * a * a * a;
}

/// Compute an approximate conversion from PQ to linear space.
///
/// PQ is very close to x^(1/8). The functions below Use the fast FfxFloat32 approximation method to do
/// PQ conversions to and from Gamma2 (4th power and fast 4th root), and PQ to and from Linear
/// (8th power and fast 8th root). The maximum error is approximately 0.2%.
///
/// @param a                    The value to convert between PQ and linear.
///
/// @returns
/// The value <c><i>a</i></c> converted into linear.
///
/// @ingroup GPU
FfxFloat32x3 ffxApproximatePQToLinear(FfxFloat32x3 a)
{
    return a * a * a * a * a * a * a * a;
}

/// Compute an approximate conversion from gamma2 to PQ space.
///
/// PQ is very close to x^(1/8). The functions below Use the fast FfxFloat32 approximation method to do
/// PQ conversions to and from Gamma2 (4th power and fast 4th root), and PQ to and from Linear
/// (8th power and fast 8th root). The maximum error is approximately 0.2%.
///
/// @param a                    The value to convert between gamma2 and PQ.
///
/// @returns
/// The value <c><i>a</i></c> converted into PQ.
///
/// @ingroup GPU
FfxFloat32x3 ffxApproximateGamma2ToPQ(FfxFloat32x3 a)
{
    return ffxAsFloat((ffxAsUInt32(a) >> ffxBroadcast3(2u)) + ffxBroadcast3(0x2F9A4E46u));
}

/// Compute a more accurate approximate conversion from gamma2 to PQ space.
///
/// PQ is very close to x^(1/8). The functions below Use the fast FfxFloat32 approximation method to do
/// PQ conversions to and from Gamma2 (4th power and fast 4th root), and PQ to and from Linear
/// (8th power and fast 8th root). The maximum error is approximately 0.2%.
///
/// @param a                    The value to convert between gamma2 and PQ.
///
/// @returns
/// The value <c><i>a</i></c> converted into PQ.
///
/// @ingroup GPU
FfxFloat32x3 ffxApproximateGamma2ToPQMedium(FfxFloat32x3 a)
{
    FfxFloat32x3 b  = ffxAsFloat((ffxAsUInt32(a) >> ffxBroadcast3(2u)) + ffxBroadcast3(0x2F9A4E46u));
    FfxFloat32x3 b4 = b * b * b * b;
    return b - b * (b4 - a) / (FfxFloat32(4.0) * b4);
}

/// Compute a high accuracy approximate conversion from gamma2 to PQ space.
///
/// PQ is very close to x^(1/8). The functions below Use the fast FfxFloat32 approximation method to do
/// PQ conversions to and from Gamma2 (4th power and fast 4th root), and PQ to and from Linear
/// (8th power and fast 8th root). The maximum error is approximately 0.2%.
///
/// @param a                    The value to convert between gamma2 and PQ.
///
/// @returns
/// The value <c><i>a</i></c> converted into PQ.
///
/// @ingroup GPU
FfxFloat32x3 ffxApproximateGamma2ToPQHigh(FfxFloat32x3 a)
{
    return ffxSqrt(ffxSqrt(a));
}

/// Compute an approximate conversion from linear to PQ space.
///
/// PQ is very close to x^(1/8). The functions below Use the fast FfxFloat32 approximation method to do
/// PQ conversions to and from Gamma2 (4th power and fast 4th root), and PQ to and from Linear
/// (8th power and fast 8th root). The maximum error is approximately 0.2%.
///
/// @param a                    The value to convert between linear and PQ.
///
/// @returns
/// The value <c><i>a</i></c> converted into PQ.
///
/// @ingroup GPU
FfxFloat32x3 ffxApproximateLinearToPQ(FfxFloat32x3 a)
{
    return ffxAsFloat((ffxAsUInt32(a) >> ffxBroadcast3(3u)) + ffxBroadcast3(0x378D8723u));
}

/// Compute a more accurate approximate conversion from linear to PQ space.
///
/// PQ is very close to x^(1/8). The functions below Use the fast FfxFloat32 approximation method to do
/// PQ conversions to and from Gamma2 (4th power and fast 4th root), and PQ to and from Linear
/// (8th power and fast 8th root). The maximum error is approximately 0.2%.
///
/// @param a                    The value to convert between linear and PQ.
///
/// @returns
/// The value <c><i>a</i></c> converted into PQ.
///
/// @ingroup GPU
FfxFloat32x3 ffxApproximateLinearToPQMedium(FfxFloat32x3 a)
{
    FfxFloat32x3 b  = ffxAsFloat((ffxAsUInt32(a) >> ffxBroadcast3(3u)) + ffxBroadcast3(0x378D8723u));
    FfxFloat32x3 b8 = b * b * b * b * b * b * b * b;
    return b - b * (b8 - a) / (FfxFloat32(8.0) * b8);
}

/// Compute a very accurate approximate conversion from linear to PQ space.
///
/// PQ is very close to x^(1/8). The functions below Use the fast FfxFloat32 approximation method to do
/// PQ conversions to and from Gamma2 (4th power and fast 4th root), and PQ to and from Linear
/// (8th power and fast 8th root). The maximum error is approximately 0.2%.
///
/// @param a                    The value to convert between linear and PQ.
///
/// @returns
/// The value <c><i>a</i></c> converted into PQ.
///
/// @ingroup GPU
FfxFloat32x3 ffxApproximateLinearToPQHigh(FfxFloat32x3 a)
{
    return ffxSqrt(ffxSqrt(ffxSqrt(a)));
}

/// Compute an approximate conversion from PQ to Gamma2 space.
///
/// PQ is very close to x^(1/8). The functions below Use the fast FfxFloat32 approximation method to do
/// PQ conversions to and from Gamma2 (4th power and fast 4th root), and PQ to and from Linear
/// (8th power and fast 8th root). The maximum error is approximately 0.2%.
///
/// @param a                    The value to convert between PQ and Gamma2.
///
/// @returns
/// The value <c><i>a</i></c> converted into Gamma2.
///
/// @ingroup GPU
FfxFloat32x4 ffxApproximatePQToGamma2Medium(FfxFloat32x4 a)
{
    return a * a * a * a;
}

/// Compute an approximate conversion from PQ to linear space.
///
/// PQ is very close to x^(1/8). The functions below Use the fast FfxFloat32 approximation method to do
/// PQ conversions to and from Gamma2 (4th power and fast 4th root), and PQ to and from Linear
/// (8th power and fast 8th root). The maximum error is approximately 0.2%.
///
/// @param a                    The value to convert between PQ and linear.
///
/// @returns
/// The value <c><i>a</i></c> converted into linear.
///
/// @ingroup GPU
FfxFloat32x4 ffxApproximatePQToLinear(FfxFloat32x4 a)
{
    return a * a * a * a * a * a * a * a;
}

/// Compute an approximate conversion from gamma2 to PQ space.
///
/// PQ is very close to x^(1/8). The functions below Use the fast FfxFloat32 approximation method to do
/// PQ conversions to and from Gamma2 (4th power and fast 4th root), and PQ to and from Linear
/// (8th power and fast 8th root). The maximum error is approximately 0.2%.
///
/// @param a                    The value to convert between gamma2 and PQ.
///
/// @returns
/// The value <c><i>a</i></c> converted into PQ.
///
/// @ingroup GPU
FfxFloat32x4 ffxApproximateGamma2ToPQ(FfxFloat32x4 a)
{
    return ffxAsFloat((ffxAsUInt32(a) >> ffxBroadcast4(2u)) + ffxBroadcast4(0x2F9A4E46u));
}

/// Compute a more accurate approximate conversion from gamma2 to PQ space.
///
/// PQ is very close to x^(1/8). The functions below Use the fast FfxFloat32 approximation method to do
/// PQ conversions to and from Gamma2 (4th power and fast 4th root), and PQ to and from Linear
/// (8th power and fast 8th root). The maximum error is approximately 0.2%.
///
/// @param a                    The value to convert between gamma2 and PQ.
///
/// @returns
/// The value <c><i>a</i></c> converted into PQ.
///
/// @ingroup GPU
FfxFloat32x4 ffxApproximateGamma2ToPQMedium(FfxFloat32x4 a)
{
    FfxFloat32x4 b  = ffxAsFloat((ffxAsUInt32(a) >> ffxBroadcast4(2u)) + ffxBroadcast4(0x2F9A4E46u));
    FfxFloat32x4 b4 = b * b * b * b * b * b * b * b;
    return b - b * (b4 - a) / (FfxFloat32(4.0) * b4);
}

/// Compute a high accuracy approximate conversion from gamma2 to PQ space.
///
/// PQ is very close to x^(1/8). The functions below Use the fast FfxFloat32 approximation method to do
/// PQ conversions to and from Gamma2 (4th power and fast 4th root), and PQ to and from Linear
/// (8th power and fast 8th root). The maximum error is approximately 0.2%.
///
/// @param a                    The value to convert between gamma2 and PQ.
///
/// @returns
/// The value <c><i>a</i></c> converted into PQ.
///
/// @ingroup GPU
FfxFloat32x4 ffxApproximateGamma2ToPQHigh(FfxFloat32x4 a)
{
    return ffxSqrt(ffxSqrt(a));
}

/// Compute an approximate conversion from linear to PQ space.
///
/// PQ is very close to x^(1/8). The functions below Use the fast FfxFloat32 approximation method to do
/// PQ conversions to and from Gamma2 (4th power and fast 4th root), and PQ to and from Linear
/// (8th power and fast 8th root). The maximum error is approximately 0.2%.
///
/// @param a                    The value to convert between linear and PQ.
///
/// @returns
/// The value <c><i>a</i></c> converted into PQ.
///
/// @ingroup GPU
FfxFloat32x4 ffxApproximateLinearToPQ(FfxFloat32x4 a)
{
    return ffxAsFloat((ffxAsUInt32(a) >> ffxBroadcast4(3u)) + ffxBroadcast4(0x378D8723u));
}

/// Compute a more accurate approximate conversion from linear to PQ space.
///
/// PQ is very close to x^(1/8). The functions below Use the fast FfxFloat32 approximation method to do
/// PQ conversions to and from Gamma2 (4th power and fast 4th root), and PQ to and from Linear
/// (8th power and fast 8th root). The maximum error is approximately 0.2%.
///
/// @param a                    The value to convert between linear and PQ.
///
/// @returns
/// The value <c><i>a</i></c> converted into PQ.
///
/// @ingroup GPU
FfxFloat32x4 ffxApproximateLinearToPQMedium(FfxFloat32x4 a)
{
    FfxFloat32x4 b  = ffxAsFloat((ffxAsUInt32(a) >> ffxBroadcast4(3u)) + ffxBroadcast4(0x378D8723u));
    FfxFloat32x4 b8 = b * b * b * b * b * b * b * b;
    return b - b * (b8 - a) / (FfxFloat32(8.0) * b8);
}

/// Compute a very accurate approximate conversion from linear to PQ space.
///
/// PQ is very close to x^(1/8). The functions below Use the fast FfxFloat32 approximation method to do
/// PQ conversions to and from Gamma2 (4th power and fast 4th root), and PQ to and from Linear
/// (8th power and fast 8th root). The maximum error is approximately 0.2%.
///
/// @param a                    The value to convert between linear and PQ.
///
/// @returns
/// The value <c><i>a</i></c> converted into PQ.
///
/// @ingroup GPU
FfxFloat32x4 ffxApproximateLinearToPQHigh(FfxFloat32x4 a)
{
    return ffxSqrt(ffxSqrt(ffxSqrt(a)));
}

// An approximation of sine.
//
// Valid input range is {-1 to 1} representing {0 to 2 pi}, and the output range 
// is {-1/4 to 1/4} representing {-1 to 1}.
//
// @param [in] value            The value to calculate approximate sine for.
//
// @returns
// The approximate sine of <c><i>value</i></c>.
FfxFloat32 ffxParabolicSin(FfxFloat32 value)
{
    return value * abs(value) - value;
}

// An approximation of sine.
//
// Valid input range is {-1 to 1} representing {0 to 2 pi}, and the output range
// is {-1/4 to 1/4} representing {-1 to 1}.
//
// @param [in] value            The value to calculate approximate sine for.
//
// @returns
// The approximate sine of <c><i>value</i></c>.
FfxFloat32x2 ffxParabolicSin(FfxFloat32x2 x)
{
    return x * abs(x) - x;
}

// An approximation of cosine.
//
// Valid input range is {-1 to 1} representing {0 to 2 pi}, and the output range
// is {-1/4 to 1/4} representing {-1 to 1}.
//
// @param [in] value            The value to calculate approximate cosine for.
//
// @returns
// The approximate cosine of <c><i>value</i></c>.
FfxFloat32 ffxParabolicCos(FfxFloat32 x)
{
    x = ffxFract(x * FfxFloat32(0.5) + FfxFloat32(0.75));
    x = x * FfxFloat32(2.0) - FfxFloat32(1.0);
    return ffxParabolicSin(x);
}

// An approximation of cosine.
//
// Valid input range is {-1 to 1} representing {0 to 2 pi}, and the output range
// is {-1/4 to 1/4} representing {-1 to 1}.
//
// @param [in] value            The value to calculate approximate cosine for.
//
// @returns
// The approximate cosine of <c><i>value</i></c>.
FfxFloat32x2 ffxParabolicCos(FfxFloat32x2 x)
{
    x = ffxFract(x * ffxBroadcast2(0.5f) + ffxBroadcast2(0.75f));
    x = x * ffxBroadcast2(2.0f) - ffxBroadcast2(1.0f);
    return ffxParabolicSin(x);
}

// An approximation of both sine and cosine.
//
// Valid input range is {-1 to 1} representing {0 to 2 pi}, and the output range
// is {-1/4 to 1/4} representing {-1 to 1}.
//
// @param [in] value            The value to calculate approximate cosine for.
//
// @returns
// A <c><i>FfxFloat32x2</i></c> containing approximations of both sine and cosine of <c><i>value</i></c>.
FfxFloat32x2 ffxParabolicSinCos(FfxFloat32 x)
{
    FfxFloat32 y = ffxFract(x * FfxFloat32(0.5) + FfxFloat32(0.75));
    y = y * FfxFloat32(2.0) - FfxFloat32(1.0);
    return ffxParabolicSin(FfxFloat32x2(x, y));
}

/// Conditional free logic AND operation using values.
///
/// @param [in] x           The first value to be fed into the AND operator.
/// @param [in] y           The second value to be fed into the AND operator.
///
/// @returns
/// Result of the AND operation.
///
/// @ingroup GPU
FfxUInt32 ffxZeroOneAnd(FfxUInt32 x, FfxUInt32 y)
{
    return min(x, y);
}

/// Conditional free logic AND operation using two values.
///
/// @param [in] x           The first value to be fed into the AND operator.
/// @param [in] y           The second value to be fed into the AND operator.
///
/// @returns
/// Result of the AND operation.
///
/// @ingroup GPU
FfxUInt32x2 ffxZeroOneAnd(FfxUInt32x2 x, FfxUInt32x2 y)
{
    return min(x, y);
}

/// Conditional free logic AND operation using two values.
///
/// @param [in] x           The first value to be fed into the AND operator.
/// @param [in] y           The second value to be fed into the AND operator.
///
/// @returns
/// Result of the AND operation.
///
/// @ingroup GPU
FfxUInt32x3 ffxZeroOneAnd(FfxUInt32x3 x, FfxUInt32x3 y)
{
    return min(x, y);
}

/// Conditional free logic AND operation using two values.
///
/// @param [in] x           The first value to be fed into the AND operator.
/// @param [in] y           The second value to be fed into the AND operator.
///
/// @returns
/// Result of the AND operation.
///
/// @ingroup GPU
FfxUInt32x4 ffxZeroOneAnd(FfxUInt32x4 x, FfxUInt32x4 y)
{
    return min(x, y);
}

/// Conditional free logic NOT operation using two values.
///
/// @param [in] x           The first value to be fed into the NOT operator.
/// @param [in] y           The second value to be fed into the NOT operator.
///
/// @returns
/// Result of the NOT operation.
///
/// @ingroup GPU
FfxUInt32 ffxZeroOneAnd(FfxUInt32 x)
{
    return x ^ FfxUInt32(1);
}

/// Conditional free logic NOT operation using two values.
///
/// @param [in] x           The first value to be fed into the NOT operator.
/// @param [in] y           The second value to be fed into the NOT operator.
///
/// @returns
/// Result of the NOT operation.
///
/// @ingroup GPU
FfxUInt32x2 ffxZeroOneAnd(FfxUInt32x2 x)
{
    return x ^ ffxBroadcast2(1u);
}

/// Conditional free logic NOT operation using two values.
///
/// @param [in] x           The first value to be fed into the NOT operator.
/// @param [in] y           The second value to be fed into the NOT operator.
///
/// @returns
/// Result of the NOT operation.
///
/// @ingroup GPU
FfxUInt32x3 ffxZeroOneAnd(FfxUInt32x3 x)
{
    return x ^ ffxBroadcast3(1u);
}

/// Conditional free logic NOT operation using two values.
///
/// @param [in] x           The first value to be fed into the NOT operator.
/// @param [in] y           The second value to be fed into the NOT operator.
///
/// @returns
/// Result of the NOT operation.
///
/// @ingroup GPU
FfxUInt32x4 ffxZeroOneAnd(FfxUInt32x4 x)
{
    return x ^ ffxBroadcast4(1u);
}

/// Conditional free logic OR operation using two values.
///
/// @param [in] x           The first value to be fed into the OR operator.
/// @param [in] y           The second value to be fed into the OR operator.
///
/// @returns
/// Result of the OR operation.
///
/// @ingroup GPU
FfxUInt32 ffxZeroOneOr(FfxUInt32 x, FfxUInt32 y)
{
    return max(x, y);
}

/// Conditional free logic OR operation using two values.
///
/// @param [in] x           The first value to be fed into the OR operator.
/// @param [in] y           The second value to be fed into the OR operator.
///
/// @returns
/// Result of the OR operation.
///
/// @ingroup GPU
FfxUInt32x2 ffxZeroOneOr(FfxUInt32x2 x, FfxUInt32x2 y)
{
    return max(x, y);
}

/// Conditional free logic OR operation using two values.
///
/// @param [in] x           The first value to be fed into the OR operator.
/// @param [in] y           The second value to be fed into the OR operator.
///
/// @returns
/// Result of the OR operation.
///
/// @ingroup GPU
FfxUInt32x3 ffxZeroOneOr(FfxUInt32x3 x, FfxUInt32x3 y)
{
    return max(x, y);
}

/// Conditional free logic OR operation using two values.
///
/// @param [in] x           The first value to be fed into the OR operator.
/// @param [in] y           The second value to be fed into the OR operator.
///
/// @returns
/// Result of the OR operation.
///
/// @ingroup GPU
FfxUInt32x4 ffxZeroOneOr(FfxUInt32x4 x, FfxUInt32x4 y)
{
    return max(x, y);
}

/// Conditional free logic signed NOT operation using two half-precision FfxFloat32 values.
///
/// @param [in] x           The first value to be fed into the AND OR operator.
///
/// @returns
/// Result of the AND OR operation.
///
/// @ingroup GPU
FfxUInt32 ffxZeroOneAndToU1(FfxFloat32 x)
{
    return FfxUInt32(FfxFloat32(1.0) - x);
}

/// Conditional free logic signed NOT operation using two half-precision FfxFloat32 values.
///
/// @param [in] x           The first value to be fed into the AND OR operator.
///
/// @returns
/// Result of the AND OR operation.
///
/// @ingroup GPU
FfxUInt32x2 ffxZeroOneAndToU2(FfxFloat32x2 x)
{
    return FfxUInt32x2(ffxBroadcast2(1.0) - x);
}

/// Conditional free logic signed NOT operation using two half-precision FfxFloat32 values.
///
/// @param [in] x           The first value to be fed into the AND OR operator.
///
/// @returns
/// Result of the AND OR operation.
///
/// @ingroup GPU
FfxUInt32x3 ffxZeroOneAndToU3(FfxFloat32x3 x)
{
    return FfxUInt32x3(ffxBroadcast3(1.0) - x);
}

/// Conditional free logic signed NOT operation using two half-precision FfxFloat32 values.
///
/// @param [in] x           The first value to be fed into the AND OR operator.
///
/// @returns
/// Result of the AND OR operation.
///
/// @ingroup GPU
FfxUInt32x4 ffxZeroOneAndToU4(FfxFloat32x4 x)
{
    return FfxUInt32x4(ffxBroadcast4(1.0) - x);
}

/// Conditional free logic AND operation using two values followed by a NOT operation
/// using the resulting value and a third value.
///
/// @param [in] x           The first value to be fed into the AND operator.
/// @param [in] y           The second value to be fed into the AND operator.
/// @param [in] z           The second value to be fed into the OR operator.
///
/// @returns
/// Result of the AND OR operation.
///
/// @ingroup GPU
FfxFloat32 ffxZeroOneAndOr(FfxFloat32 x, FfxFloat32 y, FfxFloat32 z)
{
    return ffxSaturate(x * y + z);
}

/// Conditional free logic AND operation using two values followed by a NOT operation
/// using the resulting value and a third value.
///
/// @param [in] x           The first value to be fed into the AND operator.
/// @param [in] y           The second value to be fed into the AND operator.
/// @param [in] z           The second value to be fed into the OR operator.
///
/// @returns
/// Result of the AND OR operation.
///
/// @ingroup GPU
FfxFloat32x2 ffxZeroOneAndOr(FfxFloat32x2 x, FfxFloat32x2 y, FfxFloat32x2 z)
{
    return ffxSaturate(x * y + z);
}

/// Conditional free logic AND operation using two values followed by a NOT operation
/// using the resulting value and a third value.
///
/// @param [in] x           The first value to be fed into the AND operator.
/// @param [in] y           The second value to be fed into the AND operator.
/// @param [in] z           The second value to be fed into the OR operator.
///
/// @returns
/// Result of the AND OR operation.
///
/// @ingroup GPU
FfxFloat32x3 ffxZeroOneAndOr(FfxFloat32x3 x, FfxFloat32x3 y, FfxFloat32x3 z)
{
    return ffxSaturate(x * y + z);
}

/// Conditional free logic AND operation using two values followed by a NOT operation 
/// using the resulting value and a third value.
///
/// @param [in] x           The first value to be fed into the AND operator.
/// @param [in] y           The second value to be fed into the AND operator.
/// @param [in] z           The second value to be fed into the OR operator.
///
/// @returns
/// Result of the AND OR operation.
///
/// @ingroup GPU
FfxFloat32x4 ffxZeroOneAndOr(FfxFloat32x4 x, FfxFloat32x4 y, FfxFloat32x4 z)
{
    return ffxSaturate(x * y + z);
}

/// Given a value, returns 1.0 if greater than zero and 0.0 if not.
///
/// @param [in] x           The value to be compared.
///
/// @returns
/// Result of the greater than zero comparison.
///
/// @ingroup GPU
FfxFloat32 ffxZeroOneIsGreaterThanZero(FfxFloat32 x)
{
    return ffxSaturate(x * FfxFloat32(FFX_POSITIVE_INFINITY_FLOAT));
}

/// Given a value, returns 1.0 if greater than zero and 0.0 if not.
///
/// @param [in] x           The value to be compared.
///
/// @returns
/// Result of the greater than zero comparison.
///
/// @ingroup GPU
FfxFloat32x2 ffxZeroOneIsGreaterThanZero(FfxFloat32x2 x)
{
    return ffxSaturate(x * ffxBroadcast2(FFX_POSITIVE_INFINITY_FLOAT));
}

/// Given a value, returns 1.0 if greater than zero and 0.0 if not.
///
/// @param [in] x           The value to be compared.
///
/// @returns
/// Result of the greater than zero comparison.
///
/// @ingroup GPU
FfxFloat32x3 ffxZeroOneIsGreaterThanZero(FfxFloat32x3 x)
{
    return ffxSaturate(x * ffxBroadcast3(FFX_POSITIVE_INFINITY_FLOAT));
}

/// Given a value, returns 1.0 if greater than zero and 0.0 if not.
///
/// @param [in] x           The value to be compared.
///
/// @returns
/// Result of the greater than zero comparison.
///
/// @ingroup GPU
FfxFloat32x4 ffxZeroOneIsGreaterThanZero(FfxFloat32x4 x)
{
    return ffxSaturate(x * ffxBroadcast4(FFX_POSITIVE_INFINITY_FLOAT));
}

/// Conditional free logic signed NOT operation using two FfxFloat32 values.
///
/// @param [in] x           The first value to be fed into the AND OR operator.
///
/// @returns
/// Result of the AND OR operation.
///
/// @ingroup GPU
FfxFloat32 ffxZeroOneAnd(FfxFloat32 x)
{
    return FfxFloat32(1.0) - x;
}

/// Conditional free logic signed NOT operation using two FfxFloat32 values.
///
/// @param [in] x           The first value to be fed into the AND OR operator.
///
/// @returns
/// Result of the AND OR operation.
///
/// @ingroup GPU
FfxFloat32x2 ffxZeroOneAnd(FfxFloat32x2 x)
{
    return ffxBroadcast2(1.0) - x;
}

/// Conditional free logic signed NOT operation using two FfxFloat32 values.
///
/// @param [in] x           The first value to be fed into the AND OR operator.
///
/// @returns
/// Result of the AND OR operation.
///
/// @ingroup GPU
FfxFloat32x3 ffxZeroOneAnd(FfxFloat32x3 x)
{
    return ffxBroadcast3(1.0) - x;
}

/// Conditional free logic signed NOT operation using two FfxFloat32 values.
///
/// @param [in] x           The first value to be fed into the AND OR operator.
///
/// @returns
/// Result of the AND OR operation.
///
/// @ingroup GPU
FfxFloat32x4 ffxZeroOneAnd(FfxFloat32x4 x)
{
    return ffxBroadcast4(1.0) - x;
}

/// Conditional free logic OR operation using two FfxFloat32 values.
///
/// @param [in] x           The first value to be fed into the OR operator.
/// @param [in] y           The second value to be fed into the OR operator.
///
/// @returns
/// Result of the OR operation.
///
/// @ingroup GPU
FfxFloat32 ffxZeroOneOr(FfxFloat32 x, FfxFloat32 y)
{
    return max(x, y);
}

/// Conditional free logic OR operation using two FfxFloat32 values.
///
/// @param [in] x           The first value to be fed into the OR operator.
/// @param [in] y           The second value to be fed into the OR operator.
///
/// @returns
/// Result of the OR operation.
///
/// @ingroup GPU
FfxFloat32x2 ffxZeroOneOr(FfxFloat32x2 x, FfxFloat32x2 y)
{
    return max(x, y);
}

/// Conditional free logic OR operation using two FfxFloat32 values.
///
/// @param [in] x           The first value to be fed into the OR operator.
/// @param [in] y           The second value to be fed into the OR operator.
///
/// @returns
/// Result of the OR operation.
///
/// @ingroup GPU
FfxFloat32x3 ffxZeroOneOr(FfxFloat32x3 x, FfxFloat32x3 y)
{
    return max(x, y);
}

/// Conditional free logic OR operation using two FfxFloat32 values.
///
/// @param [in] x           The first value to be fed into the OR operator.
/// @param [in] y           The second value to be fed into the OR operator.
///
/// @returns
/// Result of the OR operation.
///
/// @ingroup GPU
FfxFloat32x4 ffxZeroOneOr(FfxFloat32x4 x, FfxFloat32x4 y)
{
    return max(x, y);
}

/// Choose between two FfxFloat32 values if the first paramter is greater than zero.
///
/// @param [in] x           The value to compare against zero.
/// @param [in] y           The value to return if the comparision is greater than zero.
/// @param [in] z           The value to return if the comparision is less than or equal to zero.
///
/// @returns
/// The selected value.
///
/// @ingroup GPU
FfxFloat32 ffxZeroOneSelect(FfxFloat32 x, FfxFloat32 y, FfxFloat32 z)
{
    FfxFloat32 r = (-x) * z + z;
    return x * y + r;
}

/// Choose between two FfxFloat32 values if the first paramter is greater than zero.
///
/// @param [in] x           The value to compare against zero.
/// @param [in] y           The value to return if the comparision is greater than zero.
/// @param [in] z           The value to return if the comparision is less than or equal to zero.
///
/// @returns
/// The selected value.
///
/// @ingroup GPU
FfxFloat32x2 ffxZeroOneSelect(FfxFloat32x2 x, FfxFloat32x2 y, FfxFloat32x2 z)
{
    FfxFloat32x2 r = (-x) * z + z;
    return x * y + r;
}

/// Choose between two FfxFloat32 values if the first paramter is greater than zero.
///
/// @param [in] x           The value to compare against zero.
/// @param [in] y           The value to return if the comparision is greater than zero.
/// @param [in] z           The value to return if the comparision is less than or equal to zero.
///
/// @returns
/// The selected value.
///
/// @ingroup GPU
FfxFloat32x3 ffxZeroOneSelect(FfxFloat32x3 x, FfxFloat32x3 y, FfxFloat32x3 z)
{
    FfxFloat32x3 r = (-x) * z + z;
    return x * y + r;
}

/// Choose between two FfxFloat32 values if the first paramter is greater than zero.
///
/// @param [in] x           The value to compare against zero.
/// @param [in] y           The value to return if the comparision is greater than zero.
/// @param [in] z           The value to return if the comparision is less than or equal to zero.
///
/// @returns
/// The selected value.
///
/// @ingroup GPU
FfxFloat32x4 ffxZeroOneSelect(FfxFloat32x4 x, FfxFloat32x4 y, FfxFloat32x4 z)
{
    FfxFloat32x4 r = (-x) * z + z;
    return x * y + r;
}

/// Given a value, returns 1.0 if less than zero and 0.0 if not.
///
/// @param [in] x           The value to be compared.
///
/// @returns
/// Result of the sign value.
///
/// @ingroup GPU
FfxFloat32 ffxZeroOneIsSigned(FfxFloat32 x)
{
    return ffxSaturate(x * FfxFloat32(FFX_NEGATIVE_INFINITY_FLOAT));
}

/// Given a value, returns 1.0 if less than zero and 0.0 if not.
///
/// @param [in] x           The value to be compared.
///
/// @returns
/// Result of the sign value.
///
/// @ingroup GPU
FfxFloat32x2 ffxZeroOneIsSigned(FfxFloat32x2 x)
{
    return ffxSaturate(x * ffxBroadcast2(FFX_NEGATIVE_INFINITY_FLOAT));
}

/// Given a value, returns 1.0 if less than zero and 0.0 if not.
///
/// @param [in] x           The value to be compared.
///
/// @returns
/// Result of the sign value.
///
/// @ingroup GPU
FfxFloat32x3 ffxZeroOneIsSigned(FfxFloat32x3 x)
{
    return ffxSaturate(x * ffxBroadcast3(FFX_NEGATIVE_INFINITY_FLOAT));
}

/// Given a value, returns 1.0 if less than zero and 0.0 if not.
///
/// @param [in] x           The value to be compared.
///
/// @returns
/// Result of the sign value.
///
/// @ingroup GPU
FfxFloat32x4 ffxZeroOneIsSigned(FfxFloat32x4 x)
{
    return ffxSaturate(x * ffxBroadcast4(FFX_NEGATIVE_INFINITY_FLOAT));
}

/// Compute a Rec.709 color space.
/// 
/// Rec.709 is used for some HDTVs.
/// 
/// Both Rec.709 and sRGB have a linear segment which as spec'ed would intersect the curved segment 2 times.
///  (a.) For 8-bit sRGB, steps {0 to 10.3} are in the linear region (4% of the encoding range).
///  (b.) For 8-bit  709, steps {0 to 20.7} are in the linear region (8% of the encoding range).
///
/// @param [in] color           The color to convert to Rec. 709.
/// 
/// @returns
/// The <c><i>color</i></c> in linear space.
/// 
/// @ingroup GPU
FfxFloat32 ffxRec709FromLinear(FfxFloat32 color)
{
    FfxFloat32x3 j = FfxFloat32x3(0.018 * 4.5, 4.5, 0.45);
    FfxFloat32x2 k = FfxFloat32x2(1.099, -0.099);
    return clamp(j.x, color * j.y, pow(color, j.z) * k.x + k.y);
}

/// Compute a Rec.709 color space.
///
/// Rec.709 is used for some HDTVs.
///
/// Both Rec.709 and sRGB have a linear segment which as spec'ed would intersect the curved segment 2 times.
///  (a.) For 8-bit sRGB, steps {0 to 10.3} are in the linear region (4% of the encoding range).
///  (b.) For 8-bit  709, steps {0 to 20.7} are in the linear region (8% of the encoding range).
///
/// @param [in] color           The color to convert to Rec. 709.
///
/// @returns
/// The <c><i>color</i></c> in linear space.
///
/// @ingroup GPU
FfxFloat32x2 ffxRec709FromLinear(FfxFloat32x2 color)
{
    FfxFloat32x3 j = FfxFloat32x3(0.018 * 4.5, 4.5, 0.45);
    FfxFloat32x2 k = FfxFloat32x2(1.099, -0.099);
    return clamp(j.xx, color * j.yy, pow(color, j.zz) * k.xx + k.yy);
}

/// Compute a Rec.709 color space.
///
/// Rec.709 is used for some HDTVs.
///
/// Both Rec.709 and sRGB have a linear segment which as spec'ed would intersect the curved segment 2 times.
///  (a.) For 8-bit sRGB, steps {0 to 10.3} are in the linear region (4% of the encoding range).
///  (b.) For 8-bit  709, steps {0 to 20.7} are in the linear region (8% of the encoding range).
///
/// @param [in] color           The color to convert to Rec. 709.
///
/// @returns
/// The <c><i>color</i></c> in linear space.
///
/// @ingroup GPU
FfxFloat32x3 ffxRec709FromLinear(FfxFloat32x3 color)
{
    FfxFloat32x3 j = FfxFloat32x3(0.018 * 4.5, 4.5, 0.45);
    FfxFloat32x2 k = FfxFloat32x2(1.099, -0.099);
    return clamp(j.xxx, color * j.yyy, pow(color, j.zzz) * k.xxx + k.yyy);
}

/// Compute a gamma value from a linear value.
///
/// Typically 2.2 for some PC displays, or 2.4-2.5 for CRTs, or 2.2 FreeSync2 native.
/// 
/// Note: 'rcpX' is '1/x', where the 'x' is what would be used in <c><i>ffxLinearFromGamma</i></c>.
/// 
/// @param [in] value           The value to convert to gamma space from linear.
/// @param [in] power           The reciprocal of power value used for the gamma curve.
///
/// @returns
/// A value in gamma space.
///
/// @ingroup GPU
FfxFloat32 ffxGammaFromLinear(FfxFloat32 color, FfxFloat32 rcpX)
{
    return pow(color, FfxFloat32(rcpX));
}

/// Compute a gamma value from a linear value.
///
/// Typically 2.2 for some PC displays, or 2.4-2.5 for CRTs, or 2.2 FreeSync2 native.
/// 
/// Note: 'rcpX' is '1/x', where the 'x' is what would be used in <c><i>ffxLinearFromGamma</i></c>.
///
/// @param [in] value           The value to convert to gamma space from linear.
/// @param [in] power           The reciprocal of power value used for the gamma curve.
///
/// @returns
/// A value in gamma space.
///
/// @ingroup GPU
FfxFloat32x2 ffxGammaFromLinear(FfxFloat32x2 color, FfxFloat32 rcpX)
{
    return pow(color, ffxBroadcast2(rcpX));
}

/// Compute a gamma value from a linear value.
///
/// Typically 2.2 for some PC displays, or 2.4-2.5 for CRTs, or 2.2 FreeSync2 native.
///
/// Note: 'rcpX' is '1/x', where the 'x' is what would be used in <c><i>ffxLinearFromGamma</i></c>.
///
/// @param [in] value           The value to convert to gamma space from linear.
/// @param [in] power           The reciprocal of power value used for the gamma curve.
///
/// @returns
/// A value in gamma space.
///
/// @ingroup GPU
FfxFloat32x3 ffxGammaFromLinear(FfxFloat32x3 color, FfxFloat32 rcpX)
{
    return pow(color, ffxBroadcast3(rcpX));
}

/// Compute a PQ value from a linear value.
///
/// @param [in] value           The value to convert to PQ from linear.
///
/// @returns
/// A value in linear space.
///
/// @ingroup GPU
FfxFloat32 ffxPQToLinear(FfxFloat32 x)
{
    FfxFloat32 p = pow(x, FfxFloat32(0.159302));
    return pow((FfxFloat32(0.835938) + FfxFloat32(18.8516) * p) / (FfxFloat32(1.0) + FfxFloat32(18.6875) * p), FfxFloat32(78.8438));
}

/// Compute a PQ value from a linear value.
///
/// @param [in] value           The value to convert to PQ from linear.
///
/// @returns
/// A value in linear space.
///
/// @ingroup GPU
FfxFloat32x2 ffxPQToLinear(FfxFloat32x2 x)
{
    FfxFloat32x2 p = pow(x, ffxBroadcast2(0.159302));
    return pow((ffxBroadcast2(0.835938) + ffxBroadcast2(18.8516) * p) / (ffxBroadcast2(1.0) + ffxBroadcast2(18.6875) * p), ffxBroadcast2(78.8438));
}

/// Compute a PQ value from a linear value.
///
/// @param [in] value           The value to convert to PQ from linear.
///
/// @returns
/// A value in linear space.
///
/// @ingroup GPU
FfxFloat32x3 ffxPQToLinear(FfxFloat32x3 x)
{
    FfxFloat32x3 p = pow(x, ffxBroadcast3(0.159302));
    return pow((ffxBroadcast3(0.835938) + ffxBroadcast3(18.8516) * p) / (ffxBroadcast3(1.0) + ffxBroadcast3(18.6875) * p), ffxBroadcast3(78.8438));
}

/// Compute a linear value from a SRGB value.
///
/// @param [in] value           The value to convert to linear from SRGB.
///
/// @returns
/// A value in SRGB space.
///
/// @ingroup GPU
FfxFloat32 ffxSrgbToLinear(FfxFloat32 color)
{
    FfxFloat32x3 j = FfxFloat32x3(0.0031308 * 12.92, 12.92, 1.0 / 2.4);
    FfxFloat32x2 k = FfxFloat32x2(1.055, -0.055);
    return clamp(j.x, color * j.y, pow(color, j.z) * k.x + k.y);
}

/// Compute a linear value from a SRGB value.
///
/// @param [in] value           The value to convert to linear from SRGB.
///
/// @returns
/// A value in SRGB space.
///
/// @ingroup GPU
FfxFloat32x2 ffxSrgbToLinear(FfxFloat32x2 color)
{
    FfxFloat32x3 j = FfxFloat32x3(0.0031308 * 12.92, 12.92, 1.0 / 2.4);
    FfxFloat32x2 k = FfxFloat32x2(1.055, -0.055);
    return clamp(j.xx, color * j.yy, pow(color, j.zz) * k.xx + k.yy);
}

/// Compute a linear value from a SRGB value.
///
/// @param [in] value           The value to convert to linear from SRGB.
///
/// @returns
/// A value in SRGB space.
///
/// @ingroup GPU
FfxFloat32x3 ffxSrgbToLinear(FfxFloat32x3 color)
{
    FfxFloat32x3 j = FfxFloat32x3(0.0031308 * 12.92, 12.92, 1.0 / 2.4);
    FfxFloat32x2 k = FfxFloat32x2(1.055, -0.055);
    return clamp(j.xxx, color * j.yyy, pow(color, j.zzz) * k.xxx + k.yyy);
}

/// Compute a linear value from a REC.709 value.
///
/// @param [in] color           The value to convert to linear from REC.709.
///
/// @returns
/// A value in linear space.
///
/// @ingroup GPU
FfxFloat32 ffxLinearFromRec709(FfxFloat32 color)
{
    FfxFloat32x3 j = FfxFloat32x3(0.081 / 4.5, 1.0 / 4.5, 1.0 / 0.45);
    FfxFloat32x2 k = FfxFloat32x2(1.0 / 1.099, 0.099 / 1.099);
    return ffxZeroOneSelect(ffxZeroOneIsSigned(color - j.x), color * j.y, pow(color * k.x + k.y, j.z));
}

/// Compute a linear value from a REC.709 value.
///
/// @param [in] color           The value to convert to linear from REC.709.
///
/// @returns
/// A value in linear space.
///
/// @ingroup GPU
FfxFloat32x2 ffxLinearFromRec709(FfxFloat32x2 color)
{
    FfxFloat32x3 j = FfxFloat32x3(0.081 / 4.5, 1.0 / 4.5, 1.0 / 0.45);
    FfxFloat32x2 k = FfxFloat32x2(1.0 / 1.099, 0.099 / 1.099);
    return ffxZeroOneSelect(ffxZeroOneIsSigned(color - j.xx), color * j.yy, pow(color * k.xx + k.yy, j.zz));
}

/// Compute a linear value from a REC.709 value.
///
/// @param [in] color           The value to convert to linear from REC.709.
///
/// @returns
/// A value in linear space.
///
/// @ingroup GPU
FfxFloat32x3 ffxLinearFromRec709(FfxFloat32x3 color)
{
    FfxFloat32x3 j = FfxFloat32x3(0.081 / 4.5, 1.0 / 4.5, 1.0 / 0.45);
    FfxFloat32x2 k = FfxFloat32x2(1.0 / 1.099, 0.099 / 1.099);
    return ffxZeroOneSelect(ffxZeroOneIsSigned(color - j.xxx), color * j.yyy, pow(color * k.xxx + k.yyy, j.zzz));
}

/// Compute a linear value from a value in a gamma space.
///
/// Typically 2.2 for some PC displays, or 2.4-2.5 for CRTs, or 2.2 FreeSync2 native.
///
/// @param [in] color           The value to convert to linear in gamma space.
/// @param [in] power           The power value used for the gamma curve.
///
/// @returns
/// A value in linear space.
///
/// @ingroup GPU
FfxFloat32 ffxLinearFromGamma(FfxFloat32 color, FfxFloat32 power)
{
    return pow(color, FfxFloat32(power));
}

/// Compute a linear value from a value in a gamma space.
///
/// Typically 2.2 for some PC displays, or 2.4-2.5 for CRTs, or 2.2 FreeSync2 native.
///
/// @param [in] color           The value to convert to linear in gamma space.
/// @param [in] power           The power value used for the gamma curve.
///
/// @returns
/// A value in linear space.
///
/// @ingroup GPU
FfxFloat32x2 ffxLinearFromGamma(FfxFloat32x2 color, FfxFloat32 power)
{
    return pow(color, ffxBroadcast2(power));
}

/// Compute a linear value from a value in a gamma space.
///
/// Typically 2.2 for some PC displays, or 2.4-2.5 for CRTs, or 2.2 FreeSync2 native.
///
/// @param [in] color           The value to convert to linear in gamma space.
/// @param [in] power           The power value used for the gamma curve.
///
/// @returns
/// A value in linear space.
///
/// @ingroup GPU
FfxFloat32x3 ffxLinearFromGamma(FfxFloat32x3 color, FfxFloat32 power)
{
    return pow(color, ffxBroadcast3(power));
}

/// Compute a linear value from a value in a PQ space.
///
/// Typically 2.2 for some PC displays, or 2.4-2.5 for CRTs, or 2.2 FreeSync2 native.
///
/// @param [in] value           The value to convert to linear in PQ space.
///
/// @returns
/// A value in linear space.
///
/// @ingroup GPU
FfxFloat32 ffxLinearFromPQ(FfxFloat32 x)
{
    FfxFloat32 p = pow(x, FfxFloat32(0.0126833));
    return pow(ffxSaturate(p - FfxFloat32(0.835938)) / (FfxFloat32(18.8516) - FfxFloat32(18.6875) * p), FfxFloat32(6.27739));
}

/// Compute a linear value from a value in a PQ space.
///
/// Typically 2.2 for some PC displays, or 2.4-2.5 for CRTs, or 2.2 FreeSync2 native.
///
/// @param [in] value           The value to convert to linear in PQ space.
///
/// @returns
/// A value in linear space.
///
/// @ingroup GPU
FfxFloat32x2 ffxLinearFromPQ(FfxFloat32x2 x)
{
    FfxFloat32x2 p = pow(x, ffxBroadcast2(0.0126833));
    return pow(ffxSaturate(p - ffxBroadcast2(0.835938)) / (ffxBroadcast2(18.8516) - ffxBroadcast2(18.6875) * p), ffxBroadcast2(6.27739));
}

/// Compute a linear value from a value in a PQ space.
///
/// Typically 2.2 for some PC displays, or 2.4-2.5 for CRTs, or 2.2 FreeSync2 native.
///
/// @param [in] value           The value to convert to linear in PQ space.
///
/// @returns
/// A value in linear space.
///
/// @ingroup GPU
FfxFloat32x3 ffxLinearFromPQ(FfxFloat32x3 x)
{
    FfxFloat32x3 p = pow(x, ffxBroadcast3(0.0126833));
    return pow(ffxSaturate(p - ffxBroadcast3(0.835938)) / (ffxBroadcast3(18.8516) - ffxBroadcast3(18.6875) * p), ffxBroadcast3(6.27739));
}

/// Compute a linear value from a value in a SRGB space.
///
/// Typically 2.2 for some PC displays, or 2.4-2.5 for CRTs, or 2.2 FreeSync2 native.
///
/// @param [in] value           The value to convert to linear in SRGB space.
///
/// @returns
/// A value in linear space.
///
/// @ingroup GPU
FfxFloat32 ffxLinearFromSrgb(FfxFloat32 color)
{
    FfxFloat32x3 j = FfxFloat32x3(0.04045 / 12.92, 1.0 / 12.92, 2.4);
    FfxFloat32x2 k = FfxFloat32x2(1.0 / 1.055, 0.055 / 1.055);
    return ffxZeroOneSelect(ffxZeroOneIsSigned(color - j.x), color * j.y, pow(color * k.x + k.y, j.z));
}

/// Compute a linear value from a value in a SRGB space.
///
/// Typically 2.2 for some PC displays, or 2.4-2.5 for CRTs, or 2.2 FreeSync2 native.
///
/// @param [in] value           The value to convert to linear in SRGB space.
///
/// @returns
/// A value in linear space.
///
/// @ingroup GPU
FfxFloat32x2 ffxLinearFromSrgb(FfxFloat32x2 color)
{
    FfxFloat32x3 j = FfxFloat32x3(0.04045 / 12.92, 1.0 / 12.92, 2.4);
    FfxFloat32x2 k = FfxFloat32x2(1.0 / 1.055, 0.055 / 1.055);
    return ffxZeroOneSelect(ffxZeroOneIsSigned(color - j.xx), color * j.yy, pow(color * k.xx + k.yy, j.zz));
}

/// Compute a linear value from a value in a SRGB space.
///
/// Typically 2.2 for some PC displays, or 2.4-2.5 for CRTs, or 2.2 FreeSync2 native.
///
/// @param [in] value           The value to convert to linear in SRGB space.
///
/// @returns
/// A value in linear space.
///
/// @ingroup GPU
FfxFloat32x3 ffxLinearFromSrgb(FfxFloat32x3 color)
{
    FfxFloat32x3 j = FfxFloat32x3(0.04045 / 12.92, 1.0 / 12.92, 2.4);
    FfxFloat32x2 k = FfxFloat32x2(1.0 / 1.055, 0.055 / 1.055);
    return ffxZeroOneSelect(ffxZeroOneIsSigned(color - j.xxx), color * j.yyy, pow(color * k.xxx + k.yyy, j.zzz));
}

/// A remapping of 64x1 to 8x8 imposing rotated 2x2 pixel quads in quad linear.
/// 
///  543210
///  ======
///  ..xxx.
///  yy...y
/// 
/// @param [in] a       The input 1D coordinates to remap.
///
/// @returns
/// The remapped 2D coordinates.
///
/// @ingroup GPU
FfxUInt32x2 ffxRemapForQuad(FfxUInt32 a)
{
    return FfxUInt32x2(bitfieldExtract(a, 1u, 3u), bitfieldInsertMask(bitfieldExtract(a, 3u, 3u), a, 1u));
}

/// A helper function performing a remap 64x1 to 8x8 remapping which is necessary for 2D wave reductions.
///
/// The 64-wide lane indices to 8x8 remapping is performed as follows:
/// 
///     00 01 08 09 10 11 18 19
///     02 03 0a 0b 12 13 1a 1b
///     04 05 0c 0d 14 15 1c 1d
///     06 07 0e 0f 16 17 1e 1f
///     20 21 28 29 30 31 38 39
///     22 23 2a 2b 32 33 3a 3b
///     24 25 2c 2d 34 35 3c 3d
///     26 27 2e 2f 36 37 3e 3f
///
/// @param [in] a       The input 1D coordinate to remap.
/// 
/// @returns
/// The remapped 2D coordinates.
/// 
/// @ingroup GPU
FfxUInt32x2 ffxRemapForWaveReduction(FfxUInt32 a)
{
    return FfxUInt32x2(bitfieldInsertMask(bitfieldExtract(a, 2u, 3u), a, 1u), bitfieldInsertMask(bitfieldExtract(a, 3u, 3u), bitfieldExtract(a, 1u, 2u), 2u));
}
