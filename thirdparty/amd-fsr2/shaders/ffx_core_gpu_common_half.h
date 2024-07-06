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

#if FFX_HALF
#if FFX_HLSL_6_2
/// A define value for 16bit positive infinity.
///
/// @ingroup GPU
#define FFX_POSITIVE_INFINITY_HALF FFX_TO_FLOAT16((uint16_t)0x7c00u)

/// A define value for 16bit negative infinity.
///
/// @ingroup GPU
#define FFX_NEGATIVE_INFINITY_HALF FFX_TO_FLOAT16((uint16_t)0xfc00u)
#else
/// A define value for 16bit positive infinity.
///
/// @ingroup GPU
#define FFX_POSITIVE_INFINITY_HALF FFX_TO_FLOAT16(0x7c00u)

/// A define value for 16bit negative infinity.
///
/// @ingroup GPU
#define FFX_NEGATIVE_INFINITY_HALF FFX_TO_FLOAT16(0xfc00u)
#endif // FFX_HLSL_6_2

/// Compute the min of two values.
///
/// @param [in] x                   The first value to compute the min of.
/// @param [in] y                   The second value to compute the min of.
///
/// @returns
/// The the lowest of two values.
///
/// @ingroup GPU
FfxFloat16 ffxMin(FfxFloat16 x, FfxFloat16 y)
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
FfxFloat16x2 ffxMin(FfxFloat16x2 x, FfxFloat16x2 y)
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
FfxFloat16x3 ffxMin(FfxFloat16x3 x, FfxFloat16x3 y)
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
FfxFloat16x4 ffxMin(FfxFloat16x4 x, FfxFloat16x4 y)
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
FfxInt16 ffxMin(FfxInt16 x, FfxInt16 y)
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
FfxInt16x2 ffxMin(FfxInt16x2 x, FfxInt16x2 y)
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
FfxInt16x3 ffxMin(FfxInt16x3 x, FfxInt16x3 y)
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
FfxInt16x4 ffxMin(FfxInt16x4 x, FfxInt16x4 y)
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
FfxUInt16 ffxMin(FfxUInt16 x, FfxUInt16 y)
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
FfxUInt16x2 ffxMin(FfxUInt16x2 x, FfxUInt16x2 y)
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
FfxUInt16x3 ffxMin(FfxUInt16x3 x, FfxUInt16x3 y)
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
FfxUInt16x4 ffxMin(FfxUInt16x4 x, FfxUInt16x4 y)
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
FfxFloat16 ffxMax(FfxFloat16 x, FfxFloat16 y)
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
FfxFloat16x2 ffxMax(FfxFloat16x2 x, FfxFloat16x2 y)
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
FfxFloat16x3 ffxMax(FfxFloat16x3 x, FfxFloat16x3 y)
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
FfxFloat16x4 ffxMax(FfxFloat16x4 x, FfxFloat16x4 y)
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
FfxInt16 ffxMax(FfxInt16 x, FfxInt16 y)
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
FfxInt16x2 ffxMax(FfxInt16x2 x, FfxInt16x2 y)
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
FfxInt16x3 ffxMax(FfxInt16x3 x, FfxInt16x3 y)
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
FfxInt16x4 ffxMax(FfxInt16x4 x, FfxInt16x4 y)
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
FfxUInt16 ffxMax(FfxUInt16 x, FfxUInt16 y)
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
FfxUInt16x2 ffxMax(FfxUInt16x2 x, FfxUInt16x2 y)
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
FfxUInt16x3 ffxMax(FfxUInt16x3 x, FfxUInt16x3 y)
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
FfxUInt16x4 ffxMax(FfxUInt16x4 x, FfxUInt16x4 y)
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
FfxFloat16 ffxPow(FfxFloat16 x, FfxFloat16 y)
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
FfxFloat16x2 ffxPow(FfxFloat16x2 x, FfxFloat16x2 y)
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
FfxFloat16x3 ffxPow(FfxFloat16x3 x, FfxFloat16x3 y)
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
FfxFloat16x4 ffxPow(FfxFloat16x4 x, FfxFloat16x4 y)
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
FfxFloat16 ffxSqrt(FfxFloat16 x)
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
FfxFloat16x2 ffxSqrt(FfxFloat16x2 x)
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
FfxFloat16x3 ffxSqrt(FfxFloat16x3 x)
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
FfxFloat16x4 ffxSqrt(FfxFloat16x4 x)
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
FfxFloat16 ffxCopySignBitHalf(FfxFloat16 d, FfxFloat16 s)
{
    return FFX_TO_FLOAT16(FFX_TO_UINT16(d) | (FFX_TO_UINT16(s) & FFX_BROADCAST_UINT16(0x8000u)));
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
FfxFloat16x2 ffxCopySignBitHalf(FfxFloat16x2 d, FfxFloat16x2 s)
{
    return FFX_TO_FLOAT16X2(FFX_TO_UINT16X2(d) | (FFX_TO_UINT16X2(s) & FFX_BROADCAST_UINT16X2(0x8000u)));
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
FfxFloat16x3 ffxCopySignBitHalf(FfxFloat16x3 d, FfxFloat16x3 s)
{
    return FFX_TO_FLOAT16X3(FFX_TO_UINT16X3(d) | (FFX_TO_UINT16X3(s) & FFX_BROADCAST_UINT16X3(0x8000u)));
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
FfxFloat16x4 ffxCopySignBitHalf(FfxFloat16x4 d, FfxFloat16x4 s)
{
    return FFX_TO_FLOAT16X4(FFX_TO_UINT16X4(d) | (FFX_TO_UINT16X4(s) & FFX_BROADCAST_UINT16X4(0x8000u)));
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
FfxFloat16 ffxIsSignedHalf(FfxFloat16 m)
{
    return ffxSaturate(m * FFX_BROADCAST_FLOAT16(FFX_NEGATIVE_INFINITY_HALF));
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
FfxFloat16x2 ffxIsSignedHalf(FfxFloat16x2 m)
{
    return ffxSaturate(m * FFX_BROADCAST_FLOAT16X2(FFX_NEGATIVE_INFINITY_HALF));
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
FfxFloat16x3 ffxIsSignedHalf(FfxFloat16x3 m)
{
    return ffxSaturate(m * FFX_BROADCAST_FLOAT16X3(FFX_NEGATIVE_INFINITY_HALF));
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
FfxFloat16x4 ffxIsSignedHalf(FfxFloat16x4 m)
{
    return ffxSaturate(m * FFX_BROADCAST_FLOAT16X4(FFX_NEGATIVE_INFINITY_HALF));
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
FfxFloat16 ffxIsGreaterThanZeroHalf(FfxFloat16 m)
{
    return ffxSaturate(m * FFX_BROADCAST_FLOAT16(FFX_POSITIVE_INFINITY_HALF));
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
FfxFloat16x2 ffxIsGreaterThanZeroHalf(FfxFloat16x2 m)
{
    return ffxSaturate(m * FFX_BROADCAST_FLOAT16X2(FFX_POSITIVE_INFINITY_HALF));
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
FfxFloat16x3 ffxIsGreaterThanZeroHalf(FfxFloat16x3 m)
{
    return ffxSaturate(m * FFX_BROADCAST_FLOAT16X3(FFX_POSITIVE_INFINITY_HALF));
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
FfxFloat16x4 ffxIsGreaterThanZeroHalf(FfxFloat16x4 m)
{
    return ffxSaturate(m * FFX_BROADCAST_FLOAT16X4(FFX_POSITIVE_INFINITY_HALF));
}

/// Convert a 16bit floating point value to sortable integer.
/// 
///  - If sign bit=0, flip the sign bit (positives).
///  - If sign bit=1, flip all bits     (negatives).
/// 
/// The function has the side effects that:
///  - Larger integers are more positive values.
///  - Float zero is mapped to center of integers (so clear to integer zero is a nice default for atomic max usage).
/// 
/// @param [in] x                       The floating point value to make sortable.
/// 
/// @returns
/// The sortable integer value.
/// 
/// @ingroup GPU
FfxUInt16 ffxFloatToSortableIntegerHalf(FfxUInt16 x)
{
    return x ^ ((ffxBitShiftRightHalf(x, FFX_BROADCAST_UINT16(15))) | FFX_BROADCAST_UINT16(0x8000));
}

/// Convert a sortable integer to a 16bit floating point value.
///
/// The function has the side effects that:
///  - If sign bit=1, flip the sign bit (positives).
///  - If sign bit=0, flip all bits     (negatives).
///
/// @param [in] x                       The sortable integer value to make floating point.
///
/// @returns
/// The floating point value.
///
/// @ingroup GPU
FfxUInt16 ffxSortableIntegerToFloatHalf(FfxUInt16 x)
{
    return x ^ ((~ffxBitShiftRightHalf(x, FFX_BROADCAST_UINT16(15))) | FFX_BROADCAST_UINT16(0x8000));
}

/// Convert a pair of 16bit floating point values to a pair of sortable integers.
/// 
///  - If sign bit=0, flip the sign bit (positives).
///  - If sign bit=1, flip all bits     (negatives).
/// 
/// The function has the side effects that:
///  - Larger integers are more positive values.
///  - Float zero is mapped to center of integers (so clear to integer zero is a nice default for atomic max usage).
/// 
/// @param [in] x                       The floating point values to make sortable.
/// 
/// @returns
/// The sortable integer values.
/// 
/// @ingroup GPU
FfxUInt16x2 ffxFloatToSortableIntegerHalf(FfxUInt16x2 x)
{
    return x ^ ((ffxBitShiftRightHalf(x, FFX_BROADCAST_UINT16X2(15))) | FFX_BROADCAST_UINT16X2(0x8000));
}

/// Convert a pair of sortable integers to a pair of 16bit floating point values.
///
/// The function has the side effects that:
///  - If sign bit=1, flip the sign bit (positives).
///  - If sign bit=0, flip all bits     (negatives).
///
/// @param [in] x                       The sortable integer values to make floating point.
///
/// @returns
/// The floating point values.
///
/// @ingroup GPU
FfxUInt16x2 ffxSortableIntegerToFloatHalf(FfxUInt16x2 x)
{
    return x ^ ((~ffxBitShiftRightHalf(x, FFX_BROADCAST_UINT16X2(15))) | FFX_BROADCAST_UINT16X2(0x8000));
}

/// Packs the bytes from the X and Y components of a FfxUInt32x2 into a single 32-bit integer.
///
/// The resulting integer will contain bytes in the following order, from most to least significant:
/// [Zero] Y0 [Zero] X0
///
/// @param [in] i                       The integer pair to pack.
///
/// @returns
/// The packed integer value.
///
/// @ingroup GPU
FfxUInt32 ffxPackBytesZeroY0ZeroX0(FfxUInt32x2 i)
{
    return ((i.x) & 0xffu) | ((i.y << 16) & 0xff0000u);
}

/// Packs the bytes from the X and Y components of a FfxUInt32x2 into a single 32-bit integer.
///
/// The resulting integer will contain bytes in the following order, from most to least significant:
/// [Zero] Y1 [Zero] X1
///
/// @param [in] i                       The integer pair to pack.
///
/// @returns
/// The packed integer value.
///
/// @ingroup GPU
FfxUInt32 ffxPackBytesZeroY1ZeroX1(FfxUInt32x2 i)
{
    return ((i.x >> 8) & 0xffu) | ((i.y << 8) & 0xff0000u);
}

/// Packs the bytes from the X and Y components of a FfxUInt32x2 into a single 32-bit integer.
///
/// The resulting integer will contain bytes in the following order, from most to least significant:
/// [Zero] Y2 [Zero] X2
///
/// @param [in] i                       The integer pair to pack.
///
/// @returns
/// The packed integer value.
///
/// @ingroup GPU
FfxUInt32 ffxPackBytesZeroY2ZeroX2(FfxUInt32x2 i)
{
    return ((i.x >> 16) & 0xffu) | ((i.y) & 0xff0000u);
}

/// Packs the bytes from the X and Y components of a FfxUInt32x2 into a single 32-bit integer.
///
/// The resulting integer will contain bytes in the following order, from most to least significant:
/// [Zero] Y3 [Zero] X3
///
/// @param [in] i                       The integer pair to pack.
///
/// @returns
/// The packed integer value.
///
/// @ingroup GPU
FfxUInt32 ffxPackBytesZeroY3ZeroX3(FfxUInt32x2 i)
{
    return ((i.x >> 24) & 0xffu) | ((i.y >> 8) & 0xff0000u);
}

/// Packs the bytes from the X and Y components of a FfxUInt32x2 into a single 32-bit integer.
///
/// The resulting integer will contain bytes in the following order, from most to least significant:
/// Y3 Y2 Y1 X0
///
/// @param [in] i                       The integer pair to pack.
///
/// @returns
/// The packed integer value.
///
/// @ingroup GPU
FfxUInt32 ffxPackBytesY3Y2Y1X0(FfxUInt32x2 i)
{
    return ((i.x) & 0x000000ffu) | (i.y & 0xffffff00u);
}

/// Packs the bytes from the X and Y components of a FfxUInt32x2 into a single 32-bit integer.
///
/// The resulting integer will contain bytes in the following order, from most to least significant:
/// Y3 Y2 Y1 X2
///
/// @param [in] i                       The integer pair to pack.
///
/// @returns
/// The packed integer value.
///
/// @ingroup GPU
FfxUInt32 ffxPackBytesY3Y2Y1X2(FfxUInt32x2 i)
{
    return ((i.x >> 16) & 0x000000ffu) | (i.y & 0xffffff00u);
}

/// Packs the bytes from the X and Y components of a FfxUInt32x2 into a single 32-bit integer.
///
/// The resulting integer will contain bytes in the following order, from most to least significant:
/// Y3 Y2 X0 Y0
///
/// @param [in] i                       The integer pair to pack.
///
/// @returns
/// The packed integer value.
///
/// @ingroup GPU
FfxUInt32 ffxPackBytesY3Y2X0Y0(FfxUInt32x2 i)
{
    return ((i.x << 8) & 0x0000ff00u) | (i.y & 0xffff00ffu);
}

/// Packs the bytes from the X and Y components of a FfxUInt32x2 into a single 32-bit integer.
///
/// The resulting integer will contain bytes in the following order, from most to least significant:
/// Y3 Y2 X2 Y0
///
/// @param [in] i                       The integer pair to pack.
///
/// @returns
/// The packed integer value.
///
/// @ingroup GPU
FfxUInt32 ffxPackBytesY3Y2X2Y0(FfxUInt32x2 i)
{
    return ((i.x >> 8) & 0x0000ff00u) | (i.y & 0xffff00ffu);
}

/// Packs the bytes from the X and Y components of a FfxUInt32x2 into a single 32-bit integer.
///
/// The resulting integer will contain bytes in the following order, from most to least significant:
/// Y3 X0 Y1 Y0
///
/// @param [in] i                       The integer pair to pack.
///
/// @returns
/// The packed integer value.
///
/// @ingroup GPU
FfxUInt32 ffxPackBytesY3X0Y1Y0(FfxUInt32x2 i)
{
    return ((i.x << 16) & 0x00ff0000u) | (i.y & 0xff00ffffu);
}

/// Packs the bytes from the X and Y components of a FfxUInt32x2 into a single 32-bit integer.
///
/// The resulting integer will contain bytes in the following order, from most to least significant:
/// Y3 X2 Y1 Y0
///
/// @param [in] i                       The integer pair to pack.
///
/// @returns
/// The packed integer value.
///
/// @ingroup GPU
FfxUInt32 ffxPackBytesY3X2Y1Y0(FfxUInt32x2 i)
{
    return ((i.x) & 0x00ff0000u) | (i.y & 0xff00ffffu);
}

/// Packs the bytes from the X and Y components of a FfxUInt32x2 into a single 32-bit integer.
///
/// The resulting integer will contain bytes in the following order, from most to least significant:
/// X0 Y2 Y1 Y0
///
/// @param [in] i                       The integer pair to pack.
///
/// @returns
/// The packed integer value.
///
/// @ingroup GPU
FfxUInt32 ffxPackBytesX0Y2Y1Y0(FfxUInt32x2 i)
{
    return ((i.x << 24) & 0xff000000u) | (i.y & 0x00ffffffu);
}

/// Packs the bytes from the X and Y components of a FfxUInt32x2 into a single 32-bit integer.
///
/// The resulting integer will contain bytes in the following order, from most to least significant:
/// X2 Y2 Y1 Y0
///
/// @param [in] i                       The integer pair to pack.
///
/// @returns
/// The packed integer value.
///
/// @ingroup GPU
FfxUInt32 ffxPackBytesX2Y2Y1Y0(FfxUInt32x2 i)
{
    return ((i.x << 8) & 0xff000000u) | (i.y & 0x00ffffffu);
}

/// Packs the bytes from the X and Y components of a FfxUInt32x2 into a single 32-bit integer.
///
/// The resulting integer will contain bytes in the following order, from most to least significant:
/// Y2 X2 Y0 X0
///
/// @param [in] i                       The integer pair to pack.
///
/// @returns
/// The packed integer value.
///
/// @ingroup GPU
FfxUInt32 ffxPackBytesY2X2Y0X0(FfxUInt32x2 i)
{
    return ((i.x) & 0x00ff00ffu) | ((i.y << 8) & 0xff00ff00u);
}

/// Packs the bytes from the X and Y components of a FfxUInt32x2 into a single 32-bit integer.
///
/// The resulting integer will contain bytes in the following order, from most to least significant:
/// Y2 Y0 X2 X0
///
/// @param [in] i                       The integer pair to pack.
///
/// @returns
/// The packed integer value.
///
/// @ingroup GPU
FfxUInt32 ffxPackBytesY2Y0X2X0(FfxUInt32x2 i)
{
    return (((i.x) & 0xffu) | ((i.x >> 8) & 0xff00u) | ((i.y << 16) & 0xff0000u) | ((i.y << 8) & 0xff000000u));
}

/// Takes two Float16x2 values x and y, normalizes them and builds a single Uint16x2 value in the format {{x0,y0},{x1,y1}}.
///
/// @param [in] x                       The first float16x2 value to pack.
/// @param [in] y                       The second float16x2 value to pack.
///
/// @returns
/// The packed FfxUInt32x2 value.
///
/// @ingroup GPU
FfxUInt16x2 ffxPackX0Y0X1Y1UnsignedToUint16x2(FfxFloat16x2 x, FfxFloat16x2 y)
{
    x *= FFX_BROADCAST_FLOAT16X2(1.0 / 32768.0);
    y *= FFX_BROADCAST_FLOAT16X2(1.0 / 32768.0);
    return FFX_UINT32_TO_UINT16X2(ffxPackBytesY2X2Y0X0(FfxUInt32x2(FFX_UINT16X2_TO_UINT32(FFX_TO_UINT16X2(x)), FFX_UINT16X2_TO_UINT32(FFX_TO_UINT16X2(y)))));
}

/// Given a FfxUInt32x2 value d, Float16x2 value i and a resulting FfxUInt32x2 value r, this function packs d.x[0:7] into r.x[0:7],   
/// d.y[0:7] into r.y[0:7], i.x[8:15] into r.x[8:15], r.y[8:15] and i.y[0:15] into r.x[16:31], r.y[16:31] using 3 ops.
///
/// r=ffxPermuteUByte0Float16x2ToUint2(d,i)
///   Where 'k0' is an SGPR with {1.0/32768.0} packed into the lower 16-bits
///   Where 'k1' is an SGPR with 0x????
///   Where 'k2' is an SGPR with 0x????
///   V_PK_FMA_F16 i,i,k0.x,0
///   V_PERM_B32 r.x,i,i,k1
///   V_PERM_B32 r.y,i,i,k2
///
/// @param [in] d                       The FfxUInt32x2 value to be packed.
/// @param [in] i                       The FfxFloat16x2 value to be packed. 
///
/// @returns
/// The packed FfxUInt32x2 value.
///
/// @ingroup GPU
FfxUInt32x2 ffxPermuteUByte0Float16x2ToUint2(FfxUInt32x2 d, FfxFloat16x2 i)
{
    FfxUInt32 b = FFX_UINT16X2_TO_UINT32(FFX_TO_UINT16X2(i * FFX_BROADCAST_FLOAT16X2(1.0 / 32768.0)));
    return FfxUInt32x2(ffxPackBytesY3Y2Y1X0(FfxUInt32x2(d.x, b)), ffxPackBytesY3Y2Y1X2(FfxUInt32x2(d.y, b)));
}

/// Given a FfxUInt32x2 value d, Float16x2 value i and a resulting FfxUInt32x2 value r, this function packs d.x[0:7] into r.x[8:15],   
/// d.y[0:7] into r.y[8:15], i.x[0:7] into r.x[0:7], r.y[0:7] and i.y[0:15] into r.x[16:31], r.y[16:31] using 3 ops.
///
/// r=ffxPermuteUByte1Float16x2ToUint2(d,i)
///   Where 'k0' is an SGPR with {1.0/32768.0} packed into the lower 16-bits
///   Where 'k1' is an SGPR with 0x????
///   Where 'k2' is an SGPR with 0x????
///   V_PK_FMA_F16 i,i,k0.x,0
///   V_PERM_B32 r.x,i,i,k1
///   V_PERM_B32 r.y,i,i,k2
///
/// @param [in] d                       The FfxUInt32x2 value to be packed.
/// @param [in] i                       The FfxFloat16x2 value to be packed. 
///
/// @returns
/// The packed FfxUInt32x2 value.
///
/// @ingroup GPU
FfxUInt32x2 ffxPermuteUByte1Float16x2ToUint2(FfxUInt32x2 d, FfxFloat16x2 i)
{
    FfxUInt32 b = FFX_UINT16X2_TO_UINT32(FFX_TO_UINT16X2(i * FFX_BROADCAST_FLOAT16X2(1.0 / 32768.0)));
    return FfxUInt32x2(ffxPackBytesY3Y2X0Y0(FfxUInt32x2(d.x, b)), ffxPackBytesY3Y2X2Y0(FfxUInt32x2(d.y, b)));
}

/// Given a FfxUInt32x2 value d, Float16x2 value i and a resulting FfxUInt32x2 value r, this function packs d.x[0:7] into r.x[16:23],   
/// d.y[0:7] into r.y[16:23], i.x[0:15] into r.x[0:15], r.y[0:15] and i.y[8:15] into r.x[24:31], r.y[24:31] using 3 ops.
///
/// r=ffxPermuteUByte2Float16x2ToUint2(d,i)
///   Where 'k0' is an SGPR with {1.0/32768.0} packed into the lower 16-bits
///   Where 'k1' is an SGPR with 0x????
///   Where 'k2' is an SGPR with 0x????
///   V_PK_FMA_F16 i,i,k0.x,0
///   V_PERM_B32 r.x,i,i,k1
///   V_PERM_B32 r.y,i,i,k2
///
/// @param [in] d                       The FfxUInt32x2 value to be packed.
/// @param [in] i                       The FfxFloat16x2 value to be packed. 
///
/// @returns
/// The packed FfxUInt32x2 value.
///
/// @ingroup GPU
FfxUInt32x2 ffxPermuteUByte2Float16x2ToUint2(FfxUInt32x2 d, FfxFloat16x2 i)
{
    FfxUInt32 b = FFX_UINT16X2_TO_UINT32(FFX_TO_UINT16X2(i * FFX_BROADCAST_FLOAT16X2(1.0 / 32768.0)));
    return FfxUInt32x2(ffxPackBytesY3X0Y1Y0(FfxUInt32x2(d.x, b)), ffxPackBytesY3X2Y1Y0(FfxUInt32x2(d.y, b)));
}

/// Given a FfxUInt32x2 value d, Float16x2 value i and a resulting FfxUInt32x2 value r, this function packs d.x[0:7] into r.x[24:31],   
/// d.y[0:7] into r.y[24:31], i.x[0:15] into r.x[0:15], r.y[0:15] and i.y[0:7] into r.x[16:23], r.y[16:23] using 3 ops.
///
/// r=ffxPermuteUByte3Float16x2ToUint2(d,i)
///   Where 'k0' is an SGPR with {1.0/32768.0} packed into the lower 16-bits
///   Where 'k1' is an SGPR with 0x????
///   Where 'k2' is an SGPR with 0x????
///   V_PK_FMA_F16 i,i,k0.x,0
///   V_PERM_B32 r.x,i,i,k1
///   V_PERM_B32 r.y,i,i,k2
///
/// @param [in] d                       The FfxUInt32x2 value to be packed.
/// @param [in] i                       The FfxFloat16x2 value to be packed. 
///
/// @returns
/// The packed FfxUInt32x2 value.
///
/// @ingroup GPU
FfxUInt32x2 ffxPermuteUByte3Float16x2ToUint2(FfxUInt32x2 d, FfxFloat16x2 i)
{
    FfxUInt32 b = FFX_UINT16X2_TO_UINT32(FFX_TO_UINT16X2(i * FFX_BROADCAST_FLOAT16X2(1.0 / 32768.0)));
    return FfxUInt32x2(ffxPackBytesX0Y2Y1Y0(FfxUInt32x2(d.x, b)), ffxPackBytesX2Y2Y1Y0(FfxUInt32x2(d.y, b)));
}

/// Given a FfxUInt32x2 value i and a resulting Float16x2 value r, this function packs i.x[0:7] into r.x[0:7] and i.y[0:7] into r.y[0:7] using 2 ops.  
///
/// @param [in] i                       The FfxUInt32x2 value to be unpacked. 
///
/// @returns
/// The unpacked FfxFloat16x2.
///
/// @ingroup GPU
FfxFloat16x2 ffxPermuteUByte0Uint2ToFloat16x2(FfxUInt32x2 i)
{
    return FFX_TO_FLOAT16X2(FFX_UINT32_TO_UINT16X2(ffxPackBytesZeroY0ZeroX0(i))) * FFX_BROADCAST_FLOAT16X2(32768.0);
}

/// Given a FfxUInt32x2 value i and a resulting Float16x2 value r, this function packs i.x[8:15] into r.x[0:7] and i.y[8:15] into r.y[0:7] using 2 ops.  
///
/// @param [in] i                       The FfxUInt32x2 value to be unpacked. 
///
/// @returns
/// The unpacked FfxFloat16x2.
///
/// @ingroup GPU
FfxFloat16x2 ffxPermuteUByte1Uint2ToFloat16x2(FfxUInt32x2 i)
{
    return FFX_TO_FLOAT16X2(FFX_UINT32_TO_UINT16X2(ffxPackBytesZeroY1ZeroX1(i))) * FFX_BROADCAST_FLOAT16X2(32768.0);
}

/// Given a FfxUInt32x2 value i and a resulting Float16x2 value r, this function packs i.x[16:23] into r.x[0:7] and i.y[16:23] into r.y[0:7] using 2 ops.  
///
/// @param [in] i                       The FfxUInt32x2 value to be unpacked. 
///
/// @returns
/// The unpacked FfxFloat16x2.
///
/// @ingroup GPU
FfxFloat16x2 ffxPermuteUByte2Uint2ToFloat16x2(FfxUInt32x2 i)
{
    return FFX_TO_FLOAT16X2(FFX_UINT32_TO_UINT16X2(ffxPackBytesZeroY2ZeroX2(i))) * FFX_BROADCAST_FLOAT16X2(32768.0);
}

/// Given a FfxUInt32x2 value i and a resulting Float16x2 value r, this function packs i.x[24:31] into r.x[0:7] and i.y[24:31] into r.y[0:7] using 2 ops.  
///
/// @param [in] i                       The FfxUInt32x2 value to be unpacked. 
///
/// @returns
/// The unpacked FfxFloat16x2.
///
/// @ingroup GPU
FfxFloat16x2 ffxPermuteUByte3Uint2ToFloat16x2(FfxUInt32x2 i)
{
    return FFX_TO_FLOAT16X2(FFX_UINT32_TO_UINT16X2(ffxPackBytesZeroY3ZeroX3(i))) * FFX_BROADCAST_FLOAT16X2(32768.0);
}

/// Takes two Float16x2 values x and y, normalizes them and builds a single Uint16x2 value in the format {{x0,y0},{x1,y1}}.
///
/// @param [in] x                       The first float16x2 value to pack.
/// @param [in] y                       The second float16x2 value to pack.
///
/// @returns
/// The packed FfxUInt32x2 value.
///
/// @ingroup GPU
FfxUInt16x2 ffxPackX0Y0X1Y1SignedToUint16x2(FfxFloat16x2 x, FfxFloat16x2 y)
{
    x = x * FFX_BROADCAST_FLOAT16X2(1.0 / 32768.0) + FFX_BROADCAST_FLOAT16X2(0.25 / 32768.0);
    y = y * FFX_BROADCAST_FLOAT16X2(1.0 / 32768.0) + FFX_BROADCAST_FLOAT16X2(0.25 / 32768.0);
    return FFX_UINT32_TO_UINT16X2(ffxPackBytesY2X2Y0X0(FfxUInt32x2(FFX_UINT16X2_TO_UINT32(FFX_TO_UINT16X2(x)), FFX_UINT16X2_TO_UINT32(FFX_TO_UINT16X2(y)))));
}

/// Given a FfxUInt32x2 value d, Float16x2 value i and a resulting FfxUInt32x2 value r, this function packs d.x[0:7] into r.x[0:7],   
/// d.y[0:7] into r.y[0:7], i.x[8:15] into r.x[8:15], r.y[8:15] and i.y[0:15] into r.x[16:31], r.y[16:31] using 3 ops.
///
/// Handles signed byte values.
///
/// @param [in] d                       The FfxUInt32x2 value to be packed.
/// @param [in] i                       The FfxFloat16x2 value to be packed. 
///
/// @returns
/// The packed FfxUInt32x2 value.
///
/// @ingroup GPU
FfxUInt32x2 ffxPermuteSByte0Float16x2ToUint2(FfxUInt32x2 d, FfxFloat16x2 i)
{
    FfxUInt32 b = FFX_UINT16X2_TO_UINT32(FFX_TO_UINT16X2(i * FFX_BROADCAST_FLOAT16X2(1.0 / 32768.0) + FFX_BROADCAST_FLOAT16X2(0.25 / 32768.0)));
    return FfxUInt32x2(ffxPackBytesY3Y2Y1X0(FfxUInt32x2(d.x, b)), ffxPackBytesY3Y2Y1X2(FfxUInt32x2(d.y, b)));
}

/// Given a FfxUInt32x2 value d, Float16x2 value i and a resulting FfxUInt32x2 value r, this function packs d.x[0:7] into r.x[8:15],   
/// d.y[0:7] into r.y[8:15], i.x[0:7] into r.x[0:7], r.y[0:7] and i.y[0:15] into r.x[16:31], r.y[16:31] using 3 ops.
///
/// Handles signed byte values.
///
/// @param [in] d                       The FfxUInt32x2 value to be packed.
/// @param [in] i                       The FfxFloat16x2 value to be packed. 
///
/// @returns
/// The packed FfxUInt32x2 value.
///
/// @ingroup GPU
FfxUInt32x2 ffxPermuteSByte1Float16x2ToUint2(FfxUInt32x2 d, FfxFloat16x2 i)
{
    FfxUInt32 b = FFX_UINT16X2_TO_UINT32(FFX_TO_UINT16X2(i * FFX_BROADCAST_FLOAT16X2(1.0 / 32768.0) + FFX_BROADCAST_FLOAT16X2(0.25 / 32768.0)));
    return FfxUInt32x2(ffxPackBytesY3Y2X0Y0(FfxUInt32x2(d.x, b)), ffxPackBytesY3Y2X2Y0(FfxUInt32x2(d.y, b)));
}

/// Given a FfxUInt32x2 value d, Float16x2 value i and a resulting FfxUInt32x2 value r, this function packs d.x[0:7] into r.x[16:23],   
/// d.y[0:7] into r.y[16:23], i.x[0:15] into r.x[0:15], r.y[0:15] and i.y[8:15] into r.x[24:31], r.y[24:31] using 3 ops.
///
/// Handles signed byte values.
///
/// @param [in] d                       The FfxUInt32x2 value to be packed.
/// @param [in] i                       The FfxFloat16x2 value to be packed. 
///
/// @returns
/// The packed FfxUInt32x2 value.
///
/// @ingroup GPU
FfxUInt32x2 ffxPermuteSByte2Float16x2ToUint2(FfxUInt32x2 d, FfxFloat16x2 i)
{
    FfxUInt32 b = FFX_UINT16X2_TO_UINT32(FFX_TO_UINT16X2(i * FFX_BROADCAST_FLOAT16X2(1.0 / 32768.0) + FFX_BROADCAST_FLOAT16X2(0.25 / 32768.0)));
    return FfxUInt32x2(ffxPackBytesY3X0Y1Y0(FfxUInt32x2(d.x, b)), ffxPackBytesY3X2Y1Y0(FfxUInt32x2(d.y, b)));
}

/// Given a FfxUInt32x2 value d, Float16x2 value i and a resulting FfxUInt32x2 value r, this function packs d.x[0:7] into r.x[24:31],   
/// d.y[0:7] into r.y[24:31], i.x[0:15] into r.x[0:15], r.y[0:15] and i.y[0:7] into r.x[16:23], r.y[16:23] using 3 ops.
///
/// Handles signed byte values.
///
/// @param [in] d                       The FfxUInt32x2 value to be packed.
/// @param [in] i                       The FfxFloat16x2 value to be packed. 
///
/// @returns
/// The packed FfxUInt32x2 value.
///
/// @ingroup GPU
FfxUInt32x2 ffxPermuteSByte3Float16x2ToUint2(FfxUInt32x2 d, FfxFloat16x2 i)
{
    FfxUInt32 b = FFX_UINT16X2_TO_UINT32(FFX_TO_UINT16X2(i * FFX_BROADCAST_FLOAT16X2(1.0 / 32768.0) + FFX_BROADCAST_FLOAT16X2(0.25 / 32768.0)));
    return FfxUInt32x2(ffxPackBytesX0Y2Y1Y0(FfxUInt32x2(d.x, b)), ffxPackBytesX2Y2Y1Y0(FfxUInt32x2(d.y, b)));
}

/// Given a FfxUInt32x2 value d, Float16x2 value i and a resulting FfxUInt32x2 value r, this function packs d.x[0:7] into r.x[0:7],   
/// d.y[0:7] into r.y[0:7], i.x[8:15] into r.x[8:15], r.y[8:15] and i.y[0:15] into r.x[16:31], r.y[16:31] using 3 ops.
///
/// Zero-based flips the MSB bit of the byte (making 128 "exact zero" actually zero).
/// This is useful if there is a desire for cleared values to decode as zero.
///
/// Handles signed byte values.
///
/// @param [in] d                       The FfxUInt32x2 value to be packed.
/// @param [in] i                       The FfxFloat16x2 value to be packed. 
///
/// @returns
/// The packed FfxUInt32x2 value.
///
/// @ingroup GPU
FfxUInt32x2 ffxPermuteZeroBasedSByte0Float16x2ToUint2(FfxUInt32x2 d, FfxFloat16x2 i)
{
    FfxUInt32 b = FFX_UINT16X2_TO_UINT32(FFX_TO_UINT16X2(i * FFX_BROADCAST_FLOAT16X2(1.0 / 32768.0) + FFX_BROADCAST_FLOAT16X2(0.25 / 32768.0))) ^ 0x00800080u;
    return FfxUInt32x2(ffxPackBytesY3Y2Y1X0(FfxUInt32x2(d.x, b)), ffxPackBytesY3Y2Y1X2(FfxUInt32x2(d.y, b)));
}

/// Given a FfxUInt32x2 value d, Float16x2 value i and a resulting FfxUInt32x2 value r, this function packs d.x[0:7] into r.x[8:15],   
/// d.y[0:7] into r.y[8:15], i.x[0:7] into r.x[0:7], r.y[0:7] and i.y[0:15] into r.x[16:31], r.y[16:31] using 3 ops.
///
/// Zero-based flips the MSB bit of the byte (making 128 "exact zero" actually zero).
/// This is useful if there is a desire for cleared values to decode as zero.
///
/// Handles signed byte values.
///
/// @param [in] d                       The FfxUInt32x2 value to be packed.
/// @param [in] i                       The FfxFloat16x2 value to be packed. 
///
/// @returns
/// The packed FfxUInt32x2 value.
///
/// @ingroup GPU
FfxUInt32x2 ffxPermuteZeroBasedSByte1Float16x2ToUint2(FfxUInt32x2 d, FfxFloat16x2 i)
{
    FfxUInt32 b = FFX_UINT16X2_TO_UINT32(FFX_TO_UINT16X2(i * FFX_BROADCAST_FLOAT16X2(1.0 / 32768.0) + FFX_BROADCAST_FLOAT16X2(0.25 / 32768.0))) ^ 0x00800080u;
    return FfxUInt32x2(ffxPackBytesY3Y2X0Y0(FfxUInt32x2(d.x, b)), ffxPackBytesY3Y2X2Y0(FfxUInt32x2(d.y, b)));
}

/// Given a FfxUInt32x2 value d, Float16x2 value i and a resulting FfxUInt32x2 value r, this function packs d.x[0:7] into r.x[16:23],   
/// d.y[0:7] into r.y[16:23], i.x[0:15] into r.x[0:15], r.y[0:15] and i.y[8:15] into r.x[24:31], r.y[24:31] using 3 ops.
///
/// Zero-based flips the MSB bit of the byte (making 128 "exact zero" actually zero).
/// This is useful if there is a desire for cleared values to decode as zero.
///
/// Handles signed byte values.
///
/// @param [in] d                       The FfxUInt32x2 value to be packed.
/// @param [in] i                       The FfxFloat16x2 value to be packed. 
///
/// @returns
/// The packed FfxUInt32x2 value.
///
/// @ingroup GPU
FfxUInt32x2 ffxPermuteZeroBasedSByte2Float16x2ToUint2(FfxUInt32x2 d, FfxFloat16x2 i)
{
    FfxUInt32 b = FFX_UINT16X2_TO_UINT32(FFX_TO_UINT16X2(i * FFX_BROADCAST_FLOAT16X2(1.0 / 32768.0) + FFX_BROADCAST_FLOAT16X2(0.25 / 32768.0))) ^ 0x00800080u;
    return FfxUInt32x2(ffxPackBytesY3X0Y1Y0(FfxUInt32x2(d.x, b)), ffxPackBytesY3X2Y1Y0(FfxUInt32x2(d.y, b)));
}

/// Given a FfxUInt32x2 value d, Float16x2 value i and a resulting FfxUInt32x2 value r, this function packs d.x[0:7] into r.x[24:31],   
/// d.y[0:7] into r.y[24:31], i.x[0:15] into r.x[0:15], r.y[0:15] and i.y[0:7] into r.x[16:23], r.y[16:23] using 3 ops.
///
/// Zero-based flips the MSB bit of the byte (making 128 "exact zero" actually zero).
/// This is useful if there is a desire for cleared values to decode as zero.
///
/// Handles signed byte values.
///
/// @param [in] d                       The FfxUInt32x2 value to be packed.
/// @param [in] i                       The FfxFloat16x2 value to be packed. 
///
/// @returns
/// The packed FfxUInt32x2 value.
///
/// @ingroup GPU
FfxUInt32x2 ffxPermuteZeroBasedSByte3Float16x2ToUint2(FfxUInt32x2 d, FfxFloat16x2 i)
{
    FfxUInt32 b = FFX_UINT16X2_TO_UINT32(FFX_TO_UINT16X2(i * FFX_BROADCAST_FLOAT16X2(1.0 / 32768.0) + FFX_BROADCAST_FLOAT16X2(0.25 / 32768.0))) ^ 0x00800080u;
    return FfxUInt32x2(ffxPackBytesX0Y2Y1Y0(FfxUInt32x2(d.x, b)), ffxPackBytesX2Y2Y1Y0(FfxUInt32x2(d.y, b)));
}

/// Given a FfxUInt32x2 value i and a resulting Float16x2 value r, this function packs i.x[0:7] into r.x[0:7] and i.y[0:7] into r.y[0:7] using 2 ops.  
///
/// Handles signed byte values.
///
/// @param [in] i                       The FfxUInt32x2 value to be unpacked. 
///
/// @returns
/// The unpacked FfxFloat16x2.
///
/// @ingroup GPU
FfxFloat16x2 ffxPermuteSByte0Uint2ToFloat16x2(FfxUInt32x2 i)
{
    return FFX_TO_FLOAT16X2(FFX_UINT32_TO_UINT16X2(ffxPackBytesZeroY0ZeroX0(i))) * FFX_BROADCAST_FLOAT16X2(32768.0) - FFX_BROADCAST_FLOAT16X2(0.25);
}

/// Given a FfxUInt32x2 value i and a resulting Float16x2 value r, this function packs i.x[8:15] into r.x[0:7] and i.y[8:15] into r.y[0:7] using 2 ops.  
///
/// Handles signed byte values.
///
/// @param [in] i                       The FfxUInt32x2 value to be unpacked. 
///
/// @returns
/// The unpacked FfxFloat16x2.
///
/// @ingroup GPU
FfxFloat16x2 ffxPermuteSByte1Uint2ToFloat16x2(FfxUInt32x2 i)
{
    return FFX_TO_FLOAT16X2(FFX_UINT32_TO_UINT16X2(ffxPackBytesZeroY1ZeroX1(i))) * FFX_BROADCAST_FLOAT16X2(32768.0) - FFX_BROADCAST_FLOAT16X2(0.25);
}

/// Given a FfxUInt32x2 value i and a resulting Float16x2 value r, this function packs i.x[16:23] into r.x[0:7] and i.y[16:23] into r.y[0:7] using 2 ops.
///  
/// Handles signed byte values.
///
/// @param [in] i                       The FfxUInt32x2 value to be unpacked. 
///
/// @returns
/// The unpacked FfxFloat16x2.
///
/// @ingroup GPU
FfxFloat16x2 ffxPermuteSByte2Uint2ToFloat16x2(FfxUInt32x2 i)
{
    return FFX_TO_FLOAT16X2(FFX_UINT32_TO_UINT16X2(ffxPackBytesZeroY2ZeroX2(i))) * FFX_BROADCAST_FLOAT16X2(32768.0) - FFX_BROADCAST_FLOAT16X2(0.25);
}

/// Given a FfxUInt32x2 value i and a resulting Float16x2 value r, this function packs i.x[24:31] into r.x[0:7] and i.y[24:31] into r.y[0:7] using 2 ops.  
///
/// Handles signed byte values.
///
/// @param [in] i                       The FfxUInt32x2 value to be unpacked. 
///
/// @returns
/// The unpacked FfxFloat16x2.
///
/// @ingroup GPU
FfxFloat16x2 ffxPermuteSByte3Uint2ToFloat16x2(FfxUInt32x2 i)
{
    return FFX_TO_FLOAT16X2(FFX_UINT32_TO_UINT16X2(ffxPackBytesZeroY3ZeroX3(i))) * FFX_BROADCAST_FLOAT16X2(32768.0) - FFX_BROADCAST_FLOAT16X2(0.25);
}

/// Given a FfxUInt32x2 value i and a resulting Float16x2 value r, this function packs i.x[0:7] into r.x[0:7] and i.y[0:7] into r.y[0:7] using 2 ops.
///  
/// Handles signed byte values.
///
/// @param [in] i                       The FfxUInt32x2 value to be unpacked. 
///
/// @returns
/// The unpacked FfxFloat16x2.
///
/// @ingroup GPU
FfxFloat16x2 ffxPermuteZeroBasedSByte0Uint2ToFloat16x2(FfxUInt32x2 i)
{
    return FFX_TO_FLOAT16X2(FFX_UINT32_TO_UINT16X2(ffxPackBytesZeroY0ZeroX0(i) ^ 0x00800080u)) * FFX_BROADCAST_FLOAT16X2(32768.0) - FFX_BROADCAST_FLOAT16X2(0.25);
}

/// Given a FfxUInt32x2 value i and a resulting Float16x2 value r, this function packs i.x[8:15] into r.x[0:7] and i.y[8:15] into r.y[0:7] using 2 ops.
///  
/// Handles signed byte values.
///
/// @param [in] i                       The FfxUInt32x2 value to be unpacked. 
///
/// @returns
/// The unpacked FfxFloat16x2.
///
/// @ingroup GPU
FfxFloat16x2 ffxPermuteZeroBasedSByte1Uint2ToFloat16x2(FfxUInt32x2 i)
{
    return FFX_TO_FLOAT16X2(FFX_UINT32_TO_UINT16X2(ffxPackBytesZeroY1ZeroX1(i) ^ 0x00800080u)) * FFX_BROADCAST_FLOAT16X2(32768.0) - FFX_BROADCAST_FLOAT16X2(0.25);
}

/// Given a FfxUInt32x2 value i and a resulting Float16x2 value r, this function packs i.x[16:23] into r.x[0:7] and i.y[16:23] into r.y[0:7] using 2 ops.
///  
/// Handles signed byte values.
///
/// @param [in] i                       The FfxUInt32x2 value to be unpacked. 
///
/// @returns
/// The unpacked FfxFloat16x2.
///
/// @ingroup GPU
FfxFloat16x2 ffxPermuteZeroBasedSByte2Uint2ToFloat16x2(FfxUInt32x2 i)
{
    return FFX_TO_FLOAT16X2(FFX_UINT32_TO_UINT16X2(ffxPackBytesZeroY2ZeroX2(i) ^ 0x00800080u)) * FFX_BROADCAST_FLOAT16X2(32768.0) - FFX_BROADCAST_FLOAT16X2(0.25);
}

/// Given a FfxUInt32x2 value i and a resulting Float16x2 value r, this function packs i.x[24:31] into r.x[0:7] and i.y[24:31] into r.y[0:7] using 2 ops.
///  
/// Handles signed byte values.
///
/// @param [in] i                       The FfxUInt32x2 value to be unpacked. 
///
/// @returns
/// The unpacked FfxFloat16x2.
///
/// @ingroup GPU
FfxFloat16x2 ffxPermuteZeroBasedSByte3Uint2ToFloat16x2(FfxUInt32x2 i)
{
    return FFX_TO_FLOAT16X2(FFX_UINT32_TO_UINT16X2(ffxPackBytesZeroY3ZeroX3(i) ^ 0x00800080u)) * FFX_BROADCAST_FLOAT16X2(32768.0) - FFX_BROADCAST_FLOAT16X2(0.25);
}

/// Calculate a half-precision low-quality approximation for the square root of a value.
///
/// For additional information on the approximation family of functions, you can refer to Michal Drobot's excellent
/// presentation materials:
///
///  - https://michaldrobot.files.wordpress.com/2014/05/gcn_alu_opt_digitaldragons2014.pdf
///  - https://github.com/michaldrobot/ShaderFastLibs/blob/master/ShaderFastMathLib.h
///
/// @param [in] a           The value to calculate an approximate to the square root for.
///
/// @returns
/// An approximation of the square root, estimated to low quality.
///
/// @ingroup GPU
FfxFloat16 ffxApproximateSqrtHalf(FfxFloat16 a)
{
    return FFX_TO_FLOAT16((FFX_TO_UINT16(a) >> FFX_BROADCAST_UINT16(1)) + FFX_BROADCAST_UINT16(0x1de2));
}

/// Calculate a half-precision low-quality approximation for the square root of a value.
///
/// For additional information on the approximation family of functions, you can refer to Michal Drobot's excellent
/// presentation materials:
///
///  - https://michaldrobot.files.wordpress.com/2014/05/gcn_alu_opt_digitaldragons2014.pdf
///  - https://github.com/michaldrobot/ShaderFastLibs/blob/master/ShaderFastMathLib.h
///
/// @param [in] a           The value to calculate an approximate to the square root for.
///
/// @returns
/// An approximation of the square root, estimated to low quality.
///
/// @ingroup GPU
FfxFloat16x2 ffxApproximateSqrtHalf(FfxFloat16x2 a)
{
    return FFX_TO_FLOAT16X2((FFX_TO_UINT16X2(a) >> FFX_BROADCAST_UINT16X2(1)) + FFX_BROADCAST_UINT16X2(0x1de2));
}

/// Calculate a half-precision low-quality approximation for the square root of a value.
///
/// For additional information on the approximation family of functions, you can refer to Michal Drobot's excellent
/// presentation materials:
///
///  - https://michaldrobot.files.wordpress.com/2014/05/gcn_alu_opt_digitaldragons2014.pdf
///  - https://github.com/michaldrobot/ShaderFastLibs/blob/master/ShaderFastMathLib.h
///
/// @param [in] a           The value to calculate an approximate to the square root for.
///
/// @returns
/// An approximation of the square root, estimated to low quality.
///
/// @ingroup GPU
FfxFloat16x3 ffxApproximateSqrtHalf(FfxFloat16x3 a)
{
    return FFX_TO_FLOAT16X3((FFX_TO_UINT16X3(a) >> FFX_BROADCAST_UINT16X3(1)) + FFX_BROADCAST_UINT16X3(0x1de2));
}

/// Calculate a half-precision low-quality approximation for the reciprocal of a value.
///
/// For additional information on the approximation family of functions, you can refer to Michal Drobot's excellent
/// presentation materials:
///
///  - https://michaldrobot.files.wordpress.com/2014/05/gcn_alu_opt_digitaldragons2014.pdf
///  - https://github.com/michaldrobot/ShaderFastLibs/blob/master/ShaderFastMathLib.h
///
/// @param [in] a           The value to calculate an approximate to the reciprocal for.
///
/// @returns
/// An approximation of the reciprocal, estimated to low quality.
///
/// @ingroup GPU
FfxFloat16 ffxApproximateReciprocalHalf(FfxFloat16 a)
{
    return FFX_TO_FLOAT16(FFX_BROADCAST_UINT16(0x7784) - FFX_TO_UINT16(a));
}

/// Calculate a half-precision low-quality approximation for the reciprocal of a value.
///
/// For additional information on the approximation family of functions, you can refer to Michal Drobot's excellent
/// presentation materials:
///
///  - https://michaldrobot.files.wordpress.com/2014/05/gcn_alu_opt_digitaldragons2014.pdf
///  - https://github.com/michaldrobot/ShaderFastLibs/blob/master/ShaderFastMathLib.h
///
/// @param [in] a           The value to calculate an approximate to the reciprocal for.
///
/// @returns
/// An approximation of the reciprocal, estimated to low quality.
///
/// @ingroup GPU
FfxFloat16x2 ffxApproximateReciprocalHalf(FfxFloat16x2 a)
{
    return FFX_TO_FLOAT16X2(FFX_BROADCAST_UINT16X2(0x7784) - FFX_TO_UINT16X2(a));
}

/// Calculate a half-precision low-quality approximation for the reciprocal of a value.
///
/// For additional information on the approximation family of functions, you can refer to Michal Drobot's excellent
/// presentation materials:
///
///  - https://michaldrobot.files.wordpress.com/2014/05/gcn_alu_opt_digitaldragons2014.pdf
///  - https://github.com/michaldrobot/ShaderFastLibs/blob/master/ShaderFastMathLib.h
///
/// @param [in] a           The value to calculate an approximate to the reciprocal for.
///
/// @returns
/// An approximation of the reciprocal, estimated to low quality.
///
/// @ingroup GPU
FfxFloat16x3 ffxApproximateReciprocalHalf(FfxFloat16x3 a)
{
    return FFX_TO_FLOAT16X3(FFX_BROADCAST_UINT16X3(0x7784) - FFX_TO_UINT16X3(a));
}

/// Calculate a half-precision low-quality approximation for the reciprocal of a value.
///
/// For additional information on the approximation family of functions, you can refer to Michal Drobot's excellent
/// presentation materials:
///
///  - https://michaldrobot.files.wordpress.com/2014/05/gcn_alu_opt_digitaldragons2014.pdf
///  - https://github.com/michaldrobot/ShaderFastLibs/blob/master/ShaderFastMathLib.h
///
/// @param [in] a           The value to calculate an approximate to the reciprocal for.
///
/// @returns
/// An approximation of the reciprocal, estimated to low quality.
///
/// @ingroup GPU
FfxFloat16x4 ffxApproximateReciprocalHalf(FfxFloat16x4 a)
{
    return FFX_TO_FLOAT16X4(FFX_BROADCAST_UINT16X4(0x7784) - FFX_TO_UINT16X4(a));
}

/// Calculate a half-precision medium-quality approximation for the reciprocal of a value.
///
/// For additional information on the approximation family of functions, you can refer to Michal Drobot's excellent
/// presentation materials:
///
///  - https://michaldrobot.files.wordpress.com/2014/05/gcn_alu_opt_digitaldragons2014.pdf
///  - https://github.com/michaldrobot/ShaderFastLibs/blob/master/ShaderFastMathLib.h
///
/// @param [in] a           The value to calculate an approximate to the reciprocal for.
///
/// @returns
/// An approximation of the reciprocal, estimated to medium quality.
///
/// @ingroup GPU
FfxFloat16 ffxApproximateReciprocalMediumHalf(FfxFloat16 a)
{
    FfxFloat16 b = FFX_TO_FLOAT16(FFX_BROADCAST_UINT16(0x778d) - FFX_TO_UINT16(a));
    return b * (-b * a + FFX_BROADCAST_FLOAT16(2.0));
}

/// Calculate a half-precision medium-quality approximation for the reciprocal of a value.
///
/// For additional information on the approximation family of functions, you can refer to Michal Drobot's excellent
/// presentation materials:
///
///  - https://michaldrobot.files.wordpress.com/2014/05/gcn_alu_opt_digitaldragons2014.pdf
///  - https://github.com/michaldrobot/ShaderFastLibs/blob/master/ShaderFastMathLib.h
///
/// @param [in] a           The value to calculate an approximate to the reciprocal for.
///
/// @returns
/// An approximation of the reciprocal, estimated to medium quality.
///
/// @ingroup GPU
FfxFloat16x2 ffxApproximateReciprocalMediumHalf(FfxFloat16x2 a)
{
    FfxFloat16x2 b = FFX_TO_FLOAT16X2(FFX_BROADCAST_UINT16X2(0x778d) - FFX_TO_UINT16X2(a));
    return b * (-b * a + FFX_BROADCAST_FLOAT16X2(2.0));
}

/// Calculate a half-precision medium-quality approximation for the reciprocal of a value.
///
/// For additional information on the approximation family of functions, you can refer to Michal Drobot's excellent
/// presentation materials:
///
///  - https://michaldrobot.files.wordpress.com/2014/05/gcn_alu_opt_digitaldragons2014.pdf
///  - https://github.com/michaldrobot/ShaderFastLibs/blob/master/ShaderFastMathLib.h
///
/// @param [in] a           The value to calculate an approximate to the reciprocal for.
///
/// @returns
/// An approximation of the reciprocal, estimated to medium quality.
///
/// @ingroup GPU
FfxFloat16x3 ffxApproximateReciprocalMediumHalf(FfxFloat16x3 a)
{
    FfxFloat16x3 b = FFX_TO_FLOAT16X3(FFX_BROADCAST_UINT16X3(0x778d) - FFX_TO_UINT16X3(a));
    return b * (-b * a + FFX_BROADCAST_FLOAT16X3(2.0));
}

/// Calculate a half-precision medium-quality approximation for the reciprocal of a value.
///
/// For additional information on the approximation family of functions, you can refer to Michal Drobot's excellent
/// presentation materials:
///
///  - https://michaldrobot.files.wordpress.com/2014/05/gcn_alu_opt_digitaldragons2014.pdf
///  - https://github.com/michaldrobot/ShaderFastLibs/blob/master/ShaderFastMathLib.h
///
/// @param [in] a           The value to calculate an approximate to the reciprocal for.
///
/// @returns
/// An approximation of the reciprocal, estimated to medium quality.
///
/// @ingroup GPU
FfxFloat16x4 ffxApproximateReciprocalMediumHalf(FfxFloat16x4 a)
{
    FfxFloat16x4 b = FFX_TO_FLOAT16X4(FFX_BROADCAST_UINT16X4(0x778d) - FFX_TO_UINT16X4(a));
    return b * (-b * a + FFX_BROADCAST_FLOAT16X4(2.0));
}

/// Calculate a half-precision low-quality approximation for the reciprocal of the square root of a value.
///
/// For additional information on the approximation family of functions, you can refer to Michal Drobot's excellent
/// presentation materials:
///
///  - https://michaldrobot.files.wordpress.com/2014/05/gcn_alu_opt_digitaldragons2014.pdf
///  - https://github.com/michaldrobot/ShaderFastLibs/blob/master/ShaderFastMathLib.h
///
/// @param [in] a           The value to calculate an approximate to the reciprocal of the square root for.
///
/// @returns
/// An approximation of the reciprocal of the square root, estimated to low quality.
///
/// @ingroup GPU
FfxFloat16 ffxApproximateReciprocalSquareRootHalf(FfxFloat16 a)
{
    return FFX_TO_FLOAT16(FFX_BROADCAST_UINT16(0x59a3) - (FFX_TO_UINT16(a) >> FFX_BROADCAST_UINT16(1)));
}

/// Calculate a half-precision low-quality approximation for the reciprocal of the square root of a value.
///
/// For additional information on the approximation family of functions, you can refer to Michal Drobot's excellent
/// presentation materials:
///
///  - https://michaldrobot.files.wordpress.com/2014/05/gcn_alu_opt_digitaldragons2014.pdf
///  - https://github.com/michaldrobot/ShaderFastLibs/blob/master/ShaderFastMathLib.h
///
/// @param [in] a           The value to calculate an approximate to the reciprocal of the square root for.
///
/// @returns
/// An approximation of the reciprocal of the square root, estimated to low quality.
///
/// @ingroup GPU
FfxFloat16x2 ffxApproximateReciprocalSquareRootHalf(FfxFloat16x2 a)
{
    return FFX_TO_FLOAT16X2(FFX_BROADCAST_UINT16X2(0x59a3) - (FFX_TO_UINT16X2(a) >> FFX_BROADCAST_UINT16X2(1)));
}

/// Calculate a half-precision low-quality approximation for the reciprocal of the square root of a value.
///
/// For additional information on the approximation family of functions, you can refer to Michal Drobot's excellent
/// presentation materials:
///
///  - https://michaldrobot.files.wordpress.com/2014/05/gcn_alu_opt_digitaldragons2014.pdf
///  - https://github.com/michaldrobot/ShaderFastLibs/blob/master/ShaderFastMathLib.h
///
/// @param [in] a           The value to calculate an approximate to the reciprocal of the square root for.
///
/// @returns
/// An approximation of the reciprocal of the square root, estimated to low quality.
///
/// @ingroup GPU
FfxFloat16x3 ffxApproximateReciprocalSquareRootHalf(FfxFloat16x3 a)
{
    return FFX_TO_FLOAT16X3(FFX_BROADCAST_UINT16X3(0x59a3) - (FFX_TO_UINT16X3(a) >> FFX_BROADCAST_UINT16X3(1)));
}

/// Calculate a half-precision low-quality approximation for the reciprocal of the square root of a value.
///
/// For additional information on the approximation family of functions, you can refer to Michal Drobot's excellent
/// presentation materials:
///
///  - https://michaldrobot.files.wordpress.com/2014/05/gcn_alu_opt_digitaldragons2014.pdf
///  - https://github.com/michaldrobot/ShaderFastLibs/blob/master/ShaderFastMathLib.h
///
/// @param [in] a           The value to calculate an approximate to the reciprocal of the square root for.
///
/// @returns
/// An approximation of the reciprocal of the square root, estimated to low quality.
///
/// @ingroup GPU
FfxFloat16x4 ffxApproximateReciprocalSquareRootHalf(FfxFloat16x4 a)
{
    return FFX_TO_FLOAT16X4(FFX_BROADCAST_UINT16X4(0x59a3) - (FFX_TO_UINT16X4(a) >> FFX_BROADCAST_UINT16X4(1)));
}

/// An approximation of sine.
///
/// Valid input range is {-1 to 1} representing {0 to 2 pi}, and the output range
/// is {-1/4 to 1/4} representing {-1 to 1}.
///
/// @param [in] x            The value to calculate approximate sine for.
///
/// @returns
/// The approximate sine of <c><i>value</i></c>.
FfxFloat16 ffxParabolicSinHalf(FfxFloat16 x)
{
    return x * abs(x) - x;
}

/// An approximation of sine.
///
/// Valid input range is {-1 to 1} representing {0 to 2 pi}, and the output range
/// is {-1/4 to 1/4} representing {-1 to 1}.
///
/// @param [in] x            The value to calculate approximate sine for.
///
/// @returns
/// The approximate sine of <c><i>value</i></c>.
FfxFloat16x2 ffxParabolicSinHalf(FfxFloat16x2 x)
{
    return x * abs(x) - x;
}

/// An approximation of cosine.
///
/// Valid input range is {-1 to 1} representing {0 to 2 pi}, and the output range
/// is {-1/4 to 1/4} representing {-1 to 1}.
///
/// @param [in] x            The value to calculate approximate cosine for.
///
/// @returns
/// The approximate cosine of <c><i>value</i></c>.
FfxFloat16 ffxParabolicCosHalf(FfxFloat16 x)
{
    x = ffxFract(x * FFX_BROADCAST_FLOAT16(0.5) + FFX_BROADCAST_FLOAT16(0.75));
    x = x * FFX_BROADCAST_FLOAT16(2.0) - FFX_BROADCAST_FLOAT16(1.0);
    return ffxParabolicSinHalf(x);
}

/// An approximation of cosine.
///
/// Valid input range is {-1 to 1} representing {0 to 2 pi}, and the output range
/// is {-1/4 to 1/4} representing {-1 to 1}.
///
/// @param [in] x            The value to calculate approximate cosine for.
///
/// @returns
/// The approximate cosine of <c><i>value</i></c>.
FfxFloat16x2 ffxParabolicCosHalf(FfxFloat16x2 x)
{
    x = ffxFract(x * FFX_BROADCAST_FLOAT16X2(0.5) + FFX_BROADCAST_FLOAT16X2(0.75));
    x = x * FFX_BROADCAST_FLOAT16X2(2.0) - FFX_BROADCAST_FLOAT16X2(1.0);
    return ffxParabolicSinHalf(x);
}

/// An approximation of both sine and cosine.
///
/// Valid input range is {-1 to 1} representing {0 to 2 pi}, and the output range
/// is {-1/4 to 1/4} representing {-1 to 1}.
///
/// @param [in] x            The value to calculate approximate cosine for.
///
/// @returns
/// A <c><i>FfxFloat32x2</i></c> containing approximations of both sine and cosine of <c><i>value</i></c>.
FfxFloat16x2 ffxParabolicSinCosHalf(FfxFloat16 x)
{
    FfxFloat16 y = ffxFract(x * FFX_BROADCAST_FLOAT16(0.5) + FFX_BROADCAST_FLOAT16(0.75));
    y     = y * FFX_BROADCAST_FLOAT16(2.0) - FFX_BROADCAST_FLOAT16(1.0);
    return ffxParabolicSinHalf(FfxFloat16x2(x, y));
}

/// Conditional free logic AND operation using two half-precision values.
///
/// @param [in] x           The first value to be fed into the AND operator.
/// @param [in] y           The second value to be fed into the AND operator.
///
/// @returns
/// Result of the AND operation.
///
/// @ingroup GPU
FfxUInt16 ffxZeroOneAndHalf(FfxUInt16 x, FfxUInt16 y)
{
    return min(x, y);
}

/// Conditional free logic AND operation using two half-precision values.
///
/// @param [in] x           The first value to be fed into the AND operator.
/// @param [in] y           The second value to be fed into the AND operator.
///
/// @returns
/// Result of the AND operation.
///
/// @ingroup GPU
FfxUInt16x2 ffxZeroOneAndHalf(FfxUInt16x2 x, FfxUInt16x2 y)
{
    return min(x, y);
}

/// Conditional free logic AND operation using two half-precision values.
///
/// @param [in] x           The first value to be fed into the AND operator.
/// @param [in] y           The second value to be fed into the AND operator.
///
/// @returns
/// Result of the AND operation.
///
/// @ingroup GPU
FfxUInt16x3 ffxZeroOneAndHalf(FfxUInt16x3 x, FfxUInt16x3 y)
{
    return min(x, y);
}

/// Conditional free logic AND operation using two half-precision values.
///
/// @param [in] x           The first value to be fed into the AND operator.
/// @param [in] y           The second value to be fed into the AND operator.
///
/// @returns
/// Result of the AND operation.
///
/// @ingroup GPU
FfxUInt16x4 ffxZeroOneAndHalf(FfxUInt16x4 x, FfxUInt16x4 y)
{
    return min(x, y);
}

/// Conditional free logic NOT operation using two half-precision values.
///
/// @param [in] x           The first value to be fed into the NOT operator.
/// @param [in] y           The second value to be fed into the NOT operator.
///
/// @returns
/// Result of the NOT operation.
///
/// @ingroup GPU
FfxUInt16 ffxZeroOneNotHalf(FfxUInt16 x)
{
    return x ^ FFX_BROADCAST_UINT16(1);
}

/// Conditional free logic NOT operation using two half-precision values.
///
/// @param [in] x           The first value to be fed into the NOT operator.
/// @param [in] y           The second value to be fed into the NOT operator.
///
/// @returns
/// Result of the NOT operation.
///
/// @ingroup GPU
FfxUInt16x2 ffxZeroOneNotHalf(FfxUInt16x2 x)
{
    return x ^ FFX_BROADCAST_UINT16X2(1);
}

/// Conditional free logic NOT operation using two half-precision values.
///
/// @param [in] x           The first value to be fed into the NOT operator.
/// @param [in] y           The second value to be fed into the NOT operator.
///
/// @returns
/// Result of the NOT operation.
///
/// @ingroup GPU
FfxUInt16x3 ffxZeroOneNotHalf(FfxUInt16x3 x)
{
    return x ^ FFX_BROADCAST_UINT16X3(1);
}

/// Conditional free logic NOT operation using two half-precision values.
///
/// @param [in] x           The first value to be fed into the NOT operator.
/// @param [in] y           The second value to be fed into the NOT operator.
///
/// @returns
/// Result of the NOT operation.
///
/// @ingroup GPU
FfxUInt16x4 ffxZeroOneNotHalf(FfxUInt16x4 x)
{
    return x ^ FFX_BROADCAST_UINT16X4(1);
}

/// Conditional free logic OR operation using two half-precision values.
///
/// @param [in] x           The first value to be fed into the OR operator.
/// @param [in] y           The second value to be fed into the OR operator.
///
/// @returns
/// Result of the OR operation.
///
/// @ingroup GPU
FfxUInt16 ffxZeroOneOrHalf(FfxUInt16 x, FfxUInt16 y)
{
    return max(x, y);
}

/// Conditional free logic OR operation using two half-precision values.
///
/// @param [in] x           The first value to be fed into the OR operator.
/// @param [in] y           The second value to be fed into the OR operator.
///
/// @returns
/// Result of the OR operation.
///
/// @ingroup GPU
FfxUInt16x2 ffxZeroOneOrHalf(FfxUInt16x2 x, FfxUInt16x2 y)
{
    return max(x, y);
}

/// Conditional free logic OR operation using two half-precision values.
///
/// @param [in] x           The first value to be fed into the OR operator.
/// @param [in] y           The second value to be fed into the OR operator.
///
/// @returns
/// Result of the OR operation.
///
/// @ingroup GPU
FfxUInt16x3 ffxZeroOneOrHalf(FfxUInt16x3 x, FfxUInt16x3 y)
{
    return max(x, y);
}

/// Conditional free logic OR operation using two half-precision values.
///
/// @param [in] x           The first value to be fed into the OR operator.
/// @param [in] y           The second value to be fed into the OR operator.
///
/// @returns
/// Result of the OR operation.
///
/// @ingroup GPU
FfxUInt16x4 ffxZeroOneOrHalf(FfxUInt16x4 x, FfxUInt16x4 y)
{
    return max(x, y);
}

/// Convert a half-precision FfxFloat32 value between 0.0f and 1.0f to a half-precision Uint.
///
/// @param [in] x           The value to converted to a Uint.
///
/// @returns
/// The converted Uint value.
///
/// @ingroup GPU
FfxUInt16 ffxZeroOneFloat16ToUint16(FfxFloat16 x)
{
    return FFX_TO_UINT16(x * FFX_TO_FLOAT16(FFX_TO_UINT16(1)));
}

/// Convert a half-precision FfxFloat32 value between 0.0f and 1.0f to a half-precision Uint.
///
/// @param [in] x           The value to converted to a Uint.
///
/// @returns
/// The converted Uint value.
///
/// @ingroup GPU
FfxUInt16x2 ffxZeroOneFloat16x2ToUint16x2(FfxFloat16x2 x)
{
    return FFX_TO_UINT16X2(x * FFX_TO_FLOAT16X2(FfxUInt16x2(1, 1)));
}

/// Convert a half-precision FfxFloat32 value between 0.0f and 1.0f to a half-precision Uint.
///
/// @param [in] x           The value to converted to a Uint.
///
/// @returns
/// The converted Uint value.
///
/// @ingroup GPU
FfxUInt16x3 ffxZeroOneFloat16x3ToUint16x3(FfxFloat16x3 x)
{
    return FFX_TO_UINT16X3(x * FFX_TO_FLOAT16X3(FfxUInt16x3(1, 1, 1)));
}

/// Convert a half-precision FfxFloat32 value between 0.0f and 1.0f to a half-precision Uint.
///
/// @param [in] x           The value to converted to a Uint.
///
/// @returns
/// The converted Uint value.
///
/// @ingroup GPU
FfxUInt16x4 ffxZeroOneFloat16x4ToUint16x4(FfxFloat16x4 x)
{
    return FFX_TO_UINT16X4(x * FFX_TO_FLOAT16X4(FfxUInt16x4(1, 1, 1, 1)));
}

/// Convert a half-precision FfxUInt32 value between 0 and 1 to a half-precision FfxFloat32.
///
/// @param [in] x           The value to converted to a half-precision FfxFloat32.
///
/// @returns
/// The converted half-precision FfxFloat32 value.
///
/// @ingroup GPU
FfxFloat16 ffxZeroOneUint16ToFloat16(FfxUInt16 x)
{
    return FFX_TO_FLOAT16(x * FFX_TO_UINT16(FFX_TO_FLOAT16(1.0)));
}

/// Convert a half-precision FfxUInt32 value between 0 and 1 to a half-precision FfxFloat32.
///
/// @param [in] x           The value to converted to a half-precision FfxFloat32.
///
/// @returns
/// The converted half-precision FfxFloat32 value.
///
/// @ingroup GPU
FfxFloat16x2 ffxZeroOneUint16x2ToFloat16x2(FfxUInt16x2 x)
{
    return FFX_TO_FLOAT16X2(x * FFX_TO_UINT16X2(FfxUInt16x2(FFX_TO_FLOAT16(1.0), FFX_TO_FLOAT16(1.0))));
}

/// Convert a half-precision FfxUInt32 value between 0 and 1 to a half-precision FfxFloat32.
///
/// @param [in] x           The value to converted to a half-precision FfxFloat32.
///
/// @returns
/// The converted half-precision FfxFloat32 value.
///
/// @ingroup GPU
FfxFloat16x3 ffxZeroOneUint16x3ToFloat16x3(FfxUInt16x3 x)
{
    return FFX_TO_FLOAT16X3(x * FFX_TO_UINT16X3(FfxUInt16x3(FFX_TO_FLOAT16(1.0), FFX_TO_FLOAT16(1.0), FFX_TO_FLOAT16(1.0))));
}

/// Convert a half-precision FfxUInt32 value between 0 and 1 to a half-precision FfxFloat32.
///
/// @param [in] x           The value to converted to a half-precision FfxFloat32.
///
/// @returns
/// The converted half-precision FfxFloat32 value.
///
/// @ingroup GPU
FfxFloat16x4 ffxZeroOneUint16x4ToFloat16x4(FfxUInt16x4 x)
{
    return FFX_TO_FLOAT16X4(x * FFX_TO_UINT16X4(FfxUInt16x4(FFX_TO_FLOAT16(1.0), FFX_TO_FLOAT16(1.0), FFX_TO_FLOAT16(1.0), FFX_TO_FLOAT16(1.0))));
}

/// Conditional free logic AND operation using two half-precision values.
///
/// @param [in] x           The first value to be fed into the AND operator.
/// @param [in] y           The second value to be fed into the AND operator.
///
/// @returns
/// Result of the AND operation.
///
/// @ingroup GPU
FfxFloat16 ffxZeroOneAndHalf(FfxFloat16 x, FfxFloat16 y)
{
    return min(x, y);
}

/// Conditional free logic AND operation using two half-precision values.
///
/// @param [in] x           The first value to be fed into the AND operator.
/// @param [in] y           The second value to be fed into the AND operator.
///
/// @returns
/// Result of the AND operation.
///
/// @ingroup GPU
FfxFloat16x2 ffxZeroOneAndHalf(FfxFloat16x2 x, FfxFloat16x2 y)
{
    return min(x, y);
}

/// Conditional free logic AND operation using two half-precision values.
///
/// @param [in] x           The first value to be fed into the AND operator.
/// @param [in] y           The second value to be fed into the AND operator.
///
/// @returns
/// Result of the AND operation.
///
/// @ingroup GPU
FfxFloat16x3 ffxZeroOneAndHalf(FfxFloat16x3 x, FfxFloat16x3 y)
{
    return min(x, y);
}

/// Conditional free logic AND operation using two half-precision values.
///
/// @param [in] x           The first value to be fed into the AND operator.
/// @param [in] y           The second value to be fed into the AND operator.
///
/// @returns
/// Result of the AND operation.
///
/// @ingroup GPU
FfxFloat16x4 ffxZeroOneAndHalf(FfxFloat16x4 x, FfxFloat16x4 y)
{
    return min(x, y);
}

/// Conditional free logic AND NOT operation using two half-precision values.
///
/// @param [in] x           The first value to be fed into the AND NOT operator.
/// @param [in] y           The second value to be fed into the AND NOT operator.
///
/// @returns
/// Result of the AND NOT operation.
///
/// @ingroup GPU
FfxFloat16 ffxSignedZeroOneAndOrHalf(FfxFloat16 x, FfxFloat16 y)
{
    return (-x) * y + FFX_BROADCAST_FLOAT16(1.0);
}

/// Conditional free logic AND NOT operation using two half-precision values.
///
/// @param [in] x           The first value to be fed into the AND NOT operator.
/// @param [in] y           The second value to be fed into the AND NOT operator.
///
/// @returns
/// Result of the AND NOT operation.
///
/// @ingroup GPU
FfxFloat16x2 ffxSignedZeroOneAndOrHalf(FfxFloat16x2 x, FfxFloat16x2 y)
{
    return (-x) * y + FFX_BROADCAST_FLOAT16X2(1.0);
}

/// Conditional free logic AND NOT operation using two half-precision values.
///
/// @param [in] x           The first value to be fed into the AND NOT operator.
/// @param [in] y           The second value to be fed into the AND NOT operator.
///
/// @returns
/// Result of the AND NOT operation.
///
/// @ingroup GPU
FfxFloat16x3 ffxSignedZeroOneAndOrHalf(FfxFloat16x3 x, FfxFloat16x3 y)
{
    return (-x) * y + FFX_BROADCAST_FLOAT16X3(1.0);
}

/// Conditional free logic AND NOT operation using two half-precision values.
///
/// @param [in] x           The first value to be fed into the AND NOT operator.
/// @param [in] y           The second value to be fed into the AND NOT operator.
///
/// @returns
/// Result of the AND NOT operation.
///
/// @ingroup GPU
FfxFloat16x4 ffxSignedZeroOneAndOrHalf(FfxFloat16x4 x, FfxFloat16x4 y)
{
    return (-x) * y + FFX_BROADCAST_FLOAT16X4(1.0);
}

/// Conditional free logic AND operation using two half-precision values followed by
/// a NOT operation using the resulting value and a third half-precision value.
///
/// @param [in] x           The first value to be fed into the AND operator.
/// @param [in] y           The second value to be fed into the AND operator.
/// @param [in] z           The second value to be fed into the OR operator.
///
/// @returns
/// Result of the AND OR operation.
///
/// @ingroup GPU
FfxFloat16 ffxZeroOneAndOrHalf(FfxFloat16 x, FfxFloat16 y, FfxFloat16 z)
{
    return ffxSaturate(x * y + z);
}

/// Conditional free logic AND operation using two half-precision values followed by
/// a NOT operation using the resulting value and a third half-precision value.
///
/// @param [in] x           The first value to be fed into the AND operator.
/// @param [in] y           The second value to be fed into the AND operator.
/// @param [in] z           The second value to be fed into the OR operator.
///
/// @returns
/// Result of the AND OR operation.
///
/// @ingroup GPU
FfxFloat16x2 ffxZeroOneAndOrHalf(FfxFloat16x2 x, FfxFloat16x2 y, FfxFloat16x2 z)
{
    return ffxSaturate(x * y + z);
}

/// Conditional free logic AND operation using two half-precision values followed by
/// a NOT operation using the resulting value and a third half-precision value.
///
/// @param [in] x           The first value to be fed into the AND operator.
/// @param [in] y           The second value to be fed into the AND operator.
/// @param [in] z           The second value to be fed into the OR operator.
///
/// @returns
/// Result of the AND OR operation.
///
/// @ingroup GPU
FfxFloat16x3 ffxZeroOneAndOrHalf(FfxFloat16x3 x, FfxFloat16x3 y, FfxFloat16x3 z)
{
    return ffxSaturate(x * y + z);
}

/// Conditional free logic AND operation using two half-precision values followed by
/// a NOT operation using the resulting value and a third half-precision value.
///
/// @param [in] x           The first value to be fed into the AND operator.
/// @param [in] y           The second value to be fed into the AND operator.
/// @param [in] z           The second value to be fed into the OR operator.
///
/// @returns
/// Result of the AND OR operation.
///
/// @ingroup GPU
FfxFloat16x4 ffxZeroOneAndOrHalf(FfxFloat16x4 x, FfxFloat16x4 y, FfxFloat16x4 z)
{
    return ffxSaturate(x * y + z);
}

/// Given a half-precision value, returns 1.0 if greater than zero and 0.0 if not.
///
/// @param [in] x           The value to be compared.
///
/// @returns
/// Result of the greater than zero comparison.
///
/// @ingroup GPU
FfxFloat16 ffxZeroOneIsGreaterThanZeroHalf(FfxFloat16 x)
{
    return ffxSaturate(x * FFX_BROADCAST_FLOAT16(FFX_POSITIVE_INFINITY_HALF));
}

/// Given a half-precision value, returns 1.0 if greater than zero and 0.0 if not.
///
/// @param [in] x           The value to be compared.
///
/// @returns
/// Result of the greater than zero comparison.
///
/// @ingroup GPU
FfxFloat16x2 ffxZeroOneIsGreaterThanZeroHalf(FfxFloat16x2 x)
{
    return ffxSaturate(x * FFX_BROADCAST_FLOAT16X2(FFX_POSITIVE_INFINITY_HALF));
}

/// Given a half-precision value, returns 1.0 if greater than zero and 0.0 if not.
///
/// @param [in] x           The value to be compared.
///
/// @returns
/// Result of the greater than zero comparison.
///
/// @ingroup GPU
FfxFloat16x3 ffxZeroOneIsGreaterThanZeroHalf(FfxFloat16x3 x)
{
    return ffxSaturate(x * FFX_BROADCAST_FLOAT16X3(FFX_POSITIVE_INFINITY_HALF));
}

/// Given a half-precision value, returns 1.0 if greater than zero and 0.0 if not.
///
/// @param [in] x           The value to be compared.
///
/// @returns
/// Result of the greater than zero comparison.
///
/// @ingroup GPU
FfxFloat16x4 ffxZeroOneIsGreaterThanZeroHalf(FfxFloat16x4 x)
{
    return ffxSaturate(x * FFX_BROADCAST_FLOAT16X4(FFX_POSITIVE_INFINITY_HALF));
}

/// Conditional free logic signed NOT operation using two half-precision FfxFloat32 values.
///
/// @param [in] x           The first value to be fed into the AND OR operator.
///
/// @returns
/// Result of the AND OR operation.
///
/// @ingroup GPU
FfxFloat16 ffxZeroOneNotHalf(FfxFloat16 x)
{
    return FFX_BROADCAST_FLOAT16(1.0) - x;
}

/// Conditional free logic signed NOT operation using two half-precision FfxFloat32 values.
///
/// @param [in] x           The first value to be fed into the AND OR operator.
///
/// @returns
/// Result of the AND OR operation.
///
/// @ingroup GPU
FfxFloat16x2 ffxZeroOneNotHalf(FfxFloat16x2 x)
{
    return FFX_BROADCAST_FLOAT16X2(1.0) - x;
}

/// Conditional free logic signed NOT operation using two half-precision FfxFloat32 values.
///
/// @param [in] x           The first value to be fed into the AND OR operator.
///
/// @returns
/// Result of the AND OR operation.
///
/// @ingroup GPU
FfxFloat16x3 ffxZeroOneNotHalf(FfxFloat16x3 x)
{
    return FFX_BROADCAST_FLOAT16X3(1.0) - x;
}

/// Conditional free logic signed NOT operation using two half-precision FfxFloat32 values.
///
/// @param [in] x           The first value to be fed into the AND OR operator.
///
/// @returns
/// Result of the AND OR operation.
///
/// @ingroup GPU
FfxFloat16x4 ffxZeroOneNotHalf(FfxFloat16x4 x)
{
    return FFX_BROADCAST_FLOAT16X4(1.0) - x;
}

/// Conditional free logic OR operation using two half-precision FfxFloat32 values.
///
/// @param [in] x           The first value to be fed into the OR operator.
/// @param [in] y           The second value to be fed into the OR operator.
///
/// @returns
/// Result of the OR operation.
///
/// @ingroup GPU
FfxFloat16 ffxZeroOneOrHalf(FfxFloat16 x, FfxFloat16 y)
{
    return max(x, y);
}

/// Conditional free logic OR operation using two half-precision FfxFloat32 values.
///
/// @param [in] x           The first value to be fed into the OR operator.
/// @param [in] y           The second value to be fed into the OR operator.
///
/// @returns
/// Result of the OR operation.
///
/// @ingroup GPU
FfxFloat16x2 ffxZeroOneOrHalf(FfxFloat16x2 x, FfxFloat16x2 y)
{
    return max(x, y);
}

/// Conditional free logic OR operation using two half-precision FfxFloat32 values.
///
/// @param [in] x           The first value to be fed into the OR operator.
/// @param [in] y           The second value to be fed into the OR operator.
///
/// @returns
/// Result of the OR operation.
///
/// @ingroup GPU
FfxFloat16x3 ffxZeroOneOrHalf(FfxFloat16x3 x, FfxFloat16x3 y)
{
    return max(x, y);
}

/// Conditional free logic OR operation using two half-precision FfxFloat32 values.
///
/// @param [in] x           The first value to be fed into the OR operator.
/// @param [in] y           The second value to be fed into the OR operator.
///
/// @returns
/// Result of the OR operation.
///
/// @ingroup GPU
FfxFloat16x4 ffxZeroOneOrHalf(FfxFloat16x4 x, FfxFloat16x4 y)
{
    return max(x, y);
}

/// Choose between two half-precision FfxFloat32 values if the first paramter is greater than zero.
///
/// @param [in] x           The value to compare against zero.
/// @param [in] y           The value to return if the comparision is greater than zero.
/// @param [in] z           The value to return if the comparision is less than or equal to zero.
///
/// @returns
/// The selected value.
///
/// @ingroup GPU
FfxFloat16 ffxZeroOneSelectHalf(FfxFloat16 x, FfxFloat16 y, FfxFloat16 z)
{
    FfxFloat16 r = (-x) * z + z;
    return x * y + r;
}

/// Choose between two half-precision FfxFloat32 values if the first paramter is greater than zero.
///
/// @param [in] x           The value to compare against zero.
/// @param [in] y           The value to return if the comparision is greater than zero.
/// @param [in] z           The value to return if the comparision is less than or equal to zero.
///
/// @returns
/// The selected value.
///
/// @ingroup GPU
FfxFloat16x2 ffxZeroOneSelectHalf(FfxFloat16x2 x, FfxFloat16x2 y, FfxFloat16x2 z)
{
    FfxFloat16x2 r = (-x) * z + z;
    return x * y + r;
}

/// Choose between two half-precision FfxFloat32 values if the first paramter is greater than zero.
///
/// @param [in] x           The value to compare against zero.
/// @param [in] y           The value to return if the comparision is greater than zero.
/// @param [in] z           The value to return if the comparision is less than or equal to zero.
///
/// @returns
/// The selected value.
///
/// @ingroup GPU
FfxFloat16x3 ffxZeroOneSelectHalf(FfxFloat16x3 x, FfxFloat16x3 y, FfxFloat16x3 z)
{
    FfxFloat16x3 r = (-x) * z + z;
    return x * y + r;
}

/// Choose between two half-precision FfxFloat32 values if the first paramter is greater than zero.
///
/// @param [in] x           The value to compare against zero.
/// @param [in] y           The value to return if the comparision is greater than zero.
/// @param [in] z           The value to return if the comparision is less than or equal to zero.
///
/// @returns
/// The selected value.
///
/// @ingroup GPU
FfxFloat16x4 ffxZeroOneSelectHalf(FfxFloat16x4 x, FfxFloat16x4 y, FfxFloat16x4 z)
{
    FfxFloat16x4 r = (-x) * z + z;
    return x * y + r;
}

/// Given a half-precision value, returns 1.0 if less than zero and 0.0 if not.
///
/// @param [in] x           The value to be compared.
///
/// @returns
/// Result of the sign value.
///
/// @ingroup GPU
FfxFloat16 ffxZeroOneIsSignedHalf(FfxFloat16 x)
{
    return ffxSaturate(x * FFX_BROADCAST_FLOAT16(FFX_NEGATIVE_INFINITY_HALF));
}

/// Given a half-precision value, returns 1.0 if less than zero and 0.0 if not.
///
/// @param [in] x           The value to be compared.
///
/// @returns
/// Result of the sign value.
///
/// @ingroup GPU
FfxFloat16x2 ffxZeroOneIsSignedHalf(FfxFloat16x2 x)
{
    return ffxSaturate(x * FFX_BROADCAST_FLOAT16X2(FFX_NEGATIVE_INFINITY_HALF));
}

/// Given a half-precision value, returns 1.0 if less than zero and 0.0 if not.
///
/// @param [in] x           The value to be compared.
///
/// @returns
/// Result of the sign value.
///
/// @ingroup GPU
FfxFloat16x3 ffxZeroOneIsSignedHalf(FfxFloat16x3 x)
{
    return ffxSaturate(x * FFX_BROADCAST_FLOAT16X3(FFX_NEGATIVE_INFINITY_HALF));
}

/// Given a half-precision value, returns 1.0 if less than zero and 0.0 if not.
///
/// @param [in] x           The value to be compared.
///
/// @returns
/// Result of the sign value.
///
/// @ingroup GPU
FfxFloat16x4 ffxZeroOneIsSignedHalf(FfxFloat16x4 x)
{
    return ffxSaturate(x * FFX_BROADCAST_FLOAT16X4(FFX_NEGATIVE_INFINITY_HALF));
}

/// Compute a Rec.709 color space.
/// 
/// Rec.709 is used for some HDTVs.
/// 
/// Both Rec.709 and sRGB have a linear segment which as spec'ed would intersect the curved segment 2 times.
///  (a.) For 8-bit sRGB, steps {0 to 10.3} are in the linear region (4% of the encoding range).
///  (b.) For 8-bit  709, steps {0 to 20.7} are in the linear region (8% of the encoding range).
///
/// @param [in] c           The color to convert to Rec. 709.
/// 
/// @returns
/// The <c><i>color</i></c> in Rec.709 space.
/// 
/// @ingroup GPU
FfxFloat16 ffxRec709FromLinearHalf(FfxFloat16 c)
{
    FfxFloat16x3 j = FfxFloat16x3(0.018 * 4.5, 4.5, 0.45);
    FfxFloat16x2 k = FfxFloat16x2(1.099, -0.099);
    return clamp(j.x, c * j.y, pow(c, j.z) * k.x + k.y);
}

/// Compute a Rec.709 color space.
/// 
/// Rec.709 is used for some HDTVs.
/// 
/// Both Rec.709 and sRGB have a linear segment which as spec'ed would intersect the curved segment 2 times.
///  (a.) For 8-bit sRGB, steps {0 to 10.3} are in the linear region (4% of the encoding range).
///  (b.) For 8-bit  709, steps {0 to 20.7} are in the linear region (8% of the encoding range).
///
/// @param [in] c           The color to convert to Rec. 709.
/// 
/// @returns
/// The <c><i>color</i></c> in Rec.709 space.
/// 
/// @ingroup GPU
FfxFloat16x2 ffxRec709FromLinearHalf(FfxFloat16x2 c)
{
    FfxFloat16x3 j = FfxFloat16x3(0.018 * 4.5, 4.5, 0.45);
    FfxFloat16x2 k = FfxFloat16x2(1.099, -0.099);
    return clamp(j.xx, c * j.yy, pow(c, j.zz) * k.xx + k.yy);
}

/// Compute a Rec.709 color space.
/// 
/// Rec.709 is used for some HDTVs.
/// 
/// Both Rec.709 and sRGB have a linear segment which as spec'ed would intersect the curved segment 2 times.
///  (a.) For 8-bit sRGB, steps {0 to 10.3} are in the linear region (4% of the encoding range).
///  (b.) For 8-bit  709, steps {0 to 20.7} are in the linear region (8% of the encoding range).
///
/// @param [in] c           The color to convert to Rec. 709.
/// 
/// @returns
/// The <c><i>color</i></c> in Rec.709 space.
/// 
/// @ingroup GPU
FfxFloat16x3 ffxRec709FromLinearHalf(FfxFloat16x3 c)
{
    FfxFloat16x3 j = FfxFloat16x3(0.018 * 4.5, 4.5, 0.45);
    FfxFloat16x2 k = FfxFloat16x2(1.099, -0.099);
    return clamp(j.xxx, c * j.yyy, pow(c, j.zzz) * k.xxx + k.yyy);
}

/// Compute a gamma value from a linear value.
///
/// Typically 2.2 for some PC displays, or 2.4-2.5 for CRTs, or 2.2 FreeSync2 native.
/// 
/// Note: 'rcpX' is '1/x', where the 'x' is what would be used in <c><i>ffxLinearFromGammaHalf</i></c>.
/// 
/// @param [in] c              The value to convert to gamma space from linear.
/// @param [in] rcpX           The reciprocal of power value used for the gamma curve.
///
/// @returns
/// A value in gamma space.
///
/// @ingroup GPU
FfxFloat16 ffxGammaFromLinearHalf(FfxFloat16 c, FfxFloat16 rcpX)
{
    return pow(c, FFX_BROADCAST_FLOAT16(rcpX));
}

/// Compute a gamma value from a linear value.
///
/// Typically 2.2 for some PC displays, or 2.4-2.5 for CRTs, or 2.2 FreeSync2 native.
/// 
/// Note: 'rcpX' is '1/x', where the 'x' is what would be used in <c><i>ffxLinearFromGammaHalf</i></c>.
/// 
/// @param [in] c              The value to convert to gamma space from linear.
/// @param [in] rcpX           The reciprocal of power value used for the gamma curve.
///
/// @returns
/// A value in gamma space.
///
/// @ingroup GPU
FfxFloat16x2 ffxGammaFromLinearHalf(FfxFloat16x2 c, FfxFloat16 rcpX)
{
    return pow(c, FFX_BROADCAST_FLOAT16X2(rcpX));
}

/// Compute a gamma value from a linear value.
///
/// Typically 2.2 for some PC displays, or 2.4-2.5 for CRTs, or 2.2 FreeSync2 native.
/// 
/// Note: 'rcpX' is '1/x', where the 'x' is what would be used in <c><i>ffxLinearFromGammaHalf</i></c>.
/// 
/// @param [in] c              The value to convert to gamma space from linear.
/// @param [in] rcpX           The reciprocal of power value used for the gamma curve.
///
/// @returns
/// A value in gamma space.
///
/// @ingroup GPU
FfxFloat16x3 ffxGammaFromLinearHalf(FfxFloat16x3 c, FfxFloat16 rcpX)
{
    return pow(c, FFX_BROADCAST_FLOAT16X3(rcpX));
}

/// Compute an SRGB value from a linear value.
///
/// @param [in] c           The value to convert to SRGB from linear.
///
/// @returns
/// A value in SRGB space.
///
/// @ingroup GPU
FfxFloat16 ffxSrgbFromLinearHalf(FfxFloat16 c)
{
    FfxFloat16x3 j = FfxFloat16x3(0.0031308 * 12.92, 12.92, 1.0 / 2.4);
    FfxFloat16x2 k = FfxFloat16x2(1.055, -0.055);
    return clamp(j.x, c * j.y, pow(c, j.z) * k.x + k.y);
}

/// Compute an SRGB value from a linear value.
///
/// @param [in] c           The value to convert to SRGB from linear.
///
/// @returns
/// A value in SRGB space.
///
/// @ingroup GPU
FfxFloat16x2 ffxSrgbFromLinearHalf(FfxFloat16x2 c)
{
    FfxFloat16x3 j = FfxFloat16x3(0.0031308 * 12.92, 12.92, 1.0 / 2.4);
    FfxFloat16x2 k = FfxFloat16x2(1.055, -0.055);
    return clamp(j.xx, c * j.yy, pow(c, j.zz) * k.xx + k.yy);
}

/// Compute an SRGB value from a linear value.
///
/// @param [in] c           The value to convert to SRGB from linear.
///
/// @returns
/// A value in SRGB space.
///
/// @ingroup GPU
FfxFloat16x3 ffxSrgbFromLinearHalf(FfxFloat16x3 c)
{
    FfxFloat16x3 j = FfxFloat16x3(0.0031308 * 12.92, 12.92, 1.0 / 2.4);
    FfxFloat16x2 k = FfxFloat16x2(1.055, -0.055);
    return clamp(j.xxx, c * j.yyy, pow(c, j.zzz) * k.xxx + k.yyy);
}

/// Compute the square root of a value.
///
/// @param [in] c           The value to compute the square root for.
///
/// @returns
/// A square root of the input value.
///
/// @ingroup GPU
FfxFloat16 ffxSquareRootHalf(FfxFloat16 c)
{
    return sqrt(c);
}

/// Compute the square root of a value.
///
/// @param [in] c           The value to compute the square root for.
///
/// @returns
/// A square root of the input value.
///
/// @ingroup GPU
FfxFloat16x2 ffxSquareRootHalf(FfxFloat16x2 c)
{
    return sqrt(c);
}

/// Compute the square root of a value.
///
/// @param [in] c           The value to compute the square root for.
///
/// @returns
/// A square root of the input value.
///
/// @ingroup GPU
FfxFloat16x3 ffxSquareRootHalf(FfxFloat16x3 c)
{
    return sqrt(c);
}

/// Compute the cube root of a value.
///
/// @param [in] c           The value to compute the cube root for.
///
/// @returns
/// A cube root of the input value.
///
/// @ingroup GPU
FfxFloat16 ffxCubeRootHalf(FfxFloat16 c)
{
    return pow(c, FFX_BROADCAST_FLOAT16(1.0 / 3.0));
}

/// Compute the cube root of a value.
///
/// @param [in] c           The value to compute the cube root for.
///
/// @returns
/// A cube root of the input value.
///
/// @ingroup GPU
FfxFloat16x2 ffxCubeRootHalf(FfxFloat16x2 c)
{
    return pow(c, FFX_BROADCAST_FLOAT16X2(1.0 / 3.0));
}

/// Compute the cube root of a value.
///
/// @param [in] c           The value to compute the cube root for.
///
/// @returns
/// A cube root of the input value.
///
/// @ingroup GPU
FfxFloat16x3 ffxCubeRootHalf(FfxFloat16x3 c)
{
    return pow(c, FFX_BROADCAST_FLOAT16X3(1.0 / 3.0));
}

/// Compute a linear value from a REC.709 value.
///
/// @param [in] c           The value to convert to linear from REC.709.
///
/// @returns
/// A value in linear space.
///
/// @ingroup GPU
FfxFloat16 ffxLinearFromRec709Half(FfxFloat16 c)
{
    FfxFloat16x3 j = FfxFloat16x3(0.081 / 4.5, 1.0 / 4.5, 1.0 / 0.45);
    FfxFloat16x2 k = FfxFloat16x2(1.0 / 1.099, 0.099 / 1.099);
    return ffxZeroOneSelectHalf(ffxZeroOneIsSignedHalf(c - j.x), c * j.y, pow(c * k.x + k.y, j.z));
}

/// Compute a linear value from a REC.709 value.
///
/// @param [in] c           The value to convert to linear from REC.709.
///
/// @returns
/// A value in linear space.
///
/// @ingroup GPU
FfxFloat16x2 ffxLinearFromRec709Half(FfxFloat16x2 c)
{
    FfxFloat16x3 j = FfxFloat16x3(0.081 / 4.5, 1.0 / 4.5, 1.0 / 0.45);
    FfxFloat16x2 k = FfxFloat16x2(1.0 / 1.099, 0.099 / 1.099);
    return ffxZeroOneSelectHalf(ffxZeroOneIsSignedHalf(c - j.xx), c * j.yy, pow(c * k.xx + k.yy, j.zz));
}

/// Compute a linear value from a REC.709 value.
///
/// @param [in] c           The value to convert to linear from REC.709.
///
/// @returns
/// A value in linear space.
///
/// @ingroup GPU
FfxFloat16x3 ffxLinearFromRec709Half(FfxFloat16x3 c)
{
    FfxFloat16x3 j = FfxFloat16x3(0.081 / 4.5, 1.0 / 4.5, 1.0 / 0.45);
    FfxFloat16x2 k = FfxFloat16x2(1.0 / 1.099, 0.099 / 1.099);
    return ffxZeroOneSelectHalf(ffxZeroOneIsSignedHalf(c - j.xxx), c * j.yyy, pow(c * k.xxx + k.yyy, j.zzz));
}

/// Compute a linear value from a value in a gamma space.
///
/// Typically 2.2 for some PC displays, or 2.4-2.5 for CRTs, or 2.2 FreeSync2 native.
///
/// @param [in] c           The value to convert to linear in gamma space.
/// @param [in] x           The power value used for the gamma curve.
///
/// @returns
/// A value in linear space.
///
/// @ingroup GPU
FfxFloat16 ffxLinearFromGammaHalf(FfxFloat16 c, FfxFloat16 x)
{
    return pow(c, FFX_BROADCAST_FLOAT16(x));
}

/// Compute a linear value from a value in a gamma space.
///
/// Typically 2.2 for some PC displays, or 2.4-2.5 for CRTs, or 2.2 FreeSync2 native.
///
/// @param [in] c           The value to convert to linear in gamma space.
/// @param [in] x           The power value used for the gamma curve.
///
/// @returns
/// A value in linear space.
///
/// @ingroup GPU
FfxFloat16x2 ffxLinearFromGammaHalf(FfxFloat16x2 c, FfxFloat16 x)
{
    return pow(c, FFX_BROADCAST_FLOAT16X2(x));
}

/// Compute a linear value from a value in a gamma space.
///
/// Typically 2.2 for some PC displays, or 2.4-2.5 for CRTs, or 2.2 FreeSync2 native.
///
/// @param [in] c           The value to convert to linear in gamma space.
/// @param [in] x           The power value used for the gamma curve.
///
/// @returns
/// A value in linear space.
///
/// @ingroup GPU
FfxFloat16x3 ffxLinearFromGammaHalf(FfxFloat16x3 c, FfxFloat16 x)
{
    return pow(c, FFX_BROADCAST_FLOAT16X3(x));
}

/// Compute a linear value from a value in a SRGB space.
///
/// Typically 2.2 for some PC displays, or 2.4-2.5 for CRTs, or 2.2 FreeSync2 native.
///
/// @param [in] c           The value to convert to linear in SRGB space.
///
/// @returns
/// A value in linear space.
///
/// @ingroup GPU
FfxFloat16 ffxLinearFromSrgbHalf(FfxFloat16 c)
{
    FfxFloat16x3 j = FfxFloat16x3(0.04045 / 12.92, 1.0 / 12.92, 2.4);
    FfxFloat16x2 k = FfxFloat16x2(1.0 / 1.055, 0.055 / 1.055);
    return ffxZeroOneSelectHalf(ffxZeroOneIsSignedHalf(c - j.x), c * j.y, pow(c * k.x + k.y, j.z));
}

/// Compute a linear value from a value in a SRGB space.
///
/// Typically 2.2 for some PC displays, or 2.4-2.5 for CRTs, or 2.2 FreeSync2 native.
///
/// @param [in] c           The value to convert to linear in SRGB space.
///
/// @returns
/// A value in linear space.
///
/// @ingroup GPU
FfxFloat16x2 ffxLinearFromSrgbHalf(FfxFloat16x2 c)
{
    FfxFloat16x3 j = FfxFloat16x3(0.04045 / 12.92, 1.0 / 12.92, 2.4);
    FfxFloat16x2 k = FfxFloat16x2(1.0 / 1.055, 0.055 / 1.055);
    return ffxZeroOneSelectHalf(ffxZeroOneIsSignedHalf(c - j.xx), c * j.yy, pow(c * k.xx + k.yy, j.zz));
}

/// Compute a linear value from a value in a SRGB space.
///
/// Typically 2.2 for some PC displays, or 2.4-2.5 for CRTs, or 2.2 FreeSync2 native.
///
/// @param [in] c           The value to convert to linear in SRGB space.
///
/// @returns
/// A value in linear space.
///
/// @ingroup GPU
FfxFloat16x3 ffxLinearFromSrgbHalf(FfxFloat16x3 c)
{
    FfxFloat16x3 j = FfxFloat16x3(0.04045 / 12.92, 1.0 / 12.92, 2.4);
    FfxFloat16x2 k = FfxFloat16x2(1.0 / 1.055, 0.055 / 1.055);
    return ffxZeroOneSelectHalf(ffxZeroOneIsSignedHalf(c - j.xxx), c * j.yyy, pow(c * k.xxx + k.yyy, j.zzz));
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
FfxUInt16x2 ffxRemapForQuadHalf(FfxUInt32 a)
{
    return FfxUInt16x2(bitfieldExtract(a, 1u, 3u), bitfieldInsertMask(bitfieldExtract(a, 3u, 3u), a, 1u));
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
FfxUInt16x2 ffxRemapForWaveReductionHalf(FfxUInt32 a)
{
    return FfxUInt16x2(bitfieldInsertMask(bitfieldExtract(a, 2u, 3u), a, 1u), bitfieldInsertMask(bitfieldExtract(a, 3u, 3u), bitfieldExtract(a, 1u, 2u), 2u));
}

#endif  // FFX_HALF
