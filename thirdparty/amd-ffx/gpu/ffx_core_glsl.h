// This file is part of the FidelityFX SDK.
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
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

/// @defgroup GLSLCore GLSL Core
/// GLSL core defines and functions
///
/// @ingroup FfxGLSL

/// A define for abstracting select functionality for pre/post HLSL 21
///
/// @ingroup GLSLCore
#define FFX_SELECT(cond, arg1, arg2) cond ? arg1 : arg2

/// A define for abstracting shared memory between shading languages.
///
/// @ingroup GLSLCore
#define FFX_GROUPSHARED shared

/// A define for abstracting compute memory barriers between shading languages.
///
/// @ingroup GLSLCore
#define FFX_GROUP_MEMORY_BARRIER groupMemoryBarrier(); barrier()

/// A define for abstracting compute atomic additions between shading languages.
///
/// @ingroup GLSLCore
#define FFX_ATOMIC_ADD(x, y) atomicAdd(x, y)

/// A define for abstracting compute atomic additions between shading languages.
///
/// @ingroup GLSLCore
#define FFX_ATOMIC_ADD_RETURN(x, y, r) r = atomicAdd(x, y)

/// A define for abstracting compute atomic OR between shading languages.
///
/// @ingroup GLSLCore
#define FFX_ATOMIC_OR(x, y) atomicOr(x, y)

/// A define for abstracting compute atomic min between shading languages.
///
/// @ingroup GLSLCore
#define FFX_ATOMIC_MIN(x, y) atomicMin(x, y)

/// A define for abstracting compute atomic max between shading languages.
///
/// @ingroup GLSLCore
#define FFX_ATOMIC_MAX(x, y) atomicMax(x, y)

/// A define added to accept static markup on functions to aid CPU/GPU portability of code.
///
/// @ingroup GLSLCore
#define FFX_STATIC

/// A define for abstracting loop unrolling between shading languages.
///
/// @ingroup GLSLCore
#define FFX_UNROLL

/// A define for abstracting a 'greater than' comparison operator between two types.
///
/// @ingroup GLSLCore
#define FFX_GREATER_THAN(x, y) greaterThan(x, y)

/// A define for abstracting a 'greater than or equal' comparison operator between two types.
///
/// @ingroup GLSLCore
#define FFX_GREATER_THAN_EQUAL(x, y) greaterThanEqual(x, y)

/// A define for abstracting a 'less than' comparison operator between two types.
///
/// @ingroup GLSLCore
#define FFX_LESS_THAN(x, y) lessThan(x, y)

/// A define for abstracting a 'less than or equal' comparison operator between two types.
///
/// @ingroup GLSLCore
#define FFX_LESS_THAN_EQUAL(x, y) lessThanEqual(x, y)

/// A define for abstracting an 'equal' comparison operator between two types.
///
/// @ingroup GLSLCore
#define FFX_EQUAL(x, y) equal(x, y)

/// A define for abstracting a 'not equal' comparison operator between two types.
///
/// @ingroup GLSLCore
#define FFX_NOT_EQUAL(x, y) notEqual(x, y)

/// A define for abstracting matrix multiply operations between shading languages.
///
/// @ingroup GLSLCore
#define FFX_MATRIX_MULTIPLY(a, b) (a * b)

/// A define for abstracting vector transformations between shading languages.
///
/// @ingroup GLSLCore
#define FFX_TRANSFORM_VECTOR(a, b) (a * b)

/// A define for abstracting modulo operations between shading languages.
///
/// @ingroup GLSLCore
#define FFX_MODULO(a, b) (mod(a, b))

/// Broadcast a scalar value to a 1-dimensional floating point vector.
///
/// @ingroup GLSLCore
#define FFX_BROADCAST_FLOAT32(x)   FfxFloat32(x)

/// Broadcast a scalar value to a 2-dimensional floating point vector.
///
/// @ingroup GLSLCore
#define FFX_BROADCAST_FLOAT32X2(x) FfxFloat32x2(FfxFloat32(x))

/// Broadcast a scalar value to a 3-dimensional floating point vector.
///
/// @ingroup GLSLCore
#define FFX_BROADCAST_FLOAT32X3(x) FfxFloat32x3(FfxFloat32(x))

/// Broadcast a scalar value to a 4-dimensional floating point vector.
///
/// @ingroup GLSLCore
#define FFX_BROADCAST_FLOAT32X4(x) FfxFloat32x4(FfxFloat32(x))

/// Broadcast a scalar value to a 1-dimensional unsigned integer vector.
///
/// @ingroup GLSLCore
#define FFX_BROADCAST_UINT32(x)   FfxUInt32(x)

/// Broadcast a scalar value to a 2-dimensional unsigned integer vector.
///
/// @ingroup GLSLCore
#define FFX_BROADCAST_UINT32X2(x) FfxUInt32x2(FfxUInt32(x))

/// Broadcast a scalar value to a 3-dimensional unsigned integer vector.
///
/// @ingroup GLSLCore
#define FFX_BROADCAST_UINT32X3(x) FfxUInt32x3(FfxUInt32(x))

/// Broadcast a scalar value to a 4-dimensional unsigned integer vector.
///
/// @ingroup GLSLCore
#define FFX_BROADCAST_UINT32X4(x) FfxUInt32x4(FfxUInt32(x))

/// Broadcast a scalar value to a 1-dimensional signed integer vector.
///
/// @ingroup GLSLCore
#define FFX_BROADCAST_INT32(x)   FfxInt32(x)

/// Broadcast a scalar value to a 2-dimensional signed integer vector.
///
/// @ingroup GLSLCore
#define FFX_BROADCAST_INT32X2(x) FfxInt32x2(FfxInt32(x))

/// Broadcast a scalar value to a 3-dimensional signed integer vector.
///
/// @ingroup GLSLCore
#define FFX_BROADCAST_INT32X3(x) FfxInt32x3(FfxInt32(x))

/// Broadcast a scalar value to a 4-dimensional signed integer vector.
///
/// @ingroup GLSLCore
#define FFX_BROADCAST_INT32X4(x) FfxInt32x4(FfxInt32(x))

/// Broadcast a scalar value to a 1-dimensional half-precision floating point vector.
///
/// @ingroup GLSLCore
#define FFX_BROADCAST_MIN_FLOAT16(x)   FFX_MIN16_F(x)

/// Broadcast a scalar value to a 2-dimensional half-precision floating point vector.
///
/// @ingroup GLSLCore
#define FFX_BROADCAST_MIN_FLOAT16X2(x) FFX_MIN16_F2(FFX_MIN16_F(x))

/// Broadcast a scalar value to a 3-dimensional half-precision floating point vector.
///
/// @ingroup GLSLCore
#define FFX_BROADCAST_MIN_FLOAT16X3(x) FFX_MIN16_F3(FFX_MIN16_F(x))

/// Broadcast a scalar value to a 4-dimensional half-precision floating point vector.
///
/// @ingroup GLSLCore
#define FFX_BROADCAST_MIN_FLOAT16X4(x) FFX_MIN16_F4(FFX_MIN16_F(x))

/// Broadcast a scalar value to a 1-dimensional half-precision unsigned integer vector.
///
/// @ingroup GLSLCore
#define FFX_BROADCAST_MIN_UINT16(x)   FFX_MIN16_U(x)

/// Broadcast a scalar value to a 2-dimensional half-precision unsigned integer vector.
///
/// @ingroup GLSLCore
#define FFX_BROADCAST_MIN_UINT16X2(x) FFX_MIN16_U2(FFX_MIN16_U(x))

/// Broadcast a scalar value to a 3-dimensional half-precision unsigned integer vector.
///
/// @ingroup GLSLCore
#define FFX_BROADCAST_MIN_UINT16X3(x) FFX_MIN16_U3(FFX_MIN16_U(x))

/// Broadcast a scalar value to a 4-dimensional half-precision unsigned integer vector.
///
/// @ingroup GLSLCore
#define FFX_BROADCAST_MIN_UINT16X4(x) FFX_MIN16_U4(FFX_MIN16_U(x))

/// Broadcast a scalar value to a 1-dimensional half-precision signed integer vector.
///
/// @ingroup GLSLCore
#define FFX_BROADCAST_MIN_INT16(x)   FFX_MIN16_I(x)

/// Broadcast a scalar value to a 2-dimensional half-precision signed integer vector.
///
/// @ingroup GLSLCore
#define FFX_BROADCAST_MIN_INT16X2(x) FFX_MIN16_I2(FFX_MIN16_I(x))

/// Broadcast a scalar value to a 3-dimensional half-precision signed integer vector.
///
/// @ingroup GLSLCore
#define FFX_BROADCAST_MIN_INT16X3(x) FFX_MIN16_I3(FFX_MIN16_I(x))

/// Broadcast a scalar value to a 4-dimensional half-precision signed integer vector.
///
/// @ingroup GLSLCore
#define FFX_BROADCAST_MIN_INT16X4(x) FFX_MIN16_I4(FFX_MIN16_I(x))

    #extension GL_EXT_shader_explicit_arithmetic_types : require
#if !defined(FFX_SKIP_EXT)
#if FFX_HALF
    #extension GL_EXT_shader_16bit_storage : require
#endif // FFX_HALF

#if defined(FFX_LONG)
    #extension GL_ARB_gpu_shader_int64 : require
    #extension GL_NV_shader_atomic_int64 : require
#endif // #if defined(FFX_LONG)

#if defined(FFX_WAVE)
    #extension GL_KHR_shader_subgroup_arithmetic : require
    #extension GL_KHR_shader_subgroup_ballot : require
    #extension GL_KHR_shader_subgroup_quad : require
    #extension GL_KHR_shader_subgroup_shuffle : require
#endif // #if defined(FFX_WAVE)
#endif // #if !defined(FFX_SKIP_EXT)

// Forward declarations
FfxFloat32   ffxSqrt(FfxFloat32 x);
FfxFloat32x2 ffxSqrt(FfxFloat32x2 x);
FfxFloat32x3 ffxSqrt(FfxFloat32x3 x);
FfxFloat32x4 ffxSqrt(FfxFloat32x4 x);

/// Interprets the bit pattern of x as a floating-point number.
///
/// @param [in] x               The input value.
///
/// @returns
/// The input interpreted as a floating-point number.
///
/// @ingroup GLSLCore
FfxFloat32 ffxAsFloat(FfxUInt32 x)
{
    return uintBitsToFloat(x);
}

/// Interprets the bit pattern of x as a floating-point number.
///
/// @param [in] x               The input value.
///
/// @returns
/// The input interpreted as a floating-point number.
///
/// @ingroup GLSLCore
FfxFloat32x2 ffxAsFloat(FfxUInt32x2 x)
{
    return uintBitsToFloat(x);
}

/// Interprets the bit pattern of x as a floating-point number.
///
/// @param [in] x               The input value.
///
/// @returns
/// The input interpreted as a floating-point number.
///
/// @ingroup GLSLCore
FfxFloat32x3 ffxAsFloat(FfxUInt32x3 x)
{
    return uintBitsToFloat(x);
}

/// Interprets the bit pattern of x as a floating-point number.
///
/// @param [in] x               The input value.
///
/// @returns
/// The input interpreted as a floating-point number.
///
/// @ingroup GLSLCore
FfxFloat32x4 ffxAsFloat(FfxUInt32x4 x)
{
    return uintBitsToFloat(x);
}

/// Interprets the bit pattern of x as an unsigned integer.
///
/// @param [in] x               The input value.
///
/// @returns
/// The input interpreted as an unsigned integer.
///
/// @ingroup GLSLCore
FfxUInt32 ffxAsUInt32(FfxFloat32 x)
{
    return floatBitsToUint(x);
}

/// Interprets the bit pattern of x as an unsigned integer.
///
/// @param [in] x               The input value.
///
/// @returns
/// The input interpreted as an unsigned integer.
///
/// @ingroup GLSLCore
FfxUInt32x2 ffxAsUInt32(FfxFloat32x2 x)
{
    return floatBitsToUint(x);
}

/// Interprets the bit pattern of x as an unsigned integer.
///
/// @param [in] x               The input value.
///
/// @returns
/// The input interpreted as an unsigned integer.
///
/// @ingroup GLSLCore
FfxUInt32x3 ffxAsUInt32(FfxFloat32x3 x)
{
    return floatBitsToUint(x);
}

/// Interprets the bit pattern of x as an unsigned integer.
///
/// @param [in] x               The input value.
///
/// @returns
/// The input interpreted as an unsigned integer.
///
/// @ingroup GLSLCore
FfxUInt32x4 ffxAsUInt32(FfxFloat32x4 x)
{
    return floatBitsToUint(x);
}

/// Pack 2x32-bit floating point values in a single 32bit value.
///
/// This function first converts each component of <c><i>value</i></c> into their nearest 16-bit floating
/// point representation, and then stores the X and Y components in the lower and upper 16 bits of the
/// 32bit unsigned integer respectively.
///
/// @param [in] value               A 2-dimensional floating point value to convert and pack.
///
/// @returns
/// A packed 32bit value containing 2 16bit floating point values.
///
/// @ingroup GLSLCore
FfxUInt32 ffxPackHalf2x16(FfxFloat32x2 value)
{
    return packHalf2x16(value);
}

/// Convert a 32bit IEEE 754 floating point value to its nearest 16bit equivalent.
///
/// @param [in] value               The value to convert.
///
/// @returns
/// The nearest 16bit equivalent of <c><i>value</i></c>.
///
/// @ingroup GLSLCore
FfxUInt32 ffxF32ToF16(FfxFloat32 value)
{
    return packHalf2x16(FfxFloat32x2(value, 0.0));
}

/// Broadcast a scalar value to a 2-dimensional floating point vector.
///
/// @param [in] value               The value to to broadcast.
///
/// @returns
/// A 2-dimensional floating point vector with <c><i>value</i></c> in each component.
///
/// @ingroup GLSLCore
FfxFloat32x2 ffxBroadcast2(FfxFloat32 value)
{
    return FfxFloat32x2(value, value);
}

/// Broadcast a scalar value to a 3-dimensional floating point vector.
///
/// @param [in] value               The value to to broadcast.
///
/// @returns
/// A 3-dimensional floating point vector with <c><i>value</i></c> in each component.
///
/// @ingroup GLSLCore
FfxFloat32x3 ffxBroadcast3(FfxFloat32 value)
{
    return FfxFloat32x3(value, value, value);
}

/// Broadcast a scalar value to a 4-dimensional floating point vector.
///
/// @param [in] value               The value to to broadcast.
///
/// @returns
/// A 4-dimensional floating point vector with <c><i>value</i></c> in each component.
///
/// @ingroup GLSLCore
FfxFloat32x4 ffxBroadcast4(FfxFloat32 value)
{
    return FfxFloat32x4(value, value, value, value);
}

/// Broadcast a scalar value to a 2-dimensional signed integer vector.
///
/// @param [in] value               The value to to broadcast.
///
/// @returns
/// A 2-dimensional signed integer vector with <c><i>value</i></c> in each component.
///
/// @ingroup GLSLCore
FfxInt32x2 ffxBroadcast2(FfxInt32 value)
{
    return FfxInt32x2(value, value);
}

/// Broadcast a scalar value to a 3-dimensional signed integer vector.
///
/// @param [in] value               The value to to broadcast.
///
/// @returns
/// A 3-dimensional signed integer vector with <c><i>value</i></c> in each component.
///
/// @ingroup GLSLCore
FfxInt32x3 ffxBroadcast3(FfxInt32 value)
{
    return FfxInt32x3(value, value, value);
}

/// Broadcast a scalar value to a 4-dimensional signed integer vector.
///
/// @param [in] value               The value to to broadcast.
///
/// @returns
/// A 4-dimensional signed integer vector with <c><i>value</i></c> in each component.
///
/// @ingroup GLSLCore
FfxInt32x4 ffxBroadcast4(FfxInt32 value)
{
    return FfxInt32x4(value, value, value, value);
}

/// Broadcast a scalar value to a 2-dimensional unsigned integer vector.
///
/// @param [in] value               The value to to broadcast.
///
/// @returns
/// A 2-dimensional unsigned integer vector with <c><i>value</i></c> in each component.
///
/// @ingroup GLSLCore
FfxUInt32x2 ffxBroadcast2(FfxUInt32 value)
{
    return FfxUInt32x2(value, value);
}

/// Broadcast a scalar value to a 3-dimensional unsigned integer vector.
///
/// @param [in] value               The value to to broadcast.
///
/// @returns
/// A 3-dimensional unsigned integer vector with <c><i>value</i></c> in each component.
///
/// @ingroup GLSLCore
FfxUInt32x3 ffxBroadcast3(FfxUInt32 value)
{
    return FfxUInt32x3(value, value, value);
}

/// Broadcast a scalar value to a 4-dimensional unsigned integer vector.
///
/// @param [in] value               The value to to broadcast.
///
/// @returns
/// A 4-dimensional unsigned integer vector with <c><i>value</i></c> in each component.
///
/// @ingroup GLSLCore
FfxUInt32x4 ffxBroadcast4(FfxUInt32 value)
{
    return FfxUInt32x4(value, value, value, value);
}

///
///
/// @ingroup GLSLCore
FfxUInt32 ffxBitfieldExtract(FfxUInt32 src, FfxUInt32 off, FfxUInt32 bits)
{
    return bitfieldExtract(src, FfxInt32(off), FfxInt32(bits));
}

///
///
/// @ingroup GLSLCore
FfxUInt32 ffxBitfieldInsert(FfxUInt32 src, FfxUInt32 ins, FfxUInt32 mask)
{
    return (ins & mask) | (src & (~mask));
}

// Proxy for V_BFI_B32 where the 'mask' is set as 'bits', 'mask=(1<<bits)-1', and 'bits' needs to be an immediate.
///
///
/// @ingroup GLSLCore
FfxUInt32 ffxBitfieldInsertMask(FfxUInt32 src, FfxUInt32 ins, FfxUInt32 bits)
{
    return bitfieldInsert(src, ins, 0, FfxInt32(bits));
}

/// Compute the linear interopation between two values.
///
/// Implemented by calling the GLSL <c><i>mix</i></c> instrinsic function. Implements the
/// following math:
///
///     (1 - t) * x + t * y
///
/// @param [in] x               The first value to lerp between.
/// @param [in] y               The second value to lerp between.
/// @param [in] t               The value to determine how much of <c><i>x</i></c> and how much of <c><i>y</i></c>.
///
/// @returns
/// A linearly interpolated value between <c><i>x</i></c> and <c><i>y</i></c> according to <c><i>t</i></c>.
///
/// @ingroup GLSLCore
FfxFloat32 ffxLerp(FfxFloat32 x, FfxFloat32 y, FfxFloat32 t)
{
    return mix(x, y, t);
}

/// Compute the linear interopation between two values.
///
/// Implemented by calling the GLSL <c><i>mix</i></c> instrinsic function. Implements the
/// following math:
///
///     (1 - t) * x + t * y
///
/// @param [in] x               The first value to lerp between.
/// @param [in] y               The second value to lerp between.
/// @param [in] t               The value to determine how much of <c><i>x</i></c> and how much of <c><i>y</i></c>.
///
/// @returns
/// A linearly interpolated value between <c><i>x</i></c> and <c><i>y</i></c> according to <c><i>t</i></c>.
///
/// @ingroup GLSLCore
FfxFloat32x2 ffxLerp(FfxFloat32x2 x, FfxFloat32x2 y, FfxFloat32 t)
{
    return mix(x, y, t);
}

/// Compute the linear interopation between two values.
///
/// Implemented by calling the GLSL <c><i>mix</i></c> instrinsic function. Implements the
/// following math:
///
///     (1 - t) * x + t * y
///
/// @param [in] x               The first value to lerp between.
/// @param [in] y               The second value to lerp between.
/// @param [in] t               The value to determine how much of <c><i>x</i></c> and how much of <c><i>y</i></c>.
///
/// @returns
/// A linearly interpolated value between <c><i>x</i></c> and <c><i>y</i></c> according to <c><i>t</i></c>.
///
/// @ingroup GLSLCore
FfxFloat32x2 ffxLerp(FfxFloat32x2 x, FfxFloat32x2 y, FfxFloat32x2 t)
{
    return mix(x, y, t);
}

/// Compute the linear interopation between two values.
///
/// Implemented by calling the GLSL <c><i>mix</i></c> instrinsic function. Implements the
/// following math:
///
///     (1 - t) * x + t * y
///
/// @param [in] x               The first value to lerp between.
/// @param [in] y               The second value to lerp between.
/// @param [in] t               The value to determine how much of <c><i>x</i></c> and how much of <c><i>y</i></c>.
///
/// @returns
/// A linearly interpolated value between <c><i>x</i></c> and <c><i>y</i></c> according to <c><i>t</i></c>.
///
/// @ingroup GLSLCore
FfxFloat32x3 ffxLerp(FfxFloat32x3 x, FfxFloat32x3 y, FfxFloat32 t)
{
    return mix(x, y, t);
}

/// Compute the linear interopation between two values.
///
/// Implemented by calling the GLSL <c><i>mix</i></c> instrinsic function. Implements the
/// following math:
///
///     (1 - t) * x + t * y
///
/// @param [in] x               The first value to lerp between.
/// @param [in] y               The second value to lerp between.
/// @param [in] t               The value to determine how much of <c><i>x</i></c> and how much of <c><i>y</i></c>.
///
/// @returns
/// A linearly interpolated value between <c><i>x</i></c> and <c><i>y</i></c> according to <c><i>t</i></c>.
///
/// @ingroup GLSLCore
FfxFloat32x3 ffxLerp(FfxFloat32x3 x, FfxFloat32x3 y, FfxFloat32x3 t)
{
    return mix(x, y, t);
}

/// Compute the linear interopation between two values.
///
/// Implemented by calling the GLSL <c><i>mix</i></c> instrinsic function. Implements the
/// following math:
///
///     (1 - t) * x + t * y
///
/// @param [in] x               The first value to lerp between.
/// @param [in] y               The second value to lerp between.
/// @param [in] t               The value to determine how much of <c><i>x</i></c> and how much of <c><i>y</i></c>.
///
/// @returns
/// A linearly interpolated value between <c><i>x</i></c> and <c><i>y</i></c> according to <c><i>t</i></c>.
///
/// @ingroup GLSLCore
FfxFloat32x4 ffxLerp(FfxFloat32x4 x, FfxFloat32x4 y, FfxFloat32 t)
{
    return mix(x, y, t);
}

/// Compute the linear interopation between two values.
///
/// Implemented by calling the GLSL <c><i>mix</i></c> instrinsic function. Implements the
/// following math:
///
///     (1 - t) * x + t * y
///
/// @param [in] x               The first value to lerp between.
/// @param [in] y               The second value to lerp between.
/// @param [in] t               The value to determine how much of <c><i>x</i></c> and how much of <c><i>y</i></c>.
///
/// @returns
/// A linearly interpolated value between <c><i>x</i></c> and <c><i>y</i></c> according to <c><i>t</i></c>.
///
/// @ingroup GLSLCore
FfxFloat32x4 ffxLerp(FfxFloat32x4 x, FfxFloat32x4 y, FfxFloat32x4 t)
{
    return mix(x, y, t);
}

/// Compute the maximum of three values.
///
/// NOTE: This function should compile down to a single V_MAX3_F32 operation on
/// GCN or RDNA hardware.
///
/// @param [in] x               The first value to include in the max calculation.
/// @param [in] y               The second value to include in the max calcuation.
/// @param [in] z               The third value to include in the max calcuation.
///
/// @returns
/// The maximum value of <c><i>x</i></c>, <c><i>y</i></c>, and <c><i>z</i></c>.
///
/// @ingroup GLSLCore
FfxFloat32 ffxMax3(FfxFloat32 x, FfxFloat32 y, FfxFloat32 z)
{
    return max(x, max(y, z));
}

/// Compute the maximum of three values.
///
/// NOTE: This function should compile down to a single <c><i>V_MAX3_F32</i></c> operation on
/// GCN or RDNA hardware.
///
/// @param [in] x               The first value to include in the max calculation.
/// @param [in] y               The second value to include in the max calcuation.
/// @param [in] z               The third value to include in the max calcuation.
///
/// @returns
/// The maximum value of <c><i>x</i></c>, <c><i>y</i></c>, and <c><i>z</i></c>.
///
/// @ingroup GLSLCore
FfxFloat32x2 ffxMax3(FfxFloat32x2 x, FfxFloat32x2 y, FfxFloat32x2 z)
{
    return max(x, max(y, z));
}

/// Compute the maximum of three values.
///
/// NOTE: This function should compile down to a single <c><i>V_MAX3_F32</i></c> operation on
/// GCN or RDNA hardware.
///
/// @param [in] x               The first value to include in the max calculation.
/// @param [in] y               The second value to include in the max calcuation.
/// @param [in] z               The third value to include in the max calcuation.
///
/// @returns
/// The maximum value of <c><i>x</i></c>, <c><i>y</i></c>, and <c><i>z</i></c>.
///
/// @ingroup GLSLCore
FfxFloat32x3 ffxMax3(FfxFloat32x3 x, FfxFloat32x3 y, FfxFloat32x3 z)
{
    return max(x, max(y, z));
}

/// Compute the maximum of three values.
///
/// NOTE: This function should compile down to a single <c><i>V_MAX3_F32</i></c> operation on
/// GCN or RDNA hardware.
///
/// @param [in] x               The first value to include in the max calculation.
/// @param [in] y               The second value to include in the max calcuation.
/// @param [in] z               The third value to include in the max calcuation.
///
/// @returns
/// The maximum value of <c><i>x</i></c>, <c><i>y</i></c>, and <c><i>z</i></c>.
///
/// @ingroup GLSLCore
FfxFloat32x4 ffxMax3(FfxFloat32x4 x, FfxFloat32x4 y, FfxFloat32x4 z)
{
    return max(x, max(y, z));
}

/// Compute the maximum of three values.
///
/// NOTE: This function should compile down to a single <c><i>V_MAX3_F32</i></c> operation on
/// GCN or RDNA hardware.
///
/// @param [in] x               The first value to include in the max calculation.
/// @param [in] y               The second value to include in the max calcuation.
/// @param [in] z               The third value to include in the max calcuation.
///
/// @returns
/// The maximum value of <c><i>x</i></c>, <c><i>y</i></c>, and <c><i>z</i></c>.
///
/// @ingroup GLSLCore
FfxUInt32 ffxMax3(FfxUInt32 x, FfxUInt32 y, FfxUInt32 z)
{
    return max(x, max(y, z));
}

/// Compute the maximum of three values.
///
/// NOTE: This function should compile down to a single <c><i>V_MAX3_F32</i></c> operation on
/// GCN or RDNA hardware.
///
/// @param [in] x               The first value to include in the max calculation.
/// @param [in] y               The second value to include in the max calcuation.
/// @param [in] z               The third value to include in the max calcuation.
///
/// @returns
/// The maximum value of <c><i>x</i></c>, <c><i>y</i></c>, and <c><i>z</i></c>.
///
/// @ingroup GLSLCore
FfxUInt32x2 ffxMax3(FfxUInt32x2 x, FfxUInt32x2 y, FfxUInt32x2 z)
{
    return max(x, max(y, z));
}

/// Compute the maximum of three values.
///
/// NOTE: This function should compile down to a single <c><i>V_MAX3_F32</i></c> operation on
/// GCN/RDNA hardware.
///
/// @param [in] x               The first value to include in the max calculation.
/// @param [in] y               The second value to include in the max calcuation.
/// @param [in] z               The third value to include in the max calcuation.
///
/// @returns
/// The maximum value of <c><i>x</i></c>, <c><i>y</i></c>, and <c><i>z</i></c>.
///
/// @ingroup GLSLCore
FfxUInt32x3 ffxMax3(FfxUInt32x3 x, FfxUInt32x3 y, FfxUInt32x3 z)
{
    return max(x, max(y, z));
}

/// Compute the maximum of three values.
///
/// NOTE: This function should compile down to a single <c><i>V_MAX3_F32</i></c> operation on
/// GCN/RDNA hardware.
///
/// @param [in] x               The first value to include in the max calculation.
/// @param [in] y               The second value to include in the max calcuation.
/// @param [in] z               The third value to include in the max calcuation.
///
/// @returns
/// The maximum value of <c><i>x</i></c>, <c><i>y</i></c>, and <c><i>z</i></c>.
///
/// @ingroup GLSLCore
FfxUInt32x4 ffxMax3(FfxUInt32x4 x, FfxUInt32x4 y, FfxUInt32x4 z)
{
    return max(x, max(y, z));
}

/// Compute the median of three values.
///
/// NOTE: This function should compile down to a single <c><i>V_MED3_F32</i></c> operation on
/// GCN/RDNA hardware.
///
/// @param [in] x               The first value to include in the median calculation.
/// @param [in] y               The second value to include in the median calcuation.
/// @param [in] z               The third value to include in the median calcuation.
///
/// @returns
/// The median value of <c><i>x</i></c>, <c><i>y</i></c>, and <c><i>z</i></c>.
///
/// @ingroup GLSLCore
FfxFloat32 ffxMed3(FfxFloat32 x, FfxFloat32 y, FfxFloat32 z)
{
    return max(min(x, y), min(max(x, y), z));
}

/// Compute the median of three values.
///
/// NOTE: This function should compile down to a single <c><i>V_MED3_F32</i></c> operation on
/// GCN/RDNA hardware.
///
/// @param [in] x               The first value to include in the median calculation.
/// @param [in] y               The second value to include in the median calcuation.
/// @param [in] z               The third value to include in the median calcuation.
///
/// @returns
/// The median value of <c><i>x</i></c>, <c><i>y</i></c>, and <c><i>z</i></c>.
///
/// @ingroup GLSLCore
FfxFloat32x2 ffxMed3(FfxFloat32x2 x, FfxFloat32x2 y, FfxFloat32x2 z)
{
    return max(min(x, y), min(max(x, y), z));
}

/// Compute the median of three values.
///
/// NOTE: This function should compile down to a single <c><i>V_MED3_F32</i></c> operation on
/// GCN/RDNA hardware.
///
/// @param [in] x               The first value to include in the median calculation.
/// @param [in] y               The second value to include in the median calcuation.
/// @param [in] z               The third value to include in the median calcuation.
///
/// @returns
/// The median value of <c><i>x</i></c>, <c><i>y</i></c>, and <c><i>z</i></c>.
///
/// @ingroup GLSLCore
FfxFloat32x3 ffxMed3(FfxFloat32x3 x, FfxFloat32x3 y, FfxFloat32x3 z)
{
    return max(min(x, y), min(max(x, y), z));
}

/// Compute the median of three values.
///
/// NOTE: This function should compile down to a single <c><i>V_MED3_F32</i></c> operation on
/// GCN/RDNA hardware.
///
/// @param [in] x               The first value to include in the median calculation.
/// @param [in] y               The second value to include in the median calcuation.
/// @param [in] z               The third value to include in the median calcuation.
///
/// @returns
/// The median value of <c><i>x</i></c>, <c><i>y</i></c>, and <c><i>z</i></c>.
///
/// @ingroup GLSLCore
FfxFloat32x4 ffxMed3(FfxFloat32x4 x, FfxFloat32x4 y, FfxFloat32x4 z)
{
    return max(min(x, y), min(max(x, y), z));
}

/// Compute the median of three values.
///
/// NOTE: This function should compile down to a single <c><i>V_MED3_I32</i></c> operation on
/// GCN/RDNA hardware.
///
/// @param [in] x               The first value to include in the median calculation.
/// @param [in] y               The second value to include in the median calcuation.
/// @param [in] z               The third value to include in the median calcuation.
///
/// @returns
/// The median value of <c><i>x</i></c>, <c><i>y</i></c>, and <c><i>z</i></c>.
///
/// @ingroup GLSLCore
FfxInt32 ffxMed3(FfxInt32 x, FfxInt32 y, FfxInt32 z)
{
    return max(min(x, y), min(max(x, y), z));
}

/// Compute the median of three values.
///
/// NOTE: This function should compile down to a single <c><i>V_MED3_I32</i></c> operation on
/// GCN/RDNA hardware.
///
/// @param [in] x               The first value to include in the median calculation.
/// @param [in] y               The second value to include in the median calcuation.
/// @param [in] z               The third value to include in the median calcuation.
///
/// @returns
/// The median value of <c><i>x</i></c>, <c><i>y</i></c>, and <c><i>z</i></c>.
///
/// @ingroup GLSLCore
FfxInt32x2 ffxMed3(FfxInt32x2 x, FfxInt32x2 y, FfxInt32x2 z)
{
    return max(min(x, y), min(max(x, y), z));
}

/// Compute the median of three values.
///
/// NOTE: This function should compile down to a single <c><i>V_MED3_I32</i></c> operation on
/// GCN/RDNA hardware.
///
/// @param [in] x               The first value to include in the median calculation.
/// @param [in] y               The second value to include in the median calcuation.
/// @param [in] z               The third value to include in the median calcuation.
///
/// @returns
/// The median value of <c><i>x</i></c>, <c><i>y</i></c>, and <c><i>z</i></c>.
///
/// @ingroup GLSLCore
FfxInt32x3 ffxMed3(FfxInt32x3 x, FfxInt32x3 y, FfxInt32x3 z)
{
    return max(min(x, y), min(max(x, y), z));
}

/// Compute the median of three values.
///
/// NOTE: This function should compile down to a single <c><i>V_MED3_I32</i></c> operation on
/// GCN/RDNA hardware.
///
/// @param [in] x               The first value to include in the median calculation.
/// @param [in] y               The second value to include in the median calcuation.
/// @param [in] z               The third value to include in the median calcuation.
///
/// @returns
/// The median value of <c><i>x</i></c>, <c><i>y</i></c>, and <c><i>z</i></c>.
///
/// @ingroup GLSLCore
FfxInt32x4 ffxMed3(FfxInt32x4 x, FfxInt32x4 y, FfxInt32x4 z)
{
    return max(min(x, y), min(max(x, y), z));
}


/// Compute the minimum of three values.
///
/// NOTE: This function should compile down to a single <c><i>V_MIN3_F32</i></c> operation on
/// GCN and RDNA hardware.
///
/// @param [in] x               The first value to include in the min calculation.
/// @param [in] y               The second value to include in the min calcuation.
/// @param [in] z               The third value to include in the min calcuation.
///
/// @returns
/// The minimum value of <c><i>x</i></c>, <c><i>y</i></c>, and <c><i>z</i></c>.
///
/// @ingroup GLSLCore
FfxFloat32 ffxMin3(FfxFloat32 x, FfxFloat32 y, FfxFloat32 z)
{
    return min(x, min(y, z));
}

/// Compute the minimum of three values.
///
/// NOTE: This function should compile down to a single V_MIN3_F32 operation on
/// GCN/RDNA hardware.
///
/// @param [in] x               The first value to include in the min calculation.
/// @param [in] y               The second value to include in the min calcuation.
/// @param [in] z               The third value to include in the min calcuation.
///
/// @returns
/// The minimum value of <c><i>x</i></c>, <c><i>y</i></c>, and <c><i>z</i></c>.
///
/// @ingroup GLSLCore
FfxFloat32x2 ffxMin3(FfxFloat32x2 x, FfxFloat32x2 y, FfxFloat32x2 z)
{
    return min(x, min(y, z));
}

/// Compute the minimum of three values.
///
/// NOTE: This function should compile down to a single V_MIN3_F32 operation on
/// GCN/RDNA hardware.
///
/// @param [in] x               The first value to include in the min calculation.
/// @param [in] y               The second value to include in the min calcuation.
/// @param [in] z               The third value to include in the min calcuation.
///
/// @returns
/// The minimum value of <c><i>x</i></c>, <c><i>y</i></c>, and <c><i>z</i></c>.
///
/// @ingroup GLSLCore
FfxFloat32x3 ffxMin3(FfxFloat32x3 x, FfxFloat32x3 y, FfxFloat32x3 z)
{
    return min(x, min(y, z));
}

/// Compute the minimum of three values.
///
/// NOTE: This function should compile down to a single V_MIN3_F32 operation on
/// GCN/RDNA hardware.
///
/// @param [in] x               The first value to include in the min calculation.
/// @param [in] y               The second value to include in the min calcuation.
/// @param [in] z               The third value to include in the min calcuation.
///
/// @returns
/// The minimum value of <c><i>x</i></c>, <c><i>y</i></c>, and <c><i>z</i></c>.
///
/// @ingroup GLSLCore
FfxFloat32x4 ffxMin3(FfxFloat32x4 x, FfxFloat32x4 y, FfxFloat32x4 z)
{
    return min(x, min(y, z));
}

/// Compute the minimum of three values.
///
/// NOTE: This function should compile down to a single V_MIN3_F32 operation on
/// GCN/RDNA hardware.
///
/// @param [in] x               The first value to include in the min calculation.
/// @param [in] y               The second value to include in the min calcuation.
/// @param [in] z               The third value to include in the min calcuation.
///
/// @returns
/// The minimum value of <c><i>x</i></c>, <c><i>y</i></c>, and <c><i>z</i></c>.
///
/// @ingroup GLSLCore
FfxUInt32 ffxMin3(FfxUInt32 x, FfxUInt32 y, FfxUInt32 z)
{
    return min(x, min(y, z));
}

/// Compute the minimum of three values.
///
/// NOTE: This function should compile down to a single V_MIN3_F32 operation on
/// GCN/RDNA hardware.
///
/// @param [in] x               The first value to include in the min calculation.
/// @param [in] y               The second value to include in the min calcuation.
/// @param [in] z               The third value to include in the min calcuation.
///
/// @returns
/// The minimum value of <c><i>x</i></c>, <c><i>y</i></c>, and <c><i>z</i></c>.
///
/// @ingroup GLSLCore
FfxUInt32x2 ffxMin3(FfxUInt32x2 x, FfxUInt32x2 y, FfxUInt32x2 z)
{
    return min(x, min(y, z));
}

/// Compute the minimum of three values.
///
/// NOTE: This function should compile down to a single V_MIN3_F32 operation on
/// GCN/RDNA hardware.
///
/// @param [in] x               The first value to include in the min calculation.
/// @param [in] y               The second value to include in the min calcuation.
/// @param [in] z               The third value to include in the min calcuation.
///
/// @returns
/// The minimum value of <c><i>x</i></c>, <c><i>y</i></c>, and <c><i>z</i></c>.
///
/// @ingroup GLSLCore
FfxUInt32x3 ffxMin3(FfxUInt32x3 x, FfxUInt32x3 y, FfxUInt32x3 z)
{
    return min(x, min(y, z));
}

/// Compute the minimum of three values.
///
/// NOTE: This function should compile down to a single V_MIN3_F32 operation on
/// GCN/RDNA hardware.
///
/// @param [in] x               The first value to include in the min calculation.
/// @param [in] y               The second value to include in the min calcuation.
/// @param [in] z               The third value to include in the min calcuation.
///
/// @returns
/// The minimum value of <c><i>x</i></c>, <c><i>y</i></c>, and <c><i>z</i></c>.
///
/// @ingroup GLSLCore
FfxUInt32x4 ffxMin3(FfxUInt32x4 x, FfxUInt32x4 y, FfxUInt32x4 z)
{
    return min(x, min(y, z));
}

/// Compute the reciprocal of a value.
///
/// NOTE: This function is only provided for GLSL. In HLSL the intrinsic function <c><i>rcp</i></c> can be used.
///
/// @param [in] x               The value to compute the reciprocal for.
///
/// @returns
/// The reciprocal value of <c><i>x</i></c>.
///
/// @ingroup GLSLCore
FfxFloat32 ffxReciprocal(FfxFloat32 x)
{
    return FfxFloat32(1.0) / x;
}

/// Compute the reciprocal of a value.
///
/// NOTE: This function is only provided for GLSL. In HLSL the intrinsic function <c><i>rcp</i></c> can be used.
///
/// @param [in] x               The value to compute the reciprocal for.
///
/// @returns
/// The reciprocal value of <c><i>x</i></c>.
///
/// @ingroup GLSLCore
FfxFloat32x2 ffxReciprocal(FfxFloat32x2 x)
{
    return ffxBroadcast2(1.0) / x;
}

/// Compute the reciprocal of a value.
///
/// NOTE: This function is only provided for GLSL. In HLSL the intrinsic function <c><i>rcp</i></c> can be used.
///
/// @param [in] x               The value to compute the reciprocal for.
///
/// @returns
/// The reciprocal value of <c><i>x</i></c>.
///
/// @ingroup GLSLCore
FfxFloat32x3 ffxReciprocal(FfxFloat32x3 x)
{
    return ffxBroadcast3(1.0) / x;
}

/// Compute the reciprocal of a value.
///
/// NOTE: This function is only provided for GLSL. In HLSL the intrinsic function <c><i>rcp</i></c> can be used.
///
/// @param [in] x               The value to compute the reciprocal for.
///
/// @returns
/// The reciprocal value of <c><i>x</i></c>.
///
/// @ingroup GLSLCore
FfxFloat32x4 ffxReciprocal(FfxFloat32x4 x)
{
    return ffxBroadcast4(1.0) / x;
}

/// Compute the reciprocal square root of a value.
///
/// NOTE: This function is only provided for GLSL. In HLSL the intrinsic function <c><i>rsqrt</i></c> can be used.
///
/// @param [in] x               The value to compute the reciprocal for.
///
/// @returns
/// The reciprocal square root value of <c><i>x</i></c>.
///
/// @ingroup GLSLCore
FfxFloat32 ffxRsqrt(FfxFloat32 x)
{
    return FfxFloat32(1.0) / ffxSqrt(x);
}

/// Compute the reciprocal square root of a value.
///
/// NOTE: This function is only provided for GLSL. In HLSL the intrinsic function <c><i>rsqrt</i></c> can be used.
///
/// @param [in] x               The value to compute the reciprocal for.
///
/// @returns
/// The reciprocal square root value of <c><i>x</i></c>.
///
/// @ingroup GLSLCore
FfxFloat32x2 ffxRsqrt(FfxFloat32x2 x)
{
    return ffxBroadcast2(1.0) / ffxSqrt(x);
}

/// Compute the reciprocal square root of a value.
///
/// NOTE: This function is only provided for GLSL. In HLSL the intrinsic function <c><i>rsqrt</i></c> can be used.
///
/// @param [in] x               The value to compute the reciprocal for.
///
/// @returns
/// The reciprocal square root value of <c><i>x</i></c>.
///
/// @ingroup GLSLCore
FfxFloat32x3 ffxRsqrt(FfxFloat32x3 x)
{
    return ffxBroadcast3(1.0) / ffxSqrt(x);
}

/// Compute the reciprocal square root of a value.
///
/// NOTE: This function is only provided for GLSL. In HLSL the intrinsic function <c><i>rsqrt</i></c> can be used.
///
/// @param [in] x               The value to compute the reciprocal for.
///
/// @returns
/// The reciprocal square root value of <c><i>x</i></c>.
///
/// @ingroup GLSLCore
FfxFloat32x4 rsqrt(FfxFloat32x4 x)
{
    return ffxBroadcast4(1.0) / ffxSqrt(x);
}

/// Clamp a value to a [0..1] range.
///
/// @param [in] x               The value to clamp to [0..1] range.
///
/// @returns
/// The clamped version of <c><i>x</i></c>.
///
/// @ingroup GLSLCore
FfxFloat32 ffxSaturate(FfxFloat32 x)
{
    return clamp(x, FfxFloat32(0.0), FfxFloat32(1.0));
}

/// Clamp a value to a [0..1] range.
///
/// @param [in] x               The value to clamp to [0..1] range.
///
/// @returns
/// The clamped version of <c><i>x</i></c>.
///
/// @ingroup GLSLCore
FfxFloat32x2 ffxSaturate(FfxFloat32x2 x)
{
    return clamp(x, ffxBroadcast2(0.0), ffxBroadcast2(1.0));
}

/// Clamp a value to a [0..1] range.
///
/// @param [in] x               The value to clamp to [0..1] range.
///
/// @returns
/// The clamped version of <c><i>x</i></c>.
///
/// @ingroup GLSLCore
FfxFloat32x3 ffxSaturate(FfxFloat32x3 x)
{
    return clamp(x, ffxBroadcast3(0.0), ffxBroadcast3(1.0));
}

/// Clamp a value to a [0..1] range.
///
/// @param [in] x               The value to clamp to [0..1] range.
///
/// @returns
/// The clamped version of <c><i>x</i></c>.
///
/// @ingroup GLSLCore
FfxFloat32x4 ffxSaturate(FfxFloat32x4 x)
{
    return clamp(x, ffxBroadcast4(0.0), ffxBroadcast4(1.0));
}

/// Compute the factional part of a decimal value.
///
/// This function calculates <c><i>x - floor(x)</i></c>. Where <c><i>floor</i></c> is the intrinsic HLSL function.
///
/// NOTE: This function should compile down to a single <c><i>V_MAX3_F32</i></c> operation on GCN/RDNA hardware. It is
/// worth further noting that this function is intentionally distinct from the HLSL <c><i>frac</i></c> intrinsic
/// function.
///
/// @param [in] x               The value to compute the fractional part from.
///
/// @returns
/// The fractional part of <c><i>x</i></c>.
///
/// @ingroup GLSLCore
FfxFloat32 ffxFract(FfxFloat32 x)
{
    return fract(x);
}

/// Compute the factional part of a decimal value.
///
/// This function calculates <c><i>x - floor(x)</i></c>. Where <c><i>floor</i></c> is the intrinsic HLSL function.
///
/// NOTE: This function should compile down to a single <c><i>V_MAX3_F32</i></c> operation on GCN/RDNA hardware. It is
/// worth further noting that this function is intentionally distinct from the HLSL <c><i>frac</i></c> intrinsic
/// function.
///
/// @param [in] x               The value to compute the fractional part from.
///
/// @returns
/// The fractional part of <c><i>x</i></c>.
///
/// @ingroup GLSLCore
FfxFloat32x2 ffxFract(FfxFloat32x2 x)
{
    return fract(x);
}

/// Compute the factional part of a decimal value.
///
/// This function calculates <c><i>x - floor(x)</i></c>. Where <c><i>floor</i></c> is the intrinsic HLSL function.
///
/// NOTE: This function should compile down to a single <c><i>V_MAX3_F32</i></c> operation on GCN/RDNA hardware. It is
/// worth further noting that this function is intentionally distinct from the HLSL <c><i>frac</i></c> intrinsic
/// function.
///
/// @param [in] x               The value to compute the fractional part from.
///
/// @returns
/// The fractional part of <c><i>x</i></c>.
///
/// @ingroup GLSLCore
FfxFloat32x3 ffxFract(FfxFloat32x3 x)
{
    return fract(x);
}

/// Compute the factional part of a decimal value.
///
/// This function calculates <c><i>x - floor(x)</i></c>. Where <c><i>floor</i></c> is the intrinsic HLSL function.
///
/// NOTE: This function should compile down to a single <c><i>V_MAX3_F32</i></c> operation on GCN/RDNA hardware. It is
/// worth further noting that this function is intentionally distinct from the HLSL <c><i>frac</i></c> intrinsic
/// function.
///
/// @param [in] x               The value to compute the fractional part from.
///
/// @returns
/// The fractional part of <c><i>x</i></c>.
///
/// @ingroup GLSLCore
FfxFloat32x4 ffxFract(FfxFloat32x4 x)
{
    return fract(x);
}

/// Rounds to the nearest integer. In case the fractional part is 0.5, it will round to the nearest even integer.
///
/// @param [in] x               The value to be rounded.
///
/// @returns
/// The nearest integer from <c><i>x</i></c>. The nearest even integer from <c><i>x</i></c> if equidistant from 2 integer.
///
/// @ingroup GLSLCore
FfxFloat32 ffxRound(FfxFloat32 x)
{
    return roundEven(x);
}

/// Rounds to the nearest integer. In case the fractional part is 0.5, it will round to the nearest even integer.
///
/// @param [in] x               The value to be rounded.
///
/// @returns
/// The nearest integer from <c><i>x</i></c>. The nearest even integer from <c><i>x</i></c> if equidistant from 2 integer.
///
/// @ingroup GLSLCore
FfxFloat32x2 ffxRound(FfxFloat32x2 x)
{
    return roundEven(x);
}

/// Rounds to the nearest integer. In case the fractional part is 0.5, it will round to the nearest even integer.
///
/// @param [in] x               The value to be rounded.
///
/// @returns
/// The nearest integer from <c><i>x</i></c>. The nearest even integer from <c><i>x</i></c> if equidistant from 2 integer.
///
/// @ingroup GLSLCore
FfxFloat32x3 ffxRound(FfxFloat32x3 x)
{
    return roundEven(x);
}

/// Rounds to the nearest integer. In case the fractional part is 0.5, it will round to the nearest even integer.
///
/// @param [in] x               The value to be rounded.
///
/// @returns
/// The nearest integer from <c><i>x</i></c>. The nearest even integer from <c><i>x</i></c> if equidistant from 2 integer.
///
/// @ingroup GLSLCore
FfxFloat32x4 ffxRound(FfxFloat32x4 x)
{
    return roundEven(x);
}

FfxUInt32 ffxAShrSU1(FfxUInt32 a, FfxUInt32 b)
{
    return FfxUInt32(FfxInt32(a) >> FfxInt32(b));
}

FfxUInt32 ffxPackF32(FfxFloat32x2 v){
    return packHalf2x16(v);
}

FfxFloat32x2 ffxUnpackF32(FfxUInt32 u){
    return unpackHalf2x16(u);
}

FfxUInt32x2 ffxPackF32x2(FfxFloat32x4 v){
	return FfxUInt32x2(ffxPackF32(v.xy), ffxPackF32(v.zw));
}

FfxFloat32x4 ffxUnpackF32x2(FfxUInt32x2 a){
    return FfxFloat32x4(ffxUnpackF32(a.x), ffxUnpackF32(a.y));
}

/// @brief Inverts the value while avoiding division by zero. If the value is zero, zero is returned.
/// @param v Value to invert.
/// @return If v = 0 returns 0. If v != 0 returns 1/v.
FfxFloat32 ffxInvertSafe(FfxFloat32 v){
    FfxFloat32 s = sign(v);
    FfxFloat32 s2 = s*s;
    return s2/(v + s2 - 1.0);
}

/// @brief Inverts the value while avoiding division by zero. If the value is zero, zero is returned.
/// @param v Value to invert.
/// @return If v = 0 returns 0. If v != 0 returns 1/v.
FfxFloat32x2 ffxInvertSafe(FfxFloat32x2 v){
    FfxFloat32x2 s = sign(v);
    FfxFloat32x2 s2 = s*s;
    return s2/(v + s2 - FfxFloat32x2(1.0, 1.0));
}

/// @brief Inverts the value while avoiding division by zero. If the value is zero, zero is returned.
/// @param v Value to invert.
/// @return If v = 0 returns 0. If v != 0 returns 1/v.
FfxFloat32x3 ffxInvertSafe(FfxFloat32x3 v){
    FfxFloat32x3 s = sign(v);
    FfxFloat32x3 s2 = s*s;
    return s2/(v + s2 - FfxFloat32x3(1.0, 1.0, 1.0));
}

/// @brief Inverts the value while avoiding division by zero. If the value is zero, zero is returned.
/// @param v Value to invert.
/// @return If v = 0 returns 0. If v != 0 returns 1/v.
FfxFloat32x4 ffxInvertSafe(FfxFloat32x4 v){
    FfxFloat32x4 s = sign(v);
    FfxFloat32x4 s2 = s*s;
    return s2/(v + s2 - FfxFloat32x4(1.0, 1.0, 1.0, 1.0));
}
#if FFX_HALF
#define FFX_UINT32_TO_FLOAT16X2(x) unpackFloat2x16(FfxUInt32(x))

FfxUInt32 ffxPackF16(FfxFloat16x2 v){
    return packHalf2x16(v);
}

FfxFloat16x2 ffxUnpackF16(FfxUInt32 u){
    return FfxFloat16x2(unpackHalf2x16(u));
}

FfxFloat16x4 ffxUint32x2ToFloat16x4(FfxUInt32x2 x)
{
    return FfxFloat16x4(unpackFloat2x16(x.x), unpackFloat2x16(x.y));
}
#define FFX_UINT32X2_TO_FLOAT16X4(x) ffxUint32x2ToFloat16x4(FfxUInt32x2(x))
#define FFX_UINT32_TO_UINT16X2(x) unpackUint2x16(FfxUInt32(x))
#define FFX_UINT32X2_TO_UINT16X4(x) unpackUint4x16(pack64(FfxUInt32x2(x)))
//------------------------------------------------------------------------------------------------------------------------------
#define FFX_FLOAT16X2_TO_UINT32(x) packFloat2x16(FfxFloat16x2(x))
FfxUInt32x2 ffxFloat16x4ToUint32x2(FfxFloat16x4 x)
{
    return FfxUInt32x2(packFloat2x16(x.xy), packFloat2x16(x.zw));
}
#define FFX_FLOAT16X4_TO_UINT32X2(x) ffxFloat16x4ToUint32x2(FfxFloat16x4(x))
#define FFX_UINT16X2_TO_UINT32(x) packUint2x16(FfxUInt16x2(x))
#define FFX_UINT16X4_TO_UINT32X2(x) unpack32(packUint4x16(FfxUInt16x4(x)))
//==============================================================================================================================
#define FFX_TO_UINT16(x) halfBitsToUint16(FfxFloat16(x))
#define FFX_TO_UINT16X2(x) halfBitsToUint16(FfxFloat16x2(x))
#define FFX_TO_UINT16X3(x) halfBitsToUint16(FfxFloat16x3(x))
#define FFX_TO_UINT16X4(x) halfBitsToUint16(FfxFloat16x4(x))
//------------------------------------------------------------------------------------------------------------------------------
#define FFX_TO_FLOAT16(x) uint16BitsToHalf(FfxUInt16(x))
#define FFX_TO_FLOAT16X2(x) uint16BitsToHalf(FfxUInt16x2(x))
#define FFX_TO_FLOAT16X3(x) uint16BitsToHalf(FfxUInt16x3(x))
#define FFX_TO_FLOAT16X4(x) uint16BitsToHalf(FfxUInt16x4(x))
//==============================================================================================================================
FfxFloat16 ffxBroadcastFloat16(FfxFloat16 a)
{
    return FfxFloat16(a);
}
FfxFloat16x2 ffxBroadcastFloat16x2(FfxFloat16 a)
{
    return FfxFloat16x2(a, a);
}
FfxFloat16x3 ffxBroadcastFloat16x3(FfxFloat16 a)
{
    return FfxFloat16x3(a, a, a);
}
FfxFloat16x4 ffxBroadcastFloat16x4(FfxFloat16 a)
{
    return FfxFloat16x4(a, a, a, a);
}
#define FFX_BROADCAST_FLOAT16(a)   FfxFloat16(a)
#define FFX_BROADCAST_FLOAT16X2(a) FfxFloat16x2(FfxFloat16(a))
#define FFX_BROADCAST_FLOAT16X3(a) FfxFloat16x3(FfxFloat16(a))
#define FFX_BROADCAST_FLOAT16X4(a) FfxFloat16x4(FfxFloat16(a))
//------------------------------------------------------------------------------------------------------------------------------
FfxInt16 ffxBroadcastInt16(FfxInt16 a)
{
    return FfxInt16(a);
}
FfxInt16x2 ffxBroadcastInt16x2(FfxInt16 a)
{
    return FfxInt16x2(a, a);
}
FfxInt16x3 ffxBroadcastInt16x3(FfxInt16 a)
{
    return FfxInt16x3(a, a, a);
}
FfxInt16x4 ffxBroadcastInt16x4(FfxInt16 a)
{
    return FfxInt16x4(a, a, a, a);
}
#define FFX_BROADCAST_INT16(a)   FfxInt16(a)
#define FFX_BROADCAST_INT16X2(a) FfxInt16x2(FfxInt16(a))
#define FFX_BROADCAST_INT16X3(a) FfxInt16x3(FfxInt16(a))
#define FFX_BROADCAST_INT16X4(a) FfxInt16x4(FfxInt16(a))
//------------------------------------------------------------------------------------------------------------------------------
FfxUInt16 ffxBroadcastUInt16(FfxUInt16 a)
{
    return FfxUInt16(a);
}
FfxUInt16x2 ffxBroadcastUInt16x2(FfxUInt16 a)
{
    return FfxUInt16x2(a, a);
}
FfxUInt16x3 ffxBroadcastUInt16x3(FfxUInt16 a)
{
    return FfxUInt16x3(a, a, a);
}
FfxUInt16x4 ffxBroadcastUInt16x4(FfxUInt16 a)
{
    return FfxUInt16x4(a, a, a, a);
}
#define FFX_BROADCAST_UINT16(a)   FfxUInt16(a)
#define FFX_BROADCAST_UINT16X2(a) FfxUInt16x2(FfxUInt16(a))
#define FFX_BROADCAST_UINT16X3(a) FfxUInt16x3(FfxUInt16(a))
#define FFX_BROADCAST_UINT16X4(a) FfxUInt16x4(FfxUInt16(a))
//==============================================================================================================================
FfxUInt16 ffxAbsHalf(FfxUInt16 a)
{
    return FfxUInt16(abs(FfxInt16(a)));
}
FfxUInt16x2 ffxAbsHalf(FfxUInt16x2 a)
{
    return FfxUInt16x2(abs(FfxInt16x2(a)));
}
FfxUInt16x3 ffxAbsHalf(FfxUInt16x3 a)
{
    return FfxUInt16x3(abs(FfxInt16x3(a)));
}
FfxUInt16x4 ffxAbsHalf(FfxUInt16x4 a)
{
    return FfxUInt16x4(abs(FfxInt16x4(a)));
}
//------------------------------------------------------------------------------------------------------------------------------
FfxFloat16 ffxClampHalf(FfxFloat16 x, FfxFloat16 n, FfxFloat16 m)
{
    return clamp(x, n, m);
}
FfxFloat16x2 ffxClampHalf(FfxFloat16x2 x, FfxFloat16x2 n, FfxFloat16x2 m)
{
    return clamp(x, n, m);
}
FfxFloat16x3 ffxClampHalf(FfxFloat16x3 x, FfxFloat16x3 n, FfxFloat16x3 m)
{
    return clamp(x, n, m);
}
FfxFloat16x4 ffxClampHalf(FfxFloat16x4 x, FfxFloat16x4 n, FfxFloat16x4 m)
{
    return clamp(x, n, m);
}
//------------------------------------------------------------------------------------------------------------------------------
FfxFloat16 ffxFract(FfxFloat16 x)
{
    return fract(x);
}
FfxFloat16x2 ffxFract(FfxFloat16x2 x)
{
    return fract(x);
}
FfxFloat16x3 ffxFract(FfxFloat16x3 x)
{
    return fract(x);
}
FfxFloat16x4 ffxFract(FfxFloat16x4 x)
{
    return fract(x);
}
//------------------------------------------------------------------------------------------------------------------------------
FfxFloat16 ffxLerp(FfxFloat16 x, FfxFloat16 y, FfxFloat16 a)
{
    return mix(x, y, a);
}
FfxFloat16x2 ffxLerp(FfxFloat16x2 x, FfxFloat16x2 y, FfxFloat16 a)
{
    return mix(x, y, a);
}
FfxFloat16x2 ffxLerp(FfxFloat16x2 x, FfxFloat16x2 y, FfxFloat16x2 a)
{
    return mix(x, y, a);
}
FfxFloat16x3 ffxLerp(FfxFloat16x3 x, FfxFloat16x3 y, FfxFloat16x3 a)
{
    return mix(x, y, a);
}
FfxFloat16x3 ffxLerp(FfxFloat16x3 x, FfxFloat16x3 y, FfxFloat16 a)
{
    return mix(x, y, a);
}
FfxFloat16x4 ffxLerp(FfxFloat16x4 x, FfxFloat16x4 y, FfxFloat16 a)
{
    return mix(x, y, a);
}
FfxFloat16x4 ffxLerp(FfxFloat16x4 x, FfxFloat16x4 y, FfxFloat16x4 a)
{
    return mix(x, y, a);
}
//------------------------------------------------------------------------------------------------------------------------------
// No packed version of ffxMax3.
FfxFloat16 ffxMax3Half(FfxFloat16 x, FfxFloat16 y, FfxFloat16 z)
{
    return max(x, max(y, z));
}
FfxFloat16x2 ffxMax3Half(FfxFloat16x2 x, FfxFloat16x2 y, FfxFloat16x2 z)
{
    return max(x, max(y, z));
}
FfxFloat16x3 ffxMax3Half(FfxFloat16x3 x, FfxFloat16x3 y, FfxFloat16x3 z)
{
    return max(x, max(y, z));
}
FfxFloat16x4 ffxMax3Half(FfxFloat16x4 x, FfxFloat16x4 y, FfxFloat16x4 z)
{
    return max(x, max(y, z));
}
//------------------------------------------------------------------------------------------------------------------------------
// No packed version of ffxMin3.
FfxFloat16 ffxMin3Half(FfxFloat16 x, FfxFloat16 y, FfxFloat16 z)
{
    return min(x, min(y, z));
}
FfxFloat16x2 ffxMin3Half(FfxFloat16x2 x, FfxFloat16x2 y, FfxFloat16x2 z)
{
    return min(x, min(y, z));
}
FfxFloat16x3 ffxMin3Half(FfxFloat16x3 x, FfxFloat16x3 y, FfxFloat16x3 z)
{
    return min(x, min(y, z));
}
FfxFloat16x4 ffxMin3Half(FfxFloat16x4 x, FfxFloat16x4 y, FfxFloat16x4 z)
{
    return min(x, min(y, z));
}
//------------------------------------------------------------------------------------------------------------------------------
FfxFloat16 ffxMed3Half(FfxFloat16 x, FfxFloat16 y, FfxFloat16 z)
{
    return max(min(x, y), min(max(x, y), z));
}
FfxFloat16x2 ffxMed3Half(FfxFloat16x2 x, FfxFloat16x2 y, FfxFloat16x2 z)
{
    return max(min(x, y), min(max(x, y), z));
}
FfxFloat16x3 ffxMed3Half(FfxFloat16x3 x, FfxFloat16x3 y, FfxFloat16x3 z)
{
    return max(min(x, y), min(max(x, y), z));
}
FfxFloat16x4 ffxMed3Half(FfxFloat16x4 x, FfxFloat16x4 y, FfxFloat16x4 z)
{
    return max(min(x, y), min(max(x, y), z));
}
//------------------------------------------------------------------------------------------------------------------------------
FfxFloat16 ffxReciprocalHalf(FfxFloat16 x)
{
    return FFX_BROADCAST_FLOAT16(1.0) / x;
}
FfxFloat16x2 ffxReciprocalHalf(FfxFloat16x2 x)
{
    return FFX_BROADCAST_FLOAT16X2(1.0) / x;
}
FfxFloat16x3 ffxReciprocalHalf(FfxFloat16x3 x)
{
    return FFX_BROADCAST_FLOAT16X3(1.0) / x;
}
FfxFloat16x4 ffxReciprocalHalf(FfxFloat16x4 x)
{
    return FFX_BROADCAST_FLOAT16X4(1.0) / x;
}
//------------------------------------------------------------------------------------------------------------------------------
FfxFloat16 ffxReciprocalSquareRootHalf(FfxFloat16 x)
{
    return FFX_BROADCAST_FLOAT16(1.0) / sqrt(x);
}
FfxFloat16x2 ffxReciprocalSquareRootHalf(FfxFloat16x2 x)
{
    return FFX_BROADCAST_FLOAT16X2(1.0) / sqrt(x);
}
FfxFloat16x3 ffxReciprocalSquareRootHalf(FfxFloat16x3 x)
{
    return FFX_BROADCAST_FLOAT16X3(1.0) / sqrt(x);
}
FfxFloat16x4 ffxReciprocalSquareRootHalf(FfxFloat16x4 x)
{
    return FFX_BROADCAST_FLOAT16X4(1.0) / sqrt(x);
}
//------------------------------------------------------------------------------------------------------------------------------
FfxFloat16 ffxSaturate(FfxFloat16 x)
{
    return clamp(x, FFX_BROADCAST_FLOAT16(0.0), FFX_BROADCAST_FLOAT16(1.0));
}
FfxFloat16x2 ffxSaturate(FfxFloat16x2 x)
{
    return clamp(x, FFX_BROADCAST_FLOAT16X2(0.0), FFX_BROADCAST_FLOAT16X2(1.0));
}
FfxFloat16x3 ffxSaturate(FfxFloat16x3 x)
{
    return clamp(x, FFX_BROADCAST_FLOAT16X3(0.0), FFX_BROADCAST_FLOAT16X3(1.0));
}
FfxFloat16x4 ffxSaturate(FfxFloat16x4 x)
{
    return clamp(x, FFX_BROADCAST_FLOAT16X4(0.0), FFX_BROADCAST_FLOAT16X4(1.0));
}
//------------------------------------------------------------------------------------------------------------------------------
FfxUInt16 ffxBitShiftRightHalf(FfxUInt16 a, FfxUInt16 b)
{
    return FfxUInt16(FfxInt16(a) >> FfxInt16(b));
}
FfxUInt16x2 ffxBitShiftRightHalf(FfxUInt16x2 a, FfxUInt16x2 b)
{
    return FfxUInt16x2(FfxInt16x2(a) >> FfxInt16x2(b));
}
FfxUInt16x3 ffxBitShiftRightHalf(FfxUInt16x3 a, FfxUInt16x3 b)
{
    return FfxUInt16x3(FfxInt16x3(a) >> FfxInt16x3(b));
}
FfxUInt16x4 ffxBitShiftRightHalf(FfxUInt16x4 a, FfxUInt16x4 b)
{
    return FfxUInt16x4(FfxInt16x4(a) >> FfxInt16x4(b));
}
#endif // FFX_HALF

#if defined(FFX_WAVE)
// Where 'x' must be a compile time literal.
FfxFloat32 ffxWaveXorF1(FfxFloat32 v, FfxUInt32 x)
{
    return subgroupShuffleXor(v, x);
}
FfxFloat32x2 ffxWaveXorF2(FfxFloat32x2 v, FfxUInt32 x)
{
    return subgroupShuffleXor(v, x);
}
FfxFloat32x3 ffxWaveXorF3(FfxFloat32x3 v, FfxUInt32 x)
{
    return subgroupShuffleXor(v, x);
}
FfxFloat32x4 ffxWaveXorF4(FfxFloat32x4 v, FfxUInt32 x)
{
    return subgroupShuffleXor(v, x);
}
FfxUInt32 ffxWaveXorU1(FfxUInt32 v, FfxUInt32 x)
{
    return subgroupShuffleXor(v, x);
}
FfxUInt32x2 ffxWaveXorU2(FfxUInt32x2 v, FfxUInt32 x)
{
    return subgroupShuffleXor(v, x);
}
FfxUInt32x3 ffxWaveXorU3(FfxUInt32x3 v, FfxUInt32 x)
{
    return subgroupShuffleXor(v, x);
}
FfxUInt32x4 ffxWaveXorU4(FfxUInt32x4 v, FfxUInt32 x)
{
    return subgroupShuffleXor(v, x);
}
FfxBoolean ffxWaveIsFirstLane()
{
    return subgroupElect();
}
FfxUInt32 ffxWaveLaneIndex()
{
    return gl_SubgroupInvocationID;
}
FfxBoolean ffxWaveReadAtLaneIndexB1(FfxBoolean v, FfxUInt32 x )
{
    return subgroupShuffle(v, x);
}
FfxUInt32 ffxWavePrefixCountBits(FfxBoolean v)
{
    return subgroupBallotExclusiveBitCount(subgroupBallot(v));
}
FfxUInt32 ffxWaveActiveCountBits(FfxBoolean v)
{
    return subgroupBallotBitCount(subgroupBallot(v));
}
FfxUInt32 ffxWaveReadLaneFirstU1(FfxUInt32 v)
{
    return subgroupBroadcastFirst(v);
}
FfxUInt32x2 ffxWaveReadLaneFirstU2(FfxUInt32x2 v)
{
    return subgroupBroadcastFirst(v);
}
FfxBoolean ffxWaveReadLaneFirstB1(FfxBoolean v)
{
    return subgroupBroadcastFirst(v);
}
FfxUInt32 ffxWaveOr(FfxUInt32 a)
{
    return subgroupOr(a);
}
FfxUInt32 ffxWaveMin(FfxUInt32 a)
{
    return subgroupMin(a);
}
FfxFloat32 ffxWaveMin(FfxFloat32 a)
{
    return subgroupMin(a);
}
FfxUInt32 ffxWaveMax(FfxUInt32 a)
{
    return subgroupMax(a);
}
FfxFloat32 ffxWaveMax(FfxFloat32 a)
{
    return subgroupMax(a);
}
FfxUInt32 ffxWaveSum(FfxUInt32 a)
{
    return subgroupAdd(a);
}
FfxFloat32 ffxWaveSum(FfxFloat32 a)
{
    return subgroupAdd(a);
}
FfxUInt32 ffxWaveLaneCount()
{
    return gl_SubgroupSize;
}
#if defined(FFX_WAVE_ALL_TRUE)
FfxBoolean ffxWaveAllTrue(FfxBoolean v)
{
    return subgroupAll(v);
}
#endif
FfxFloat32 ffxQuadReadX(FfxFloat32 v)
{
    return subgroupQuadSwapHorizontal(v);
}
FfxFloat32x2 ffxQuadReadX(FfxFloat32x2 v)
{
    return subgroupQuadSwapHorizontal(v);
}
FfxFloat32 ffxQuadReadY(FfxFloat32 v)
{
    return subgroupQuadSwapVertical(v);
}
FfxFloat32x2 ffxQuadReadY(FfxFloat32x2 v)
{
    return subgroupQuadSwapVertical(v);
}

//------------------------------------------------------------------------------------------------------------------------------
#if FFX_HALF
FfxFloat16x2 ffxWaveXorFloat16x2(FfxFloat16x2 v, FfxUInt32 x)
{
    return FFX_UINT32_TO_FLOAT16X2(subgroupShuffleXor(FFX_FLOAT16X2_TO_UINT32(v), x));
}
FfxFloat16x4 ffxWaveXorFloat16x4(FfxFloat16x4 v, FfxUInt32 x)
{
    return FFX_UINT32X2_TO_FLOAT16X4(subgroupShuffleXor(FFX_FLOAT16X4_TO_UINT32X2(v), x));
}
FfxUInt16x2 ffxWaveXorUint16x2(FfxUInt16x2 v, FfxUInt32 x)
{
    return FFX_UINT32_TO_UINT16X2(subgroupShuffleXor(FFX_UINT16X2_TO_UINT32(v), x));
}
FfxUInt16x4 ffxWaveXorUint16x4(FfxUInt16x4 v, FfxUInt32 x)
{
    return FFX_UINT32X2_TO_UINT16X4(subgroupShuffleXor(FFX_UINT16X4_TO_UINT32X2(v), x));
}
#endif // FFX_HALF
#endif // #if defined(FFX_WAVE)
