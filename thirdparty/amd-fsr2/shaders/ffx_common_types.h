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
#ifndef FFX_COMMON_TYPES_H
#define FFX_COMMON_TYPES_H

#if defined(FFX_CPU)
#define FFX_PARAMETER_IN
#define FFX_PARAMETER_OUT
#define FFX_PARAMETER_INOUT
#elif defined(FFX_HLSL)
#define FFX_PARAMETER_IN        in
#define FFX_PARAMETER_OUT       out
#define FFX_PARAMETER_INOUT     inout
#elif defined(FFX_GLSL)
#define FFX_PARAMETER_IN        in
#define FFX_PARAMETER_OUT       out
#define FFX_PARAMETER_INOUT     inout
#endif // #if defined(FFX_CPU)

#if defined(FFX_CPU)
/// A typedef for a boolean value.
///
/// @ingroup CPU
typedef bool FfxBoolean;

/// A typedef for a unsigned 8bit integer.
///
/// @ingroup CPU
typedef uint8_t FfxUInt8;

/// A typedef for a unsigned 16bit integer.
///
/// @ingroup CPU
typedef uint16_t FfxUInt16;

/// A typedef for a unsigned 32bit integer.
///
/// @ingroup CPU
typedef uint32_t FfxUInt32;

/// A typedef for a unsigned 64bit integer.
///
/// @ingroup CPU
typedef uint64_t FfxUInt64;

/// A typedef for a signed 8bit integer.
///
/// @ingroup CPU
typedef int8_t FfxInt8;

/// A typedef for a signed 16bit integer.
///
/// @ingroup CPU
typedef int16_t FfxInt16;

/// A typedef for a signed 32bit integer.
///
/// @ingroup CPU
typedef int32_t FfxInt32;

/// A typedef for a signed 64bit integer.
///
/// @ingroup CPU
typedef int64_t FfxInt64;

/// A typedef for a floating point value.
///
/// @ingroup CPU
typedef float FfxFloat32;

/// A typedef for a 2-dimensional floating point value.
///
/// @ingroup CPU
typedef float FfxFloat32x2[2];

/// A typedef for a 3-dimensional floating point value.
///
/// @ingroup CPU
typedef float FfxFloat32x3[3];

/// A typedef for a 4-dimensional floating point value.
///
/// @ingroup CPU
typedef float FfxFloat32x4[4];

/// A typedef for a 2-dimensional 32bit unsigned integer.
///
/// @ingroup CPU
typedef uint32_t FfxUInt32x2[2];

/// A typedef for a 3-dimensional 32bit unsigned integer.
///
/// @ingroup CPU
typedef uint32_t FfxUInt32x3[3];

/// A typedef for a 4-dimensional 32bit unsigned integer.
///
/// @ingroup CPU
typedef uint32_t FfxUInt32x4[4];
#endif // #if defined(FFX_CPU)

#if defined(FFX_HLSL)
/// A typedef for a boolean value.
///
/// @ingroup GPU
typedef bool FfxBoolean;

#if FFX_HLSL_6_2
typedef float32_t   FfxFloat32;
typedef float32_t2  FfxFloat32x2;
typedef float32_t3  FfxFloat32x3;
typedef float32_t4  FfxFloat32x4;

/// A typedef for a unsigned 32bit integer.
///
/// @ingroup GPU
typedef uint32_t    FfxUInt32;
typedef uint32_t2   FfxUInt32x2;
typedef uint32_t3   FfxUInt32x3;
typedef uint32_t4   FfxUInt32x4;
typedef int32_t     FfxInt32;
typedef int32_t2    FfxInt32x2;
typedef int32_t3    FfxInt32x3;
typedef int32_t4    FfxInt32x4;
#else
#define FfxFloat32   float
#define FfxFloat32x2 float2
#define FfxFloat32x3 float3
#define FfxFloat32x4 float4

/// A typedef for a unsigned 32bit integer.
///
/// @ingroup GPU
typedef uint        FfxUInt32;
typedef uint2       FfxUInt32x2;
typedef uint3       FfxUInt32x3;
typedef uint4       FfxUInt32x4;
typedef int         FfxInt32;
typedef int2        FfxInt32x2;
typedef int3        FfxInt32x3;
typedef int4        FfxInt32x4;
#endif // #if defined(FFX_HLSL_6_2)

#if FFX_HALF
#if FFX_HLSL_6_2
typedef float16_t   FfxFloat16;
typedef float16_t2  FfxFloat16x2;
typedef float16_t3  FfxFloat16x3;
typedef float16_t4  FfxFloat16x4;

/// A typedef for an unsigned 16bit integer.
///
/// @ingroup GPU
typedef uint16_t    FfxUInt16;
typedef uint16_t2   FfxUInt16x2;
typedef uint16_t3   FfxUInt16x3;
typedef uint16_t4   FfxUInt16x4;

/// A typedef for a signed 16bit integer.
///
/// @ingroup GPU
typedef int16_t     FfxInt16;
typedef int16_t2    FfxInt16x2;
typedef int16_t3    FfxInt16x3;
typedef int16_t4    FfxInt16x4;
#else
typedef min16float  FfxFloat16;
typedef min16float2 FfxFloat16x2;
typedef min16float3 FfxFloat16x3;
typedef min16float4 FfxFloat16x4;

/// A typedef for an unsigned 16bit integer.
///
/// @ingroup GPU
typedef min16uint   FfxUInt16;
typedef min16uint2  FfxUInt16x2;
typedef min16uint3  FfxUInt16x3;
typedef min16uint4  FfxUInt16x4;

/// A typedef for a signed 16bit integer.
///
/// @ingroup GPU
typedef min16int    FfxInt16;
typedef min16int2   FfxInt16x2;
typedef min16int3   FfxInt16x3;
typedef min16int4   FfxInt16x4;
#endif  // FFX_HLSL_6_2
#endif // FFX_HALF
#endif // #if defined(FFX_HLSL)

#if defined(FFX_GLSL)
/// A typedef for a boolean value.
///
/// @ingroup GPU
#define FfxBoolean   bool
#define FfxFloat32   float
#define FfxFloat32x2 vec2
#define FfxFloat32x3 vec3
#define FfxFloat32x4 vec4
#define FfxUInt32    uint
#define FfxUInt32x2  uvec2
#define FfxUInt32x3  uvec3
#define FfxUInt32x4  uvec4
#define FfxInt32     int
#define FfxInt32x2   ivec2
#define FfxInt32x3   ivec3
#define FfxInt32x4   ivec4
#if FFX_HALF
#define FfxFloat16   float16_t
#define FfxFloat16x2 f16vec2
#define FfxFloat16x3 f16vec3
#define FfxFloat16x4 f16vec4
#define FfxUInt16    uint16_t
#define FfxUInt16x2  u16vec2
#define FfxUInt16x3  u16vec3
#define FfxUInt16x4  u16vec4
#define FfxInt16     int16_t
#define FfxInt16x2   i16vec2
#define FfxInt16x3   i16vec3
#define FfxInt16x4   i16vec4
#endif // FFX_HALF
#endif // #if defined(FFX_GLSL)

// Global toggles:
// #define FFX_HALF            (1)
// #define FFX_HLSL_6_2        (1)

#if FFX_HALF

#if FFX_HLSL_6_2

#define FFX_MIN16_SCALAR( TypeName, BaseComponentType )           typedef BaseComponentType##16_t TypeName;
#define FFX_MIN16_VECTOR( TypeName, BaseComponentType, COL )      typedef vector<BaseComponentType##16_t, COL> TypeName;
#define FFX_MIN16_MATRIX( TypeName, BaseComponentType, ROW, COL ) typedef matrix<BaseComponentType##16_t, ROW, COL> TypeName;

#define FFX_16BIT_SCALAR( TypeName, BaseComponentType )           typedef BaseComponentType##16_t TypeName;
#define FFX_16BIT_VECTOR( TypeName, BaseComponentType, COL )      typedef vector<BaseComponentType##16_t, COL> TypeName;
#define FFX_16BIT_MATRIX( TypeName, BaseComponentType, ROW, COL ) typedef matrix<BaseComponentType##16_t, ROW, COL> TypeName;

#else //FFX_HLSL_6_2

#define FFX_MIN16_SCALAR( TypeName, BaseComponentType )           typedef min16##BaseComponentType TypeName;
#define FFX_MIN16_VECTOR( TypeName, BaseComponentType, COL )      typedef vector<min16##BaseComponentType, COL> TypeName;
#define FFX_MIN16_MATRIX( TypeName, BaseComponentType, ROW, COL ) typedef matrix<min16##BaseComponentType, ROW, COL> TypeName;

#define FFX_16BIT_SCALAR( TypeName, BaseComponentType )           FFX_MIN16_SCALAR( TypeName, BaseComponentType );
#define FFX_16BIT_VECTOR( TypeName, BaseComponentType, COL )      FFX_MIN16_VECTOR( TypeName, BaseComponentType, COL );
#define FFX_16BIT_MATRIX( TypeName, BaseComponentType, ROW, COL ) FFX_MIN16_MATRIX( TypeName, BaseComponentType, ROW, COL );

#endif //FFX_HLSL_6_2

#else //FFX_HALF

#define FFX_MIN16_SCALAR( TypeName, BaseComponentType )           typedef BaseComponentType TypeName;
#define FFX_MIN16_VECTOR( TypeName, BaseComponentType, COL )      typedef vector<BaseComponentType, COL> TypeName;
#define FFX_MIN16_MATRIX( TypeName, BaseComponentType, ROW, COL ) typedef matrix<BaseComponentType, ROW, COL> TypeName;

#define FFX_16BIT_SCALAR( TypeName, BaseComponentType )           typedef BaseComponentType TypeName;
#define FFX_16BIT_VECTOR( TypeName, BaseComponentType, COL )      typedef vector<BaseComponentType, COL> TypeName;
#define FFX_16BIT_MATRIX( TypeName, BaseComponentType, ROW, COL ) typedef matrix<BaseComponentType, ROW, COL> TypeName;

#endif //FFX_HALF

#if defined(FFX_GPU)
// Common typedefs:
#if defined(FFX_HLSL)
FFX_MIN16_SCALAR( FFX_MIN16_F , float );
FFX_MIN16_VECTOR( FFX_MIN16_F2, float, 2 );
FFX_MIN16_VECTOR( FFX_MIN16_F3, float, 3 );
FFX_MIN16_VECTOR( FFX_MIN16_F4, float, 4 );

FFX_MIN16_SCALAR( FFX_MIN16_I,  int );
FFX_MIN16_VECTOR( FFX_MIN16_I2, int, 2 );
FFX_MIN16_VECTOR( FFX_MIN16_I3, int, 3 );
FFX_MIN16_VECTOR( FFX_MIN16_I4, int, 4 );

FFX_MIN16_SCALAR( FFX_MIN16_U,  uint );
FFX_MIN16_VECTOR( FFX_MIN16_U2, uint, 2 );
FFX_MIN16_VECTOR( FFX_MIN16_U3, uint, 3 );
FFX_MIN16_VECTOR( FFX_MIN16_U4, uint, 4 );

FFX_16BIT_SCALAR( FFX_F16_t , float );
FFX_16BIT_VECTOR( FFX_F16_t2, float, 2 );
FFX_16BIT_VECTOR( FFX_F16_t3, float, 3 );
FFX_16BIT_VECTOR( FFX_F16_t4, float, 4 );

FFX_16BIT_SCALAR( FFX_I16_t,  int );
FFX_16BIT_VECTOR( FFX_I16_t2, int, 2 );
FFX_16BIT_VECTOR( FFX_I16_t3, int, 3 );
FFX_16BIT_VECTOR( FFX_I16_t4, int, 4 );

FFX_16BIT_SCALAR( FFX_U16_t,  uint );
FFX_16BIT_VECTOR( FFX_U16_t2, uint, 2 );
FFX_16BIT_VECTOR( FFX_U16_t3, uint, 3 );
FFX_16BIT_VECTOR( FFX_U16_t4, uint, 4 );

#define TYPEDEF_MIN16_TYPES(Prefix)           \
typedef FFX_MIN16_F     Prefix##_F;           \
typedef FFX_MIN16_F2    Prefix##_F2;          \
typedef FFX_MIN16_F3    Prefix##_F3;          \
typedef FFX_MIN16_F4    Prefix##_F4;          \
typedef FFX_MIN16_I     Prefix##_I;           \
typedef FFX_MIN16_I2    Prefix##_I2;          \
typedef FFX_MIN16_I3    Prefix##_I3;          \
typedef FFX_MIN16_I4    Prefix##_I4;          \
typedef FFX_MIN16_U     Prefix##_U;           \
typedef FFX_MIN16_U2    Prefix##_U2;          \
typedef FFX_MIN16_U3    Prefix##_U3;          \
typedef FFX_MIN16_U4    Prefix##_U4;

#define TYPEDEF_16BIT_TYPES(Prefix)           \
typedef FFX_16BIT_F     Prefix##_F;           \
typedef FFX_16BIT_F2    Prefix##_F2;          \
typedef FFX_16BIT_F3    Prefix##_F3;          \
typedef FFX_16BIT_F4    Prefix##_F4;          \
typedef FFX_16BIT_I     Prefix##_I;           \
typedef FFX_16BIT_I2    Prefix##_I2;          \
typedef FFX_16BIT_I3    Prefix##_I3;          \
typedef FFX_16BIT_I4    Prefix##_I4;          \
typedef FFX_16BIT_U     Prefix##_U;           \
typedef FFX_16BIT_U2    Prefix##_U2;          \
typedef FFX_16BIT_U3    Prefix##_U3;          \
typedef FFX_16BIT_U4    Prefix##_U4;

#define TYPEDEF_FULL_PRECISION_TYPES(Prefix)  \
typedef FfxFloat32      Prefix##_F;           \
typedef FfxFloat32x2    Prefix##_F2;          \
typedef FfxFloat32x3    Prefix##_F3;          \
typedef FfxFloat32x4    Prefix##_F4;          \
typedef FfxInt32        Prefix##_I;           \
typedef FfxInt32x2      Prefix##_I2;          \
typedef FfxInt32x3      Prefix##_I3;          \
typedef FfxInt32x4      Prefix##_I4;          \
typedef FfxUInt32       Prefix##_U;           \
typedef FfxUInt32x2     Prefix##_U2;          \
typedef FfxUInt32x3     Prefix##_U3;          \
typedef FfxUInt32x4     Prefix##_U4;
#endif // #if defined(FFX_HLSL)

#if defined(FFX_GLSL)

#if FFX_HALF

#define  FFX_MIN16_F  float16_t
#define  FFX_MIN16_F2 f16vec2
#define  FFX_MIN16_F3 f16vec3
#define  FFX_MIN16_F4 f16vec4

#define  FFX_MIN16_I  int16_t
#define  FFX_MIN16_I2 i16vec2
#define  FFX_MIN16_I3 i16vec3
#define  FFX_MIN16_I4 i16vec4

#define  FFX_MIN16_U  uint16_t
#define  FFX_MIN16_U2 u16vec2
#define  FFX_MIN16_U3 u16vec3
#define  FFX_MIN16_U4 u16vec4

#define FFX_16BIT_F  float16_t
#define FFX_16BIT_F2 f16vec2
#define FFX_16BIT_F3 f16vec3
#define FFX_16BIT_F4 f16vec4

#define FFX_16BIT_I  int16_t
#define FFX_16BIT_I2 i16vec2
#define FFX_16BIT_I3 i16vec3
#define FFX_16BIT_I4 i16vec4

#define FFX_16BIT_U  uint16_t
#define FFX_16BIT_U2 u16vec2
#define FFX_16BIT_U3 u16vec3
#define FFX_16BIT_U4 u16vec4

#else // FFX_HALF

#define  FFX_MIN16_F  float
#define  FFX_MIN16_F2 vec2
#define  FFX_MIN16_F3 vec3
#define  FFX_MIN16_F4 vec4

#define  FFX_MIN16_I  int
#define  FFX_MIN16_I2 ivec2
#define  FFX_MIN16_I3 ivec3
#define  FFX_MIN16_I4 ivec4

#define  FFX_MIN16_U  uint
#define  FFX_MIN16_U2 uvec2
#define  FFX_MIN16_U3 uvec3
#define  FFX_MIN16_U4 uvec4

#define FFX_16BIT_F  float
#define FFX_16BIT_F2 vec2
#define FFX_16BIT_F3 vec3
#define FFX_16BIT_F4 vec4

#define FFX_16BIT_I  int
#define FFX_16BIT_I2 ivec2
#define FFX_16BIT_I3 ivec3
#define FFX_16BIT_I4 ivec4

#define FFX_16BIT_U  uint
#define FFX_16BIT_U2 uvec2
#define FFX_16BIT_U3 uvec3
#define FFX_16BIT_U4 uvec4

#endif // FFX_HALF

#endif // #if defined(FFX_GLSL)

#endif // #if defined(FFX_GPU)
#endif // #ifndef FFX_COMMON_TYPES_H
