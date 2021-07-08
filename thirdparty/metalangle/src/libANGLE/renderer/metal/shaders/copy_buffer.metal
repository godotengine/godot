//
// Copyright 2020 The ANGLE Project. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// copy_buffer.metal: implements compute shader that copy formatted data from buffer to texture,
// from texture to buffer and from buffer to buffer.
// NOTE(hqle): This file is a bit hard to read but there are a lot of repeated works, and it would
// be a pain to implement without the use of macros.
//

#include <metal_pack>

#include "common.h"
#include "format_autogen.h"

using namespace rx::mtl_shader;

constant int kCopyFormatType [[function_constant(1)]];

/* -------- copy pixel data between buffer and texture ---------*/
constant int kCopyTextureType [[function_constant(2)]];
constant bool kCopyTextureType2D      = kCopyTextureType == kTextureType2D;
constant bool kCopyTextureType2DArray = kCopyTextureType == kTextureType2DArray;
constant bool kCopyTextureType2DMS    = kCopyTextureType == kTextureType2DMultisample;
constant bool kCopyTextureTypeCube    = kCopyTextureType == kTextureTypeCube;
constant bool kCopyTextureType3D      = kCopyTextureType == kTextureType3D;

struct CopyPixelParams
{
    uint3 copySize;
    uint3 textureOffset;

    uint bufferStartOffset;
    uint pixelSize;
    uint bufferRowPitch;
    uint bufferDepthPitch;
};

struct WritePixelParams
{
    uint2 copySize;
    uint2 textureOffset;

    uint bufferStartOffset;

    uint pixelSize;
    uint bufferRowPitch;

    uint textureLevel;
    uint textureLayer;

    bool reverseTextureRowOrder;
};

static inline float4 sRGBtoLinear(float4 color)
{
    float3 linear1 = color.rgb / 12.92;
    float3 linear2 = pow((color.rgb + float3(0.055)) / 1.055, 2.4);
    float3 factor  = float3(color.rgb <= float3(0.04045));
    float4 linear  = float4(factor * linear1 + float3(1.0 - factor) * linear2, color.a);

    return linear;
}

// clang-format off
#define TEXTURE_PARAMS(TYPE, ACCESS, NAME_PREFIX)               \
    texture2d<TYPE, ACCESS> NAME_PREFIX##Texture2d              \
    [[texture(0), function_constant(kCopyTextureType2D)]],      \
    texture2d_array<TYPE, ACCESS> NAME_PREFIX##Texture2dArray   \
    [[texture(0), function_constant(kCopyTextureType2DArray)]], \
    texture3d<TYPE, ACCESS> NAME_PREFIX##Texture3d              \
    [[texture(0), function_constant(kCopyTextureType3D)]],      \
    texturecube<TYPE, ACCESS> NAME_PREFIX##TextureCube          \
    [[texture(0), function_constant(kCopyTextureTypeCube)]]

#define FORWARD_TEXTURE_PARAMS(NAME_PREFIX) \
    NAME_PREFIX##Texture2d,                 \
    NAME_PREFIX##Texture2dArray,            \
    NAME_PREFIX##Texture3d,                 \
    NAME_PREFIX##TextureCube               

// Params for reading from buffer to texture
#define DEST_TEXTURE_PARAMS(TYPE)  TEXTURE_PARAMS(TYPE, access::write, dst)
#define FORWARD_DEST_TEXTURE_PARAMS FORWARD_TEXTURE_PARAMS(dst)

#define COMMON_READ_KERNEL_PARAMS(TEXTURE_TYPE)     \
    ushort3 gIndices [[thread_position_in_grid]],   \
    constant CopyPixelParams &options[[buffer(0)]], \
    constant uchar *buffer [[buffer(1)]],           \
    DEST_TEXTURE_PARAMS(TEXTURE_TYPE)

#define COMMON_READ_FUNC_PARAMS        \
    uint bufferOffset,                 \
    constant uchar *buffer

#define FORWARD_COMMON_READ_FUNC_PARAMS bufferOffset, buffer

// Params for writing to buffer by coping from texture.
// (NOTE: it has additional multisample source texture parameter)
#define SRC_TEXTURE_PARAMS(TYPE)                             \
    TEXTURE_PARAMS(TYPE, access::read, src),                 \
    texture2d_ms<TYPE, access::read> srcTexture2dMS          \
    [[texture(0), function_constant(kCopyTextureType2DMS)]]  \

#define FORWARD_SRC_TEXTURE_PARAMS FORWARD_TEXTURE_PARAMS(src), srcTexture2dMS

#define COMMON_WRITE_KERNEL_PARAMS(TEXTURE_TYPE)     \
    ushort2 gIndices [[thread_position_in_grid]],    \
    constant WritePixelParams &options[[buffer(0)]], \
    SRC_TEXTURE_PARAMS(TEXTURE_TYPE),                \
    device uchar *buffer [[buffer(1)]]               \

#define COMMON_WRITE_FUNC_PARAMS(TYPE) \
    ushort2 gIndices,                  \
    constant WritePixelParams &options,\
    uint bufferOffset,                 \
    vec<TYPE, 4> color,                \
    device uchar *buffer               \

#define COMMON_WRITE_FLOAT_FUNC_PARAMS COMMON_WRITE_FUNC_PARAMS(float)
#define COMMON_WRITE_SINT_FUNC_PARAMS COMMON_WRITE_FUNC_PARAMS(int)
#define COMMON_WRITE_UINT_FUNC_PARAMS COMMON_WRITE_FUNC_PARAMS(uint)

#define FORWARD_COMMON_WRITE_FUNC_PARAMS gIndices, options, bufferOffset, color, buffer

// clang-format on

// Write to texture code based on texture type:
template <typename T>
static inline void textureWrite(ushort3 gIndices,
                                constant CopyPixelParams &options,
                                vec<T, 4> color,
                                DEST_TEXTURE_PARAMS(T))
{
    uint3 writeIndices = options.textureOffset + uint3(gIndices);
    switch (kCopyTextureType)
    {
        case kTextureType2D:
            dstTexture2d.write(color, writeIndices.xy);
            break;
        case kTextureType2DArray:
            dstTexture2dArray.write(color, writeIndices.xy, writeIndices.z);
            break;
        case kTextureType3D:
            dstTexture3d.write(color, writeIndices);
            break;
        case kTextureTypeCube:
            dstTextureCube.write(color, writeIndices.xy, writeIndices.z);
            break;
    }
}

// Read from texture code based on texture type:
template <typename T>
static inline vec<T, 4> textureRead(ushort2 gIndices,
                                    constant WritePixelParams &options,
                                    SRC_TEXTURE_PARAMS(T))
{
    vec<T, 4> color;
    uint2 coords = uint2(gIndices);
    if (options.reverseTextureRowOrder)
    {
        coords.y = options.copySize.y - 1 - gIndices.y;
    }
    coords += options.textureOffset;
    switch (kCopyTextureType)
    {
        case kTextureType2D:
            color = srcTexture2d.read(coords.xy, options.textureLevel);
            break;
        case kTextureType2DArray:
            color = srcTexture2dArray.read(coords.xy, options.textureLayer, options.textureLevel);
            break;
        case kTextureType2DMultisample:
            color = resolveTextureMS(srcTexture2dMS, coords.xy);
            break;
        case kTextureType3D:
            color = srcTexture3d.read(uint3(coords, options.textureLayer), options.textureLevel);
            break;
        case kTextureTypeCube:
            color = srcTextureCube.read(coords.xy, options.textureLayer, options.textureLevel);
            break;
    }
    return color;
}

// Calculate offset into buffer:
#define CALC_BUFFER_READ_OFFSET(pixelSize)                               \
    options.bufferStartOffset + (gIndices.z * options.bufferDepthPitch + \
                                 gIndices.y * options.bufferRowPitch + gIndices.x * pixelSize)

#define CALC_BUFFER_WRITE_OFFSET(pixelSize) \
    options.bufferStartOffset + (gIndices.y * options.bufferRowPitch + gIndices.x * pixelSize)

// Per format handling code:
#define READ_FORMAT_SWITCH_CASE(format)                                      \
    case FormatID::format: {                                                 \
        auto color = read##format(FORWARD_COMMON_READ_FUNC_PARAMS);          \
        textureWrite(gIndices, options, color, FORWARD_DEST_TEXTURE_PARAMS); \
    }                                                                        \
    break;

#define WRITE_FORMAT_SWITCH_CASE(format)                                         \
    case FormatID::format: {                                                     \
        auto color = textureRead(gIndices, options, FORWARD_SRC_TEXTURE_PARAMS); \
        write##format(FORWARD_COMMON_WRITE_FUNC_PARAMS);                         \
    }                                                                            \
    break;

#define READ_KERNEL_GUARD                                                       \
    if (gIndices.x >= options.copySize.x || gIndices.y >= options.copySize.y || \
        gIndices.z >= options.copySize.z)                                       \
    {                                                                           \
        return;                                                                 \
    }

#define WRITE_KERNEL_GUARD                                                    \
    if (gIndices.x >= options.copySize.x || gIndices.y >= options.copySize.y) \
    {                                                                         \
        return;                                                               \
    }

// R5G6B5
static inline float4 readR5G6B5_UNORM(COMMON_READ_FUNC_PARAMS)
{
    float4 color;
    ushort src = bytesToShort<ushort>(buffer, bufferOffset);

    color.r = normalizedToFloat<5>(getShiftedData<5, 11>(src));
    color.g = normalizedToFloat<6>(getShiftedData<6, 5>(src));
    color.b = normalizedToFloat<5>(getShiftedData<5, 0>(src));
    color.a = 1.0;
    return color;
}
static inline void writeR5G6B5_UNORM(COMMON_WRITE_FLOAT_FUNC_PARAMS)
{
    ushort dst = shiftData<5, 11>(floatToNormalized<5, ushort>(color.r)) |
                 shiftData<6, 5>(floatToNormalized<6, ushort>(color.g)) |
                 shiftData<5, 0>(floatToNormalized<5, ushort>(color.b));

    shortToBytes(dst, bufferOffset, buffer);
}

// R4G4B4A4
static inline float4 readR4G4B4A4_UNORM(COMMON_READ_FUNC_PARAMS)
{
    float4 color;
    ushort src = bytesToShort<ushort>(buffer, bufferOffset);

    color.r = normalizedToFloat<4>(getShiftedData<4, 12>(src));
    color.g = normalizedToFloat<4>(getShiftedData<4, 8>(src));
    color.b = normalizedToFloat<4>(getShiftedData<4, 4>(src));
    color.a = normalizedToFloat<4>(getShiftedData<4, 0>(src));
    return color;
}
static inline void writeR4G4B4A4_UNORM(COMMON_WRITE_FLOAT_FUNC_PARAMS)
{
    ushort dst = shiftData<4, 12>(floatToNormalized<4, ushort>(color.r)) |
                 shiftData<4, 8>(floatToNormalized<4, ushort>(color.g)) |
                 shiftData<4, 4>(floatToNormalized<4, ushort>(color.b)) |
                 shiftData<4, 0>(floatToNormalized<4, ushort>(color.a));
    ;

    shortToBytes(dst, bufferOffset, buffer);
}

// R5G5B5A1
static inline float4 readR5G5B5A1_UNORM(COMMON_READ_FUNC_PARAMS)
{
    float4 color;
    ushort src = bytesToShort<ushort>(buffer, bufferOffset);

    color.r = normalizedToFloat<5>(getShiftedData<5, 11>(src));
    color.g = normalizedToFloat<5>(getShiftedData<5, 6>(src));
    color.b = normalizedToFloat<5>(getShiftedData<5, 1>(src));
    color.a = normalizedToFloat<1>(getShiftedData<1, 0>(src));
    return color;
}
static inline void writeR5G5B5A1_UNORM(COMMON_WRITE_FLOAT_FUNC_PARAMS)
{
    ushort dst = shiftData<5, 11>(floatToNormalized<5, ushort>(color.r)) |
                 shiftData<5, 6>(floatToNormalized<5, ushort>(color.g)) |
                 shiftData<5, 1>(floatToNormalized<5, ushort>(color.b)) |
                 shiftData<1, 0>(floatToNormalized<1, ushort>(color.a));
    ;

    shortToBytes(dst, bufferOffset, buffer);
}

// R10G10B10A2_SINT
static inline int4 readR10G10B10A2_SINT(COMMON_READ_FUNC_PARAMS)
{
    int4 color;
    int src = bytesToInt<int>(buffer, bufferOffset);

    constexpr int3 rgbSignMask(0x200);        // 1 set at the 9 bit
    constexpr int3 negativeMask(0xfffffc00);  // All bits from 10 to 31 set to 1
    constexpr int alphaSignMask = 0x2;
    constexpr int alphaNegMask  = 0xfffffffc;

    color.r = getShiftedData<10, 0>(src);
    color.g = getShiftedData<10, 10>(src);
    color.b = getShiftedData<10, 20>(src);

    int3 isRgbNegative = (color.rgb & rgbSignMask) >> 9;
    color.rgb          = (isRgbNegative * negativeMask) | color.rgb;

    color.a             = getShiftedData<2, 30>(src);
    int isAlphaNegative = color.a & alphaSignMask >> 1;
    color.a             = (isAlphaNegative * alphaNegMask) | color.a;
    return color;
}
// R10G10B10A2_UINT
static inline uint4 readR10G10B10A2_UINT(COMMON_READ_FUNC_PARAMS)
{
    uint4 color;
    uint src = bytesToInt<uint>(buffer, bufferOffset);

    color.r = getShiftedData<10, 0>(src);
    color.g = getShiftedData<10, 10>(src);
    color.b = getShiftedData<10, 20>(src);
    color.a = getShiftedData<2, 30>(src);
    return color;
}

// R8G8B8A8 generic
static inline float4 readR8G8B8A8(COMMON_READ_FUNC_PARAMS, bool isSRGB)
{
    float4 color;
    uint src = bytesToInt<uint>(buffer, bufferOffset);

    if (isSRGB)
    {
        color = unpack_unorm4x8_srgb_to_float(src);
    }
    else
    {
        color = unpack_unorm4x8_to_float(src);
    }
    return color;
}
static inline void writeR8G8B8A8(COMMON_WRITE_FLOAT_FUNC_PARAMS, bool isSRGB)
{
    uint dst;

    if (isSRGB)
    {
        dst = pack_float_to_srgb_unorm4x8(color);
    }
    else
    {
        dst = pack_float_to_unorm4x8(color);
    }

    intToBytes(dst, bufferOffset, buffer);
}

static inline float4 readR8G8B8(COMMON_READ_FUNC_PARAMS, bool isSRGB)
{
    float4 color;
    color.r = normalizedToFloat<uchar>(buffer[bufferOffset]);
    color.g = normalizedToFloat<uchar>(buffer[bufferOffset + 1]);
    color.b = normalizedToFloat<uchar>(buffer[bufferOffset + 2]);
    color.a = 1.0;

    if (isSRGB)
    {
        color = sRGBtoLinear(color);
    }
    return color;
}
static inline void writeR8G8B8(COMMON_WRITE_FLOAT_FUNC_PARAMS, bool isSRGB)
{
    color.a = 1.0;
    uint dst;

    if (isSRGB)
    {
        dst = pack_float_to_srgb_unorm4x8(color);
    }
    else
    {
        dst = pack_float_to_unorm4x8(color);
    }
    int24bitToBytes(dst, bufferOffset, buffer);
}

// RGBA8_SNORM
static inline float4 readR8G8B8A8_SNORM(COMMON_READ_FUNC_PARAMS)
{
    float4 color;
    uint src = bytesToInt<uint>(buffer, bufferOffset);

    color = unpack_snorm4x8_to_float(src);

    return color;
}
static inline void writeR8G8B8A8_SNORM(COMMON_WRITE_FLOAT_FUNC_PARAMS)
{
    uint dst = pack_float_to_snorm4x8(color);

    intToBytes(dst, bufferOffset, buffer);
}

// RGB8_SNORM
static inline float4 readR8G8B8_SNORM(COMMON_READ_FUNC_PARAMS)
{
    float4 color;
    color.r = normalizedToFloat<7, char>(buffer[bufferOffset]);
    color.g = normalizedToFloat<7, char>(buffer[bufferOffset + 1]);
    color.b = normalizedToFloat<7, char>(buffer[bufferOffset + 2]);
    color.a = 1.0;

    return color;
}
static inline void writeR8G8B8_SNORM(COMMON_WRITE_FLOAT_FUNC_PARAMS)
{
    uint dst = pack_float_to_snorm4x8(color);

    int24bitToBytes(dst, bufferOffset, buffer);
}

// RGBA8
static inline float4 readR8G8B8A8_UNORM(COMMON_READ_FUNC_PARAMS)
{
    return readR8G8B8A8(FORWARD_COMMON_READ_FUNC_PARAMS, false);
}
static inline void writeR8G8B8A8_UNORM(COMMON_WRITE_FLOAT_FUNC_PARAMS)
{
    return writeR8G8B8A8(FORWARD_COMMON_WRITE_FUNC_PARAMS, false);
}

static inline float4 readR8G8B8A8_UNORM_SRGB(COMMON_READ_FUNC_PARAMS)
{
    return readR8G8B8A8(FORWARD_COMMON_READ_FUNC_PARAMS, true);
}
static inline void writeR8G8B8A8_UNORM_SRGB(COMMON_WRITE_FLOAT_FUNC_PARAMS)
{
    return writeR8G8B8A8(FORWARD_COMMON_WRITE_FUNC_PARAMS, true);
}

// BGRA8
static inline float4 readB8G8R8A8_UNORM(COMMON_READ_FUNC_PARAMS)
{
    return readR8G8B8A8(FORWARD_COMMON_READ_FUNC_PARAMS, false).bgra;
}
static inline void writeB8G8R8A8_UNORM(COMMON_WRITE_FLOAT_FUNC_PARAMS)
{
    color.rgba = color.bgra;
    return writeR8G8B8A8(FORWARD_COMMON_WRITE_FUNC_PARAMS, false);
}

static inline float4 readB8G8R8A8_UNORM_SRGB(COMMON_READ_FUNC_PARAMS)
{
    return readR8G8B8A8(FORWARD_COMMON_READ_FUNC_PARAMS, true).bgra;
}
static inline void writeB8G8R8A8_UNORM_SRGB(COMMON_WRITE_FLOAT_FUNC_PARAMS)
{
    color.rgba = color.bgra;
    return writeR8G8B8A8(FORWARD_COMMON_WRITE_FUNC_PARAMS, true);
}

// RGB8
static inline float4 readR8G8B8_UNORM(COMMON_READ_FUNC_PARAMS)
{
    return readR8G8B8(FORWARD_COMMON_READ_FUNC_PARAMS, false);
}
static inline void writeR8G8B8_UNORM(COMMON_WRITE_FLOAT_FUNC_PARAMS)
{
    return writeR8G8B8(FORWARD_COMMON_WRITE_FUNC_PARAMS, false);
}

static inline float4 readR8G8B8_UNORM_SRGB(COMMON_READ_FUNC_PARAMS)
{
    return readR8G8B8(FORWARD_COMMON_READ_FUNC_PARAMS, true);
}
static inline void writeR8G8B8_UNORM_SRGB(COMMON_WRITE_FLOAT_FUNC_PARAMS)
{
    return writeR8G8B8(FORWARD_COMMON_WRITE_FUNC_PARAMS, true);
}

// L8
static inline float4 readL8_UNORM(COMMON_READ_FUNC_PARAMS)
{
    float4 color;
    color.rgb = float3(normalizedToFloat<uchar>(buffer[bufferOffset]));
    color.a   = 1.0;
    return color;
}
static inline void writeL8_UNORM(COMMON_WRITE_FLOAT_FUNC_PARAMS)
{
    buffer[bufferOffset] = floatToNormalized<uchar>(color.r);
}

// A8
static inline void writeA8_UNORM(COMMON_WRITE_FLOAT_FUNC_PARAMS)
{
    buffer[bufferOffset] = floatToNormalized<uchar>(color.a);
}

// L8A8
static inline float4 readL8A8_UNORM(COMMON_READ_FUNC_PARAMS)
{
    float4 color;
    color.rgb = float3(normalizedToFloat<uchar>(buffer[bufferOffset]));
    color.a   = normalizedToFloat<uchar>(buffer[bufferOffset + 1]);
    return color;
}
static inline void writeL8A8_UNORM(COMMON_WRITE_FLOAT_FUNC_PARAMS)
{
    buffer[bufferOffset]     = floatToNormalized<uchar>(color.r);
    buffer[bufferOffset + 1] = floatToNormalized<uchar>(color.a);
}

// R8
static inline float4 readR8_UNORM(COMMON_READ_FUNC_PARAMS)
{
    float4 color;
    color.r = normalizedToFloat<uchar>(buffer[bufferOffset]);
    color.g = color.b = 0.0;
    color.a           = 1.0;
    return color;
}
static inline void writeR8_UNORM(COMMON_WRITE_FLOAT_FUNC_PARAMS)
{
    buffer[bufferOffset] = floatToNormalized<uchar>(color.r);
}

static inline float4 readR8_SNORM(COMMON_READ_FUNC_PARAMS)
{
    float4 color;
    color.r = normalizedToFloat<7, char>(buffer[bufferOffset]);
    color.g = color.b = 0.0;
    color.a           = 1.0;
    return color;
}
static inline void writeR8_SNORM(COMMON_WRITE_FLOAT_FUNC_PARAMS)
{
    buffer[bufferOffset] = as_type<uchar>(floatToNormalized<7, char>(color.r));
}

// R8_SINT
static inline int4 readR8_SINT(COMMON_READ_FUNC_PARAMS)
{
    int4 color;
    color.r = as_type<char>(buffer[bufferOffset]);
    color.g = color.b = 0;
    color.a           = 1;
    return color;
}
static inline void writeR8_SINT(COMMON_WRITE_SINT_FUNC_PARAMS)
{
    buffer[bufferOffset] = static_cast<uchar>(color.r);
}

// R8_UINT
static inline uint4 readR8_UINT(COMMON_READ_FUNC_PARAMS)
{
    uint4 color;
    color.r = as_type<uchar>(buffer[bufferOffset]);
    color.g = color.b = 0;
    color.a           = 1;
    return color;
}
static inline void writeR8_UINT(COMMON_WRITE_UINT_FUNC_PARAMS)
{
    buffer[bufferOffset] = static_cast<uchar>(color.r);
}

// R8G8
static inline float4 readR8G8_UNORM(COMMON_READ_FUNC_PARAMS)
{
    float4 color;
    color.r = normalizedToFloat<uchar>(buffer[bufferOffset]);
    color.g = normalizedToFloat<uchar>(buffer[bufferOffset + 1]);
    color.b = 0.0;
    color.a = 1.0;
    return color;
}
static inline void writeR8G8_UNORM(COMMON_WRITE_FLOAT_FUNC_PARAMS)
{
    buffer[bufferOffset]     = floatToNormalized<uchar>(color.r);
    buffer[bufferOffset + 1] = floatToNormalized<uchar>(color.g);
}

static inline float4 readR8G8_SNORM(COMMON_READ_FUNC_PARAMS)
{
    float4 color;
    color.r = normalizedToFloat<7, char>(buffer[bufferOffset]);
    color.g = normalizedToFloat<7, char>(buffer[bufferOffset + 1]);
    color.b = 0.0;
    color.a = 1.0;
    return color;
}
static inline void writeR8G8_SNORM(COMMON_WRITE_FLOAT_FUNC_PARAMS)
{
    buffer[bufferOffset]     = as_type<uchar>(floatToNormalized<7, char>(color.r));
    buffer[bufferOffset + 1] = as_type<uchar>(floatToNormalized<7, char>(color.g));
}

// RG8_SINT
static inline int4 readR8G8_SINT(COMMON_READ_FUNC_PARAMS)
{
    int4 color;
    color.r = as_type<char>(buffer[bufferOffset]);
    color.g = as_type<char>(buffer[bufferOffset + 1]);
    color.b = 0;
    color.a = 1;
    return color;
}
static inline void writeR8G8_SINT(COMMON_WRITE_SINT_FUNC_PARAMS)
{
    buffer[bufferOffset]     = static_cast<uchar>(color.r);
    buffer[bufferOffset + 1] = static_cast<uchar>(color.g);
}

// RG8_UINT
static inline uint4 readR8G8_UINT(COMMON_READ_FUNC_PARAMS)
{
    uint4 color;
    color.r = as_type<uchar>(buffer[bufferOffset]);
    color.g = as_type<uchar>(buffer[bufferOffset + 1]);
    color.b = 0;
    color.a = 1;
    return color;
}
static inline void writeR8G8_UINT(COMMON_WRITE_UINT_FUNC_PARAMS)
{
    buffer[bufferOffset]     = static_cast<uchar>(color.r);
    buffer[bufferOffset + 1] = static_cast<uchar>(color.g);
}

// R8G8B8_SINT
static inline int4 readR8G8B8_SINT(COMMON_READ_FUNC_PARAMS)
{
    int4 color;
    color.r = as_type<char>(buffer[bufferOffset]);
    color.g = as_type<char>(buffer[bufferOffset + 1]);
    color.b = as_type<char>(buffer[bufferOffset + 2]);
    color.a = 1;
    return color;
}

// R8G8B8_UINT
static inline uint4 readR8G8B8_UINT(COMMON_READ_FUNC_PARAMS)
{
    uint4 color;
    color.r = as_type<uchar>(buffer[bufferOffset]);
    color.g = as_type<uchar>(buffer[bufferOffset + 1]);
    color.b = as_type<uchar>(buffer[bufferOffset + 2]);
    color.a = 1;
    return color;
}

// R8G8G8A8_SINT
static inline int4 readR8G8B8A8_SINT(COMMON_READ_FUNC_PARAMS)
{
    int4 color;
    color.r = as_type<char>(buffer[bufferOffset]);
    color.g = as_type<char>(buffer[bufferOffset + 1]);
    color.b = as_type<char>(buffer[bufferOffset + 2]);
    color.a = as_type<char>(buffer[bufferOffset + 3]);
    return color;
}
static inline void writeR8G8B8A8_SINT(COMMON_WRITE_SINT_FUNC_PARAMS)
{
    buffer[bufferOffset]     = static_cast<uchar>(color.r);
    buffer[bufferOffset + 1] = static_cast<uchar>(color.g);
    buffer[bufferOffset + 2] = static_cast<uchar>(color.b);
    buffer[bufferOffset + 3] = static_cast<uchar>(color.a);
}

// R8G8G8A8_UINT
static inline uint4 readR8G8B8A8_UINT(COMMON_READ_FUNC_PARAMS)
{
    uint4 color;
    color.r = as_type<uchar>(buffer[bufferOffset]);
    color.g = as_type<uchar>(buffer[bufferOffset + 1]);
    color.b = as_type<uchar>(buffer[bufferOffset + 2]);
    color.a = as_type<uchar>(buffer[bufferOffset + 3]);
    return color;
}
static inline void writeR8G8B8A8_UINT(COMMON_WRITE_UINT_FUNC_PARAMS)
{
    buffer[bufferOffset]     = static_cast<uchar>(color.r);
    buffer[bufferOffset + 1] = static_cast<uchar>(color.g);
    buffer[bufferOffset + 2] = static_cast<uchar>(color.b);
    buffer[bufferOffset + 3] = static_cast<uchar>(color.a);
}

// R16_FLOAT
static inline float4 readR16_FLOAT(COMMON_READ_FUNC_PARAMS)
{
    float4 color;
    color.r = as_type<half>(bytesToShort<ushort>(buffer, bufferOffset));
    color.g = color.b = 0.0;
    color.a           = 1.0;
    return color;
}
static inline void writeR16_FLOAT(COMMON_WRITE_FLOAT_FUNC_PARAMS)
{
    shortToBytes(as_type<ushort>(static_cast<half>(color.r)), bufferOffset, buffer);
}
// R16_NORM
template <typename ShortType>
static inline float4 readR16_NORM(COMMON_READ_FUNC_PARAMS)
{
    float4 color;
    color.r = normalizedToFloat<ShortType>(bytesToShort<ShortType>(buffer, bufferOffset));
    color.g = color.b = 0.0;
    color.a           = 1.0;
    return color;
}
#define readR16_SNORM readR16_NORM<short>
#define readR16_UNORM readR16_NORM<ushort>

// R16_SINT
static inline int4 readR16_SINT(COMMON_READ_FUNC_PARAMS)
{
    int4 color;
    color.r = bytesToShort<short>(buffer, bufferOffset);
    color.g = color.b = 0;
    color.a           = 1;
    return color;
}
static inline void writeR16_SINT(COMMON_WRITE_SINT_FUNC_PARAMS)
{
    shortToBytes(static_cast<short>(color.r), bufferOffset, buffer);
}

// R16_UINT
static inline uint4 readR16_UINT(COMMON_READ_FUNC_PARAMS)
{
    uint4 color;
    color.r = bytesToShort<ushort>(buffer, bufferOffset);
    color.g = color.b = 0;
    color.a           = 1;
    return color;
}
static inline void writeR16_UINT(COMMON_WRITE_UINT_FUNC_PARAMS)
{
    shortToBytes(static_cast<ushort>(color.r), bufferOffset, buffer);
}

// A16_FLOAT
static inline float4 readA16_FLOAT(COMMON_READ_FUNC_PARAMS)
{
    float4 color;
    color.a   = as_type<half>(bytesToShort<ushort>(buffer, bufferOffset));
    color.rgb = 0.0;
    return color;
}
static inline void writeA16_FLOAT(COMMON_WRITE_FLOAT_FUNC_PARAMS)
{
    shortToBytes(as_type<ushort>(static_cast<half>(color.a)), bufferOffset, buffer);
}

// L16_FLOAT
static inline float4 readL16_FLOAT(COMMON_READ_FUNC_PARAMS)
{
    float4 color;
    color.rgb = as_type<half>(bytesToShort<ushort>(buffer, bufferOffset));
    color.a   = 1.0;
    return color;
}
static inline void writeL16_FLOAT(COMMON_WRITE_FLOAT_FUNC_PARAMS)
{
    shortToBytes(as_type<ushort>(static_cast<half>(color.r)), bufferOffset, buffer);
}

// L16A16_FLOAT
static inline float4 readL16A16_FLOAT(COMMON_READ_FUNC_PARAMS)
{
    float4 color;
    color.rgb = as_type<half>(bytesToShort<ushort>(buffer, bufferOffset));
    color.a   = as_type<half>(bytesToShort<ushort>(buffer, bufferOffset + 2));
    return color;
}
static inline void writeL16A16_FLOAT(COMMON_WRITE_FLOAT_FUNC_PARAMS)
{
    shortToBytes(as_type<ushort>(static_cast<half>(color.r)), bufferOffset, buffer);
    shortToBytes(as_type<ushort>(static_cast<half>(color.a)), bufferOffset + 2, buffer);
}

// R16G16_FLOAT
static inline float4 readR16G16_FLOAT(COMMON_READ_FUNC_PARAMS)
{
    float4 color;
    color.r = as_type<half>(bytesToShort<ushort>(buffer, bufferOffset));
    color.g = as_type<half>(bytesToShort<ushort>(buffer, bufferOffset + 2));
    color.b = 0.0;
    color.a = 1.0;
    return color;
}
static inline void writeR16G16_FLOAT(COMMON_WRITE_FLOAT_FUNC_PARAMS)
{
    shortToBytes(as_type<ushort>(static_cast<half>(color.r)), bufferOffset, buffer);
    shortToBytes(as_type<ushort>(static_cast<half>(color.g)), bufferOffset + 2, buffer);
}

// R16G16_NORM
template <typename ShortType>
static inline float4 readR16G16_NORM(COMMON_READ_FUNC_PARAMS)
{
    float4 color;
    color.r = normalizedToFloat<ShortType>(bytesToShort<ShortType>(buffer, bufferOffset));
    color.g = normalizedToFloat<ShortType>(bytesToShort<ShortType>(buffer, bufferOffset + 2));
    color.b = 0.0;
    color.a = 1.0;
    return color;
}
#define readR16G16_SNORM readR16G16_NORM<short>
#define readR16G16_UNORM readR16G16_NORM<ushort>

// R16G16_SINT
static inline int4 readR16G16_SINT(COMMON_READ_FUNC_PARAMS)
{
    int4 color;
    color.r = bytesToShort<short>(buffer, bufferOffset);
    color.g = bytesToShort<short>(buffer, bufferOffset + 2);
    color.b = 0;
    color.a = 1;
    return color;
}
static inline void writeR16G16_SINT(COMMON_WRITE_SINT_FUNC_PARAMS)
{
    shortToBytes(static_cast<short>(color.r), bufferOffset, buffer);
    shortToBytes(static_cast<short>(color.g), bufferOffset + 2, buffer);
}

// R16G16_UINT
static inline uint4 readR16G16_UINT(COMMON_READ_FUNC_PARAMS)
{
    uint4 color;
    color.r = bytesToShort<ushort>(buffer, bufferOffset);
    color.g = bytesToShort<ushort>(buffer, bufferOffset + 2);
    color.b = 0;
    color.a = 1;
    return color;
}
static inline void writeR16G16_UINT(COMMON_WRITE_UINT_FUNC_PARAMS)
{
    shortToBytes(static_cast<ushort>(color.r), bufferOffset, buffer);
    shortToBytes(static_cast<ushort>(color.g), bufferOffset + 2, buffer);
}

// R16G16B16_FLOAT
static inline float4 readR16G16B16_FLOAT(COMMON_READ_FUNC_PARAMS)
{
    float4 color;
    color.r = as_type<half>(bytesToShort<ushort>(buffer, bufferOffset));
    color.g = as_type<half>(bytesToShort<ushort>(buffer, bufferOffset + 2));
    color.b = as_type<half>(bytesToShort<ushort>(buffer, bufferOffset + 4));
    color.a = 1.0;
    return color;
}

// R16G16B16_NORM
template <typename ShortType>
static inline float4 readR16G16B16_NORM(COMMON_READ_FUNC_PARAMS)
{
    float4 color;
    color.r = normalizedToFloat<ShortType>(bytesToShort<ShortType>(buffer, bufferOffset));
    color.g = normalizedToFloat<ShortType>(bytesToShort<ShortType>(buffer, bufferOffset + 2));
    color.b = normalizedToFloat<ShortType>(bytesToShort<ShortType>(buffer, bufferOffset + 4));
    color.a = 1.0;
    return color;
}
#define readR16G16B16_SNORM readR16G16B16_NORM<short>
#define readR16G16B16_UNORM readR16G16B16_NORM<ushort>
// R16G16B16_SINT
static inline int4 readR16G16B16_SINT(COMMON_READ_FUNC_PARAMS)
{
    int4 color;
    color.r = bytesToShort<short>(buffer, bufferOffset);
    color.g = bytesToShort<short>(buffer, bufferOffset + 2);
    color.b = bytesToShort<short>(buffer, bufferOffset + 4);
    color.a = 1;
    return color;
}

// R16G16B16_UINT
static inline uint4 readR16G16B16_UINT(COMMON_READ_FUNC_PARAMS)
{
    uint4 color;
    color.r = bytesToShort<ushort>(buffer, bufferOffset);
    color.g = bytesToShort<ushort>(buffer, bufferOffset + 2);
    color.b = bytesToShort<ushort>(buffer, bufferOffset + 4);
    color.a = 1;
    return color;
}

// R16G16B16A16_FLOAT
static inline float4 readR16G16B16A16_FLOAT(COMMON_READ_FUNC_PARAMS)
{
    float4 color;
    color.r = as_type<half>(bytesToShort<ushort>(buffer, bufferOffset));
    color.g = as_type<half>(bytesToShort<ushort>(buffer, bufferOffset + 2));
    color.b = as_type<half>(bytesToShort<ushort>(buffer, bufferOffset + 4));
    color.a = as_type<half>(bytesToShort<ushort>(buffer, bufferOffset + 6));
    return color;
}
static inline void writeR16G16B16A16_FLOAT(COMMON_WRITE_FLOAT_FUNC_PARAMS)
{
    shortToBytes(as_type<ushort>(static_cast<half>(color.r)), bufferOffset, buffer);
    shortToBytes(as_type<ushort>(static_cast<half>(color.g)), bufferOffset + 2, buffer);
    shortToBytes(as_type<ushort>(static_cast<half>(color.b)), bufferOffset + 4, buffer);
    shortToBytes(as_type<ushort>(static_cast<half>(color.a)), bufferOffset + 6, buffer);
}

// R16G16B16A16_NORM
template <typename ShortType>
static inline float4 readR16G16B16A16_NORM(COMMON_READ_FUNC_PARAMS)
{
    float4 color;
    color.r = normalizedToFloat<ShortType>(bytesToShort<ShortType>(buffer, bufferOffset));
    color.g = normalizedToFloat<ShortType>(bytesToShort<ShortType>(buffer, bufferOffset + 2));
    color.b = normalizedToFloat<ShortType>(bytesToShort<ShortType>(buffer, bufferOffset + 4));
    color.a = normalizedToFloat<ShortType>(bytesToShort<ShortType>(buffer, bufferOffset + 6));
    return color;
}
#define readR16G16B16A16_SNORM readR16G16B16A16_NORM<short>
#define readR16G16B16A16_UNORM readR16G16B16A16_NORM<ushort>

// R16G16B16A16_SINT
static inline int4 readR16G16B16A16_SINT(COMMON_READ_FUNC_PARAMS)
{
    int4 color;
    color.r = bytesToShort<short>(buffer, bufferOffset);
    color.g = bytesToShort<short>(buffer, bufferOffset + 2);
    color.b = bytesToShort<short>(buffer, bufferOffset + 4);
    color.a = bytesToShort<short>(buffer, bufferOffset + 6);
    return color;
}
static inline void writeR16G16B16A16_SINT(COMMON_WRITE_SINT_FUNC_PARAMS)
{
    shortToBytes(static_cast<short>(color.r), bufferOffset, buffer);
    shortToBytes(static_cast<short>(color.g), bufferOffset + 2, buffer);
    shortToBytes(static_cast<short>(color.b), bufferOffset + 4, buffer);
    shortToBytes(static_cast<short>(color.a), bufferOffset + 6, buffer);
}

// R16G16B16A16_UINT
static inline uint4 readR16G16B16A16_UINT(COMMON_READ_FUNC_PARAMS)
{
    uint4 color;
    color.r = bytesToShort<ushort>(buffer, bufferOffset);
    color.g = bytesToShort<ushort>(buffer, bufferOffset + 2);
    color.b = bytesToShort<ushort>(buffer, bufferOffset + 4);
    color.a = bytesToShort<ushort>(buffer, bufferOffset + 6);
    return color;
}
static inline void writeR16G16B16A16_UINT(COMMON_WRITE_UINT_FUNC_PARAMS)
{
    shortToBytes(static_cast<ushort>(color.r), bufferOffset, buffer);
    shortToBytes(static_cast<ushort>(color.g), bufferOffset + 2, buffer);
    shortToBytes(static_cast<ushort>(color.b), bufferOffset + 4, buffer);
    shortToBytes(static_cast<ushort>(color.a), bufferOffset + 6, buffer);
}

// R32_FLOAT
static inline float4 readR32_FLOAT(COMMON_READ_FUNC_PARAMS)
{
    float4 color;
    color.r = as_type<float>(bytesToInt<uint>(buffer, bufferOffset));
    color.g = color.b = 0.0;
    color.a           = 1.0;
    return color;
}
static inline void writeR32_FLOAT(COMMON_WRITE_FLOAT_FUNC_PARAMS)
{
    intToBytes(as_type<uint>(color.r), bufferOffset, buffer);
}

// R32_NORM
template <typename IntType>
static inline float4 readR32_NORM(COMMON_READ_FUNC_PARAMS)
{
    float4 color;
    color.r = normalizedToFloat<IntType>(bytesToInt<IntType>(buffer, bufferOffset));
    color.g = color.b = 0.0;
    color.a           = 1.0;
    return color;
}
#define readR32_SNORM readR32_NORM<int>
#define readR32_UNORM readR32_NORM<uint>

// A32_FLOAT
static inline float4 readA32_FLOAT(COMMON_READ_FUNC_PARAMS)
{
    float4 color;
    color.a   = as_type<float>(bytesToInt<uint>(buffer, bufferOffset));
    color.rgb = 0.0;
    return color;
}
static inline void writeA32_FLOAT(COMMON_WRITE_FLOAT_FUNC_PARAMS)
{
    intToBytes(as_type<uint>(color.a), bufferOffset, buffer);
}

// L32_FLOAT
static inline float4 readL32_FLOAT(COMMON_READ_FUNC_PARAMS)
{
    float4 color;
    color.rgb = as_type<float>(bytesToInt<uint>(buffer, bufferOffset));
    color.a   = 1.0;
    return color;
}
static inline void writeL32_FLOAT(COMMON_WRITE_FLOAT_FUNC_PARAMS)
{
    intToBytes(as_type<uint>(color.r), bufferOffset, buffer);
}

// R32_SINT
static inline int4 readR32_SINT(COMMON_READ_FUNC_PARAMS)
{
    int4 color;
    color.r = bytesToInt<int>(buffer, bufferOffset);
    color.g = color.b = 0;
    color.a           = 1;
    return color;
}
static inline void writeR32_SINT(COMMON_WRITE_SINT_FUNC_PARAMS)
{
    intToBytes(color.r, bufferOffset, buffer);
}

// R32_FIXED
static inline float4 readR32_FIXED(COMMON_READ_FUNC_PARAMS)
{
    float4 color;
    constexpr float kDivisor = 1.0f / (1 << 16);
    color.r                  = bytesToInt<int>(buffer, bufferOffset) * kDivisor;
    color.g = color.b = 0.0;
    color.a           = 1.0;
    return color;
}

// R32_UINT
static inline uint4 readR32_UINT(COMMON_READ_FUNC_PARAMS)
{
    uint4 color;
    color.r = bytesToInt<uint>(buffer, bufferOffset);
    color.g = color.b = 0;
    color.a           = 1;
    return color;
}
static inline void writeR32_UINT(COMMON_WRITE_UINT_FUNC_PARAMS)
{
    intToBytes(color.r, bufferOffset, buffer);
}

// L32A32_FLOAT
static inline float4 readL32A32_FLOAT(COMMON_READ_FUNC_PARAMS)
{
    float4 color;
    color.rgb = as_type<float>(bytesToInt<uint>(buffer, bufferOffset));
    color.a   = as_type<float>(bytesToInt<uint>(buffer, bufferOffset + 4));
    return color;
}
static inline void writeL32A32_FLOAT(COMMON_WRITE_FLOAT_FUNC_PARAMS)
{
    intToBytes(as_type<uint>(color.r), bufferOffset, buffer);
    intToBytes(as_type<uint>(color.a), bufferOffset + 4, buffer);
}

// R32G32_FLOAT
static inline float4 readR32G32_FLOAT(COMMON_READ_FUNC_PARAMS)
{
    float4 color;
    color.r = as_type<float>(bytesToInt<uint>(buffer, bufferOffset));
    color.g = as_type<float>(bytesToInt<uint>(buffer, bufferOffset + 4));
    color.b = 0.0;
    color.a = 1.0;
    return color;
}
static inline void writeR32G32_FLOAT(COMMON_WRITE_FLOAT_FUNC_PARAMS)
{
    intToBytes(as_type<uint>(color.r), bufferOffset, buffer);
    intToBytes(as_type<uint>(color.g), bufferOffset + 4, buffer);
}

// R32G32_NORM
template <typename IntType>
static inline float4 readR32G32_NORM(COMMON_READ_FUNC_PARAMS)
{
    float4 color;
    color.r = normalizedToFloat<IntType>(bytesToInt<IntType>(buffer, bufferOffset));
    color.g = normalizedToFloat<IntType>(bytesToInt<IntType>(buffer, bufferOffset + 4));
    color.b = 0.0;
    color.a = 1.0;
    return color;
}
#define readR32G32_SNORM readR32G32_NORM<int>
#define readR32G32_UNORM readR32G32_NORM<uint>

// R32G32_SINT
static inline int4 readR32G32_SINT(COMMON_READ_FUNC_PARAMS)
{
    int4 color;
    color.r = bytesToInt<int>(buffer, bufferOffset);
    color.g = bytesToInt<int>(buffer, bufferOffset + 4);
    color.b = 0;
    color.a = 1;
    return color;
}
static inline void writeR32G32_SINT(COMMON_WRITE_SINT_FUNC_PARAMS)
{
    intToBytes(color.r, bufferOffset, buffer);
    intToBytes(color.g, bufferOffset + 4, buffer);
}

// R32G32_FIXED
static inline float4 readR32G32_FIXED(COMMON_READ_FUNC_PARAMS)
{
    float4 color;
    constexpr float kDivisor = 1.0f / (1 << 16);
    color.r                  = bytesToInt<int>(buffer, bufferOffset) * kDivisor;
    color.g                  = bytesToInt<int>(buffer, bufferOffset + 4) * kDivisor;
    color.b                  = 0.0;
    color.a                  = 1.0;
    return color;
}

// R32G32_UINT
static inline uint4 readR32G32_UINT(COMMON_READ_FUNC_PARAMS)
{
    uint4 color;
    color.r = bytesToInt<uint>(buffer, bufferOffset);
    color.g = bytesToInt<uint>(buffer, bufferOffset + 4);
    color.b = 0;
    color.a = 1;
    return color;
}
static inline void writeR32G32_UINT(COMMON_WRITE_UINT_FUNC_PARAMS)
{
    intToBytes(color.r, bufferOffset, buffer);
    intToBytes(color.g, bufferOffset + 4, buffer);
}

// R32G32B32_FLOAT
static inline float4 readR32G32B32_FLOAT(COMMON_READ_FUNC_PARAMS)
{
    float4 color;
    color.r = as_type<float>(bytesToInt<uint>(buffer, bufferOffset));
    color.g = as_type<float>(bytesToInt<uint>(buffer, bufferOffset + 4));
    color.b = as_type<float>(bytesToInt<uint>(buffer, bufferOffset + 8));
    color.a = 1.0;
    return color;
}

// R32G32B32_NORM
template <typename IntType>
static inline float4 readR32G32B32_NORM(COMMON_READ_FUNC_PARAMS)
{
    float4 color;
    color.r = normalizedToFloat<IntType>(bytesToInt<IntType>(buffer, bufferOffset));
    color.g = normalizedToFloat<IntType>(bytesToInt<IntType>(buffer, bufferOffset + 4));
    color.b = normalizedToFloat<IntType>(bytesToInt<IntType>(buffer, bufferOffset + 8));
    color.a = 1.0;
    return color;
}
#define readR32G32B32_SNORM readR32G32B32_NORM<int>
#define readR32G32B32_UNORM readR32G32B32_NORM<uint>

// R32G32B32_SINT
static inline int4 readR32G32B32_SINT(COMMON_READ_FUNC_PARAMS)
{
    int4 color;
    color.r = bytesToInt<int>(buffer, bufferOffset);
    color.g = bytesToInt<int>(buffer, bufferOffset + 4);
    color.b = bytesToInt<int>(buffer, bufferOffset + 8);
    color.a = 1;
    return color;
}

// R32G32B32_FIXED
static inline float4 readR32G32B32_FIXED(COMMON_READ_FUNC_PARAMS)
{
    float4 color;
    constexpr float kDivisor = 1.0f / (1 << 16);
    color.r                  = bytesToInt<int>(buffer, bufferOffset) * kDivisor;
    color.g                  = bytesToInt<int>(buffer, bufferOffset + 4) * kDivisor;
    color.b                  = bytesToInt<int>(buffer, bufferOffset + 8) * kDivisor;
    color.a                  = 1.0;
    return color;
}

// R32G32B32_UINT
static inline uint4 readR32G32B32_UINT(COMMON_READ_FUNC_PARAMS)
{
    uint4 color;
    color.r = bytesToInt<uint>(buffer, bufferOffset);
    color.g = bytesToInt<uint>(buffer, bufferOffset + 4);
    color.b = bytesToInt<uint>(buffer, bufferOffset + 8);
    color.a = 1;
    return color;
}

// R32G32B32A32_FLOAT
static inline float4 readR32G32B32A32_FLOAT(COMMON_READ_FUNC_PARAMS)
{
    float4 color;
    color.r = as_type<float>(bytesToInt<uint>(buffer, bufferOffset));
    color.g = as_type<float>(bytesToInt<uint>(buffer, bufferOffset + 4));
    color.b = as_type<float>(bytesToInt<uint>(buffer, bufferOffset + 8));
    color.a = as_type<float>(bytesToInt<uint>(buffer, bufferOffset + 12));
    return color;
}
static inline void writeR32G32B32A32_FLOAT(COMMON_WRITE_FLOAT_FUNC_PARAMS)
{
    intToBytes(as_type<uint>(color.r), bufferOffset, buffer);
    intToBytes(as_type<uint>(color.g), bufferOffset + 4, buffer);
    intToBytes(as_type<uint>(color.b), bufferOffset + 8, buffer);
    intToBytes(as_type<uint>(color.a), bufferOffset + 12, buffer);
}

// R32G32B32A32_NORM
template <typename IntType>
static inline float4 readR32G32B32A32_NORM(COMMON_READ_FUNC_PARAMS)
{
    float4 color;
    color.r = normalizedToFloat<IntType>(bytesToInt<IntType>(buffer, bufferOffset));
    color.g = normalizedToFloat<IntType>(bytesToInt<IntType>(buffer, bufferOffset + 4));
    color.b = normalizedToFloat<IntType>(bytesToInt<IntType>(buffer, bufferOffset + 8));
    color.a = normalizedToFloat<IntType>(bytesToInt<IntType>(buffer, bufferOffset + 12));
    return color;
}
#define readR32G32B32A32_SNORM readR32G32B32A32_NORM<int>
#define readR32G32B32A32_UNORM readR32G32B32A32_NORM<uint>

// R32G32B32A32_SINT
static inline int4 readR32G32B32A32_SINT(COMMON_READ_FUNC_PARAMS)
{
    int4 color;
    color.r = bytesToInt<int>(buffer, bufferOffset);
    color.g = bytesToInt<int>(buffer, bufferOffset + 4);
    color.b = bytesToInt<int>(buffer, bufferOffset + 8);
    color.a = bytesToInt<int>(buffer, bufferOffset + 12);
    return color;
}
static inline void writeR32G32B32A32_SINT(COMMON_WRITE_SINT_FUNC_PARAMS)
{
    intToBytes(color.r, bufferOffset, buffer);
    intToBytes(color.g, bufferOffset + 4, buffer);
    intToBytes(color.b, bufferOffset + 8, buffer);
    intToBytes(color.a, bufferOffset + 12, buffer);
}
// R32G32B32A32_FIXED
static inline float4 readR32G32B32A32_FIXED(COMMON_READ_FUNC_PARAMS)
{
    float4 color;
    constexpr float kDivisor = 1.0f / (1 << 16);
    color.r                  = bytesToInt<int>(buffer, bufferOffset) * kDivisor;
    color.g                  = bytesToInt<int>(buffer, bufferOffset + 4) * kDivisor;
    color.b                  = bytesToInt<int>(buffer, bufferOffset + 8) * kDivisor;
    color.a                  = bytesToInt<int>(buffer, bufferOffset + 12) * kDivisor;
    return color;
}

// R32G32B32A32_UINT
static inline uint4 readR32G32B32A32_UINT(COMMON_READ_FUNC_PARAMS)
{
    uint4 color;
    color.r = bytesToInt<uint>(buffer, bufferOffset);
    color.g = bytesToInt<uint>(buffer, bufferOffset + 4);
    color.b = bytesToInt<uint>(buffer, bufferOffset + 8);
    color.a = bytesToInt<uint>(buffer, bufferOffset + 12);
    return color;
}
static inline void writeR32G32B32A32_UINT(COMMON_WRITE_UINT_FUNC_PARAMS)
{
    intToBytes(color.r, bufferOffset, buffer);
    intToBytes(color.g, bufferOffset + 4, buffer);
    intToBytes(color.b, bufferOffset + 8, buffer);
    intToBytes(color.a, bufferOffset + 12, buffer);
}

#define ALIAS_READ_SINT_FUNC(FORMAT)                                   \
    static inline int4 read##FORMAT##_SSCALED(COMMON_READ_FUNC_PARAMS) \
    {                                                                  \
        return read##FORMAT##_SINT(FORWARD_COMMON_READ_FUNC_PARAMS);   \
    }

#define ALIAS_READ_UINT_FUNC(FORMAT)                                    \
    static inline uint4 read##FORMAT##_USCALED(COMMON_READ_FUNC_PARAMS) \
    {                                                                   \
        return read##FORMAT##_UINT(FORWARD_COMMON_READ_FUNC_PARAMS);    \
    }

#define ALIAS_READ_INT_FUNC(FORMAT) \
    ALIAS_READ_SINT_FUNC(FORMAT)    \
    ALIAS_READ_UINT_FUNC(FORMAT)

#define ALIAS_READ_INT_FUNCS(BITS)                 \
    ALIAS_READ_INT_FUNC(R##BITS)                   \
    ALIAS_READ_INT_FUNC(R##BITS##G##BITS)          \
    ALIAS_READ_INT_FUNC(R##BITS##G##BITS##B##BITS) \
    ALIAS_READ_INT_FUNC(R##BITS##G##BITS##B##BITS##A##BITS)

ALIAS_READ_INT_FUNCS(8)
ALIAS_READ_INT_FUNCS(16)
ALIAS_READ_INT_FUNCS(32)

ALIAS_READ_INT_FUNC(R10G10B10A2)

// Copy pixels from buffer to texture
kernel void readFromBufferToFloatTexture(COMMON_READ_KERNEL_PARAMS(float))
{
    READ_KERNEL_GUARD

#define SUPPORTED_FORMATS(PROC) \
    PROC(R5G6B5_UNORM)          \
    PROC(R8G8B8A8_UNORM)        \
    PROC(R8G8B8A8_UNORM_SRGB)   \
    PROC(R8G8B8A8_SNORM)        \
    PROC(B8G8R8A8_UNORM)        \
    PROC(B8G8R8A8_UNORM_SRGB)   \
    PROC(R8G8B8_UNORM)          \
    PROC(R8G8B8_UNORM_SRGB)     \
    PROC(R8G8B8_SNORM)          \
    PROC(L8_UNORM)              \
    PROC(L8A8_UNORM)            \
    PROC(R5G5B5A1_UNORM)        \
    PROC(R4G4B4A4_UNORM)        \
    PROC(R8_UNORM)              \
    PROC(R8_SNORM)              \
    PROC(R8G8_UNORM)            \
    PROC(R8G8_SNORM)            \
    PROC(R16_FLOAT)             \
    PROC(A16_FLOAT)             \
    PROC(L16_FLOAT)             \
    PROC(L16A16_FLOAT)          \
    PROC(R16G16_FLOAT)          \
    PROC(R16G16B16_FLOAT)       \
    PROC(R16G16B16A16_FLOAT)    \
    PROC(R32_FLOAT)             \
    PROC(A32_FLOAT)             \
    PROC(L32_FLOAT)             \
    PROC(L32A32_FLOAT)          \
    PROC(R32G32_FLOAT)          \
    PROC(R32G32B32_FLOAT)       \
    PROC(R32G32B32A32_FLOAT)

    uint bufferOffset = CALC_BUFFER_READ_OFFSET(options.pixelSize);

    switch (kCopyFormatType)
    {
        SUPPORTED_FORMATS(READ_FORMAT_SWITCH_CASE)
    }

#undef SUPPORTED_FORMATS
}

kernel void readFromBufferToIntTexture(COMMON_READ_KERNEL_PARAMS(int))
{
    READ_KERNEL_GUARD

#define SUPPORTED_FORMATS(PROC) \
    PROC(R8_SINT)               \
    PROC(R8G8_SINT)             \
    PROC(R8G8B8_SINT)           \
    PROC(R8G8B8A8_SINT)         \
    PROC(R16_SINT)              \
    PROC(R16G16_SINT)           \
    PROC(R16G16B16_SINT)        \
    PROC(R16G16B16A16_SINT)     \
    PROC(R32_SINT)              \
    PROC(R32G32_SINT)           \
    PROC(R32G32B32_SINT)        \
    PROC(R32G32B32A32_SINT)

    uint bufferOffset = CALC_BUFFER_READ_OFFSET(options.pixelSize);

    switch (kCopyFormatType)
    {
        SUPPORTED_FORMATS(READ_FORMAT_SWITCH_CASE)
    }

#undef SUPPORTED_FORMATS
}

kernel void readFromBufferToUIntTexture(COMMON_READ_KERNEL_PARAMS(uint))
{
    READ_KERNEL_GUARD

#define SUPPORTED_FORMATS(PROC) \
    PROC(R8_UINT)               \
    PROC(R8G8_UINT)             \
    PROC(R8G8B8_UINT)           \
    PROC(R8G8B8A8_UINT)         \
    PROC(R16_UINT)              \
    PROC(R16G16_UINT)           \
    PROC(R16G16B16_UINT)        \
    PROC(R16G16B16A16_UINT)     \
    PROC(R32_UINT)              \
    PROC(R32G32_UINT)           \
    PROC(R32G32B32_UINT)        \
    PROC(R32G32B32A32_UINT)

    uint bufferOffset = CALC_BUFFER_READ_OFFSET(options.pixelSize);

    switch (kCopyFormatType)
    {
        SUPPORTED_FORMATS(READ_FORMAT_SWITCH_CASE)
    }

#undef SUPPORTED_FORMATS
}

// Copy pixels from texture to buffer
kernel void writeFromFloatTextureToBuffer(COMMON_WRITE_KERNEL_PARAMS(float))
{
    WRITE_KERNEL_GUARD

#define SUPPORTED_FORMATS(PROC) \
    PROC(R5G6B5_UNORM)          \
    PROC(R8G8B8A8_UNORM)        \
    PROC(R8G8B8A8_UNORM_SRGB)   \
    PROC(R8G8B8A8_SNORM)        \
    PROC(B8G8R8A8_UNORM)        \
    PROC(B8G8R8A8_UNORM_SRGB)   \
    PROC(R8G8B8_UNORM)          \
    PROC(R8G8B8_UNORM_SRGB)     \
    PROC(R8G8B8_SNORM)          \
    PROC(L8_UNORM)              \
    PROC(A8_UNORM)              \
    PROC(L8A8_UNORM)            \
    PROC(R5G5B5A1_UNORM)        \
    PROC(R4G4B4A4_UNORM)        \
    PROC(R8_UNORM)              \
    PROC(R8_SNORM)              \
    PROC(R8G8_UNORM)            \
    PROC(R8G8_SNORM)            \
    PROC(R16_FLOAT)             \
    PROC(A16_FLOAT)             \
    PROC(L16_FLOAT)             \
    PROC(L16A16_FLOAT)          \
    PROC(R16G16_FLOAT)          \
    PROC(R16G16B16A16_FLOAT)    \
    PROC(R32_FLOAT)             \
    PROC(A32_FLOAT)             \
    PROC(L32_FLOAT)             \
    PROC(L32A32_FLOAT)          \
    PROC(R32G32_FLOAT)          \
    PROC(R32G32B32A32_FLOAT)

    uint bufferOffset = CALC_BUFFER_WRITE_OFFSET(options.pixelSize);

    switch (kCopyFormatType)
    {
        SUPPORTED_FORMATS(WRITE_FORMAT_SWITCH_CASE)
    }

#undef SUPPORTED_FORMATS
}

kernel void writeFromIntTextureToBuffer(COMMON_WRITE_KERNEL_PARAMS(int))
{
    WRITE_KERNEL_GUARD

#define SUPPORTED_FORMATS(PROC) \
    PROC(R8_SINT)               \
    PROC(R8G8_SINT)             \
    PROC(R8G8B8A8_SINT)         \
    PROC(R16_SINT)              \
    PROC(R16G16_SINT)           \
    PROC(R16G16B16A16_SINT)     \
    PROC(R32_SINT)              \
    PROC(R32G32_SINT)           \
    PROC(R32G32B32A32_SINT)

    uint bufferOffset = CALC_BUFFER_WRITE_OFFSET(options.pixelSize);

    switch (kCopyFormatType)
    {
        SUPPORTED_FORMATS(WRITE_FORMAT_SWITCH_CASE)
    }

#undef SUPPORTED_FORMATS
}

kernel void writeFromUIntTextureToBuffer(COMMON_WRITE_KERNEL_PARAMS(uint))
{
    WRITE_KERNEL_GUARD

#define SUPPORTED_FORMATS(PROC) \
    PROC(R8_UINT)               \
    PROC(R8G8_UINT)             \
    PROC(R8G8B8A8_UINT)         \
    PROC(R16_UINT)              \
    PROC(R16G16_UINT)           \
    PROC(R16G16B16A16_UINT)     \
    PROC(R32_UINT)              \
    PROC(R32G32_UINT)           \
    PROC(R32G32B32A32_UINT)

    uint bufferOffset = CALC_BUFFER_WRITE_OFFSET(options.pixelSize);

    switch (kCopyFormatType)
    {
        SUPPORTED_FORMATS(WRITE_FORMAT_SWITCH_CASE)
    }

#undef SUPPORTED_FORMATS
}

/** -----  vertex format conversion --------*/
struct CopyVertexParams
{
    uint srcBufferStartOffset;
    uint srcStride;
    uint srcComponentBytes;  // unused when convert to float
    uint srcComponents;      // unused when convert to float
    // Default source alpha when expanding the number of components.
    // if source has less than 32 bits per component, only those bits are usable in
    // srcDefaultAlpha
    uchar4 srcDefaultAlphaData;  // unused when convert to float

    uint dstBufferStartOffset;
    uint dstStride;
    uint dstComponents;

    uint vertexCount;
};

#define INT_FORMAT_PROC(FORMAT, PROC) \
    PROC(FORMAT##_UNORM)              \
    PROC(FORMAT##_SNORM)              \
    PROC(FORMAT##_UINT)               \
    PROC(FORMAT##_SINT)               \
    PROC(FORMAT##_USCALED)            \
    PROC(FORMAT##_SSCALED)

#define PURE_INT_FORMAT_PROC(FORMAT, PROC) \
    PROC(FORMAT##_UINT)                    \
    PROC(FORMAT##_SINT)

#define FLOAT_FORMAT_PROC(FORMAT, PROC) PROC(FORMAT##_FLOAT)
#define FIXED_FORMAT_PROC(FORMAT, PROC) PROC(FORMAT##_FIXED)

#define FORMAT_BITS_PROC(BITS, PROC1, PROC2) \
    PROC1(R##BITS, PROC2)                    \
    PROC1(R##BITS##G##BITS, PROC2)           \
    PROC1(R##BITS##G##BITS##B##BITS, PROC2)  \
    PROC1(R##BITS##G##BITS##B##BITS##A##BITS, PROC2)

template <typename IntType>
static inline void writeFloatVertex(constant CopyVertexParams &options,
                                    uint idx,
                                    vec<IntType, 4> data,
                                    device uchar *dst)
{
    uint dstOffset = idx * options.dstStride + options.dstBufferStartOffset;

    for (uint component = 0; component < options.dstComponents; ++component, dstOffset += 4)
    {
        floatToBytes(static_cast<float>(data[component]), dstOffset, dst);
    }
}

template <>
inline void writeFloatVertex(constant CopyVertexParams &options,
                             uint idx,
                             vec<float, 4> data,
                             device uchar *dst)
{
    uint dstOffset = idx * options.dstStride + options.dstBufferStartOffset;

    for (uint component = 0; component < options.dstComponents; ++component, dstOffset += 4)
    {
        floatToBytes(data[component], dstOffset, dst);
    }
}

// Function to convert from any vertex format to float vertex format
static inline void convertToFloatVertexFormat(uint index,
                                              constant CopyVertexParams &options,
                                              constant uchar *srcBuffer,
                                              device uchar *dstBuffer)
{
#define SUPPORTED_FORMATS(PROC)                   \
    FORMAT_BITS_PROC(8, INT_FORMAT_PROC, PROC)    \
    FORMAT_BITS_PROC(16, INT_FORMAT_PROC, PROC)   \
    FORMAT_BITS_PROC(32, INT_FORMAT_PROC, PROC)   \
    FORMAT_BITS_PROC(16, FLOAT_FORMAT_PROC, PROC) \
    FORMAT_BITS_PROC(32, FLOAT_FORMAT_PROC, PROC) \
    FORMAT_BITS_PROC(32, FIXED_FORMAT_PROC, PROC) \
    PROC(R10G10B10A2_SINT)                        \
    PROC(R10G10B10A2_UINT)                        \
    PROC(R10G10B10A2_SSCALED)                     \
    PROC(R10G10B10A2_USCALED)

    uint bufferOffset = options.srcBufferStartOffset + options.srcStride * index;

#define COMVERT_FLOAT_VERTEX_SWITCH_CASE(FORMAT)           \
    case FormatID::FORMAT: {                               \
        auto data = read##FORMAT(bufferOffset, srcBuffer); \
        writeFloatVertex(options, index, data, dstBuffer); \
    }                                                      \
    break;

    switch (kCopyFormatType)
    {
        SUPPORTED_FORMATS(COMVERT_FLOAT_VERTEX_SWITCH_CASE)
    }

#undef SUPPORTED_FORMATS
}

// Kernel to convert from any vertex format to float vertex format
kernel void convertToFloatVertexFormatCS(uint index [[thread_position_in_grid]],
                                         constant CopyVertexParams &options [[buffer(0)]],
                                         constant uchar *srcBuffer [[buffer(1)]],
                                         device uchar *dstBuffer [[buffer(2)]])
{
    ANGLE_KERNEL_GUARD(index, options.vertexCount);
    convertToFloatVertexFormat(index, options, srcBuffer, dstBuffer);
}

// Vertex shader to convert from any vertex format to float vertex format
vertex void convertToFloatVertexFormatVS(uint index [[vertex_id]],
                                         constant CopyVertexParams &options [[buffer(0)]],
                                         constant uchar *srcBuffer [[buffer(1)]],
                                         device uchar *dstBuffer [[buffer(2)]])
{
    convertToFloatVertexFormat(index, options, srcBuffer, dstBuffer);
}

// Function to expand (or just simply copy) the components of the vertex
static inline void expandVertexFormatComponents(uint index,
                                                constant CopyVertexParams &options,
                                                constant uchar *srcBuffer,
                                                device uchar *dstBuffer)
{
    uint srcOffset = options.srcBufferStartOffset + options.srcStride * index;
    uint dstOffset = options.dstBufferStartOffset + options.dstStride * index;

    uint dstComponentsBeforeAlpha = min(options.dstComponents, 3u);
    uint component;
    for (component = 0; component < options.srcComponents; ++component,
        srcOffset += options.srcComponentBytes, dstOffset += options.srcComponentBytes)
    {
        for (uint byte = 0; byte < options.srcComponentBytes; ++byte)
        {
            dstBuffer[dstOffset + byte] = srcBuffer[srcOffset + byte];
        }
    }

    for (; component < dstComponentsBeforeAlpha;
         ++component, dstOffset += options.srcComponentBytes)
    {
        for (uint byte = 0; byte < options.srcComponentBytes; ++byte)
        {
            dstBuffer[dstOffset + byte] = 0;
        }
    }

    if (component < options.dstComponents)
    {
        // Last alpha component
        for (uint byte = 0; byte < options.srcComponentBytes; ++byte)
        {
            dstBuffer[dstOffset + byte] = options.srcDefaultAlphaData[byte];
        }
    }
}

// Kernel to expand (or just simply copy) the components of the vertex
kernel void expandVertexFormatComponentsCS(uint index [[thread_position_in_grid]],
                                           constant CopyVertexParams &options [[buffer(0)]],
                                           constant uchar *srcBuffer [[buffer(1)]],
                                           device uchar *dstBuffer [[buffer(2)]])
{
    ANGLE_KERNEL_GUARD(index, options.vertexCount);

    expandVertexFormatComponents(index, options, srcBuffer, dstBuffer);
}

// Vertex shader to expand (or just simply copy) the components of the vertex
vertex void expandVertexFormatComponentsVS(uint index [[vertex_id]],
                                           constant CopyVertexParams &options [[buffer(0)]],
                                           constant uchar *srcBuffer [[buffer(1)]],
                                           device uchar *dstBuffer [[buffer(2)]])
{
    expandVertexFormatComponents(index, options, srcBuffer, dstBuffer);
}