/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/

#include "SDL_internal.h"

#ifdef SDL_GPU_METAL

#include <Metal/Metal.h>
#include <QuartzCore/CoreAnimation.h>

#include "../SDL_sysgpu.h"

// Defines

#define METAL_FIRST_VERTEX_BUFFER_SLOT 14
#define WINDOW_PROPERTY_DATA           "SDL_GPUMetalWindowPropertyData"
#define SDL_GPU_SHADERSTAGE_COMPUTE    2

#define TRACK_RESOURCE(resource, type, array, count, capacity)   \
    do {                                                         \
        Uint32 i;                                                \
                                                                 \
        for (i = 0; i < commandBuffer->count; i += 1) {          \
            if (commandBuffer->array[i] == (resource)) {         \
                return;                                          \
            }                                                    \
        }                                                        \
                                                                 \
        if (commandBuffer->count == commandBuffer->capacity) {   \
            commandBuffer->capacity += 1;                        \
            commandBuffer->array = SDL_realloc(                  \
                commandBuffer->array,                            \
                commandBuffer->capacity * sizeof(type));         \
        }                                                        \
        commandBuffer->array[commandBuffer->count] = (resource); \
        commandBuffer->count += 1;                               \
        SDL_AtomicIncRef(&(resource)->referenceCount);           \
    } while (0)

#define SET_ERROR_AND_RETURN(fmt, msg, ret)               \
    do {                                                  \
        if (renderer->debugMode) {                        \
            SDL_LogError(SDL_LOG_CATEGORY_GPU, fmt, msg); \
        }                                                 \
        SDL_SetError(fmt, msg);                           \
        return ret;                                       \
    } while (0)

#define SET_STRING_ERROR_AND_RETURN(msg, ret) SET_ERROR_AND_RETURN("%s", msg, ret)

// Blit Shaders

#include "Metal_Blit.h"

// Forward Declarations

static bool METAL_Wait(SDL_GPURenderer *driverData);
static void METAL_ReleaseWindow(
    SDL_GPURenderer *driverData,
    SDL_Window *window);
static void METAL_INTERNAL_DestroyBlitResources(SDL_GPURenderer *driverData);

// Conversions

#define RETURN_FORMAT(availability, format) \
    if (availability) { return format; } else { return MTLPixelFormatInvalid; }

static MTLPixelFormat SDLToMetal_TextureFormat(SDL_GPUTextureFormat format)
{
    switch (format) {
        case SDL_GPU_TEXTUREFORMAT_INVALID: return MTLPixelFormatInvalid;
        case SDL_GPU_TEXTUREFORMAT_A8_UNORM: return MTLPixelFormatA8Unorm;
        case SDL_GPU_TEXTUREFORMAT_R8_UNORM: return MTLPixelFormatR8Unorm;
        case SDL_GPU_TEXTUREFORMAT_R8G8_UNORM: return MTLPixelFormatRG8Unorm;
        case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM: return MTLPixelFormatRGBA8Unorm;
        case SDL_GPU_TEXTUREFORMAT_R16_UNORM: return MTLPixelFormatR16Unorm;
        case SDL_GPU_TEXTUREFORMAT_R16G16_UNORM: return MTLPixelFormatRG16Unorm;
        case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_UNORM: return MTLPixelFormatRGBA16Unorm;
        case SDL_GPU_TEXTUREFORMAT_R10G10B10A2_UNORM: return MTLPixelFormatRGB10A2Unorm;
        case SDL_GPU_TEXTUREFORMAT_B5G6R5_UNORM: RETURN_FORMAT(@available(macOS 11.0, *), MTLPixelFormatB5G6R5Unorm);
        case SDL_GPU_TEXTUREFORMAT_B5G5R5A1_UNORM: RETURN_FORMAT(@available(macOS 11.0, *), MTLPixelFormatBGR5A1Unorm);
        case SDL_GPU_TEXTUREFORMAT_B4G4R4A4_UNORM: RETURN_FORMAT(@available(macOS 11.0, *), MTLPixelFormatABGR4Unorm);
        case SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM: return MTLPixelFormatBGRA8Unorm;
        case SDL_GPU_TEXTUREFORMAT_BC1_RGBA_UNORM: RETURN_FORMAT(@available(iOS 16.4, tvOS 16.4, *), MTLPixelFormatBC1_RGBA);
        case SDL_GPU_TEXTUREFORMAT_BC2_RGBA_UNORM: RETURN_FORMAT(@available(iOS 16.4, tvOS 16.4, *), MTLPixelFormatBC2_RGBA);
        case SDL_GPU_TEXTUREFORMAT_BC3_RGBA_UNORM: RETURN_FORMAT(@available(iOS 16.4, tvOS 16.4, *), MTLPixelFormatBC3_RGBA);
        case SDL_GPU_TEXTUREFORMAT_BC4_R_UNORM: RETURN_FORMAT(@available(iOS 16.4, tvOS 16.4, *), MTLPixelFormatBC4_RUnorm);
        case SDL_GPU_TEXTUREFORMAT_BC5_RG_UNORM: RETURN_FORMAT(@available(iOS 16.4, tvOS 16.4, *), MTLPixelFormatBC5_RGUnorm);
        case SDL_GPU_TEXTUREFORMAT_BC7_RGBA_UNORM: RETURN_FORMAT(@available(iOS 16.4, tvOS 16.4, *), MTLPixelFormatBC7_RGBAUnorm);
        case SDL_GPU_TEXTUREFORMAT_BC6H_RGB_FLOAT: RETURN_FORMAT(@available(iOS 16.4, tvOS 16.4, *), MTLPixelFormatBC6H_RGBFloat);
        case SDL_GPU_TEXTUREFORMAT_BC6H_RGB_UFLOAT: RETURN_FORMAT(@available(iOS 16.4, tvOS 16.4, *), MTLPixelFormatBC6H_RGBUfloat);
        case SDL_GPU_TEXTUREFORMAT_R8_SNORM: return MTLPixelFormatR8Snorm;
        case SDL_GPU_TEXTUREFORMAT_R8G8_SNORM: return MTLPixelFormatRG8Snorm;
        case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_SNORM: return MTLPixelFormatRGBA8Snorm;
        case SDL_GPU_TEXTUREFORMAT_R16_SNORM: return MTLPixelFormatR16Snorm;
        case SDL_GPU_TEXTUREFORMAT_R16G16_SNORM: return MTLPixelFormatRG16Snorm;
        case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_SNORM: return MTLPixelFormatRGBA16Snorm;
        case SDL_GPU_TEXTUREFORMAT_R16_FLOAT: return MTLPixelFormatR16Float;
        case SDL_GPU_TEXTUREFORMAT_R16G16_FLOAT: return MTLPixelFormatRG16Float;
        case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT: return MTLPixelFormatRGBA16Float;
        case SDL_GPU_TEXTUREFORMAT_R32_FLOAT: return MTLPixelFormatR32Float;
        case SDL_GPU_TEXTUREFORMAT_R32G32_FLOAT: return MTLPixelFormatRG32Float;
        case SDL_GPU_TEXTUREFORMAT_R32G32B32A32_FLOAT: return MTLPixelFormatRGBA32Float;
        case SDL_GPU_TEXTUREFORMAT_R11G11B10_UFLOAT: return MTLPixelFormatRG11B10Float;
        case SDL_GPU_TEXTUREFORMAT_R8_UINT: return MTLPixelFormatR8Uint;
        case SDL_GPU_TEXTUREFORMAT_R8G8_UINT: return MTLPixelFormatRG8Uint;
        case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UINT: return MTLPixelFormatRGBA8Uint;
        case SDL_GPU_TEXTUREFORMAT_R16_UINT: return MTLPixelFormatR16Uint;
        case SDL_GPU_TEXTUREFORMAT_R16G16_UINT: return MTLPixelFormatRG16Uint;
        case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_UINT: return MTLPixelFormatRGBA16Uint;
        case SDL_GPU_TEXTUREFORMAT_R32_UINT: return MTLPixelFormatR32Uint;
        case SDL_GPU_TEXTUREFORMAT_R32G32_UINT: return MTLPixelFormatRG32Uint;
        case SDL_GPU_TEXTUREFORMAT_R32G32B32A32_UINT: return MTLPixelFormatRGBA32Uint;
        case SDL_GPU_TEXTUREFORMAT_R8_INT: return MTLPixelFormatR8Sint;
        case SDL_GPU_TEXTUREFORMAT_R8G8_INT: return MTLPixelFormatRG8Sint;
        case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_INT: return MTLPixelFormatRGBA8Sint;
        case SDL_GPU_TEXTUREFORMAT_R16_INT: return MTLPixelFormatR16Sint;
        case SDL_GPU_TEXTUREFORMAT_R16G16_INT: return MTLPixelFormatRG16Sint;
        case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_INT: return MTLPixelFormatRGBA16Sint;
        case SDL_GPU_TEXTUREFORMAT_R32_INT: return MTLPixelFormatR32Sint;
        case SDL_GPU_TEXTUREFORMAT_R32G32_INT: return MTLPixelFormatRG32Sint;
        case SDL_GPU_TEXTUREFORMAT_R32G32B32A32_INT: return MTLPixelFormatRGBA32Sint;
        case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM_SRGB: return MTLPixelFormatRGBA8Unorm_sRGB;
        case SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM_SRGB: return MTLPixelFormatBGRA8Unorm_sRGB;
        case SDL_GPU_TEXTUREFORMAT_BC1_RGBA_UNORM_SRGB: RETURN_FORMAT(@available(iOS 16.4, tvOS 16.4, *), MTLPixelFormatBC1_RGBA_sRGB);
        case SDL_GPU_TEXTUREFORMAT_BC2_RGBA_UNORM_SRGB: RETURN_FORMAT(@available(iOS 16.4, tvOS 16.4, *), MTLPixelFormatBC2_RGBA_sRGB);
        case SDL_GPU_TEXTUREFORMAT_BC3_RGBA_UNORM_SRGB: RETURN_FORMAT(@available(iOS 16.4, tvOS 16.4, *), MTLPixelFormatBC3_RGBA_sRGB);
        case SDL_GPU_TEXTUREFORMAT_BC7_RGBA_UNORM_SRGB: RETURN_FORMAT(@available(iOS 16.4, tvOS 16.4, *), MTLPixelFormatBC7_RGBAUnorm_sRGB);
        case SDL_GPU_TEXTUREFORMAT_D16_UNORM: RETURN_FORMAT(@available(iOS 13.0, tvOS 13.0, *), MTLPixelFormatDepth16Unorm);
        case SDL_GPU_TEXTUREFORMAT_D24_UNORM:
#ifdef SDL_PLATFORM_MACOS
            return MTLPixelFormatDepth24Unorm_Stencil8;
#else
            return MTLPixelFormatInvalid;
#endif
        case SDL_GPU_TEXTUREFORMAT_D32_FLOAT: return MTLPixelFormatDepth32Float;
        case SDL_GPU_TEXTUREFORMAT_D24_UNORM_S8_UINT:
#ifdef SDL_PLATFORM_MACOS
            return MTLPixelFormatDepth24Unorm_Stencil8;
#else
            return MTLPixelFormatInvalid;
#endif
        case SDL_GPU_TEXTUREFORMAT_D32_FLOAT_S8_UINT: return MTLPixelFormatDepth32Float_Stencil8;
        case SDL_GPU_TEXTUREFORMAT_ASTC_4x4_UNORM: RETURN_FORMAT(@available(macOS 11.0, *), MTLPixelFormatASTC_4x4_LDR);
        case SDL_GPU_TEXTUREFORMAT_ASTC_5x4_UNORM: RETURN_FORMAT(@available(macOS 11.0, *), MTLPixelFormatASTC_5x4_LDR);
        case SDL_GPU_TEXTUREFORMAT_ASTC_5x5_UNORM: RETURN_FORMAT(@available(macOS 11.0, *), MTLPixelFormatASTC_5x5_LDR);
        case SDL_GPU_TEXTUREFORMAT_ASTC_6x5_UNORM: RETURN_FORMAT(@available(macOS 11.0, *), MTLPixelFormatASTC_6x5_LDR);
        case SDL_GPU_TEXTUREFORMAT_ASTC_6x6_UNORM: RETURN_FORMAT(@available(macOS 11.0, *), MTLPixelFormatASTC_6x6_LDR);
        case SDL_GPU_TEXTUREFORMAT_ASTC_8x5_UNORM: RETURN_FORMAT(@available(macOS 11.0, *), MTLPixelFormatASTC_8x5_LDR);
        case SDL_GPU_TEXTUREFORMAT_ASTC_8x6_UNORM: RETURN_FORMAT(@available(macOS 11.0, *), MTLPixelFormatASTC_8x6_LDR);
        case SDL_GPU_TEXTUREFORMAT_ASTC_8x8_UNORM: RETURN_FORMAT(@available(macOS 11.0, *), MTLPixelFormatASTC_8x8_LDR);
        case SDL_GPU_TEXTUREFORMAT_ASTC_10x5_UNORM: RETURN_FORMAT(@available(macOS 11.0, *), MTLPixelFormatASTC_10x5_LDR);
        case SDL_GPU_TEXTUREFORMAT_ASTC_10x6_UNORM: RETURN_FORMAT(@available(macOS 11.0, *), MTLPixelFormatASTC_10x6_LDR);
        case SDL_GPU_TEXTUREFORMAT_ASTC_10x8_UNORM: RETURN_FORMAT(@available(macOS 11.0, *), MTLPixelFormatASTC_10x8_LDR);
        case SDL_GPU_TEXTUREFORMAT_ASTC_10x10_UNORM: RETURN_FORMAT(@available(macOS 11.0, *), MTLPixelFormatASTC_10x10_LDR);
        case SDL_GPU_TEXTUREFORMAT_ASTC_12x10_UNORM: RETURN_FORMAT(@available(macOS 11.0, *), MTLPixelFormatASTC_12x10_LDR);
        case SDL_GPU_TEXTUREFORMAT_ASTC_12x12_UNORM: RETURN_FORMAT(@available(macOS 11.0, *), MTLPixelFormatASTC_12x12_LDR);
        case SDL_GPU_TEXTUREFORMAT_ASTC_4x4_UNORM_SRGB: RETURN_FORMAT(@available(macOS 11.0, *), MTLPixelFormatASTC_4x4_sRGB);
        case SDL_GPU_TEXTUREFORMAT_ASTC_5x4_UNORM_SRGB: RETURN_FORMAT(@available(macOS 11.0, *), MTLPixelFormatASTC_5x4_sRGB);
        case SDL_GPU_TEXTUREFORMAT_ASTC_5x5_UNORM_SRGB: RETURN_FORMAT(@available(macOS 11.0, *), MTLPixelFormatASTC_5x5_sRGB);
        case SDL_GPU_TEXTUREFORMAT_ASTC_6x5_UNORM_SRGB: RETURN_FORMAT(@available(macOS 11.0, *), MTLPixelFormatASTC_6x5_sRGB);
        case SDL_GPU_TEXTUREFORMAT_ASTC_6x6_UNORM_SRGB: RETURN_FORMAT(@available(macOS 11.0, *), MTLPixelFormatASTC_6x6_sRGB);
        case SDL_GPU_TEXTUREFORMAT_ASTC_8x5_UNORM_SRGB: RETURN_FORMAT(@available(macOS 11.0, *), MTLPixelFormatASTC_8x5_sRGB);
        case SDL_GPU_TEXTUREFORMAT_ASTC_8x6_UNORM_SRGB: RETURN_FORMAT(@available(macOS 11.0, *), MTLPixelFormatASTC_8x6_sRGB);
        case SDL_GPU_TEXTUREFORMAT_ASTC_8x8_UNORM_SRGB: RETURN_FORMAT(@available(macOS 11.0, *), MTLPixelFormatASTC_8x8_sRGB);
        case SDL_GPU_TEXTUREFORMAT_ASTC_10x5_UNORM_SRGB: RETURN_FORMAT(@available(macOS 11.0, *), MTLPixelFormatASTC_10x5_sRGB);
        case SDL_GPU_TEXTUREFORMAT_ASTC_10x6_UNORM_SRGB: RETURN_FORMAT(@available(macOS 11.0, *), MTLPixelFormatASTC_10x6_sRGB);
        case SDL_GPU_TEXTUREFORMAT_ASTC_10x8_UNORM_SRGB: RETURN_FORMAT(@available(macOS 11.0, *), MTLPixelFormatASTC_10x8_sRGB);
        case SDL_GPU_TEXTUREFORMAT_ASTC_10x10_UNORM_SRGB: RETURN_FORMAT(@available(macOS 11.0, *), MTLPixelFormatASTC_10x10_sRGB);
        case SDL_GPU_TEXTUREFORMAT_ASTC_12x10_UNORM_SRGB: RETURN_FORMAT(@available(macOS 11.0, *), MTLPixelFormatASTC_12x10_sRGB);
        case SDL_GPU_TEXTUREFORMAT_ASTC_12x12_UNORM_SRGB: RETURN_FORMAT(@available(macOS 11.0, *), MTLPixelFormatASTC_12x12_sRGB);
        case SDL_GPU_TEXTUREFORMAT_ASTC_4x4_FLOAT: RETURN_FORMAT(@available(macOS 11.0, iOS 13.0, tvOS 16.0, *), MTLPixelFormatASTC_4x4_HDR);
        case SDL_GPU_TEXTUREFORMAT_ASTC_5x4_FLOAT: RETURN_FORMAT(@available(macOS 11.0, iOS 13.0, tvOS 16.0, *), MTLPixelFormatASTC_5x4_HDR);
        case SDL_GPU_TEXTUREFORMAT_ASTC_5x5_FLOAT: RETURN_FORMAT(@available(macOS 11.0, iOS 13.0, tvOS 16.0, *), MTLPixelFormatASTC_5x5_HDR);
        case SDL_GPU_TEXTUREFORMAT_ASTC_6x5_FLOAT: RETURN_FORMAT(@available(macOS 11.0, iOS 13.0, tvOS 16.0, *), MTLPixelFormatASTC_6x5_HDR);
        case SDL_GPU_TEXTUREFORMAT_ASTC_6x6_FLOAT: RETURN_FORMAT(@available(macOS 11.0, iOS 13.0, tvOS 16.0, *), MTLPixelFormatASTC_6x6_HDR);
        case SDL_GPU_TEXTUREFORMAT_ASTC_8x5_FLOAT: RETURN_FORMAT(@available(macOS 11.0, iOS 13.0, tvOS 16.0, *), MTLPixelFormatASTC_8x5_HDR);
        case SDL_GPU_TEXTUREFORMAT_ASTC_8x6_FLOAT: RETURN_FORMAT(@available(macOS 11.0, iOS 13.0, tvOS 16.0, *), MTLPixelFormatASTC_8x6_HDR);
        case SDL_GPU_TEXTUREFORMAT_ASTC_8x8_FLOAT: RETURN_FORMAT(@available(macOS 11.0, iOS 13.0, tvOS 16.0, *), MTLPixelFormatASTC_8x8_HDR);
        case SDL_GPU_TEXTUREFORMAT_ASTC_10x5_FLOAT: RETURN_FORMAT(@available(macOS 11.0, iOS 13.0, tvOS 16.0, *), MTLPixelFormatASTC_10x5_HDR);
        case SDL_GPU_TEXTUREFORMAT_ASTC_10x6_FLOAT: RETURN_FORMAT(@available(macOS 11.0, iOS 13.0, tvOS 16.0, *), MTLPixelFormatASTC_10x6_HDR);
        case SDL_GPU_TEXTUREFORMAT_ASTC_10x8_FLOAT: RETURN_FORMAT(@available(macOS 11.0, iOS 13.0, tvOS 16.0, *), MTLPixelFormatASTC_10x8_HDR);
        case SDL_GPU_TEXTUREFORMAT_ASTC_10x10_FLOAT: RETURN_FORMAT(@available(macOS 11.0, iOS 13.0, tvOS 16.0, *), MTLPixelFormatASTC_10x10_HDR);
        case SDL_GPU_TEXTUREFORMAT_ASTC_12x10_FLOAT: RETURN_FORMAT(@available(macOS 11.0, iOS 13.0, tvOS 16.0, *), MTLPixelFormatASTC_12x10_HDR);
        case SDL_GPU_TEXTUREFORMAT_ASTC_12x12_FLOAT: RETURN_FORMAT(@available(macOS 11.0, iOS 13.0, tvOS 16.0, *), MTLPixelFormatASTC_12x12_HDR);
    }
}

#undef RETURN_FORMAT

static MTLVertexFormat SDLToMetal_VertexFormat[] = {
    MTLVertexFormatInvalid,           // INVALID
    MTLVertexFormatInt,               // INT
    MTLVertexFormatInt2,              // INT2
    MTLVertexFormatInt3,              // INT3
    MTLVertexFormatInt4,              // INT4
    MTLVertexFormatUInt,              // UINT
    MTLVertexFormatUInt2,             // UINT2
    MTLVertexFormatUInt3,             // UINT3
    MTLVertexFormatUInt4,             // UINT4
    MTLVertexFormatFloat,             // FLOAT
    MTLVertexFormatFloat2,            // FLOAT2
    MTLVertexFormatFloat3,            // FLOAT3
    MTLVertexFormatFloat4,            // FLOAT4
    MTLVertexFormatChar2,             // BYTE2
    MTLVertexFormatChar4,             // BYTE4
    MTLVertexFormatUChar2,            // UBYTE2
    MTLVertexFormatUChar4,            // UBYTE4
    MTLVertexFormatChar2Normalized,   // BYTE2_NORM
    MTLVertexFormatChar4Normalized,   // BYTE4_NORM
    MTLVertexFormatUChar2Normalized,  // UBYTE2_NORM
    MTLVertexFormatUChar4Normalized,  // UBYTE4_NORM
    MTLVertexFormatShort2,            // SHORT2
    MTLVertexFormatShort4,            // SHORT4
    MTLVertexFormatUShort2,           // USHORT2
    MTLVertexFormatUShort4,           // USHORT4
    MTLVertexFormatShort2Normalized,  // SHORT2_NORM
    MTLVertexFormatShort4Normalized,  // SHORT4_NORM
    MTLVertexFormatUShort2Normalized, // USHORT2_NORM
    MTLVertexFormatUShort4Normalized, // USHORT4_NORM
    MTLVertexFormatHalf2,             // HALF2
    MTLVertexFormatHalf4              // HALF4
};
SDL_COMPILE_TIME_ASSERT(SDLToMetal_VertexFormat, SDL_arraysize(SDLToMetal_VertexFormat) == SDL_GPU_VERTEXELEMENTFORMAT_MAX_ENUM_VALUE);

static MTLIndexType SDLToMetal_IndexType[] = {
    MTLIndexTypeUInt16, // 16BIT
    MTLIndexTypeUInt32, // 32BIT
};

static MTLPrimitiveType SDLToMetal_PrimitiveType[] = {
    MTLPrimitiveTypeTriangle,      // TRIANGLELIST
    MTLPrimitiveTypeTriangleStrip, // TRIANGLESTRIP
    MTLPrimitiveTypeLine,          // LINELIST
    MTLPrimitiveTypeLineStrip,     // LINESTRIP
    MTLPrimitiveTypePoint          // POINTLIST
};

static MTLTriangleFillMode SDLToMetal_PolygonMode[] = {
    MTLTriangleFillModeFill,  // FILL
    MTLTriangleFillModeLines, // LINE
};

static MTLCullMode SDLToMetal_CullMode[] = {
    MTLCullModeNone,  // NONE
    MTLCullModeFront, // FRONT
    MTLCullModeBack,  // BACK
};

static MTLWinding SDLToMetal_FrontFace[] = {
    MTLWindingCounterClockwise, // COUNTER_CLOCKWISE
    MTLWindingClockwise,        // CLOCKWISE
};

static MTLBlendFactor SDLToMetal_BlendFactor[] = {
    MTLBlendFactorZero,                     // INVALID
    MTLBlendFactorZero,                     // ZERO
    MTLBlendFactorOne,                      // ONE
    MTLBlendFactorSourceColor,              // SRC_COLOR
    MTLBlendFactorOneMinusSourceColor,      // ONE_MINUS_SRC_COLOR
    MTLBlendFactorDestinationColor,         // DST_COLOR
    MTLBlendFactorOneMinusDestinationColor, // ONE_MINUS_DST_COLOR
    MTLBlendFactorSourceAlpha,              // SRC_ALPHA
    MTLBlendFactorOneMinusSourceAlpha,      // ONE_MINUS_SRC_ALPHA
    MTLBlendFactorDestinationAlpha,         // DST_ALPHA
    MTLBlendFactorOneMinusDestinationAlpha, // ONE_MINUS_DST_ALPHA
    MTLBlendFactorBlendColor,               // CONSTANT_COLOR
    MTLBlendFactorOneMinusBlendColor,       // ONE_MINUS_CONSTANT_COLOR
    MTLBlendFactorSourceAlphaSaturated,     // SRC_ALPHA_SATURATE
};
SDL_COMPILE_TIME_ASSERT(SDLToMetal_BlendFactor, SDL_arraysize(SDLToMetal_BlendFactor) == SDL_GPU_BLENDFACTOR_MAX_ENUM_VALUE);

static MTLBlendOperation SDLToMetal_BlendOp[] = {
    MTLBlendOperationAdd,             // INVALID
    MTLBlendOperationAdd,             // ADD
    MTLBlendOperationSubtract,        // SUBTRACT
    MTLBlendOperationReverseSubtract, // REVERSE_SUBTRACT
    MTLBlendOperationMin,             // MIN
    MTLBlendOperationMax,             // MAX
};
SDL_COMPILE_TIME_ASSERT(SDLToMetal_BlendOp, SDL_arraysize(SDLToMetal_BlendOp) == SDL_GPU_BLENDOP_MAX_ENUM_VALUE);

static MTLCompareFunction SDLToMetal_CompareOp[] = {
    MTLCompareFunctionNever,        // INVALID
    MTLCompareFunctionNever,        // NEVER
    MTLCompareFunctionLess,         // LESS
    MTLCompareFunctionEqual,        // EQUAL
    MTLCompareFunctionLessEqual,    // LESS_OR_EQUAL
    MTLCompareFunctionGreater,      // GREATER
    MTLCompareFunctionNotEqual,     // NOT_EQUAL
    MTLCompareFunctionGreaterEqual, // GREATER_OR_EQUAL
    MTLCompareFunctionAlways,       // ALWAYS
};
SDL_COMPILE_TIME_ASSERT(SDLToMetal_CompareOp, SDL_arraysize(SDLToMetal_CompareOp) == SDL_GPU_COMPAREOP_MAX_ENUM_VALUE);

static MTLStencilOperation SDLToMetal_StencilOp[] = {
    MTLStencilOperationKeep,           // INVALID
    MTLStencilOperationKeep,           // KEEP
    MTLStencilOperationZero,           // ZERO
    MTLStencilOperationReplace,        // REPLACE
    MTLStencilOperationIncrementClamp, // INCREMENT_AND_CLAMP
    MTLStencilOperationDecrementClamp, // DECREMENT_AND_CLAMP
    MTLStencilOperationInvert,         // INVERT
    MTLStencilOperationIncrementWrap,  // INCREMENT_AND_WRAP
    MTLStencilOperationDecrementWrap,  // DECREMENT_AND_WRAP
};
SDL_COMPILE_TIME_ASSERT(SDLToMetal_StencilOp, SDL_arraysize(SDLToMetal_StencilOp) == SDL_GPU_STENCILOP_MAX_ENUM_VALUE);

static MTLSamplerAddressMode SDLToMetal_SamplerAddressMode[] = {
    MTLSamplerAddressModeRepeat,       // REPEAT
    MTLSamplerAddressModeMirrorRepeat, // MIRRORED_REPEAT
    MTLSamplerAddressModeClampToEdge   // CLAMP_TO_EDGE
};

static MTLSamplerMinMagFilter SDLToMetal_MinMagFilter[] = {
    MTLSamplerMinMagFilterNearest, // NEAREST
    MTLSamplerMinMagFilterLinear,  // LINEAR
};

static MTLSamplerMipFilter SDLToMetal_MipFilter[] = {
    MTLSamplerMipFilterNearest, // NEAREST
    MTLSamplerMipFilterLinear,  // LINEAR
};

static MTLLoadAction SDLToMetal_LoadOp[] = {
    MTLLoadActionLoad,     // LOAD
    MTLLoadActionClear,    // CLEAR
    MTLLoadActionDontCare, // DONT_CARE
};

static MTLStoreAction SDLToMetal_StoreOp[] = {
    MTLStoreActionStore,
    MTLStoreActionDontCare,
    MTLStoreActionMultisampleResolve,
    MTLStoreActionStoreAndMultisampleResolve
};

static MTLVertexStepFunction SDLToMetal_StepFunction[] = {
    MTLVertexStepFunctionPerVertex,
    MTLVertexStepFunctionPerInstance,
};

static NSUInteger SDLToMetal_SampleCount[] = {
    1, // SDL_GPU_SAMPLECOUNT_1
    2, // SDL_GPU_SAMPLECOUNT_2
    4, // SDL_GPU_SAMPLECOUNT_4
    8  // SDL_GPU_SAMPLECOUNT_8
};

static SDL_GPUTextureFormat SwapchainCompositionToFormat[] = {
    SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM,      // SDR
    SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM_SRGB, // SDR_LINEAR
    SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT,  // HDR_EXTENDED_LINEAR
    SDL_GPU_TEXTUREFORMAT_R10G10B10A2_UNORM,   // HDR10_ST2084
};

static CFStringRef SwapchainCompositionToColorSpace[4]; // initialized on device creation

static MTLTextureType SDLToMetal_TextureType(SDL_GPUTextureType textureType, bool isMSAA)
{
    switch (textureType) {
    case SDL_GPU_TEXTURETYPE_2D:
        return isMSAA ? MTLTextureType2DMultisample : MTLTextureType2D;
    case SDL_GPU_TEXTURETYPE_2D_ARRAY:
        return MTLTextureType2DArray;
    case SDL_GPU_TEXTURETYPE_3D:
        return MTLTextureType3D;
    case SDL_GPU_TEXTURETYPE_CUBE:
        return MTLTextureTypeCube;
    case SDL_GPU_TEXTURETYPE_CUBE_ARRAY:
        return MTLTextureTypeCubeArray;
    default:
        return MTLTextureType2D;
    }
}

static MTLColorWriteMask SDLToMetal_ColorWriteMask(
    SDL_GPUColorComponentFlags mask)
{
    MTLColorWriteMask result = 0;
    if (mask & SDL_GPU_COLORCOMPONENT_R) {
        result |= MTLColorWriteMaskRed;
    }
    if (mask & SDL_GPU_COLORCOMPONENT_G) {
        result |= MTLColorWriteMaskGreen;
    }
    if (mask & SDL_GPU_COLORCOMPONENT_B) {
        result |= MTLColorWriteMaskBlue;
    }
    if (mask & SDL_GPU_COLORCOMPONENT_A) {
        result |= MTLColorWriteMaskAlpha;
    }
    return result;
}

static MTLDepthClipMode SDLToMetal_DepthClipMode(
    bool enableDepthClip
) {
    if (enableDepthClip) {
        return MTLDepthClipModeClip;
    } else {
        return MTLDepthClipModeClamp;
    }
}

// Structs

typedef struct MetalTexture
{
    id<MTLTexture> handle;
    SDL_AtomicInt referenceCount;
} MetalTexture;

typedef struct MetalTextureContainer
{
    TextureCommonHeader header;

    MetalTexture *activeTexture;
    Uint8 canBeCycled;

    Uint32 textureCapacity;
    Uint32 textureCount;
    MetalTexture **textures;

    char *debugName;
} MetalTextureContainer;

typedef struct MetalFence
{
    SDL_AtomicInt complete;
    SDL_AtomicInt referenceCount;
} MetalFence;

typedef struct MetalWindowData
{
    SDL_Window *window;
    SDL_MetalView view;
    CAMetalLayer *layer;
    SDL_GPUPresentMode presentMode;
    id<CAMetalDrawable> drawable;
    MetalTexture texture;
    MetalTextureContainer textureContainer;
    SDL_GPUFence *inFlightFences[MAX_FRAMES_IN_FLIGHT];
    Uint32 frameCounter;
} MetalWindowData;

typedef struct MetalShader
{
    id<MTLLibrary> library;
    id<MTLFunction> function;

    SDL_GPUShaderStage stage;
    Uint32 numSamplers;
    Uint32 numUniformBuffers;
    Uint32 numStorageBuffers;
    Uint32 numStorageTextures;
} MetalShader;

typedef struct MetalGraphicsPipeline
{
    id<MTLRenderPipelineState> handle;

    SDL_GPURasterizerState rasterizerState;
    SDL_GPUPrimitiveType primitiveType;

    id<MTLDepthStencilState> depth_stencil_state;

    Uint32 vertexSamplerCount;
    Uint32 vertexUniformBufferCount;
    Uint32 vertexStorageBufferCount;
    Uint32 vertexStorageTextureCount;

    Uint32 fragmentSamplerCount;
    Uint32 fragmentUniformBufferCount;
    Uint32 fragmentStorageBufferCount;
    Uint32 fragmentStorageTextureCount;
} MetalGraphicsPipeline;

typedef struct MetalComputePipeline
{
    id<MTLComputePipelineState> handle;
    Uint32 numSamplers;
    Uint32 numReadonlyStorageTextures;
    Uint32 numReadWriteStorageTextures;
    Uint32 numReadonlyStorageBuffers;
    Uint32 numReadWriteStorageBuffers;
    Uint32 numUniformBuffers;
    Uint32 threadcountX;
    Uint32 threadcountY;
    Uint32 threadcountZ;
} MetalComputePipeline;

typedef struct MetalBuffer
{
    id<MTLBuffer> handle;
    SDL_AtomicInt referenceCount;
} MetalBuffer;

typedef struct MetalBufferContainer
{
    MetalBuffer *activeBuffer;
    Uint32 size;

    Uint32 bufferCapacity;
    Uint32 bufferCount;
    MetalBuffer **buffers;

    bool isPrivate;
    bool isWriteOnly;
    char *debugName;
} MetalBufferContainer;

typedef struct MetalUniformBuffer
{
    id<MTLBuffer> handle;
    Uint32 writeOffset;
    Uint32 drawOffset;
} MetalUniformBuffer;

typedef struct MetalRenderer MetalRenderer;

typedef struct MetalCommandBuffer
{
    CommandBufferCommonHeader common;
    MetalRenderer *renderer;

    // Native Handle
    id<MTLCommandBuffer> handle;

    // Presentation
    MetalWindowData **windowDatas;
    Uint32 windowDataCount;
    Uint32 windowDataCapacity;

    // Render Pass
    id<MTLRenderCommandEncoder> renderEncoder;
    MetalGraphicsPipeline *graphics_pipeline;
    MetalBuffer *indexBuffer;
    Uint32 indexBufferOffset;
    SDL_GPUIndexElementSize index_element_size;

    // Copy Pass
    id<MTLBlitCommandEncoder> blitEncoder;

    // Compute Pass
    id<MTLComputeCommandEncoder> computeEncoder;
    MetalComputePipeline *compute_pipeline;

    // Resource slot state
    bool needVertexBufferBind;
    bool needVertexSamplerBind;
    bool needVertexStorageTextureBind;
    bool needVertexStorageBufferBind;
    bool needVertexUniformBufferBind[MAX_UNIFORM_BUFFERS_PER_STAGE];

    bool needFragmentSamplerBind;
    bool needFragmentStorageTextureBind;
    bool needFragmentStorageBufferBind;
    bool needFragmentUniformBufferBind[MAX_UNIFORM_BUFFERS_PER_STAGE];

    bool needComputeSamplerBind;
    bool needComputeReadOnlyStorageTextureBind;
    bool needComputeReadOnlyStorageBufferBind;
    bool needComputeUniformBufferBind[MAX_UNIFORM_BUFFERS_PER_STAGE];

    id<MTLBuffer> vertexBuffers[MAX_VERTEX_BUFFERS];
    Uint32 vertexBufferOffsets[MAX_VERTEX_BUFFERS];
    Uint32 vertexBufferCount;

    id<MTLSamplerState> vertexSamplers[MAX_TEXTURE_SAMPLERS_PER_STAGE];
    id<MTLTexture> vertexTextures[MAX_TEXTURE_SAMPLERS_PER_STAGE];
    id<MTLTexture> vertexStorageTextures[MAX_STORAGE_TEXTURES_PER_STAGE];
    id<MTLBuffer> vertexStorageBuffers[MAX_STORAGE_BUFFERS_PER_STAGE];
    MetalUniformBuffer *vertexUniformBuffers[MAX_UNIFORM_BUFFERS_PER_STAGE];

    id<MTLSamplerState> fragmentSamplers[MAX_TEXTURE_SAMPLERS_PER_STAGE];
    id<MTLTexture> fragmentTextures[MAX_TEXTURE_SAMPLERS_PER_STAGE];
    id<MTLTexture> fragmentStorageTextures[MAX_STORAGE_TEXTURES_PER_STAGE];
    id<MTLBuffer> fragmentStorageBuffers[MAX_STORAGE_BUFFERS_PER_STAGE];
    MetalUniformBuffer *fragmentUniformBuffers[MAX_UNIFORM_BUFFERS_PER_STAGE];

    id<MTLTexture> computeSamplerTextures[MAX_TEXTURE_SAMPLERS_PER_STAGE];
    id<MTLSamplerState> computeSamplers[MAX_TEXTURE_SAMPLERS_PER_STAGE];
    id<MTLTexture> computeReadOnlyTextures[MAX_STORAGE_TEXTURES_PER_STAGE];
    id<MTLBuffer> computeReadOnlyBuffers[MAX_STORAGE_BUFFERS_PER_STAGE];
    id<MTLTexture> computeReadWriteTextures[MAX_COMPUTE_WRITE_TEXTURES];
    id<MTLBuffer> computeReadWriteBuffers[MAX_COMPUTE_WRITE_BUFFERS];
    MetalUniformBuffer *computeUniformBuffers[MAX_UNIFORM_BUFFERS_PER_STAGE];

    MetalUniformBuffer **usedUniformBuffers;
    Uint32 usedUniformBufferCount;
    Uint32 usedUniformBufferCapacity;

    // Fences
    MetalFence *fence;
    bool autoReleaseFence;

    // Reference Counting
    MetalBuffer **usedBuffers;
    Uint32 usedBufferCount;
    Uint32 usedBufferCapacity;

    MetalTexture **usedTextures;
    Uint32 usedTextureCount;
    Uint32 usedTextureCapacity;
} MetalCommandBuffer;

typedef struct MetalSampler
{
    id<MTLSamplerState> handle;
} MetalSampler;

typedef struct BlitPipeline
{
    SDL_GPUGraphicsPipeline *pipeline;
    SDL_GPUTextureFormat format;
} BlitPipeline;

struct MetalRenderer
{
    // Reference to the parent device
    SDL_GPUDevice *sdlGPUDevice;

    id<MTLDevice> device;
    id<MTLCommandQueue> queue;

    bool debugMode;
    SDL_PropertiesID props;
    Uint32 allowedFramesInFlight;

    MetalWindowData **claimedWindows;
    Uint32 claimedWindowCount;
    Uint32 claimedWindowCapacity;

    MetalCommandBuffer **availableCommandBuffers;
    Uint32 availableCommandBufferCount;
    Uint32 availableCommandBufferCapacity;

    MetalCommandBuffer **submittedCommandBuffers;
    Uint32 submittedCommandBufferCount;
    Uint32 submittedCommandBufferCapacity;

    MetalFence **availableFences;
    Uint32 availableFenceCount;
    Uint32 availableFenceCapacity;

    MetalUniformBuffer **uniformBufferPool;
    Uint32 uniformBufferPoolCount;
    Uint32 uniformBufferPoolCapacity;

    MetalBufferContainer **bufferContainersToDestroy;
    Uint32 bufferContainersToDestroyCount;
    Uint32 bufferContainersToDestroyCapacity;

    MetalTextureContainer **textureContainersToDestroy;
    Uint32 textureContainersToDestroyCount;
    Uint32 textureContainersToDestroyCapacity;

    // Blit
    SDL_GPUShader *blitVertexShader;
    SDL_GPUShader *blitFrom2DShader;
    SDL_GPUShader *blitFrom2DArrayShader;
    SDL_GPUShader *blitFrom3DShader;
    SDL_GPUShader *blitFromCubeShader;
    SDL_GPUShader *blitFromCubeArrayShader;

    SDL_GPUSampler *blitNearestSampler;
    SDL_GPUSampler *blitLinearSampler;

    BlitPipelineCacheEntry *blitPipelines;
    Uint32 blitPipelineCount;
    Uint32 blitPipelineCapacity;

    // Mutexes
    SDL_Mutex *submitLock;
    SDL_Mutex *acquireCommandBufferLock;
    SDL_Mutex *acquireUniformBufferLock;
    SDL_Mutex *disposeLock;
    SDL_Mutex *fenceLock;
    SDL_Mutex *windowLock;
};

// Helper Functions

// FIXME: This should be moved into SDL_sysgpu.h
static inline Uint32 METAL_INTERNAL_NextHighestAlignment(
    Uint32 n,
    Uint32 align)
{
    return align * ((n + align - 1) / align);
}

// Quit

static void METAL_DestroyDevice(SDL_GPUDevice *device)
{
    MetalRenderer *renderer = (MetalRenderer *)device->driverData;

    // Flush any remaining GPU work...
    METAL_Wait(device->driverData);

    // Release the window data
    for (Sint32 i = renderer->claimedWindowCount - 1; i >= 0; i -= 1) {
        METAL_ReleaseWindow(device->driverData, renderer->claimedWindows[i]->window);
    }
    SDL_free(renderer->claimedWindows);

    // Release the blit resources
    METAL_INTERNAL_DestroyBlitResources(device->driverData);

    // Release uniform buffers
    for (Uint32 i = 0; i < renderer->uniformBufferPoolCount; i += 1) {
        renderer->uniformBufferPool[i]->handle = nil;
        SDL_free(renderer->uniformBufferPool[i]);
    }
    SDL_free(renderer->uniformBufferPool);

    // Release destroyed resource lists
    SDL_free(renderer->bufferContainersToDestroy);
    SDL_free(renderer->textureContainersToDestroy);

    // Release command buffer infrastructure
    for (Uint32 i = 0; i < renderer->availableCommandBufferCount; i += 1) {
        MetalCommandBuffer *commandBuffer = renderer->availableCommandBuffers[i];
        SDL_free(commandBuffer->usedBuffers);
        SDL_free(commandBuffer->usedTextures);
        SDL_free(commandBuffer->usedUniformBuffers);
        SDL_free(commandBuffer->windowDatas);
        SDL_free(commandBuffer);
    }
    SDL_free(renderer->availableCommandBuffers);
    SDL_free(renderer->submittedCommandBuffers);

    // Release fence infrastructure
    for (Uint32 i = 0; i < renderer->availableFenceCount; i += 1) {
        SDL_free(renderer->availableFences[i]);
    }
    SDL_free(renderer->availableFences);

    // Release the mutexes
    SDL_DestroyMutex(renderer->submitLock);
    SDL_DestroyMutex(renderer->acquireCommandBufferLock);
    SDL_DestroyMutex(renderer->acquireUniformBufferLock);
    SDL_DestroyMutex(renderer->disposeLock);
    SDL_DestroyMutex(renderer->fenceLock);
    SDL_DestroyMutex(renderer->windowLock);

    // Release the command queue
    renderer->queue = nil;

    // Release properties
    SDL_DestroyProperties(renderer->props);

    // Free the primary structures
    SDL_free(renderer);
    SDL_free(device);
}

static SDL_PropertiesID METAL_GetDeviceProperties(SDL_GPUDevice *device)
{
    MetalRenderer *renderer = (MetalRenderer *)device->driverData;
    return renderer->props;
}

// Resource tracking

static void METAL_INTERNAL_TrackBuffer(
    MetalCommandBuffer *commandBuffer,
    MetalBuffer *buffer)
{
    TRACK_RESOURCE(
        buffer,
        MetalBuffer *,
        usedBuffers,
        usedBufferCount,
        usedBufferCapacity);
}

static void METAL_INTERNAL_TrackTexture(
    MetalCommandBuffer *commandBuffer,
    MetalTexture *texture)
{
    TRACK_RESOURCE(
        texture,
        MetalTexture *,
        usedTextures,
        usedTextureCount,
        usedTextureCapacity);
}

static void METAL_INTERNAL_TrackUniformBuffer(
    MetalCommandBuffer *commandBuffer,
    MetalUniformBuffer *uniformBuffer)
{
    Uint32 i;
    for (i = 0; i < commandBuffer->usedUniformBufferCount; i += 1) {
        if (commandBuffer->usedUniformBuffers[i] == uniformBuffer) {
            return;
        }
    }

    if (commandBuffer->usedUniformBufferCount == commandBuffer->usedUniformBufferCapacity) {
        commandBuffer->usedUniformBufferCapacity += 1;
        commandBuffer->usedUniformBuffers = SDL_realloc(
            commandBuffer->usedUniformBuffers,
            commandBuffer->usedUniformBufferCapacity * sizeof(MetalUniformBuffer *));
    }

    commandBuffer->usedUniformBuffers[commandBuffer->usedUniformBufferCount] = uniformBuffer;
    commandBuffer->usedUniformBufferCount += 1;
}

// Shader Compilation

typedef struct MetalLibraryFunction
{
    id<MTLLibrary> library;
    id<MTLFunction> function;
} MetalLibraryFunction;

static bool METAL_INTERNAL_IsValidMetalLibrary(
    const Uint8 *code,
    size_t codeSize)
{
    // Metal libraries have a 4 byte header containing `MTLB`.
    if (codeSize < 4 || code == NULL) {
        return false;
    }
    return SDL_memcmp(code, "MTLB", 4) == 0;
}

// This function assumes that it's called from within an autorelease pool
static MetalLibraryFunction METAL_INTERNAL_CompileShader(
    MetalRenderer *renderer,
    SDL_GPUShaderFormat format,
    const Uint8 *code,
    size_t codeSize,
    const char *entrypoint)
{
    MetalLibraryFunction libraryFunction = { nil, nil };
    id<MTLLibrary> library;
    NSError *error;
    dispatch_data_t data;
    id<MTLFunction> function;

    if (!entrypoint) {
        entrypoint = "main0";
    }

    if (format == SDL_GPU_SHADERFORMAT_MSL) {
        NSString *codeString = [[NSString alloc]
            initWithBytes:code
                   length:codeSize
                 encoding:NSUTF8StringEncoding];
        library = [renderer->device
            newLibraryWithSource:codeString
                         options:nil
                           error:&error];
    } else if (format == SDL_GPU_SHADERFORMAT_METALLIB) {
        if (!METAL_INTERNAL_IsValidMetalLibrary(code, codeSize)) {
            SET_STRING_ERROR_AND_RETURN(
                "The provided shader code is not a valid Metal library!",
                libraryFunction);
        }
        data = dispatch_data_create(
            code,
            codeSize,
            dispatch_get_global_queue(0, 0),
            DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        library = [renderer->device newLibraryWithData:data error:&error];
    } else {
        SDL_assert(!"SDL_gpu.c should have already validated this!");
        return libraryFunction;
    }

    if (library == nil) {
        SDL_LogError(
            SDL_LOG_CATEGORY_GPU,
            "Creating MTLLibrary failed: %s",
            [[error description] cStringUsingEncoding:[NSString defaultCStringEncoding]]);
        return libraryFunction;
    } else if (error != nil) {
        SDL_LogWarn(
            SDL_LOG_CATEGORY_GPU,
            "Creating MTLLibrary failed: %s",
            [[error description] cStringUsingEncoding:[NSString defaultCStringEncoding]]);
    }

    function = [library newFunctionWithName:@(entrypoint)];
    if (function == nil) {
        SDL_LogError(
            SDL_LOG_CATEGORY_GPU,
            "Creating MTLFunction failed");
        return libraryFunction;
    }

    libraryFunction.library = library;
    libraryFunction.function = function;
    return libraryFunction;
}

// Disposal

static void METAL_INTERNAL_DestroyTextureContainer(
    MetalTextureContainer *container)
{
    for (Uint32 i = 0; i < container->textureCount; i += 1) {
        container->textures[i]->handle = nil;
        SDL_free(container->textures[i]);
    }
    if (container->debugName != NULL) {
        SDL_free(container->debugName);
    }
    SDL_free(container->textures);
    SDL_free(container);
}

static void METAL_ReleaseTexture(
    SDL_GPURenderer *driverData,
    SDL_GPUTexture *texture)
{
    MetalRenderer *renderer = (MetalRenderer *)driverData;
    MetalTextureContainer *container = (MetalTextureContainer *)texture;

    SDL_LockMutex(renderer->disposeLock);

    EXPAND_ARRAY_IF_NEEDED(
        renderer->textureContainersToDestroy,
        MetalTextureContainer *,
        renderer->textureContainersToDestroyCount + 1,
        renderer->textureContainersToDestroyCapacity,
        renderer->textureContainersToDestroyCapacity + 1);

    renderer->textureContainersToDestroy[renderer->textureContainersToDestroyCount] = container;
    renderer->textureContainersToDestroyCount += 1;

    SDL_UnlockMutex(renderer->disposeLock);
}

static void METAL_ReleaseSampler(
    SDL_GPURenderer *driverData,
    SDL_GPUSampler *sampler)
{
    @autoreleasepool {
        MetalSampler *metalSampler = (MetalSampler *)sampler;
        metalSampler->handle = nil;
        SDL_free(metalSampler);
    }
}

static void METAL_INTERNAL_DestroyBufferContainer(
    MetalBufferContainer *container)
{
    for (Uint32 i = 0; i < container->bufferCount; i += 1) {
        container->buffers[i]->handle = nil;
        SDL_free(container->buffers[i]);
    }
    if (container->debugName != NULL) {
        SDL_free(container->debugName);
    }
    SDL_free(container->buffers);
    SDL_free(container);
}

static void METAL_ReleaseBuffer(
    SDL_GPURenderer *driverData,
    SDL_GPUBuffer *buffer)
{
    MetalRenderer *renderer = (MetalRenderer *)driverData;
    MetalBufferContainer *container = (MetalBufferContainer *)buffer;

    SDL_LockMutex(renderer->disposeLock);

    EXPAND_ARRAY_IF_NEEDED(
        renderer->bufferContainersToDestroy,
        MetalBufferContainer *,
        renderer->bufferContainersToDestroyCount + 1,
        renderer->bufferContainersToDestroyCapacity,
        renderer->bufferContainersToDestroyCapacity + 1);

    renderer->bufferContainersToDestroy[renderer->bufferContainersToDestroyCount] = container;
    renderer->bufferContainersToDestroyCount += 1;

    SDL_UnlockMutex(renderer->disposeLock);
}

static void METAL_ReleaseTransferBuffer(
    SDL_GPURenderer *driverData,
    SDL_GPUTransferBuffer *transferBuffer)
{
    METAL_ReleaseBuffer(
        driverData,
        (SDL_GPUBuffer *)transferBuffer);
}

static void METAL_ReleaseShader(
    SDL_GPURenderer *driverData,
    SDL_GPUShader *shader)
{
    @autoreleasepool {
        MetalShader *metalShader = (MetalShader *)shader;
        metalShader->function = nil;
        metalShader->library = nil;
        SDL_free(metalShader);
    }
}

static void METAL_ReleaseComputePipeline(
    SDL_GPURenderer *driverData,
    SDL_GPUComputePipeline *computePipeline)
{
    @autoreleasepool {
        MetalComputePipeline *metalComputePipeline = (MetalComputePipeline *)computePipeline;
        metalComputePipeline->handle = nil;
        SDL_free(metalComputePipeline);
    }
}

static void METAL_ReleaseGraphicsPipeline(
    SDL_GPURenderer *driverData,
    SDL_GPUGraphicsPipeline *graphicsPipeline)
{
    @autoreleasepool {
        MetalGraphicsPipeline *metalGraphicsPipeline = (MetalGraphicsPipeline *)graphicsPipeline;
        metalGraphicsPipeline->handle = nil;
        metalGraphicsPipeline->depth_stencil_state = nil;
        SDL_free(metalGraphicsPipeline);
    }
}

// Pipeline Creation

static SDL_GPUComputePipeline *METAL_CreateComputePipeline(
    SDL_GPURenderer *driverData,
    const SDL_GPUComputePipelineCreateInfo *createinfo)
{
    @autoreleasepool {
        MetalRenderer *renderer = (MetalRenderer *)driverData;
        MetalLibraryFunction libraryFunction;
        id<MTLComputePipelineState> handle;
        MetalComputePipeline *pipeline;
        NSError *error;

        libraryFunction = METAL_INTERNAL_CompileShader(
            renderer,
            createinfo->format,
            createinfo->code,
            createinfo->code_size,
            createinfo->entrypoint);

        if (libraryFunction.library == nil || libraryFunction.function == nil) {
            return NULL;
        }

        MTLComputePipelineDescriptor *descriptor = [MTLComputePipelineDescriptor new];
        descriptor.computeFunction = libraryFunction.function;

        if (renderer->debugMode && SDL_HasProperty(createinfo->props, SDL_PROP_GPU_COMPUTEPIPELINE_CREATE_NAME_STRING)) {
            const char *name = SDL_GetStringProperty(createinfo->props, SDL_PROP_GPU_COMPUTEPIPELINE_CREATE_NAME_STRING, NULL);
            descriptor.label = @(name);
        }

        handle = [renderer->device newComputePipelineStateWithDescriptor:descriptor options:MTLPipelineOptionNone reflection: nil error:&error];
        if (error != NULL) {
            SET_ERROR_AND_RETURN("Creating compute pipeline failed: %s", [[error description] UTF8String], NULL);
        }

        pipeline = SDL_calloc(1, sizeof(MetalComputePipeline));
        pipeline->handle = handle;
        pipeline->numSamplers = createinfo->num_samplers;
        pipeline->numReadonlyStorageTextures = createinfo->num_readonly_storage_textures;
        pipeline->numReadWriteStorageTextures = createinfo->num_readwrite_storage_textures;
        pipeline->numReadonlyStorageBuffers = createinfo->num_readonly_storage_buffers;
        pipeline->numReadWriteStorageBuffers = createinfo->num_readwrite_storage_buffers;
        pipeline->numUniformBuffers = createinfo->num_uniform_buffers;
        pipeline->threadcountX = createinfo->threadcount_x;
        pipeline->threadcountY = createinfo->threadcount_y;
        pipeline->threadcountZ = createinfo->threadcount_z;

        return (SDL_GPUComputePipeline *)pipeline;
    }
}

static SDL_GPUGraphicsPipeline *METAL_CreateGraphicsPipeline(
    SDL_GPURenderer *driverData,
    const SDL_GPUGraphicsPipelineCreateInfo *createinfo)
{
    @autoreleasepool {
        MetalRenderer *renderer = (MetalRenderer *)driverData;
        MetalShader *vertexShader = (MetalShader *)createinfo->vertex_shader;
        MetalShader *fragmentShader = (MetalShader *)createinfo->fragment_shader;
        MTLRenderPipelineDescriptor *pipelineDescriptor;
        const SDL_GPUColorTargetBlendState *blendState;
        MTLVertexDescriptor *vertexDescriptor;
        Uint32 binding;
        MTLDepthStencilDescriptor *depthStencilDescriptor;
        MTLStencilDescriptor *frontStencilDescriptor = NULL;
        MTLStencilDescriptor *backStencilDescriptor = NULL;
        id<MTLDepthStencilState> depthStencilState = nil;
        id<MTLRenderPipelineState> pipelineState = nil;
        NSError *error = NULL;
        MetalGraphicsPipeline *result = NULL;

        if (renderer->debugMode) {
            if (vertexShader->stage != SDL_GPU_SHADERSTAGE_VERTEX) {
                SDL_assert_release(!"CreateGraphicsPipeline was passed a fragment shader for the vertex stage");
            }
            if (fragmentShader->stage != SDL_GPU_SHADERSTAGE_FRAGMENT) {
                SDL_assert_release(!"CreateGraphicsPipeline was passed a vertex shader for the fragment stage");
            }
        }
        pipelineDescriptor = [MTLRenderPipelineDescriptor new];

        // Blend

        for (Uint32 i = 0; i < createinfo->target_info.num_color_targets; i += 1) {
            blendState = &createinfo->target_info.color_target_descriptions[i].blend_state;
            SDL_GPUColorComponentFlags colorWriteMask = blendState->enable_color_write_mask ?
                blendState->color_write_mask :
                0xF;

            pipelineDescriptor.colorAttachments[i].pixelFormat = SDLToMetal_TextureFormat(createinfo->target_info.color_target_descriptions[i].format);
            pipelineDescriptor.colorAttachments[i].writeMask = SDLToMetal_ColorWriteMask(colorWriteMask);
            pipelineDescriptor.colorAttachments[i].blendingEnabled = blendState->enable_blend;
            pipelineDescriptor.colorAttachments[i].rgbBlendOperation = SDLToMetal_BlendOp[blendState->color_blend_op];
            pipelineDescriptor.colorAttachments[i].alphaBlendOperation = SDLToMetal_BlendOp[blendState->alpha_blend_op];
            pipelineDescriptor.colorAttachments[i].sourceRGBBlendFactor = SDLToMetal_BlendFactor[blendState->src_color_blendfactor];
            pipelineDescriptor.colorAttachments[i].sourceAlphaBlendFactor = SDLToMetal_BlendFactor[blendState->src_alpha_blendfactor];
            pipelineDescriptor.colorAttachments[i].destinationRGBBlendFactor = SDLToMetal_BlendFactor[blendState->dst_color_blendfactor];
            pipelineDescriptor.colorAttachments[i].destinationAlphaBlendFactor = SDLToMetal_BlendFactor[blendState->dst_alpha_blendfactor];
        }

        // Multisample

        pipelineDescriptor.rasterSampleCount = SDLToMetal_SampleCount[createinfo->multisample_state.sample_count];
        pipelineDescriptor.alphaToCoverageEnabled = createinfo->multisample_state.enable_alpha_to_coverage;

        // Depth Stencil

        if (createinfo->target_info.has_depth_stencil_target) {
            pipelineDescriptor.depthAttachmentPixelFormat = SDLToMetal_TextureFormat(createinfo->target_info.depth_stencil_format);
            if (IsStencilFormat(createinfo->target_info.depth_stencil_format)) {
                pipelineDescriptor.stencilAttachmentPixelFormat = SDLToMetal_TextureFormat(createinfo->target_info.depth_stencil_format);
            }

            if (createinfo->depth_stencil_state.enable_stencil_test) {
                frontStencilDescriptor = [MTLStencilDescriptor new];
                frontStencilDescriptor.stencilCompareFunction = SDLToMetal_CompareOp[createinfo->depth_stencil_state.front_stencil_state.compare_op];
                frontStencilDescriptor.stencilFailureOperation = SDLToMetal_StencilOp[createinfo->depth_stencil_state.front_stencil_state.fail_op];
                frontStencilDescriptor.depthStencilPassOperation = SDLToMetal_StencilOp[createinfo->depth_stencil_state.front_stencil_state.pass_op];
                frontStencilDescriptor.depthFailureOperation = SDLToMetal_StencilOp[createinfo->depth_stencil_state.front_stencil_state.depth_fail_op];
                frontStencilDescriptor.readMask = createinfo->depth_stencil_state.compare_mask;
                frontStencilDescriptor.writeMask = createinfo->depth_stencil_state.write_mask;

                backStencilDescriptor = [MTLStencilDescriptor new];
                backStencilDescriptor.stencilCompareFunction = SDLToMetal_CompareOp[createinfo->depth_stencil_state.back_stencil_state.compare_op];
                backStencilDescriptor.stencilFailureOperation = SDLToMetal_StencilOp[createinfo->depth_stencil_state.back_stencil_state.fail_op];
                backStencilDescriptor.depthStencilPassOperation = SDLToMetal_StencilOp[createinfo->depth_stencil_state.back_stencil_state.pass_op];
                backStencilDescriptor.depthFailureOperation = SDLToMetal_StencilOp[createinfo->depth_stencil_state.back_stencil_state.depth_fail_op];
                backStencilDescriptor.readMask = createinfo->depth_stencil_state.compare_mask;
                backStencilDescriptor.writeMask = createinfo->depth_stencil_state.write_mask;
            }

            depthStencilDescriptor = [MTLDepthStencilDescriptor new];
            depthStencilDescriptor.depthCompareFunction = createinfo->depth_stencil_state.enable_depth_test ? SDLToMetal_CompareOp[createinfo->depth_stencil_state.compare_op] : MTLCompareFunctionAlways;
            // Disable write when test is disabled, to match other APIs' behavior
            depthStencilDescriptor.depthWriteEnabled = createinfo->depth_stencil_state.enable_depth_write && createinfo->depth_stencil_state.enable_depth_test;
            depthStencilDescriptor.frontFaceStencil = frontStencilDescriptor;
            depthStencilDescriptor.backFaceStencil = backStencilDescriptor;

            depthStencilState = [renderer->device newDepthStencilStateWithDescriptor:depthStencilDescriptor];
        }

        // Shaders

        pipelineDescriptor.vertexFunction = vertexShader->function;
        pipelineDescriptor.fragmentFunction = fragmentShader->function;

        // Vertex Descriptor

        if (createinfo->vertex_input_state.num_vertex_buffers > 0) {
            vertexDescriptor = [MTLVertexDescriptor vertexDescriptor];

            for (Uint32 i = 0; i < createinfo->vertex_input_state.num_vertex_attributes; i += 1) {
                Uint32 loc = createinfo->vertex_input_state.vertex_attributes[i].location;
                vertexDescriptor.attributes[loc].format = SDLToMetal_VertexFormat[createinfo->vertex_input_state.vertex_attributes[i].format];
                vertexDescriptor.attributes[loc].offset = createinfo->vertex_input_state.vertex_attributes[i].offset;
                vertexDescriptor.attributes[loc].bufferIndex =
                    METAL_FIRST_VERTEX_BUFFER_SLOT + createinfo->vertex_input_state.vertex_attributes[i].buffer_slot;
            }

            for (Uint32 i = 0; i < createinfo->vertex_input_state.num_vertex_buffers; i += 1) {
                binding = METAL_FIRST_VERTEX_BUFFER_SLOT + createinfo->vertex_input_state.vertex_buffer_descriptions[i].slot;
                vertexDescriptor.layouts[binding].stepFunction = SDLToMetal_StepFunction[createinfo->vertex_input_state.vertex_buffer_descriptions[i].input_rate];
                vertexDescriptor.layouts[binding].stepRate = 1;
                vertexDescriptor.layouts[binding].stride = createinfo->vertex_input_state.vertex_buffer_descriptions[i].pitch;
            }

            pipelineDescriptor.vertexDescriptor = vertexDescriptor;
        }

        if (renderer->debugMode && SDL_HasProperty(createinfo->props, SDL_PROP_GPU_GRAPHICSPIPELINE_CREATE_NAME_STRING)) {
            const char *name = SDL_GetStringProperty(createinfo->props, SDL_PROP_GPU_GRAPHICSPIPELINE_CREATE_NAME_STRING, NULL);
            pipelineDescriptor.label = @(name);
        }

        // Create the graphics pipeline

        pipelineState = [renderer->device newRenderPipelineStateWithDescriptor:pipelineDescriptor error:&error];
        if (error != NULL) {
            SET_ERROR_AND_RETURN("Creating render pipeline failed: %s", [[error description] UTF8String], NULL);
        }

        result = SDL_calloc(1, sizeof(MetalGraphicsPipeline));
        result->handle = pipelineState;
        result->depth_stencil_state = depthStencilState;
        result->rasterizerState = createinfo->rasterizer_state;
        result->primitiveType = createinfo->primitive_type;
        result->vertexSamplerCount = vertexShader->numSamplers;
        result->vertexUniformBufferCount = vertexShader->numUniformBuffers;
        result->vertexStorageBufferCount = vertexShader->numStorageBuffers;
        result->vertexStorageTextureCount = vertexShader->numStorageTextures;
        result->fragmentSamplerCount = fragmentShader->numSamplers;
        result->fragmentUniformBufferCount = fragmentShader->numUniformBuffers;
        result->fragmentStorageBufferCount = fragmentShader->numStorageBuffers;
        result->fragmentStorageTextureCount = fragmentShader->numStorageTextures;
        return (SDL_GPUGraphicsPipeline *)result;
    }
}

// Debug Naming

static void METAL_SetBufferName(
    SDL_GPURenderer *driverData,
    SDL_GPUBuffer *buffer,
    const char *text)
{
    @autoreleasepool {
        MetalRenderer *renderer = (MetalRenderer *)driverData;
        MetalBufferContainer *container = (MetalBufferContainer *)buffer;

        if (renderer->debugMode && text != NULL) {
            if (container->debugName != NULL) {
                SDL_free(container->debugName);
            }

            container->debugName = SDL_strdup(text);

            for (Uint32 i = 0; i < container->bufferCount; i += 1) {
                container->buffers[i]->handle.label = @(text);
            }
        }
    }
}

static void METAL_SetTextureName(
    SDL_GPURenderer *driverData,
    SDL_GPUTexture *texture,
    const char *text)
{
    @autoreleasepool {
        MetalRenderer *renderer = (MetalRenderer *)driverData;
        MetalTextureContainer *container = (MetalTextureContainer *)texture;

        if (renderer->debugMode && text != NULL) {
            if (container->debugName != NULL) {
                SDL_free(container->debugName);
            }

            container->debugName = SDL_strdup(text);

            for (Uint32 i = 0; i < container->textureCount; i += 1) {
                container->textures[i]->handle.label = @(text);
            }
        }
    }
}

static void METAL_InsertDebugLabel(
    SDL_GPUCommandBuffer *commandBuffer,
    const char *text)
{
    @autoreleasepool {
        MetalCommandBuffer *metalCommandBuffer = (MetalCommandBuffer *)commandBuffer;
        NSString *label = @(text);

        if (metalCommandBuffer->renderEncoder) {
            [metalCommandBuffer->renderEncoder insertDebugSignpost:label];
        } else if (metalCommandBuffer->blitEncoder) {
            [metalCommandBuffer->blitEncoder insertDebugSignpost:label];
        } else if (metalCommandBuffer->computeEncoder) {
            [metalCommandBuffer->computeEncoder insertDebugSignpost:label];
        } else {
            // Metal doesn't have insertDebugSignpost for command buffers...
            [metalCommandBuffer->handle pushDebugGroup:label];
            [metalCommandBuffer->handle popDebugGroup];
        }
    }
}

static void METAL_PushDebugGroup(
    SDL_GPUCommandBuffer *commandBuffer,
    const char *name)
{
    @autoreleasepool {
        MetalCommandBuffer *metalCommandBuffer = (MetalCommandBuffer *)commandBuffer;
        NSString *label = @(name);

        if (metalCommandBuffer->renderEncoder) {
            [metalCommandBuffer->renderEncoder pushDebugGroup:label];
        } else if (metalCommandBuffer->blitEncoder) {
            [metalCommandBuffer->blitEncoder pushDebugGroup:label];
        } else if (metalCommandBuffer->computeEncoder) {
            [metalCommandBuffer->computeEncoder pushDebugGroup:label];
        } else {
            [metalCommandBuffer->handle pushDebugGroup:label];
        }
    }
}

static void METAL_PopDebugGroup(
    SDL_GPUCommandBuffer *commandBuffer)
{
    @autoreleasepool {
        MetalCommandBuffer *metalCommandBuffer = (MetalCommandBuffer *)commandBuffer;

        if (metalCommandBuffer->renderEncoder) {
            [metalCommandBuffer->renderEncoder popDebugGroup];
        } else if (metalCommandBuffer->blitEncoder) {
            [metalCommandBuffer->blitEncoder popDebugGroup];
        } else if (metalCommandBuffer->computeEncoder) {
            [metalCommandBuffer->computeEncoder popDebugGroup];
        } else {
            [metalCommandBuffer->handle popDebugGroup];
        }
    }
}

// Resource Creation

static SDL_GPUSampler *METAL_CreateSampler(
    SDL_GPURenderer *driverData,
    const SDL_GPUSamplerCreateInfo *createinfo)
{
    @autoreleasepool {
        MetalRenderer *renderer = (MetalRenderer *)driverData;
        MTLSamplerDescriptor *samplerDesc = [MTLSamplerDescriptor new];
        id<MTLSamplerState> sampler;
        MetalSampler *metalSampler;

        samplerDesc.sAddressMode = SDLToMetal_SamplerAddressMode[createinfo->address_mode_u];
        samplerDesc.tAddressMode = SDLToMetal_SamplerAddressMode[createinfo->address_mode_v];
        samplerDesc.rAddressMode = SDLToMetal_SamplerAddressMode[createinfo->address_mode_w];
        samplerDesc.minFilter = SDLToMetal_MinMagFilter[createinfo->min_filter];
        samplerDesc.magFilter = SDLToMetal_MinMagFilter[createinfo->mag_filter];
        samplerDesc.mipFilter = SDLToMetal_MipFilter[createinfo->mipmap_mode]; // FIXME: Is this right with non-mipmapped samplers?
        samplerDesc.lodMinClamp = createinfo->min_lod;
        samplerDesc.lodMaxClamp = createinfo->max_lod;
        samplerDesc.maxAnisotropy = (NSUInteger)((createinfo->enable_anisotropy) ? createinfo->max_anisotropy : 1);
        samplerDesc.compareFunction = (createinfo->enable_compare) ? SDLToMetal_CompareOp[createinfo->compare_op] : MTLCompareFunctionAlways;

        if (renderer->debugMode && SDL_HasProperty(createinfo->props, SDL_PROP_GPU_SAMPLER_CREATE_NAME_STRING)) {
            const char *name = SDL_GetStringProperty(createinfo->props, SDL_PROP_GPU_SAMPLER_CREATE_NAME_STRING, NULL);
            samplerDesc.label = @(name);
        }

        sampler = [renderer->device newSamplerStateWithDescriptor:samplerDesc];
        if (sampler == NULL) {
            SET_STRING_ERROR_AND_RETURN("Failed to create sampler", NULL);
        }

        metalSampler = (MetalSampler *)SDL_calloc(1, sizeof(MetalSampler));
        metalSampler->handle = sampler;
        return (SDL_GPUSampler *)metalSampler;
    }
}

static SDL_GPUShader *METAL_CreateShader(
    SDL_GPURenderer *driverData,
    const SDL_GPUShaderCreateInfo *createinfo)
{
    @autoreleasepool {
        MetalLibraryFunction libraryFunction;
        MetalShader *result;

        libraryFunction = METAL_INTERNAL_CompileShader(
            (MetalRenderer *)driverData,
            createinfo->format,
            createinfo->code,
            createinfo->code_size,
            createinfo->entrypoint);

        if (libraryFunction.library == nil || libraryFunction.function == nil) {
            return NULL;
        }

        result = SDL_calloc(1, sizeof(MetalShader));
        result->library = libraryFunction.library;
        result->function = libraryFunction.function;
        result->stage = createinfo->stage;
        result->numSamplers = createinfo->num_samplers;
        result->numStorageBuffers = createinfo->num_storage_buffers;
        result->numStorageTextures = createinfo->num_storage_textures;
        result->numUniformBuffers = createinfo->num_uniform_buffers;
        return (SDL_GPUShader *)result;
    }
}

// This function assumes that it's called from within an autorelease pool
static MetalTexture *METAL_INTERNAL_CreateTexture(
    MetalRenderer *renderer,
    const SDL_GPUTextureCreateInfo *createinfo)
{
    MTLTextureDescriptor *textureDescriptor = [MTLTextureDescriptor new];
    id<MTLTexture> texture;
    MetalTexture *metalTexture;

    textureDescriptor.textureType = SDLToMetal_TextureType(createinfo->type, createinfo->sample_count > SDL_GPU_SAMPLECOUNT_1);
    textureDescriptor.pixelFormat = SDLToMetal_TextureFormat(createinfo->format);
    // This format isn't natively supported so let's swizzle!
    if (createinfo->format == SDL_GPU_TEXTUREFORMAT_B4G4R4A4_UNORM) {
        if (@available(macOS 10.15, iOS 13.0, tvOS 13.0, *)) {
            textureDescriptor.swizzle = MTLTextureSwizzleChannelsMake(MTLTextureSwizzleBlue,
                                                                      MTLTextureSwizzleGreen,
                                                                      MTLTextureSwizzleRed,
                                                                      MTLTextureSwizzleAlpha);
        } else {
            SET_STRING_ERROR_AND_RETURN("SDL_GPU_TEXTUREFORMAT_B4G4R4A4_UNORM is not supported", NULL);
        }
    }

    textureDescriptor.width = createinfo->width;
    textureDescriptor.height = createinfo->height;
    textureDescriptor.depth = (createinfo->type == SDL_GPU_TEXTURETYPE_3D) ? createinfo->layer_count_or_depth : 1;
    textureDescriptor.mipmapLevelCount = createinfo->num_levels;
    textureDescriptor.sampleCount = SDLToMetal_SampleCount[createinfo->sample_count];
    textureDescriptor.arrayLength =
        (createinfo->type == SDL_GPU_TEXTURETYPE_2D_ARRAY || createinfo->type == SDL_GPU_TEXTURETYPE_CUBE_ARRAY)
            ? createinfo->layer_count_or_depth
            : 1;
    textureDescriptor.storageMode = MTLStorageModePrivate;

    textureDescriptor.usage = 0;
    if (createinfo->usage & (SDL_GPU_TEXTUREUSAGE_COLOR_TARGET |
                             SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET)) {
        textureDescriptor.usage |= MTLTextureUsageRenderTarget;
    }
    if (createinfo->usage & (SDL_GPU_TEXTUREUSAGE_SAMPLER |
                             SDL_GPU_TEXTUREUSAGE_GRAPHICS_STORAGE_READ |
                             SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_READ)) {
        textureDescriptor.usage |= MTLTextureUsageShaderRead;
    }
    if (createinfo->usage & (SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE |
                             SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_SIMULTANEOUS_READ_WRITE)) {
        textureDescriptor.usage |= MTLTextureUsageShaderWrite;
    }

    texture = [renderer->device newTextureWithDescriptor:textureDescriptor];
    if (texture == NULL) {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Failed to create MTLTexture!");
        return NULL;
    }

    metalTexture = (MetalTexture *)SDL_calloc(1, sizeof(MetalTexture));
    metalTexture->handle = texture;
    SDL_SetAtomicInt(&metalTexture->referenceCount, 0);

    if (renderer->debugMode && SDL_HasProperty(createinfo->props, SDL_PROP_GPU_TEXTURE_CREATE_NAME_STRING)) {
        metalTexture->handle.label = @(SDL_GetStringProperty(createinfo->props, SDL_PROP_GPU_TEXTURE_CREATE_NAME_STRING, NULL));
    }

    return metalTexture;
}

static bool METAL_SupportsSampleCount(
    SDL_GPURenderer *driverData,
    SDL_GPUTextureFormat format,
    SDL_GPUSampleCount sampleCount)
{
    @autoreleasepool {
        MetalRenderer *renderer = (MetalRenderer *)driverData;
        NSUInteger mtlSampleCount = SDLToMetal_SampleCount[sampleCount];
        return [renderer->device supportsTextureSampleCount:mtlSampleCount];
    }
}

static SDL_GPUTexture *METAL_CreateTexture(
    SDL_GPURenderer *driverData,
    const SDL_GPUTextureCreateInfo *createinfo)
{
    @autoreleasepool {
        MetalRenderer *renderer = (MetalRenderer *)driverData;
        MetalTextureContainer *container;
        MetalTexture *texture;

        texture = METAL_INTERNAL_CreateTexture(
            renderer,
            createinfo);

        if (texture == NULL) {
            SET_STRING_ERROR_AND_RETURN("Failed to create texture", NULL);
        }

        container = SDL_calloc(1, sizeof(MetalTextureContainer));
        container->canBeCycled = 1;

        // Copy properties so we don't lose information when the client destroys them
        container->header.info = *createinfo;
        container->header.info.props = SDL_CreateProperties();
        if (createinfo->props) {
            SDL_CopyProperties(createinfo->props, container->header.info.props);
        }

        container->activeTexture = texture;
        container->textureCapacity = 1;
        container->textureCount = 1;
        container->textures = SDL_calloc(
            container->textureCapacity, sizeof(MetalTexture *));
        container->textures[0] = texture;
        container->debugName = NULL;

        if (SDL_HasProperty(createinfo->props, SDL_PROP_GPU_TEXTURE_CREATE_NAME_STRING)) {
            container->debugName = SDL_strdup(SDL_GetStringProperty(createinfo->props, SDL_PROP_GPU_TEXTURE_CREATE_NAME_STRING, NULL));
        }

        return (SDL_GPUTexture *)container;
    }
}

// This function assumes that it's called from within an autorelease pool
static MetalTexture *METAL_INTERNAL_PrepareTextureForWrite(
    MetalRenderer *renderer,
    MetalTextureContainer *container,
    bool cycle)
{
    Uint32 i;

    // Cycle the active texture handle if needed
    if (cycle && container->canBeCycled) {
        for (i = 0; i < container->textureCount; i += 1) {
            if (SDL_GetAtomicInt(&container->textures[i]->referenceCount) == 0) {
                container->activeTexture = container->textures[i];
                return container->activeTexture;
            }
        }

        EXPAND_ARRAY_IF_NEEDED(
            container->textures,
            MetalTexture *,
            container->textureCount + 1,
            container->textureCapacity,
            container->textureCapacity + 1);

        container->textures[container->textureCount] = METAL_INTERNAL_CreateTexture(
            renderer,
            &container->header.info);
        container->textureCount += 1;

        container->activeTexture = container->textures[container->textureCount - 1];
    }

    return container->activeTexture;
}

// This function assumes that it's called from within an autorelease pool
static MetalBuffer *METAL_INTERNAL_CreateBuffer(
    MetalRenderer *renderer,
    Uint32 size,
    MTLResourceOptions resourceOptions,
    const char *debugName)
{
    id<MTLBuffer> bufferHandle;
    MetalBuffer *metalBuffer;

    // Storage buffers have to be 4-aligned, so might as well align them all
    size = METAL_INTERNAL_NextHighestAlignment(size, 4);

    bufferHandle = [renderer->device newBufferWithLength:size options:resourceOptions];
    if (bufferHandle == NULL) {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Could not create buffer");
        return NULL;
    }

    metalBuffer = SDL_calloc(1, sizeof(MetalBuffer));
    metalBuffer->handle = bufferHandle;
    SDL_SetAtomicInt(&metalBuffer->referenceCount, 0);

    if (debugName != NULL) {
        metalBuffer->handle.label = @(debugName);
    }

    return metalBuffer;
}

// This function assumes that it's called from within an autorelease pool
static MetalBufferContainer *METAL_INTERNAL_CreateBufferContainer(
    MetalRenderer *renderer,
    Uint32 size,
    bool isPrivate,
    bool isWriteOnly,
    const char *debugName)
{
    MetalBufferContainer *container = SDL_calloc(1, sizeof(MetalBufferContainer));
    MTLResourceOptions resourceOptions;

    container->size = size;
    container->bufferCapacity = 1;
    container->bufferCount = 1;
    container->buffers = SDL_calloc(
        container->bufferCapacity, sizeof(MetalBuffer *));
    container->isPrivate = isPrivate;
    container->isWriteOnly = isWriteOnly;
    container->debugName = NULL;
    if (container->debugName != NULL) {
        container->debugName = SDL_strdup(debugName);
    }

    if (isPrivate) {
        resourceOptions = MTLResourceStorageModePrivate;
    } else {
        if (isWriteOnly) {
            resourceOptions = MTLResourceCPUCacheModeWriteCombined;
        } else {
            resourceOptions = MTLResourceCPUCacheModeDefaultCache;
        }
    }

    container->buffers[0] = METAL_INTERNAL_CreateBuffer(
        renderer,
        size,
        resourceOptions,
        debugName);

    container->activeBuffer = container->buffers[0];

    return container;
}

static SDL_GPUBuffer *METAL_CreateBuffer(
    SDL_GPURenderer *driverData,
    SDL_GPUBufferUsageFlags usage,
    Uint32 size,
    const char *debugName)
{
    @autoreleasepool {
        return (SDL_GPUBuffer *)METAL_INTERNAL_CreateBufferContainer(
            (MetalRenderer *)driverData,
            size,
            true,
            false,
            debugName);
    }
}

static SDL_GPUTransferBuffer *METAL_CreateTransferBuffer(
    SDL_GPURenderer *driverData,
    SDL_GPUTransferBufferUsage usage,
    Uint32 size,
    const char *debugName)
{
    @autoreleasepool {
        return (SDL_GPUTransferBuffer *)METAL_INTERNAL_CreateBufferContainer(
            (MetalRenderer *)driverData,
            size,
            false,
            usage == SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
            debugName);
    }
}

// This function assumes that it's called from within an autorelease pool
static MetalUniformBuffer *METAL_INTERNAL_CreateUniformBuffer(
    MetalRenderer *renderer,
    Uint32 size)
{
    MetalUniformBuffer *uniformBuffer;
    id<MTLBuffer> bufferHandle;

    bufferHandle = [renderer->device newBufferWithLength:size options:MTLResourceCPUCacheModeWriteCombined];
    if (bufferHandle == nil) {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Could not create uniform buffer");
        return NULL;
    }

    uniformBuffer = SDL_calloc(1, sizeof(MetalUniformBuffer));
    uniformBuffer->handle = bufferHandle;
    uniformBuffer->writeOffset = 0;
    uniformBuffer->drawOffset = 0;

    return uniformBuffer;
}

// This function assumes that it's called from within an autorelease pool
static MetalBuffer *METAL_INTERNAL_PrepareBufferForWrite(
    MetalRenderer *renderer,
    MetalBufferContainer *container,
    bool cycle)
{
    MTLResourceOptions resourceOptions;
    Uint32 i;

    // Cycle if needed
    if (cycle && SDL_GetAtomicInt(&container->activeBuffer->referenceCount) > 0) {
        for (i = 0; i < container->bufferCount; i += 1) {
            if (SDL_GetAtomicInt(&container->buffers[i]->referenceCount) == 0) {
                container->activeBuffer = container->buffers[i];
                return container->activeBuffer;
            }
        }

        EXPAND_ARRAY_IF_NEEDED(
            container->buffers,
            MetalBuffer *,
            container->bufferCount + 1,
            container->bufferCapacity,
            container->bufferCapacity + 1);

        if (container->isPrivate) {
            resourceOptions = MTLResourceStorageModePrivate;
        } else {
            if (container->isWriteOnly) {
                resourceOptions = MTLResourceCPUCacheModeWriteCombined;
            } else {
                resourceOptions = MTLResourceCPUCacheModeDefaultCache;
            }
        }

        container->buffers[container->bufferCount] = METAL_INTERNAL_CreateBuffer(
            renderer,
            container->size,
            resourceOptions,
            container->debugName);
        container->bufferCount += 1;

        container->activeBuffer = container->buffers[container->bufferCount - 1];
    }

    return container->activeBuffer;
}

// TransferBuffer Data

static void *METAL_MapTransferBuffer(
    SDL_GPURenderer *driverData,
    SDL_GPUTransferBuffer *transferBuffer,
    bool cycle)
{
    @autoreleasepool {
        MetalRenderer *renderer = (MetalRenderer *)driverData;
        MetalBufferContainer *container = (MetalBufferContainer *)transferBuffer;
        MetalBuffer *buffer = METAL_INTERNAL_PrepareBufferForWrite(renderer, container, cycle);
        return [buffer->handle contents];
    }
}

static void METAL_UnmapTransferBuffer(
    SDL_GPURenderer *driverData,
    SDL_GPUTransferBuffer *transferBuffer)
{
#ifdef SDL_PLATFORM_MACOS
    @autoreleasepool {
        // FIXME: Is this necessary?
        MetalBufferContainer *container = (MetalBufferContainer *)transferBuffer;
        MetalBuffer *buffer = container->activeBuffer;
        if (buffer->handle.storageMode == MTLStorageModeManaged) {
            [buffer->handle didModifyRange:NSMakeRange(0, container->size)];
        }
    }
#endif
}

// Copy Pass

static void METAL_BeginCopyPass(
    SDL_GPUCommandBuffer *commandBuffer)
{
    @autoreleasepool {
        MetalCommandBuffer *metalCommandBuffer = (MetalCommandBuffer *)commandBuffer;
        metalCommandBuffer->blitEncoder = [metalCommandBuffer->handle blitCommandEncoder];
    }
}

static void METAL_UploadToTexture(
    SDL_GPUCommandBuffer *commandBuffer,
    const SDL_GPUTextureTransferInfo *source,
    const SDL_GPUTextureRegion *destination,
    bool cycle)
{
    @autoreleasepool {
        MetalCommandBuffer *metalCommandBuffer = (MetalCommandBuffer *)commandBuffer;
        MetalRenderer *renderer = metalCommandBuffer->renderer;
        MetalBufferContainer *bufferContainer = (MetalBufferContainer *)source->transfer_buffer;
        MetalTextureContainer *textureContainer = (MetalTextureContainer *)destination->texture;

        MetalTexture *metalTexture = METAL_INTERNAL_PrepareTextureForWrite(renderer, textureContainer, cycle);

        [metalCommandBuffer->blitEncoder
                 copyFromBuffer:bufferContainer->activeBuffer->handle
                   sourceOffset:source->offset
              sourceBytesPerRow:BytesPerRow(destination->w, textureContainer->header.info.format)
            sourceBytesPerImage:SDL_CalculateGPUTextureFormatSize(textureContainer->header.info.format, destination->w, destination->h, destination->d)
                     sourceSize:MTLSizeMake(destination->w, destination->h, destination->d)
                      toTexture:metalTexture->handle
               destinationSlice:destination->layer
               destinationLevel:destination->mip_level
              destinationOrigin:MTLOriginMake(destination->x, destination->y, destination->z)];

        METAL_INTERNAL_TrackTexture(metalCommandBuffer, metalTexture);
        METAL_INTERNAL_TrackBuffer(metalCommandBuffer, bufferContainer->activeBuffer);
    }
}

static void METAL_UploadToBuffer(
    SDL_GPUCommandBuffer *commandBuffer,
    const SDL_GPUTransferBufferLocation *source,
    const SDL_GPUBufferRegion *destination,
    bool cycle)
{
    @autoreleasepool {
        MetalCommandBuffer *metalCommandBuffer = (MetalCommandBuffer *)commandBuffer;
        MetalRenderer *renderer = metalCommandBuffer->renderer;
        MetalBufferContainer *transferContainer = (MetalBufferContainer *)source->transfer_buffer;
        MetalBufferContainer *bufferContainer = (MetalBufferContainer *)destination->buffer;

        MetalBuffer *metalBuffer = METAL_INTERNAL_PrepareBufferForWrite(
            renderer,
            bufferContainer,
            cycle);

        [metalCommandBuffer->blitEncoder
               copyFromBuffer:transferContainer->activeBuffer->handle
                 sourceOffset:source->offset
                     toBuffer:metalBuffer->handle
            destinationOffset:destination->offset
                         size:destination->size];

        METAL_INTERNAL_TrackBuffer(metalCommandBuffer, metalBuffer);
        METAL_INTERNAL_TrackBuffer(metalCommandBuffer, transferContainer->activeBuffer);
    }
}

static void METAL_CopyTextureToTexture(
    SDL_GPUCommandBuffer *commandBuffer,
    const SDL_GPUTextureLocation *source,
    const SDL_GPUTextureLocation *destination,
    Uint32 w,
    Uint32 h,
    Uint32 d,
    bool cycle)
{
    @autoreleasepool {
        MetalCommandBuffer *metalCommandBuffer = (MetalCommandBuffer *)commandBuffer;
        MetalRenderer *renderer = metalCommandBuffer->renderer;
        MetalTextureContainer *srcContainer = (MetalTextureContainer *)source->texture;
        MetalTextureContainer *dstContainer = (MetalTextureContainer *)destination->texture;

        MetalTexture *srcTexture = srcContainer->activeTexture;
        MetalTexture *dstTexture = METAL_INTERNAL_PrepareTextureForWrite(
            renderer,
            dstContainer,
            cycle);

        [metalCommandBuffer->blitEncoder
              copyFromTexture:srcTexture->handle
                  sourceSlice:source->layer
                  sourceLevel:source->mip_level
                 sourceOrigin:MTLOriginMake(source->x, source->y, source->z)
                   sourceSize:MTLSizeMake(w, h, d)
                    toTexture:dstTexture->handle
             destinationSlice:destination->layer
             destinationLevel:destination->mip_level
            destinationOrigin:MTLOriginMake(destination->x, destination->y, destination->z)];

        METAL_INTERNAL_TrackTexture(metalCommandBuffer, srcTexture);
        METAL_INTERNAL_TrackTexture(metalCommandBuffer, dstTexture);
    }
}

static void METAL_CopyBufferToBuffer(
    SDL_GPUCommandBuffer *commandBuffer,
    const SDL_GPUBufferLocation *source,
    const SDL_GPUBufferLocation *destination,
    Uint32 size,
    bool cycle)
{
    @autoreleasepool {
        MetalCommandBuffer *metalCommandBuffer = (MetalCommandBuffer *)commandBuffer;
        MetalRenderer *renderer = metalCommandBuffer->renderer;
        MetalBufferContainer *srcContainer = (MetalBufferContainer *)source->buffer;
        MetalBufferContainer *dstContainer = (MetalBufferContainer *)destination->buffer;

        MetalBuffer *srcBuffer = srcContainer->activeBuffer;
        MetalBuffer *dstBuffer = METAL_INTERNAL_PrepareBufferForWrite(
            renderer,
            dstContainer,
            cycle);

        [metalCommandBuffer->blitEncoder
               copyFromBuffer:srcBuffer->handle
                 sourceOffset:source->offset
                     toBuffer:dstBuffer->handle
            destinationOffset:destination->offset
                         size:size];

        METAL_INTERNAL_TrackBuffer(metalCommandBuffer, srcBuffer);
        METAL_INTERNAL_TrackBuffer(metalCommandBuffer, dstBuffer);
    }
}

static void METAL_DownloadFromTexture(
    SDL_GPUCommandBuffer *commandBuffer,
    const SDL_GPUTextureRegion *source,
    const SDL_GPUTextureTransferInfo *destination)
{
    @autoreleasepool {
        MetalCommandBuffer *metalCommandBuffer = (MetalCommandBuffer *)commandBuffer;
        MetalRenderer *renderer = metalCommandBuffer->renderer;
        MetalTextureContainer *textureContainer = (MetalTextureContainer *)source->texture;
        MetalTexture *metalTexture = textureContainer->activeTexture;
        MetalBufferContainer *bufferContainer = (MetalBufferContainer *)destination->transfer_buffer;
        Uint32 bufferStride = destination->pixels_per_row;
        Uint32 bufferImageHeight = destination->rows_per_layer;
        Uint32 bytesPerRow, bytesPerDepthSlice;

        MetalBuffer *dstBuffer = METAL_INTERNAL_PrepareBufferForWrite(
            renderer,
            bufferContainer,
            false);

        MTLOrigin regionOrigin = MTLOriginMake(
            source->x,
            source->y,
            source->z);

        MTLSize regionSize = MTLSizeMake(
            source->w,
            source->h,
            source->d);

        if (bufferStride == 0 || bufferImageHeight == 0) {
            bufferStride = source->w;
            bufferImageHeight = source->h;
        }

        bytesPerRow = BytesPerRow(bufferStride, textureContainer->header.info.format);
        bytesPerDepthSlice = bytesPerRow * bufferImageHeight;

        [metalCommandBuffer->blitEncoder
                     copyFromTexture:metalTexture->handle
                         sourceSlice:source->layer
                         sourceLevel:source->mip_level
                        sourceOrigin:regionOrigin
                          sourceSize:regionSize
                            toBuffer:dstBuffer->handle
                   destinationOffset:destination->offset
              destinationBytesPerRow:bytesPerRow
            destinationBytesPerImage:bytesPerDepthSlice];

        METAL_INTERNAL_TrackTexture(metalCommandBuffer, metalTexture);
        METAL_INTERNAL_TrackBuffer(metalCommandBuffer, dstBuffer);
    }
}

static void METAL_DownloadFromBuffer(
    SDL_GPUCommandBuffer *commandBuffer,
    const SDL_GPUBufferRegion *source,
    const SDL_GPUTransferBufferLocation *destination)
{
    SDL_GPUBufferLocation sourceLocation;
    sourceLocation.buffer = source->buffer;
    sourceLocation.offset = source->offset;

    METAL_CopyBufferToBuffer(
        commandBuffer,
        &sourceLocation,
        (SDL_GPUBufferLocation *)destination,
        source->size,
        false);
}

static void METAL_EndCopyPass(
    SDL_GPUCommandBuffer *commandBuffer)
{
    @autoreleasepool {
        MetalCommandBuffer *metalCommandBuffer = (MetalCommandBuffer *)commandBuffer;
        [metalCommandBuffer->blitEncoder endEncoding];
        metalCommandBuffer->blitEncoder = nil;
    }
}

static void METAL_GenerateMipmaps(
    SDL_GPUCommandBuffer *commandBuffer,
    SDL_GPUTexture *texture)
{
    @autoreleasepool {
        MetalCommandBuffer *metalCommandBuffer = (MetalCommandBuffer *)commandBuffer;
        MetalTextureContainer *container = (MetalTextureContainer *)texture;
        MetalTexture *metalTexture = container->activeTexture;

        METAL_BeginCopyPass(commandBuffer);
        [metalCommandBuffer->blitEncoder
            generateMipmapsForTexture:metalTexture->handle];
        METAL_EndCopyPass(commandBuffer);

        METAL_INTERNAL_TrackTexture(metalCommandBuffer, metalTexture);
    }
}

// Graphics State

static void METAL_INTERNAL_AllocateCommandBuffers(
    MetalRenderer *renderer,
    Uint32 allocateCount)
{
    MetalCommandBuffer *commandBuffer;

    renderer->availableCommandBufferCapacity += allocateCount;

    renderer->availableCommandBuffers = SDL_realloc(
        renderer->availableCommandBuffers,
        sizeof(MetalCommandBuffer *) * renderer->availableCommandBufferCapacity);

    for (Uint32 i = 0; i < allocateCount; i += 1) {
        commandBuffer = SDL_calloc(1, sizeof(MetalCommandBuffer));
        commandBuffer->renderer = renderer;

        // The native Metal command buffer is created in METAL_AcquireCommandBuffer

        commandBuffer->windowDataCapacity = 1;
        commandBuffer->windowDataCount = 0;
        commandBuffer->windowDatas = SDL_calloc(
            commandBuffer->windowDataCapacity, sizeof(MetalWindowData *));

        // Reference Counting
        commandBuffer->usedBufferCapacity = 4;
        commandBuffer->usedBufferCount = 0;
        commandBuffer->usedBuffers = SDL_calloc(
            commandBuffer->usedBufferCapacity, sizeof(MetalBuffer *));

        commandBuffer->usedTextureCapacity = 4;
        commandBuffer->usedTextureCount = 0;
        commandBuffer->usedTextures = SDL_calloc(
            commandBuffer->usedTextureCapacity, sizeof(MetalTexture *));

        renderer->availableCommandBuffers[renderer->availableCommandBufferCount] = commandBuffer;
        renderer->availableCommandBufferCount += 1;
    }
}

static MetalCommandBuffer *METAL_INTERNAL_GetInactiveCommandBufferFromPool(
    MetalRenderer *renderer)
{
    MetalCommandBuffer *commandBuffer;

    if (renderer->availableCommandBufferCount == 0) {
        METAL_INTERNAL_AllocateCommandBuffers(
            renderer,
            renderer->availableCommandBufferCapacity);
    }

    commandBuffer = renderer->availableCommandBuffers[renderer->availableCommandBufferCount - 1];
    renderer->availableCommandBufferCount -= 1;

    return commandBuffer;
}

static Uint8 METAL_INTERNAL_CreateFence(
    MetalRenderer *renderer)
{
    MetalFence *fence;

    fence = SDL_calloc(1, sizeof(MetalFence));
    SDL_SetAtomicInt(&fence->complete, 0);
    SDL_SetAtomicInt(&fence->referenceCount, 0);

    // Add it to the available pool
    // FIXME: Should this be EXPAND_IF_NEEDED?
    if (renderer->availableFenceCount >= renderer->availableFenceCapacity) {
        renderer->availableFenceCapacity *= 2;

        renderer->availableFences = SDL_realloc(
            renderer->availableFences,
            sizeof(MetalFence *) * renderer->availableFenceCapacity);
    }

    renderer->availableFences[renderer->availableFenceCount] = fence;
    renderer->availableFenceCount += 1;

    return 1;
}

static bool METAL_INTERNAL_AcquireFence(
    MetalRenderer *renderer,
    MetalCommandBuffer *commandBuffer)
{
    MetalFence *fence;

    // Acquire a fence from the pool
    SDL_LockMutex(renderer->fenceLock);

    if (renderer->availableFenceCount == 0) {
        if (!METAL_INTERNAL_CreateFence(renderer)) {
            SDL_UnlockMutex(renderer->fenceLock);
            SDL_LogError(SDL_LOG_CATEGORY_GPU, "Failed to create fence!");
            return false;
        }
    }

    fence = renderer->availableFences[renderer->availableFenceCount - 1];
    renderer->availableFenceCount -= 1;

    SDL_UnlockMutex(renderer->fenceLock);

    // Associate the fence with the command buffer
    commandBuffer->fence = fence;
    SDL_SetAtomicInt(&fence->complete, 0); // FIXME: Is this right?
    (void)SDL_AtomicIncRef(&commandBuffer->fence->referenceCount);

    return true;
}

static SDL_GPUCommandBuffer *METAL_AcquireCommandBuffer(
    SDL_GPURenderer *driverData)
{
    @autoreleasepool {
        MetalRenderer *renderer = (MetalRenderer *)driverData;
        MetalCommandBuffer *commandBuffer;

        SDL_LockMutex(renderer->acquireCommandBufferLock);

        commandBuffer = METAL_INTERNAL_GetInactiveCommandBufferFromPool(renderer);
        commandBuffer->handle = [renderer->queue commandBuffer];

        commandBuffer->graphics_pipeline = NULL;
        commandBuffer->compute_pipeline = NULL;
        for (Uint32 i = 0; i < MAX_UNIFORM_BUFFERS_PER_STAGE; i += 1) {
            commandBuffer->vertexUniformBuffers[i] = NULL;
            commandBuffer->fragmentUniformBuffers[i] = NULL;
            commandBuffer->computeUniformBuffers[i] = NULL;
        }

        commandBuffer->autoReleaseFence = true;

        SDL_UnlockMutex(renderer->acquireCommandBufferLock);

        return (SDL_GPUCommandBuffer *)commandBuffer;
    }
}

// This function assumes that it's called from within an autorelease pool
static MetalUniformBuffer *METAL_INTERNAL_AcquireUniformBufferFromPool(
    MetalCommandBuffer *commandBuffer)
{
    MetalRenderer *renderer = commandBuffer->renderer;
    MetalUniformBuffer *uniformBuffer;

    SDL_LockMutex(renderer->acquireUniformBufferLock);

    if (renderer->uniformBufferPoolCount > 0) {
        uniformBuffer = renderer->uniformBufferPool[renderer->uniformBufferPoolCount - 1];
        renderer->uniformBufferPoolCount -= 1;
    } else {
        uniformBuffer = METAL_INTERNAL_CreateUniformBuffer(
            renderer,
            UNIFORM_BUFFER_SIZE);
    }

    SDL_UnlockMutex(renderer->acquireUniformBufferLock);

    METAL_INTERNAL_TrackUniformBuffer(commandBuffer, uniformBuffer);

    return uniformBuffer;
}

static void METAL_INTERNAL_ReturnUniformBufferToPool(
    MetalRenderer *renderer,
    MetalUniformBuffer *uniformBuffer)
{
    if (renderer->uniformBufferPoolCount >= renderer->uniformBufferPoolCapacity) {
        renderer->uniformBufferPoolCapacity *= 2;
        renderer->uniformBufferPool = SDL_realloc(
            renderer->uniformBufferPool,
            renderer->uniformBufferPoolCapacity * sizeof(MetalUniformBuffer *));
    }

    renderer->uniformBufferPool[renderer->uniformBufferPoolCount] = uniformBuffer;
    renderer->uniformBufferPoolCount += 1;

    uniformBuffer->writeOffset = 0;
    uniformBuffer->drawOffset = 0;
}

static void METAL_SetViewport(
    SDL_GPUCommandBuffer *commandBuffer,
    const SDL_GPUViewport *viewport)
{
    @autoreleasepool {
        MetalCommandBuffer *metalCommandBuffer = (MetalCommandBuffer *)commandBuffer;
        MTLViewport metalViewport;

        metalViewport.originX = viewport->x;
        metalViewport.originY = viewport->y;
        metalViewport.width = viewport->w;
        metalViewport.height = viewport->h;
        metalViewport.znear = viewport->min_depth;
        metalViewport.zfar = viewport->max_depth;

        [metalCommandBuffer->renderEncoder setViewport:metalViewport];
    }
}

static void METAL_SetScissor(
    SDL_GPUCommandBuffer *commandBuffer,
    const SDL_Rect *scissor)
{
    @autoreleasepool {
        MetalCommandBuffer *metalCommandBuffer = (MetalCommandBuffer *)commandBuffer;
        MTLScissorRect metalScissor;

        metalScissor.x = scissor->x;
        metalScissor.y = scissor->y;
        metalScissor.width = scissor->w;
        metalScissor.height = scissor->h;

        [metalCommandBuffer->renderEncoder setScissorRect:metalScissor];
    }
}

static void METAL_SetBlendConstants(
    SDL_GPUCommandBuffer *commandBuffer,
    SDL_FColor blendConstants)
{
    @autoreleasepool {
        MetalCommandBuffer *metalCommandBuffer = (MetalCommandBuffer *)commandBuffer;
        [metalCommandBuffer->renderEncoder setBlendColorRed:blendConstants.r
                                                      green:blendConstants.g
                                                       blue:blendConstants.b
                                                      alpha:blendConstants.a];
    }
}

static void METAL_SetStencilReference(
    SDL_GPUCommandBuffer *commandBuffer,
    Uint8 reference)
{
    @autoreleasepool {
        MetalCommandBuffer *metalCommandBuffer = (MetalCommandBuffer *)commandBuffer;
        [metalCommandBuffer->renderEncoder setStencilReferenceValue:reference];
    }
}

static void METAL_BeginRenderPass(
    SDL_GPUCommandBuffer *commandBuffer,
    const SDL_GPUColorTargetInfo *colorTargetInfos,
    Uint32 numColorTargets,
    const SDL_GPUDepthStencilTargetInfo *depthStencilTargetInfo)
{
    @autoreleasepool {
        MetalCommandBuffer *metalCommandBuffer = (MetalCommandBuffer *)commandBuffer;
        MetalRenderer *renderer = metalCommandBuffer->renderer;
        MTLRenderPassDescriptor *passDescriptor = [MTLRenderPassDescriptor renderPassDescriptor];
        Uint32 vpWidth = UINT_MAX;
        Uint32 vpHeight = UINT_MAX;
        SDL_GPUViewport viewport;
        SDL_Rect scissorRect;
        SDL_FColor blendConstants;

        for (Uint32 i = 0; i < numColorTargets; i += 1) {
            MetalTextureContainer *container = (MetalTextureContainer *)colorTargetInfos[i].texture;
            MetalTexture *texture = METAL_INTERNAL_PrepareTextureForWrite(
                renderer,
                container,
                colorTargetInfos[i].cycle);

            passDescriptor.colorAttachments[i].texture = texture->handle;
            passDescriptor.colorAttachments[i].level = colorTargetInfos[i].mip_level;
            if (container->header.info.type == SDL_GPU_TEXTURETYPE_3D) {
                passDescriptor.colorAttachments[i].depthPlane = colorTargetInfos[i].layer_or_depth_plane;
            } else {
                passDescriptor.colorAttachments[i].slice = colorTargetInfos[i].layer_or_depth_plane;
            }
            passDescriptor.colorAttachments[i].clearColor = MTLClearColorMake(
                colorTargetInfos[i].clear_color.r,
                colorTargetInfos[i].clear_color.g,
                colorTargetInfos[i].clear_color.b,
                colorTargetInfos[i].clear_color.a);
            passDescriptor.colorAttachments[i].loadAction = SDLToMetal_LoadOp[colorTargetInfos[i].load_op];
            passDescriptor.colorAttachments[i].storeAction = SDLToMetal_StoreOp[colorTargetInfos[i].store_op];

            METAL_INTERNAL_TrackTexture(metalCommandBuffer, texture);

            if (colorTargetInfos[i].store_op == SDL_GPU_STOREOP_RESOLVE || colorTargetInfos[i].store_op == SDL_GPU_STOREOP_RESOLVE_AND_STORE) {
                MetalTextureContainer *resolveContainer = (MetalTextureContainer *)colorTargetInfos[i].resolve_texture;
                MetalTexture *resolveTexture = METAL_INTERNAL_PrepareTextureForWrite(
                    renderer,
                    resolveContainer,
                    colorTargetInfos[i].cycle_resolve_texture);

                passDescriptor.colorAttachments[i].resolveTexture = resolveTexture->handle;
                passDescriptor.colorAttachments[i].resolveSlice = colorTargetInfos[i].resolve_layer;
                passDescriptor.colorAttachments[i].resolveLevel = colorTargetInfos[i].resolve_mip_level;

                METAL_INTERNAL_TrackTexture(metalCommandBuffer, resolveTexture);
            }
        }

        if (depthStencilTargetInfo != NULL) {
            MetalTextureContainer *container = (MetalTextureContainer *)depthStencilTargetInfo->texture;
            MetalTexture *texture = METAL_INTERNAL_PrepareTextureForWrite(
                renderer,
                container,
                depthStencilTargetInfo->cycle);

            passDescriptor.depthAttachment.texture = texture->handle;
            passDescriptor.depthAttachment.loadAction = SDLToMetal_LoadOp[depthStencilTargetInfo->load_op];
            passDescriptor.depthAttachment.storeAction = SDLToMetal_StoreOp[depthStencilTargetInfo->store_op];
            passDescriptor.depthAttachment.clearDepth = depthStencilTargetInfo->clear_depth;

            if (IsStencilFormat(container->header.info.format)) {
                passDescriptor.stencilAttachment.texture = texture->handle;
                passDescriptor.stencilAttachment.loadAction = SDLToMetal_LoadOp[depthStencilTargetInfo->stencil_load_op];
                passDescriptor.stencilAttachment.storeAction = SDLToMetal_StoreOp[depthStencilTargetInfo->stencil_store_op];
                passDescriptor.stencilAttachment.clearStencil = depthStencilTargetInfo->clear_stencil;
            }

            METAL_INTERNAL_TrackTexture(metalCommandBuffer, texture);
        }

        metalCommandBuffer->renderEncoder = [metalCommandBuffer->handle renderCommandEncoderWithDescriptor:passDescriptor];

        // The viewport cannot be larger than the smallest target.
        for (Uint32 i = 0; i < numColorTargets; i += 1) {
            MetalTextureContainer *container = (MetalTextureContainer *)colorTargetInfos[i].texture;
            Uint32 w = container->header.info.width >> colorTargetInfos[i].mip_level;
            Uint32 h = container->header.info.height >> colorTargetInfos[i].mip_level;

            if (w < vpWidth) {
                vpWidth = w;
            }

            if (h < vpHeight) {
                vpHeight = h;
            }
        }

        if (depthStencilTargetInfo != NULL) {
            MetalTextureContainer *container = (MetalTextureContainer *)depthStencilTargetInfo->texture;
            Uint32 w = container->header.info.width;
            Uint32 h = container->header.info.height;

            if (w < vpWidth) {
                vpWidth = w;
            }

            if (h < vpHeight) {
                vpHeight = h;
            }
        }

        // Set sensible default states
        viewport.x = 0;
        viewport.y = 0;
        viewport.w = vpWidth;
        viewport.h = vpHeight;
        viewport.min_depth = 0;
        viewport.max_depth = 1;
        METAL_SetViewport(commandBuffer, &viewport);

        scissorRect.x = 0;
        scissorRect.y = 0;
        scissorRect.w = vpWidth;
        scissorRect.h = vpHeight;
        METAL_SetScissor(commandBuffer, &scissorRect);

        blendConstants.r = 1.0f;
        blendConstants.g = 1.0f;
        blendConstants.b = 1.0f;
        blendConstants.a = 1.0f;
        METAL_SetBlendConstants(
            commandBuffer,
            blendConstants);

        METAL_SetStencilReference(
            commandBuffer,
            0);
    }
}

static void METAL_BindGraphicsPipeline(
    SDL_GPUCommandBuffer *commandBuffer,
    SDL_GPUGraphicsPipeline *graphicsPipeline)
{
    @autoreleasepool {
        MetalCommandBuffer *metalCommandBuffer = (MetalCommandBuffer *)commandBuffer;
        MetalGraphicsPipeline *pipeline = (MetalGraphicsPipeline *)graphicsPipeline;
        SDL_GPURasterizerState *rast = &pipeline->rasterizerState;
        Uint32 i;

        metalCommandBuffer->graphics_pipeline = pipeline;

        [metalCommandBuffer->renderEncoder setRenderPipelineState:pipeline->handle];

        // Apply rasterizer state
        [metalCommandBuffer->renderEncoder setTriangleFillMode:SDLToMetal_PolygonMode[pipeline->rasterizerState.fill_mode]];
        [metalCommandBuffer->renderEncoder setCullMode:SDLToMetal_CullMode[pipeline->rasterizerState.cull_mode]];
        [metalCommandBuffer->renderEncoder setFrontFacingWinding:SDLToMetal_FrontFace[pipeline->rasterizerState.front_face]];
        [metalCommandBuffer->renderEncoder setDepthClipMode:SDLToMetal_DepthClipMode(pipeline->rasterizerState.enable_depth_clip)];
        [metalCommandBuffer->renderEncoder
            setDepthBias:((rast->enable_depth_bias) ? rast->depth_bias_constant_factor : 0)
              slopeScale:((rast->enable_depth_bias) ? rast->depth_bias_slope_factor : 0)
              clamp:((rast->enable_depth_bias) ? rast->depth_bias_clamp : 0)];

        // Apply depth-stencil state
        if (pipeline->depth_stencil_state != NULL) {
            [metalCommandBuffer->renderEncoder
                setDepthStencilState:pipeline->depth_stencil_state];
        }

        for (i = 0; i < MAX_UNIFORM_BUFFERS_PER_STAGE; i += 1) {
            metalCommandBuffer->needVertexUniformBufferBind[i] = true;
            metalCommandBuffer->needFragmentUniformBufferBind[i] = true;
        }

        for (i = 0; i < pipeline->vertexUniformBufferCount; i += 1) {
            if (metalCommandBuffer->vertexUniformBuffers[i] == NULL) {
                metalCommandBuffer->vertexUniformBuffers[i] = METAL_INTERNAL_AcquireUniformBufferFromPool(
                    metalCommandBuffer);
            }
        }

        for (i = 0; i < pipeline->fragmentUniformBufferCount; i += 1) {
            if (metalCommandBuffer->fragmentUniformBuffers[i] == NULL) {
                metalCommandBuffer->fragmentUniformBuffers[i] = METAL_INTERNAL_AcquireUniformBufferFromPool(
                    metalCommandBuffer);
            }
        }
    }
}

static void METAL_BindVertexBuffers(
    SDL_GPUCommandBuffer *commandBuffer,
    Uint32 firstSlot,
    const SDL_GPUBufferBinding *bindings,
    Uint32 numBindings)
{
    MetalCommandBuffer *metalCommandBuffer = (MetalCommandBuffer *)commandBuffer;

    for (Uint32 i = 0; i < numBindings; i += 1) {
        MetalBuffer *currentBuffer = ((MetalBufferContainer *)bindings[i].buffer)->activeBuffer;
        if (metalCommandBuffer->vertexBuffers[firstSlot + i] != currentBuffer->handle || metalCommandBuffer->vertexBufferOffsets[firstSlot + i] != bindings[i].offset) {
            metalCommandBuffer->vertexBuffers[firstSlot + i] = currentBuffer->handle;
            metalCommandBuffer->vertexBufferOffsets[firstSlot + i] = bindings[i].offset;
            metalCommandBuffer->needVertexBufferBind = true;
            METAL_INTERNAL_TrackBuffer(metalCommandBuffer, currentBuffer);
        }
    }

    metalCommandBuffer->vertexBufferCount =
        SDL_max(metalCommandBuffer->vertexBufferCount, firstSlot + numBindings);
}

static void METAL_BindIndexBuffer(
    SDL_GPUCommandBuffer *commandBuffer,
    const SDL_GPUBufferBinding *binding,
    SDL_GPUIndexElementSize indexElementSize)
{
    MetalCommandBuffer *metalCommandBuffer = (MetalCommandBuffer *)commandBuffer;
    metalCommandBuffer->indexBuffer = ((MetalBufferContainer *)binding->buffer)->activeBuffer;
    metalCommandBuffer->indexBufferOffset = binding->offset;
    metalCommandBuffer->index_element_size = indexElementSize;

    METAL_INTERNAL_TrackBuffer(metalCommandBuffer, metalCommandBuffer->indexBuffer);
}

static void METAL_BindVertexSamplers(
    SDL_GPUCommandBuffer *commandBuffer,
    Uint32 firstSlot,
    const SDL_GPUTextureSamplerBinding *textureSamplerBindings,
    Uint32 numBindings)
{
    MetalCommandBuffer *metalCommandBuffer = (MetalCommandBuffer *)commandBuffer;
    MetalTextureContainer *textureContainer;
    MetalSampler *sampler;

    for (Uint32 i = 0; i < numBindings; i += 1) {
        textureContainer = (MetalTextureContainer *)textureSamplerBindings[i].texture;
        sampler = (MetalSampler *)textureSamplerBindings[i].sampler;

        if (metalCommandBuffer->vertexSamplers[firstSlot + i] != sampler->handle) {
            metalCommandBuffer->vertexSamplers[firstSlot + i] = sampler->handle;
            metalCommandBuffer->needVertexSamplerBind  = true;
        }

        if (metalCommandBuffer->vertexTextures[firstSlot + i] != textureContainer->activeTexture->handle) {
            METAL_INTERNAL_TrackTexture(
                metalCommandBuffer,
                textureContainer->activeTexture);

            metalCommandBuffer->vertexTextures[firstSlot + i] =
                textureContainer->activeTexture->handle;

            metalCommandBuffer->needVertexSamplerBind  = true;
        }
    }
}

static void METAL_BindVertexStorageTextures(
    SDL_GPUCommandBuffer *commandBuffer,
    Uint32 firstSlot,
    SDL_GPUTexture *const *storageTextures,
    Uint32 numBindings)
{
    MetalCommandBuffer *metalCommandBuffer = (MetalCommandBuffer *)commandBuffer;
    MetalTextureContainer *textureContainer;

    for (Uint32 i = 0; i < numBindings; i += 1) {
        textureContainer = (MetalTextureContainer *)storageTextures[i];

        if (metalCommandBuffer->vertexStorageTextures[firstSlot + i] != textureContainer->activeTexture->handle) {
            METAL_INTERNAL_TrackTexture(
                metalCommandBuffer,
                textureContainer->activeTexture);

            metalCommandBuffer->vertexStorageTextures[firstSlot + i] =
                textureContainer->activeTexture->handle;

            metalCommandBuffer->needVertexStorageTextureBind = true;
        }
    }
}

static void METAL_BindVertexStorageBuffers(
    SDL_GPUCommandBuffer *commandBuffer,
    Uint32 firstSlot,
    SDL_GPUBuffer *const *storageBuffers,
    Uint32 numBindings)
{
    MetalCommandBuffer *metalCommandBuffer = (MetalCommandBuffer *)commandBuffer;
    MetalBufferContainer *bufferContainer;

    for (Uint32 i = 0; i < numBindings; i += 1) {
        bufferContainer = (MetalBufferContainer *)storageBuffers[i];

        if (metalCommandBuffer->vertexStorageBuffers[firstSlot + i] != bufferContainer->activeBuffer->handle) {
            METAL_INTERNAL_TrackBuffer(
                metalCommandBuffer,
                bufferContainer->activeBuffer);

            metalCommandBuffer->vertexStorageBuffers[firstSlot + i] =
                bufferContainer->activeBuffer->handle;

            metalCommandBuffer->needVertexStorageBufferBind = true;
        }
    }
}

static void METAL_BindFragmentSamplers(
    SDL_GPUCommandBuffer *commandBuffer,
    Uint32 firstSlot,
    const SDL_GPUTextureSamplerBinding *textureSamplerBindings,
    Uint32 numBindings)
{
    MetalCommandBuffer *metalCommandBuffer = (MetalCommandBuffer *)commandBuffer;
    MetalTextureContainer *textureContainer;
    MetalSampler *sampler;

    for (Uint32 i = 0; i < numBindings; i += 1) {
        textureContainer = (MetalTextureContainer *)textureSamplerBindings[i].texture;
        sampler = (MetalSampler *)textureSamplerBindings[i].sampler;

        if (metalCommandBuffer->fragmentSamplers[firstSlot + i] != sampler->handle) {
            metalCommandBuffer->fragmentSamplers[firstSlot + i] = sampler->handle;
            metalCommandBuffer->needFragmentSamplerBind  = true;
        }

        if (metalCommandBuffer->fragmentTextures[firstSlot + i] != textureContainer->activeTexture->handle) {
            METAL_INTERNAL_TrackTexture(
                metalCommandBuffer,
                textureContainer->activeTexture);

            metalCommandBuffer->fragmentTextures[firstSlot + i] =
                textureContainer->activeTexture->handle;

            metalCommandBuffer->needFragmentSamplerBind  = true;
        }
    }
}

static void METAL_BindFragmentStorageTextures(
    SDL_GPUCommandBuffer *commandBuffer,
    Uint32 firstSlot,
    SDL_GPUTexture *const *storageTextures,
    Uint32 numBindings)
{
    MetalCommandBuffer *metalCommandBuffer = (MetalCommandBuffer *)commandBuffer;
    MetalTextureContainer *textureContainer;

    for (Uint32 i = 0; i < numBindings; i += 1) {
        textureContainer = (MetalTextureContainer *)storageTextures[i];

        if (metalCommandBuffer->fragmentStorageTextures[firstSlot + i] != textureContainer->activeTexture->handle) {
            METAL_INTERNAL_TrackTexture(
                metalCommandBuffer,
                textureContainer->activeTexture);

            metalCommandBuffer->fragmentStorageTextures[firstSlot + i] =
                textureContainer->activeTexture->handle;

            metalCommandBuffer->needFragmentStorageTextureBind = true;
        }
    }
}

static void METAL_BindFragmentStorageBuffers(
    SDL_GPUCommandBuffer *commandBuffer,
    Uint32 firstSlot,
    SDL_GPUBuffer *const *storageBuffers,
    Uint32 numBindings)
{
    MetalCommandBuffer *metalCommandBuffer = (MetalCommandBuffer *)commandBuffer;
    MetalBufferContainer *bufferContainer;

    for (Uint32 i = 0; i < numBindings; i += 1) {
        bufferContainer = (MetalBufferContainer *)storageBuffers[i];

        if (metalCommandBuffer->fragmentStorageBuffers[firstSlot + i] != bufferContainer->activeBuffer->handle) {
            METAL_INTERNAL_TrackBuffer(
                metalCommandBuffer,
                bufferContainer->activeBuffer);

            metalCommandBuffer->fragmentStorageBuffers[firstSlot + i] =
                bufferContainer->activeBuffer->handle;

            metalCommandBuffer->needFragmentStorageBufferBind = true;
        }
    }
}

// This function assumes that it's called from within an autorelease pool
static void METAL_INTERNAL_BindGraphicsResources(
    MetalCommandBuffer *commandBuffer)
{
    MetalGraphicsPipeline *graphicsPipeline = commandBuffer->graphics_pipeline;
    NSUInteger offsets[MAX_STORAGE_BUFFERS_PER_STAGE] = { 0 };

    // Vertex Buffers
    if (commandBuffer->needVertexBufferBind) {
        id<MTLBuffer> metalBuffers[MAX_VERTEX_BUFFERS];
        NSUInteger bufferOffsets[MAX_VERTEX_BUFFERS];
        NSRange range = NSMakeRange(METAL_FIRST_VERTEX_BUFFER_SLOT, commandBuffer->vertexBufferCount);
        for (Uint32 i = 0; i < commandBuffer->vertexBufferCount; i += 1) {
            metalBuffers[i] = commandBuffer->vertexBuffers[i];
            bufferOffsets[i] = commandBuffer->vertexBufferOffsets[i];
        }
        [commandBuffer->renderEncoder setVertexBuffers:metalBuffers offsets:bufferOffsets withRange:range];
        commandBuffer->needVertexBufferBind = false;
    }

    // Vertex Samplers+Textures

    if (commandBuffer->needVertexSamplerBind) {
        if (graphicsPipeline->vertexSamplerCount > 0) {
            [commandBuffer->renderEncoder setVertexSamplerStates:commandBuffer->vertexSamplers
                                                       withRange:NSMakeRange(0, graphicsPipeline->vertexSamplerCount)];
            [commandBuffer->renderEncoder setVertexTextures:commandBuffer->vertexTextures
                                                  withRange:NSMakeRange(0, graphicsPipeline->vertexSamplerCount)];
        }
        commandBuffer->needVertexSamplerBind = false;
    }

    // Vertex Storage Textures

    if (commandBuffer->needVertexStorageTextureBind) {
        if (graphicsPipeline->vertexStorageTextureCount > 0) {
            [commandBuffer->renderEncoder setVertexTextures:commandBuffer->vertexStorageTextures
                                                  withRange:NSMakeRange(graphicsPipeline->vertexSamplerCount,
                                                                        graphicsPipeline->vertexStorageTextureCount)];
        }
        commandBuffer->needVertexStorageTextureBind = false;
    }

    // Vertex Storage Buffers

    if (commandBuffer->needVertexStorageBufferBind) {
        if (graphicsPipeline->vertexStorageBufferCount > 0) {
            [commandBuffer->renderEncoder setVertexBuffers:commandBuffer->vertexStorageBuffers
                                                   offsets:offsets
                                                 withRange:NSMakeRange(graphicsPipeline->vertexUniformBufferCount,
                                                                       graphicsPipeline->vertexStorageBufferCount)];
        }
        commandBuffer->needVertexStorageBufferBind = false;
    }

    // Vertex Uniform Buffers

    for (Uint32 i = 0; i < graphicsPipeline->vertexUniformBufferCount; i += 1) {
        if (commandBuffer->needVertexUniformBufferBind[i]) {
            if (graphicsPipeline->vertexUniformBufferCount > i) {
                [commandBuffer->renderEncoder
                    setVertexBuffer:commandBuffer->vertexUniformBuffers[i]->handle
                             offset:commandBuffer->vertexUniformBuffers[i]->drawOffset
                            atIndex:i];
            }
            commandBuffer->needVertexUniformBufferBind[i] = false;
        }
    }

    // Fragment Samplers+Textures

    if (commandBuffer->needFragmentSamplerBind) {
        if (graphicsPipeline->fragmentSamplerCount > 0) {
            [commandBuffer->renderEncoder setFragmentSamplerStates:commandBuffer->fragmentSamplers
                                                         withRange:NSMakeRange(0, graphicsPipeline->fragmentSamplerCount)];
            [commandBuffer->renderEncoder setFragmentTextures:commandBuffer->fragmentTextures
                                                    withRange:NSMakeRange(0, graphicsPipeline->fragmentSamplerCount)];
        }
        commandBuffer->needFragmentSamplerBind = false;
    }

    // Fragment Storage Textures

    if (commandBuffer->needFragmentStorageTextureBind) {
        if (graphicsPipeline->fragmentStorageTextureCount > 0) {
            [commandBuffer->renderEncoder setFragmentTextures:commandBuffer->fragmentStorageTextures
                                                    withRange:NSMakeRange(graphicsPipeline->fragmentSamplerCount,
                                                                          graphicsPipeline->fragmentStorageTextureCount)];
        }
        commandBuffer->needFragmentStorageTextureBind = false;
    }

    // Fragment Storage Buffers

    if (commandBuffer->needFragmentStorageBufferBind) {
        if (graphicsPipeline->fragmentStorageBufferCount > 0) {
            [commandBuffer->renderEncoder setFragmentBuffers:commandBuffer->fragmentStorageBuffers
                                                     offsets:offsets
                                                   withRange:NSMakeRange(graphicsPipeline->fragmentUniformBufferCount,
                                                                         graphicsPipeline->fragmentStorageBufferCount)];
        }
        commandBuffer->needFragmentStorageBufferBind = false;
    }

    // Fragment Uniform Buffers

    for (Uint32 i = 0; i < graphicsPipeline->fragmentUniformBufferCount; i += 1) {
        if (commandBuffer->needFragmentUniformBufferBind[i]) {
            if (graphicsPipeline->fragmentUniformBufferCount > i) {
                [commandBuffer->renderEncoder
                    setFragmentBuffer:commandBuffer->fragmentUniformBuffers[i]->handle
                            offset:commandBuffer->fragmentUniformBuffers[i]->drawOffset
                            atIndex:i];
            }
            commandBuffer->needFragmentUniformBufferBind[i] = false;
        }
    }
}

// This function assumes that it's called from within an autorelease pool
static void METAL_INTERNAL_BindComputeResources(
    MetalCommandBuffer *commandBuffer)
{
    MetalComputePipeline *computePipeline = commandBuffer->compute_pipeline;
    NSUInteger offsets[MAX_STORAGE_BUFFERS_PER_STAGE] = { 0 };

    if (commandBuffer->needComputeSamplerBind) {
        if (computePipeline->numSamplers > 0) {
            [commandBuffer->computeEncoder setTextures:commandBuffer->computeSamplerTextures
                                             withRange:NSMakeRange(0, computePipeline->numSamplers)];
            [commandBuffer->computeEncoder setSamplerStates:commandBuffer->computeSamplers
                                                  withRange:NSMakeRange(0, computePipeline->numSamplers)];
        }
        commandBuffer->needComputeSamplerBind = false;
    }

    if (commandBuffer->needComputeReadOnlyStorageTextureBind) {
        if (computePipeline->numReadonlyStorageTextures > 0) {
            [commandBuffer->computeEncoder setTextures:commandBuffer->computeReadOnlyTextures
                                             withRange:NSMakeRange(
                                                           computePipeline->numSamplers,
                                                           computePipeline->numReadonlyStorageTextures)];
        }
        commandBuffer->needComputeReadOnlyStorageTextureBind = false;
    }

    if (commandBuffer->needComputeReadOnlyStorageBufferBind) {
        if (computePipeline->numReadonlyStorageBuffers > 0) {
            [commandBuffer->computeEncoder setBuffers:commandBuffer->computeReadOnlyBuffers
                                              offsets:offsets
                                            withRange:NSMakeRange(computePipeline->numUniformBuffers,
                                                                  computePipeline->numReadonlyStorageBuffers)];
        }
        commandBuffer->needComputeReadOnlyStorageBufferBind = false;
    }

    for (Uint32 i = 0; i < MAX_UNIFORM_BUFFERS_PER_STAGE; i += 1) {
        if (commandBuffer->needComputeUniformBufferBind[i]) {
            if (computePipeline->numUniformBuffers > i) {
                [commandBuffer->computeEncoder
                    setBuffer:commandBuffer->computeUniformBuffers[i]->handle
                    offset:commandBuffer->computeUniformBuffers[i]->drawOffset
                    atIndex:i];
            }
        }
        commandBuffer->needComputeUniformBufferBind[i] = false;
    }
}

static void METAL_DrawIndexedPrimitives(
    SDL_GPUCommandBuffer *commandBuffer,
    Uint32 numIndices,
    Uint32 numInstances,
    Uint32 firstIndex,
    Sint32 vertexOffset,
    Uint32 firstInstance)
{
    @autoreleasepool {
        MetalCommandBuffer *metalCommandBuffer = (MetalCommandBuffer *)commandBuffer;
        SDL_GPUPrimitiveType primitiveType = metalCommandBuffer->graphics_pipeline->primitiveType;
        Uint32 indexSize = IndexSize(metalCommandBuffer->index_element_size);

        METAL_INTERNAL_BindGraphicsResources(metalCommandBuffer);

        [metalCommandBuffer->renderEncoder
            drawIndexedPrimitives:SDLToMetal_PrimitiveType[primitiveType]
                       indexCount:numIndices
                        indexType:SDLToMetal_IndexType[metalCommandBuffer->index_element_size]
                      indexBuffer:metalCommandBuffer->indexBuffer->handle
                indexBufferOffset:metalCommandBuffer->indexBufferOffset + (firstIndex * indexSize)
                    instanceCount:numInstances
                       baseVertex:vertexOffset
                     baseInstance:firstInstance];
    }
}

static void METAL_DrawPrimitives(
    SDL_GPUCommandBuffer *commandBuffer,
    Uint32 numVertices,
    Uint32 numInstances,
    Uint32 firstVertex,
    Uint32 firstInstance)
{
    @autoreleasepool {
        MetalCommandBuffer *metalCommandBuffer = (MetalCommandBuffer *)commandBuffer;
        SDL_GPUPrimitiveType primitiveType = metalCommandBuffer->graphics_pipeline->primitiveType;

        METAL_INTERNAL_BindGraphicsResources(metalCommandBuffer);

        [metalCommandBuffer->renderEncoder
            drawPrimitives:SDLToMetal_PrimitiveType[primitiveType]
               vertexStart:firstVertex
               vertexCount:numVertices
             instanceCount:numInstances
              baseInstance:firstInstance];
    }
}

static void METAL_DrawPrimitivesIndirect(
    SDL_GPUCommandBuffer *commandBuffer,
    SDL_GPUBuffer *buffer,
    Uint32 offset,
    Uint32 drawCount)
{
    @autoreleasepool {
        MetalCommandBuffer *metalCommandBuffer = (MetalCommandBuffer *)commandBuffer;
        MetalBuffer *metalBuffer = ((MetalBufferContainer *)buffer)->activeBuffer;
        SDL_GPUPrimitiveType primitiveType = metalCommandBuffer->graphics_pipeline->primitiveType;

        METAL_INTERNAL_BindGraphicsResources(metalCommandBuffer);

        /* Metal: "We have multi-draw at home!"
         * Multi-draw at home:
         */
        for (Uint32 i = 0; i < drawCount; i += 1) {
            [metalCommandBuffer->renderEncoder
                      drawPrimitives:SDLToMetal_PrimitiveType[primitiveType]
                      indirectBuffer:metalBuffer->handle
                indirectBufferOffset:offset + (sizeof(SDL_GPUIndirectDrawCommand) * i)];
        }

        METAL_INTERNAL_TrackBuffer(metalCommandBuffer, metalBuffer);
    }
}

static void METAL_DrawIndexedPrimitivesIndirect(
    SDL_GPUCommandBuffer *commandBuffer,
    SDL_GPUBuffer *buffer,
    Uint32 offset,
    Uint32 drawCount)
{
    @autoreleasepool {
        MetalCommandBuffer *metalCommandBuffer = (MetalCommandBuffer *)commandBuffer;
        MetalBuffer *metalBuffer = ((MetalBufferContainer *)buffer)->activeBuffer;
        SDL_GPUPrimitiveType primitiveType = metalCommandBuffer->graphics_pipeline->primitiveType;

        METAL_INTERNAL_BindGraphicsResources(metalCommandBuffer);

        for (Uint32 i = 0; i < drawCount; i += 1) {
            [metalCommandBuffer->renderEncoder
                drawIndexedPrimitives:SDLToMetal_PrimitiveType[primitiveType]
                            indexType:SDLToMetal_IndexType[metalCommandBuffer->index_element_size]
                          indexBuffer:metalCommandBuffer->indexBuffer->handle
                    indexBufferOffset:metalCommandBuffer->indexBufferOffset
                       indirectBuffer:metalBuffer->handle
                 indirectBufferOffset:offset + (sizeof(SDL_GPUIndexedIndirectDrawCommand) * i)];
        }

        METAL_INTERNAL_TrackBuffer(metalCommandBuffer, metalBuffer);
    }
}

static void METAL_EndRenderPass(
    SDL_GPUCommandBuffer *commandBuffer)
{
    @autoreleasepool {
        MetalCommandBuffer *metalCommandBuffer = (MetalCommandBuffer *)commandBuffer;
        [metalCommandBuffer->renderEncoder endEncoding];
        metalCommandBuffer->renderEncoder = nil;

        for (Uint32 i = 0; i < MAX_VERTEX_BUFFERS; i += 1) {
            metalCommandBuffer->vertexBuffers[i] = nil;
            metalCommandBuffer->vertexBufferOffsets[i] = 0;
            metalCommandBuffer->vertexBufferCount = 0;
        }
        for (Uint32 i = 0; i < MAX_TEXTURE_SAMPLERS_PER_STAGE; i += 1) {
            metalCommandBuffer->vertexSamplers[i] = nil;
            metalCommandBuffer->vertexTextures[i] = nil;
            metalCommandBuffer->fragmentSamplers[i] = nil;
            metalCommandBuffer->fragmentTextures[i] = nil;
        }
        for (Uint32 i = 0; i < MAX_STORAGE_TEXTURES_PER_STAGE; i += 1) {
            metalCommandBuffer->vertexStorageTextures[i] = nil;
            metalCommandBuffer->fragmentStorageTextures[i] = nil;
        }
        for (Uint32 i = 0; i < MAX_STORAGE_BUFFERS_PER_STAGE; i += 1) {
            metalCommandBuffer->vertexStorageBuffers[i] = nil;
            metalCommandBuffer->fragmentStorageBuffers[i] = nil;
        }
    }
}

// This function assumes that it's called from within an autorelease pool
static void METAL_INTERNAL_PushUniformData(
    MetalCommandBuffer *metalCommandBuffer,
    SDL_GPUShaderStage shaderStage,
    Uint32 slotIndex,
    const void *data,
    Uint32 length)
{
    MetalUniformBuffer *metalUniformBuffer;
    Uint32 alignedDataLength;

    if (shaderStage == SDL_GPU_SHADERSTAGE_VERTEX) {
        if (metalCommandBuffer->vertexUniformBuffers[slotIndex] == NULL) {
            metalCommandBuffer->vertexUniformBuffers[slotIndex] = METAL_INTERNAL_AcquireUniformBufferFromPool(
                metalCommandBuffer);
        }
        metalUniformBuffer = metalCommandBuffer->vertexUniformBuffers[slotIndex];
    } else if (shaderStage == SDL_GPU_SHADERSTAGE_FRAGMENT) {
        if (metalCommandBuffer->fragmentUniformBuffers[slotIndex] == NULL) {
            metalCommandBuffer->fragmentUniformBuffers[slotIndex] = METAL_INTERNAL_AcquireUniformBufferFromPool(
                metalCommandBuffer);
        }
        metalUniformBuffer = metalCommandBuffer->fragmentUniformBuffers[slotIndex];
    } else if (shaderStage == SDL_GPU_SHADERSTAGE_COMPUTE) {
        if (metalCommandBuffer->computeUniformBuffers[slotIndex] == NULL) {
            metalCommandBuffer->computeUniformBuffers[slotIndex] = METAL_INTERNAL_AcquireUniformBufferFromPool(
                metalCommandBuffer);
        }
        metalUniformBuffer = metalCommandBuffer->computeUniformBuffers[slotIndex];
    } else {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Unrecognized shader stage!");
        return;
    }

    alignedDataLength = METAL_INTERNAL_NextHighestAlignment(
        length,
        256);

    if (metalUniformBuffer->writeOffset + alignedDataLength >= UNIFORM_BUFFER_SIZE) {
        metalUniformBuffer = METAL_INTERNAL_AcquireUniformBufferFromPool(
            metalCommandBuffer);

        metalUniformBuffer->writeOffset = 0;
        metalUniformBuffer->drawOffset = 0;

        if (shaderStage == SDL_GPU_SHADERSTAGE_VERTEX) {
            metalCommandBuffer->vertexUniformBuffers[slotIndex] = metalUniformBuffer;
        } else if (shaderStage == SDL_GPU_SHADERSTAGE_FRAGMENT) {
            metalCommandBuffer->fragmentUniformBuffers[slotIndex] = metalUniformBuffer;
        } else if (shaderStage == SDL_GPU_SHADERSTAGE_COMPUTE) {
            metalCommandBuffer->computeUniformBuffers[slotIndex] = metalUniformBuffer;
        } else {
            SDL_LogError(SDL_LOG_CATEGORY_GPU, "Unrecognized shader stage!");
            return;
        }
    }

    metalUniformBuffer->drawOffset = metalUniformBuffer->writeOffset;

    SDL_memcpy(
        (metalUniformBuffer->handle).contents + metalUniformBuffer->writeOffset,
        data,
        length);

    metalUniformBuffer->writeOffset += alignedDataLength;

    if (shaderStage == SDL_GPU_SHADERSTAGE_VERTEX) {
        metalCommandBuffer->needVertexUniformBufferBind[slotIndex] = true;
    } else if (shaderStage == SDL_GPU_SHADERSTAGE_FRAGMENT) {
        metalCommandBuffer->needFragmentUniformBufferBind[slotIndex] = true;
    } else if (shaderStage == SDL_GPU_SHADERSTAGE_COMPUTE) {
        metalCommandBuffer->needComputeUniformBufferBind[slotIndex] = true;
    } else {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Unrecognized shader stage!");
    }
}

static void METAL_PushVertexUniformData(
    SDL_GPUCommandBuffer *commandBuffer,
    Uint32 slotIndex,
    const void *data,
    Uint32 length)
{
    @autoreleasepool {
        METAL_INTERNAL_PushUniformData(
            (MetalCommandBuffer *)commandBuffer,
            SDL_GPU_SHADERSTAGE_VERTEX,
            slotIndex,
            data,
            length);
    }
}

static void METAL_PushFragmentUniformData(
    SDL_GPUCommandBuffer *commandBuffer,
    Uint32 slotIndex,
    const void *data,
    Uint32 length)
{
    @autoreleasepool {
        METAL_INTERNAL_PushUniformData(
            (MetalCommandBuffer *)commandBuffer,
            SDL_GPU_SHADERSTAGE_FRAGMENT,
            slotIndex,
            data,
            length);
    }
}

// Blit

static void METAL_Blit(
    SDL_GPUCommandBuffer *commandBuffer,
    const SDL_GPUBlitInfo *info)
{
    MetalCommandBuffer *metalCommandBuffer = (MetalCommandBuffer *)commandBuffer;
    MetalRenderer *renderer = (MetalRenderer *)metalCommandBuffer->renderer;

    SDL_GPU_BlitCommon(
        commandBuffer,
        info,
        renderer->blitLinearSampler,
        renderer->blitNearestSampler,
        renderer->blitVertexShader,
        renderer->blitFrom2DShader,
        renderer->blitFrom2DArrayShader,
        renderer->blitFrom3DShader,
        renderer->blitFromCubeShader,
        renderer->blitFromCubeArrayShader,
        &renderer->blitPipelines,
        &renderer->blitPipelineCount,
        &renderer->blitPipelineCapacity);
}

// Compute State

static void METAL_BeginComputePass(
    SDL_GPUCommandBuffer *commandBuffer,
    const SDL_GPUStorageTextureReadWriteBinding *storageTextureBindings,
    Uint32 numStorageTextureBindings,
    const SDL_GPUStorageBufferReadWriteBinding *storageBufferBindings,
    Uint32 numStorageBufferBindings)
{
    @autoreleasepool {
        MetalCommandBuffer *metalCommandBuffer = (MetalCommandBuffer *)commandBuffer;
        MetalTextureContainer *textureContainer;
        MetalTexture *texture;
        id<MTLTexture> textureView;
        MetalBufferContainer *bufferContainer;
        MetalBuffer *buffer;

        metalCommandBuffer->computeEncoder = [metalCommandBuffer->handle computeCommandEncoder];

        for (Uint32 i = 0; i < numStorageTextureBindings; i += 1) {
            textureContainer = (MetalTextureContainer *)storageTextureBindings[i].texture;

            texture = METAL_INTERNAL_PrepareTextureForWrite(
                metalCommandBuffer->renderer,
                textureContainer,
                storageTextureBindings[i].cycle);

            METAL_INTERNAL_TrackTexture(metalCommandBuffer, texture);

            textureView = [texture->handle newTextureViewWithPixelFormat:SDLToMetal_TextureFormat(textureContainer->header.info.format)
                                                             textureType:SDLToMetal_TextureType(textureContainer->header.info.type, false)
                                                                  levels:NSMakeRange(storageTextureBindings[i].mip_level, 1)
                                                                  slices:NSMakeRange(storageTextureBindings[i].layer, 1)];

            metalCommandBuffer->computeReadWriteTextures[i] = textureView;
        }

        for (Uint32 i = 0; i < numStorageBufferBindings; i += 1) {
            bufferContainer = (MetalBufferContainer *)storageBufferBindings[i].buffer;

            buffer = METAL_INTERNAL_PrepareBufferForWrite(
                metalCommandBuffer->renderer,
                bufferContainer,
                storageBufferBindings[i].cycle);

            METAL_INTERNAL_TrackBuffer(
                metalCommandBuffer,
                buffer);

            metalCommandBuffer->computeReadWriteBuffers[i] = buffer->handle;
        }
    }
}

static void METAL_BindComputePipeline(
    SDL_GPUCommandBuffer *commandBuffer,
    SDL_GPUComputePipeline *computePipeline)
{
    @autoreleasepool {
        MetalCommandBuffer *metalCommandBuffer = (MetalCommandBuffer *)commandBuffer;
        MetalComputePipeline *pipeline = (MetalComputePipeline *)computePipeline;

        metalCommandBuffer->compute_pipeline = pipeline;

        [metalCommandBuffer->computeEncoder setComputePipelineState:pipeline->handle];

        for (Uint32 i = 0; i < MAX_UNIFORM_BUFFERS_PER_STAGE; i += 1) {
            metalCommandBuffer->needComputeUniformBufferBind[i] = true;
        }

        for (Uint32 i = 0; i < pipeline->numUniformBuffers; i += 1) {
            if (metalCommandBuffer->computeUniformBuffers[i] == NULL) {
                metalCommandBuffer->computeUniformBuffers[i] = METAL_INTERNAL_AcquireUniformBufferFromPool(
                    metalCommandBuffer);
            }
        }

        // Bind write-only resources
        if (pipeline->numReadWriteStorageTextures > 0) {
            [metalCommandBuffer->computeEncoder setTextures:metalCommandBuffer->computeReadWriteTextures
                                                  withRange:NSMakeRange(
                                                        pipeline->numSamplers +
                                                            pipeline->numReadonlyStorageTextures,
                                                        pipeline->numReadWriteStorageTextures)];
        }

        NSUInteger offsets[MAX_COMPUTE_WRITE_BUFFERS] = { 0 };
        if (pipeline->numReadWriteStorageBuffers > 0) {
            [metalCommandBuffer->computeEncoder setBuffers:metalCommandBuffer->computeReadWriteBuffers
                                                   offsets:offsets
                                                 withRange:NSMakeRange(
                                                        pipeline->numUniformBuffers +
                                                            pipeline->numReadonlyStorageBuffers,
                                                        pipeline->numReadWriteStorageBuffers)];
        }
    }
}

static void METAL_BindComputeSamplers(
    SDL_GPUCommandBuffer *commandBuffer,
    Uint32 firstSlot,
    const SDL_GPUTextureSamplerBinding *textureSamplerBindings,
    Uint32 numBindings)
{
    MetalCommandBuffer *metalCommandBuffer = (MetalCommandBuffer *)commandBuffer;
    MetalTextureContainer *textureContainer;
    MetalSampler *sampler;

    for (Uint32 i = 0; i < numBindings; i += 1) {
        textureContainer = (MetalTextureContainer *)textureSamplerBindings[i].texture;
        sampler = (MetalSampler *)textureSamplerBindings[i].sampler;

        if (metalCommandBuffer->computeSamplers[firstSlot + i] != sampler->handle) {
            metalCommandBuffer->computeSamplers[firstSlot + i] = sampler->handle;
            metalCommandBuffer->needComputeSamplerBind = true;
        }

        if (metalCommandBuffer->computeSamplerTextures[firstSlot + i] != textureContainer->activeTexture->handle) {
            METAL_INTERNAL_TrackTexture(
                metalCommandBuffer,
                textureContainer->activeTexture);

            metalCommandBuffer->computeSamplerTextures[firstSlot + i] =
                textureContainer->activeTexture->handle;

            metalCommandBuffer->needComputeSamplerBind = true;
        }
    }
}

static void METAL_BindComputeStorageTextures(
    SDL_GPUCommandBuffer *commandBuffer,
    Uint32 firstSlot,
    SDL_GPUTexture *const *storageTextures,
    Uint32 numBindings)
{
    MetalCommandBuffer *metalCommandBuffer = (MetalCommandBuffer *)commandBuffer;
    MetalTextureContainer *textureContainer;

    for (Uint32 i = 0; i < numBindings; i += 1) {
        textureContainer = (MetalTextureContainer *)storageTextures[i];

        if (metalCommandBuffer->computeReadOnlyTextures[firstSlot + i] != textureContainer->activeTexture->handle) {
            METAL_INTERNAL_TrackTexture(
                metalCommandBuffer,
                textureContainer->activeTexture);

            metalCommandBuffer->computeReadOnlyTextures[firstSlot + i] =
                textureContainer->activeTexture->handle;

            metalCommandBuffer->needComputeReadOnlyStorageTextureBind = true;
        }
    }
}

static void METAL_BindComputeStorageBuffers(
    SDL_GPUCommandBuffer *commandBuffer,
    Uint32 firstSlot,
    SDL_GPUBuffer *const *storageBuffers,
    Uint32 numBindings)
{
    MetalCommandBuffer *metalCommandBuffer = (MetalCommandBuffer *)commandBuffer;
    MetalBufferContainer *bufferContainer;

    for (Uint32 i = 0; i < numBindings; i += 1) {
        bufferContainer = (MetalBufferContainer *)storageBuffers[i];

        if (metalCommandBuffer->computeReadOnlyBuffers[firstSlot + i] != bufferContainer->activeBuffer->handle) {
            METAL_INTERNAL_TrackBuffer(
                metalCommandBuffer,
                bufferContainer->activeBuffer);

            metalCommandBuffer->computeReadOnlyBuffers[firstSlot + i] =
                bufferContainer->activeBuffer->handle;

            metalCommandBuffer->needComputeReadOnlyStorageBufferBind = true;
        }
    }
}

static void METAL_PushComputeUniformData(
    SDL_GPUCommandBuffer *commandBuffer,
    Uint32 slotIndex,
    const void *data,
    Uint32 length)
{
    @autoreleasepool {
        METAL_INTERNAL_PushUniformData(
            (MetalCommandBuffer *)commandBuffer,
            SDL_GPU_SHADERSTAGE_COMPUTE,
            slotIndex,
            data,
            length);
    }
}

static void METAL_DispatchCompute(
    SDL_GPUCommandBuffer *commandBuffer,
    Uint32 groupcountX,
    Uint32 groupcountY,
    Uint32 groupcountZ)
{
    @autoreleasepool {
        MetalCommandBuffer *metalCommandBuffer = (MetalCommandBuffer *)commandBuffer;
        MTLSize threadgroups = MTLSizeMake(groupcountX, groupcountY, groupcountZ);
        MTLSize threadsPerThreadgroup = MTLSizeMake(
            metalCommandBuffer->compute_pipeline->threadcountX,
            metalCommandBuffer->compute_pipeline->threadcountY,
            metalCommandBuffer->compute_pipeline->threadcountZ);

        METAL_INTERNAL_BindComputeResources(metalCommandBuffer);

        [metalCommandBuffer->computeEncoder
             dispatchThreadgroups:threadgroups
            threadsPerThreadgroup:threadsPerThreadgroup];
    }
}

static void METAL_DispatchComputeIndirect(
    SDL_GPUCommandBuffer *commandBuffer,
    SDL_GPUBuffer *buffer,
    Uint32 offset)
{
    @autoreleasepool {
        MetalCommandBuffer *metalCommandBuffer = (MetalCommandBuffer *)commandBuffer;
        MetalBuffer *metalBuffer = ((MetalBufferContainer *)buffer)->activeBuffer;
        MTLSize threadsPerThreadgroup = MTLSizeMake(
            metalCommandBuffer->compute_pipeline->threadcountX,
            metalCommandBuffer->compute_pipeline->threadcountY,
            metalCommandBuffer->compute_pipeline->threadcountZ);

        METAL_INTERNAL_BindComputeResources(metalCommandBuffer);

        [metalCommandBuffer->computeEncoder
            dispatchThreadgroupsWithIndirectBuffer:metalBuffer->handle
                              indirectBufferOffset:offset
                             threadsPerThreadgroup:threadsPerThreadgroup];

        METAL_INTERNAL_TrackBuffer(metalCommandBuffer, metalBuffer);
    }
}

static void METAL_EndComputePass(
    SDL_GPUCommandBuffer *commandBuffer)
{
    @autoreleasepool {
        MetalCommandBuffer *metalCommandBuffer = (MetalCommandBuffer *)commandBuffer;
        [metalCommandBuffer->computeEncoder endEncoding];
        metalCommandBuffer->computeEncoder = nil;

        for (Uint32 i = 0; i < MAX_TEXTURE_SAMPLERS_PER_STAGE; i += 1) {
            metalCommandBuffer->computeSamplers[i] = nil;
            metalCommandBuffer->computeSamplerTextures[i] = nil;
        }
        for (Uint32 i = 0; i < MAX_COMPUTE_WRITE_TEXTURES; i += 1) {
            metalCommandBuffer->computeReadWriteTextures[i] = nil;
        }
        for (Uint32 i = 0; i < MAX_COMPUTE_WRITE_BUFFERS; i += 1) {
            metalCommandBuffer->computeReadWriteBuffers[i] = nil;
        }
        for (Uint32 i = 0; i < MAX_STORAGE_TEXTURES_PER_STAGE; i += 1) {
            metalCommandBuffer->computeReadOnlyTextures[i] = nil;
        }
        for (Uint32 i = 0; i < MAX_STORAGE_BUFFERS_PER_STAGE; i += 1) {
            metalCommandBuffer->computeReadOnlyBuffers[i] = nil;
        }
    }
}

// Fence Cleanup

static void METAL_INTERNAL_ReleaseFenceToPool(
    MetalRenderer *renderer,
    MetalFence *fence)
{
    SDL_LockMutex(renderer->fenceLock);

    // FIXME: Should this use EXPAND_IF_NEEDED?
    if (renderer->availableFenceCount == renderer->availableFenceCapacity) {
        renderer->availableFenceCapacity *= 2;
        renderer->availableFences = SDL_realloc(
            renderer->availableFences,
            renderer->availableFenceCapacity * sizeof(MetalFence *));
    }
    renderer->availableFences[renderer->availableFenceCount] = fence;
    renderer->availableFenceCount += 1;

    SDL_UnlockMutex(renderer->fenceLock);
}

static void METAL_ReleaseFence(
    SDL_GPURenderer *driverData,
    SDL_GPUFence *fence)
{
    MetalFence *metalFence = (MetalFence *)fence;
    if (SDL_AtomicDecRef(&metalFence->referenceCount)) {
        METAL_INTERNAL_ReleaseFenceToPool(
            (MetalRenderer *)driverData,
            (MetalFence *)fence);
    }
}

// Cleanup

static void METAL_INTERNAL_CleanCommandBuffer(
    MetalRenderer *renderer,
    MetalCommandBuffer *commandBuffer,
    bool cancel)
{
    Uint32 i;

    // End any active passes
    if (commandBuffer->renderEncoder) {
        [commandBuffer->renderEncoder endEncoding];
        commandBuffer->renderEncoder = nil;
    }
    if (commandBuffer->computeEncoder) {
        [commandBuffer->computeEncoder endEncoding];
        commandBuffer->computeEncoder = nil;
    }
    if (commandBuffer->blitEncoder) {
        [commandBuffer->blitEncoder endEncoding];
        commandBuffer->blitEncoder = nil;
    }

    // Uniform buffers are now available

    SDL_LockMutex(renderer->acquireUniformBufferLock);

    for (i = 0; i < commandBuffer->usedUniformBufferCount; i += 1) {
        METAL_INTERNAL_ReturnUniformBufferToPool(
            renderer,
            commandBuffer->usedUniformBuffers[i]);
    }
    commandBuffer->usedUniformBufferCount = 0;

    SDL_UnlockMutex(renderer->acquireUniformBufferLock);

    // Reference Counting

    for (i = 0; i < commandBuffer->usedBufferCount; i += 1) {
        (void)SDL_AtomicDecRef(&commandBuffer->usedBuffers[i]->referenceCount);
    }
    commandBuffer->usedBufferCount = 0;

    for (i = 0; i < commandBuffer->usedTextureCount; i += 1) {
        (void)SDL_AtomicDecRef(&commandBuffer->usedTextures[i]->referenceCount);
    }
    commandBuffer->usedTextureCount = 0;

    // Reset presentation
    commandBuffer->windowDataCount = 0;

    // Reset bindings
    for (i = 0; i < MAX_VERTEX_BUFFERS; i += 1) {
        commandBuffer->vertexBuffers[i] = nil;
        commandBuffer->vertexBufferOffsets[i] = 0;
    }
    commandBuffer->vertexBufferCount = 0;
    commandBuffer->indexBuffer = NULL;
    for (i = 0; i < MAX_TEXTURE_SAMPLERS_PER_STAGE; i += 1) {
        commandBuffer->vertexSamplers[i] = nil;
        commandBuffer->vertexTextures[i] = nil;
        commandBuffer->fragmentSamplers[i] = nil;
        commandBuffer->fragmentTextures[i] = nil;
        commandBuffer->computeSamplers[i] = nil;
        commandBuffer->computeSamplerTextures[i] = nil;
    }
    for (i = 0; i < MAX_STORAGE_TEXTURES_PER_STAGE; i += 1) {
        commandBuffer->vertexStorageTextures[i] = nil;
        commandBuffer->fragmentStorageTextures[i] = nil;
        commandBuffer->computeReadOnlyTextures[i] = nil;
    }
    for (i = 0; i < MAX_STORAGE_BUFFERS_PER_STAGE; i += 1) {
        commandBuffer->vertexStorageBuffers[i] = nil;
        commandBuffer->fragmentStorageBuffers[i] = nil;
        commandBuffer->computeReadOnlyBuffers[i] = nil;
    }
    for (i = 0; i < MAX_COMPUTE_WRITE_TEXTURES; i += 1) {
        commandBuffer->computeReadWriteTextures[i] = nil;
    }
    for (i = 0; i < MAX_COMPUTE_WRITE_BUFFERS; i += 1) {
        commandBuffer->computeReadWriteBuffers[i] = nil;
    }

    commandBuffer->needVertexBufferBind = false;
    commandBuffer->needVertexSamplerBind = false;
    commandBuffer->needVertexStorageBufferBind = false;
    commandBuffer->needVertexStorageTextureBind = false;
    SDL_zeroa(commandBuffer->needVertexUniformBufferBind);

    commandBuffer->needFragmentSamplerBind = false;
    commandBuffer->needFragmentStorageBufferBind = false;
    commandBuffer->needFragmentStorageTextureBind = false;
    SDL_zeroa(commandBuffer->needFragmentUniformBufferBind);

    commandBuffer->needComputeSamplerBind = false;
    commandBuffer->needComputeReadOnlyStorageBufferBind = false;
    commandBuffer->needComputeReadOnlyStorageTextureBind = false;
    SDL_zeroa(commandBuffer->needComputeUniformBufferBind);

    // The fence is now available (unless SubmitAndAcquireFence was called)
    if (commandBuffer->autoReleaseFence) {
        METAL_ReleaseFence(
            (SDL_GPURenderer *)renderer,
            (SDL_GPUFence *)commandBuffer->fence);
    }

    // Return command buffer to pool
    SDL_LockMutex(renderer->acquireCommandBufferLock);
    // FIXME: Should this use EXPAND_IF_NEEDED?
    if (renderer->availableCommandBufferCount == renderer->availableCommandBufferCapacity) {
        renderer->availableCommandBufferCapacity += 1;
        renderer->availableCommandBuffers = SDL_realloc(
            renderer->availableCommandBuffers,
            renderer->availableCommandBufferCapacity * sizeof(MetalCommandBuffer *));
    }
    renderer->availableCommandBuffers[renderer->availableCommandBufferCount] = commandBuffer;
    renderer->availableCommandBufferCount += 1;
    SDL_UnlockMutex(renderer->acquireCommandBufferLock);

    // Remove this command buffer from the submitted list
    if (!cancel) {
        for (i = 0; i < renderer->submittedCommandBufferCount; i += 1) {
            if (renderer->submittedCommandBuffers[i] == commandBuffer) {
                renderer->submittedCommandBuffers[i] = renderer->submittedCommandBuffers[renderer->submittedCommandBufferCount - 1];
                renderer->submittedCommandBufferCount -= 1;
            }
        }
    }
}

// This function assumes that it's called from within an autorelease pool
static void METAL_INTERNAL_PerformPendingDestroys(
    MetalRenderer *renderer)
{
    Sint32 referenceCount = 0;
    Sint32 i;
    Uint32 j;

    for (i = renderer->bufferContainersToDestroyCount - 1; i >= 0; i -= 1) {
        referenceCount = 0;
        for (j = 0; j < renderer->bufferContainersToDestroy[i]->bufferCount; j += 1) {
            referenceCount += SDL_GetAtomicInt(&renderer->bufferContainersToDestroy[i]->buffers[j]->referenceCount);
        }

        if (referenceCount == 0) {
            METAL_INTERNAL_DestroyBufferContainer(
                renderer->bufferContainersToDestroy[i]);

            renderer->bufferContainersToDestroy[i] = renderer->bufferContainersToDestroy[renderer->bufferContainersToDestroyCount - 1];
            renderer->bufferContainersToDestroyCount -= 1;
        }
    }

    for (i = renderer->textureContainersToDestroyCount - 1; i >= 0; i -= 1) {
        referenceCount = 0;
        for (j = 0; j < renderer->textureContainersToDestroy[i]->textureCount; j += 1) {
            referenceCount += SDL_GetAtomicInt(&renderer->textureContainersToDestroy[i]->textures[j]->referenceCount);
        }

        if (referenceCount == 0) {
            METAL_INTERNAL_DestroyTextureContainer(
                renderer->textureContainersToDestroy[i]);

            renderer->textureContainersToDestroy[i] = renderer->textureContainersToDestroy[renderer->textureContainersToDestroyCount - 1];
            renderer->textureContainersToDestroyCount -= 1;
        }
    }
}

// Fences

static bool METAL_WaitForFences(
    SDL_GPURenderer *driverData,
    bool waitAll,
    SDL_GPUFence *const *fences,
    Uint32 numFences)
{
    @autoreleasepool {
        MetalRenderer *renderer = (MetalRenderer *)driverData;
        bool waiting;

        if (waitAll) {
            for (Uint32 i = 0; i < numFences; i += 1) {
                while (!SDL_GetAtomicInt(&((MetalFence *)fences[i])->complete)) {
                    // Spin!
                }
            }
        } else {
            waiting = 1;
            while (waiting) {
                for (Uint32 i = 0; i < numFences; i += 1) {
                    if (SDL_GetAtomicInt(&((MetalFence *)fences[i])->complete) > 0) {
                        waiting = 0;
                        break;
                    }
                }
            }
        }

        METAL_INTERNAL_PerformPendingDestroys(renderer);

        return true;
    }
}

static bool METAL_QueryFence(
    SDL_GPURenderer *driverData,
    SDL_GPUFence *fence)
{
    MetalFence *metalFence = (MetalFence *)fence;
    return SDL_GetAtomicInt(&metalFence->complete) == 1;
}

// Window and Swapchain Management

static MetalWindowData *METAL_INTERNAL_FetchWindowData(SDL_Window *window)
{
    SDL_PropertiesID properties = SDL_GetWindowProperties(window);
    return (MetalWindowData *)SDL_GetPointerProperty(properties, WINDOW_PROPERTY_DATA, NULL);
}

static bool METAL_SupportsSwapchainComposition(
    SDL_GPURenderer *driverData,
    SDL_Window *window,
    SDL_GPUSwapchainComposition swapchainComposition)
{
#ifndef SDL_PLATFORM_MACOS
    if (swapchainComposition == SDL_GPU_SWAPCHAINCOMPOSITION_HDR10_ST2084) {
        return false;
    }
#endif

    if (@available(macOS 11.0, *)) {
        return true;
    } else {
        return swapchainComposition != SDL_GPU_SWAPCHAINCOMPOSITION_HDR10_ST2084;
    }
}

// This function assumes that it's called from within an autorelease pool
static Uint8 METAL_INTERNAL_CreateSwapchain(
    MetalRenderer *renderer,
    MetalWindowData *windowData,
    SDL_GPUSwapchainComposition swapchainComposition,
    SDL_GPUPresentMode presentMode)
{
    CGColorSpaceRef colorspace;
    CGSize drawableSize;

    windowData->view = SDL_Metal_CreateView(windowData->window);
    windowData->drawable = nil;
    windowData->presentMode = SDL_GPU_PRESENTMODE_VSYNC;
    windowData->frameCounter = 0;

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i += 1) {
        windowData->inFlightFences[i] = NULL;
    }

    windowData->layer = (__bridge CAMetalLayer *)(SDL_Metal_GetLayer(windowData->view));
    windowData->layer.device = renderer->device;
#ifdef SDL_PLATFORM_MACOS
    if (@available(macOS 10.13, *)) {
        windowData->layer.displaySyncEnabled = (presentMode != SDL_GPU_PRESENTMODE_IMMEDIATE);
        windowData->presentMode = presentMode;
    }
#endif
    windowData->layer.pixelFormat = SDLToMetal_TextureFormat(SwapchainCompositionToFormat[swapchainComposition]);
#ifndef SDL_PLATFORM_TVOS
    if (@available(iOS 16.0, *)) {
        windowData->layer.wantsExtendedDynamicRangeContent = (swapchainComposition != SDL_GPU_SWAPCHAINCOMPOSITION_SDR);
    }
#endif

    colorspace = CGColorSpaceCreateWithName(SwapchainCompositionToColorSpace[swapchainComposition]);
    windowData->layer.colorspace = colorspace;
    CGColorSpaceRelease(colorspace);

    windowData->texture.handle = nil; // This will be set in AcquireSwapchainTexture.

    // Precache blit pipelines for the swapchain format
    for (Uint32 i = 0; i < 4; i += 1) {
        SDL_GPU_FetchBlitPipeline(
            renderer->sdlGPUDevice,
            (SDL_GPUTextureType)i,
            SwapchainCompositionToFormat[swapchainComposition],
            renderer->blitVertexShader,
            renderer->blitFrom2DShader,
            renderer->blitFrom2DArrayShader,
            renderer->blitFrom3DShader,
            renderer->blitFromCubeShader,
            renderer->blitFromCubeArrayShader,
            &renderer->blitPipelines,
            &renderer->blitPipelineCount,
            &renderer->blitPipelineCapacity);
    }

    // Set up the texture container
    SDL_zero(windowData->textureContainer);
    windowData->textureContainer.canBeCycled = 0;
    windowData->textureContainer.activeTexture = &windowData->texture;
    windowData->textureContainer.textureCapacity = 1;
    windowData->textureContainer.textureCount = 1;
    windowData->textureContainer.header.info.format = SwapchainCompositionToFormat[swapchainComposition];
    windowData->textureContainer.header.info.num_levels = 1;
    windowData->textureContainer.header.info.layer_count_or_depth = 1;
    windowData->textureContainer.header.info.type = SDL_GPU_TEXTURETYPE_2D;
    windowData->textureContainer.header.info.usage = SDL_GPU_TEXTUREUSAGE_COLOR_TARGET;

    drawableSize = windowData->layer.drawableSize;
    windowData->textureContainer.header.info.width = (Uint32)drawableSize.width;
    windowData->textureContainer.header.info.height = (Uint32)drawableSize.height;

    return 1;
}

static bool METAL_SupportsPresentMode(
    SDL_GPURenderer *driverData,
    SDL_Window *window,
    SDL_GPUPresentMode presentMode)
{
    switch (presentMode) {
#ifdef SDL_PLATFORM_MACOS
    case SDL_GPU_PRESENTMODE_IMMEDIATE:
#endif
    case SDL_GPU_PRESENTMODE_VSYNC:
        return true;
    default:
        return false;
    }
}

static bool METAL_ClaimWindow(
    SDL_GPURenderer *driverData,
    SDL_Window *window)
{
    @autoreleasepool {
        MetalRenderer *renderer = (MetalRenderer *)driverData;
        MetalWindowData *windowData = METAL_INTERNAL_FetchWindowData(window);

        if (windowData == NULL) {
            windowData = (MetalWindowData *)SDL_calloc(1, sizeof(MetalWindowData));
            windowData->window = window;

            if (METAL_INTERNAL_CreateSwapchain(renderer, windowData, SDL_GPU_SWAPCHAINCOMPOSITION_SDR, SDL_GPU_PRESENTMODE_VSYNC)) {
                SDL_SetPointerProperty(SDL_GetWindowProperties(window), WINDOW_PROPERTY_DATA, windowData);

                SDL_LockMutex(renderer->windowLock);

                if (renderer->claimedWindowCount >= renderer->claimedWindowCapacity) {
                    renderer->claimedWindowCapacity *= 2;
                    renderer->claimedWindows = SDL_realloc(
                        renderer->claimedWindows,
                        renderer->claimedWindowCapacity * sizeof(MetalWindowData *));
                }
                renderer->claimedWindows[renderer->claimedWindowCount] = windowData;
                renderer->claimedWindowCount += 1;

                SDL_UnlockMutex(renderer->windowLock);

                return true;
            } else {
                SDL_free(windowData);
                SET_STRING_ERROR_AND_RETURN("Could not create swapchain, failed to claim window", false);
            }
        } else {
            SET_ERROR_AND_RETURN("%s", "Window already claimed!", false);
        }
    }
}

static void METAL_ReleaseWindow(
    SDL_GPURenderer *driverData,
    SDL_Window *window)
{
    @autoreleasepool {
        MetalRenderer *renderer = (MetalRenderer *)driverData;
        MetalWindowData *windowData = METAL_INTERNAL_FetchWindowData(window);

        if (windowData == NULL) {
            SET_STRING_ERROR_AND_RETURN("Window is not claimed by this SDL_GpuDevice", );
        }

        METAL_Wait(driverData);
        SDL_Metal_DestroyView(windowData->view);
        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i += 1) {
            if (windowData->inFlightFences[i] != NULL) {
                METAL_ReleaseFence(
                    (SDL_GPURenderer *)renderer,
                    windowData->inFlightFences[i]);
            }
        }

        SDL_LockMutex(renderer->windowLock);
        for (Uint32 i = 0; i < renderer->claimedWindowCount; i += 1) {
            if (renderer->claimedWindows[i]->window == window) {
                renderer->claimedWindows[i] = renderer->claimedWindows[renderer->claimedWindowCount - 1];
                renderer->claimedWindowCount -= 1;
                break;
            }
        }
        SDL_UnlockMutex(renderer->windowLock);

        SDL_free(windowData);

        SDL_ClearProperty(SDL_GetWindowProperties(window), WINDOW_PROPERTY_DATA);
    }
}

static bool METAL_WaitForSwapchain(
    SDL_GPURenderer *driverData,
    SDL_Window *window)
{
    @autoreleasepool {
        MetalRenderer *renderer = (MetalRenderer *)driverData;
        MetalWindowData *windowData = METAL_INTERNAL_FetchWindowData(window);

        if (windowData == NULL) {
            SET_STRING_ERROR_AND_RETURN("Cannot wait for a swapchain from an unclaimed window!", false);
        }

        if (windowData->inFlightFences[windowData->frameCounter] != NULL) {
            if (!METAL_WaitForFences(
                driverData,
                true,
                &windowData->inFlightFences[windowData->frameCounter],
                1)) {
                return false;
            }
        }

        return true;
    }
}

static bool METAL_INTERNAL_AcquireSwapchainTexture(
    bool block,
    SDL_GPUCommandBuffer *commandBuffer,
    SDL_Window *window,
    SDL_GPUTexture **texture,
    Uint32 *swapchainTextureWidth,
    Uint32 *swapchainTextureHeight)
{
    @autoreleasepool {
        MetalCommandBuffer *metalCommandBuffer = (MetalCommandBuffer *)commandBuffer;
        MetalRenderer *renderer = metalCommandBuffer->renderer;
        MetalWindowData *windowData;
        CGSize drawableSize;

        *texture = NULL;
        if (swapchainTextureWidth) {
            *swapchainTextureWidth = 0;
        }
        if (swapchainTextureHeight) {
            *swapchainTextureHeight = 0;
        }

        windowData = METAL_INTERNAL_FetchWindowData(window);
        if (windowData == NULL) {
            SET_STRING_ERROR_AND_RETURN("Window is not claimed by this SDL_GpuDevice", false);
        }

        // Update the window size
        drawableSize = windowData->layer.drawableSize;
        windowData->textureContainer.header.info.width = (Uint32)drawableSize.width;
        windowData->textureContainer.header.info.height = (Uint32)drawableSize.height;
        if (swapchainTextureWidth) {
            *swapchainTextureWidth = (Uint32)drawableSize.width;
        }
        if (swapchainTextureHeight) {
            *swapchainTextureHeight = (Uint32)drawableSize.height;
        }

        if (windowData->inFlightFences[windowData->frameCounter] != NULL) {
            if (block) {
                // If we are blocking, just wait for the fence!
                if (!METAL_WaitForFences(
                    (SDL_GPURenderer *)renderer,
                    true,
                    &windowData->inFlightFences[windowData->frameCounter],
                    1)) {
                    return false;
                }
            } else {
                // If we are not blocking and the least recent fence is not signaled,
                // return true to indicate that there is no error but rendering should be skipped.
                if (!METAL_QueryFence(
                        (SDL_GPURenderer *)metalCommandBuffer->renderer,
                        windowData->inFlightFences[windowData->frameCounter])) {
                    return true;
                }
            }

            METAL_ReleaseFence(
                (SDL_GPURenderer *)metalCommandBuffer->renderer,
                windowData->inFlightFences[windowData->frameCounter]);

            windowData->inFlightFences[windowData->frameCounter] = NULL;
        }

        // Get the drawable and its underlying texture
        windowData->drawable = [windowData->layer nextDrawable];
        windowData->texture.handle = [windowData->drawable texture];

        // Set up presentation
        if (metalCommandBuffer->windowDataCount == metalCommandBuffer->windowDataCapacity) {
            metalCommandBuffer->windowDataCapacity += 1;
            metalCommandBuffer->windowDatas = SDL_realloc(
                metalCommandBuffer->windowDatas,
                metalCommandBuffer->windowDataCapacity * sizeof(MetalWindowData *));
        }
        metalCommandBuffer->windowDatas[metalCommandBuffer->windowDataCount] = windowData;
        metalCommandBuffer->windowDataCount += 1;

        // Return the swapchain texture
        *texture = (SDL_GPUTexture *)&windowData->textureContainer;
        return true;
    }
}

static bool METAL_AcquireSwapchainTexture(
    SDL_GPUCommandBuffer *command_buffer,
    SDL_Window *window,
    SDL_GPUTexture **swapchain_texture,
    Uint32 *swapchain_texture_width,
    Uint32 *swapchain_texture_height
) {
    return METAL_INTERNAL_AcquireSwapchainTexture(
        false,
        command_buffer,
        window,
        swapchain_texture,
        swapchain_texture_width,
        swapchain_texture_height);
}

static bool METAL_WaitAndAcquireSwapchainTexture(
    SDL_GPUCommandBuffer *command_buffer,
    SDL_Window *window,
    SDL_GPUTexture **swapchain_texture,
    Uint32 *swapchain_texture_width,
    Uint32 *swapchain_texture_height
) {
    return METAL_INTERNAL_AcquireSwapchainTexture(
        true,
        command_buffer,
        window,
        swapchain_texture,
        swapchain_texture_width,
        swapchain_texture_height);
}

static SDL_GPUTextureFormat METAL_GetSwapchainTextureFormat(
    SDL_GPURenderer *driverData,
    SDL_Window *window)
{
    MetalRenderer *renderer = (MetalRenderer *)driverData;
    MetalWindowData *windowData = METAL_INTERNAL_FetchWindowData(window);

    if (windowData == NULL) {
        SET_STRING_ERROR_AND_RETURN("Cannot get swapchain format, window has not been claimed", SDL_GPU_TEXTUREFORMAT_INVALID);
    }

    return windowData->textureContainer.header.info.format;
}

static bool METAL_SetSwapchainParameters(
    SDL_GPURenderer *driverData,
    SDL_Window *window,
    SDL_GPUSwapchainComposition swapchainComposition,
    SDL_GPUPresentMode presentMode)
{
    @autoreleasepool {
        MetalRenderer *renderer = (MetalRenderer *)driverData;
        MetalWindowData *windowData = METAL_INTERNAL_FetchWindowData(window);
        CGColorSpaceRef colorspace;

        if (windowData == NULL) {
            SET_STRING_ERROR_AND_RETURN("Cannot set swapchain parameters, window has not been claimed!", false);
        }

        if (!METAL_SupportsSwapchainComposition(driverData, window, swapchainComposition)) {
            SET_STRING_ERROR_AND_RETURN("Swapchain composition not supported", false);
        }

        if (!METAL_SupportsPresentMode(driverData, window, presentMode)) {
            SET_STRING_ERROR_AND_RETURN("Present mode not supported", false);
        }

        METAL_Wait(driverData);

        windowData->presentMode = SDL_GPU_PRESENTMODE_VSYNC;

#ifdef SDL_PLATFORM_MACOS
        if (@available(macOS 10.13, *)) {
            windowData->layer.displaySyncEnabled = (presentMode != SDL_GPU_PRESENTMODE_IMMEDIATE);
            windowData->presentMode = presentMode;
        }
#endif
        windowData->layer.pixelFormat = SDLToMetal_TextureFormat(SwapchainCompositionToFormat[swapchainComposition]);
#ifndef SDL_PLATFORM_TVOS
        if (@available(iOS 16.0, *)) {
            windowData->layer.wantsExtendedDynamicRangeContent = (swapchainComposition != SDL_GPU_SWAPCHAINCOMPOSITION_SDR);
        }
#endif

        colorspace = CGColorSpaceCreateWithName(SwapchainCompositionToColorSpace[swapchainComposition]);
        windowData->layer.colorspace = colorspace;
        CGColorSpaceRelease(colorspace);

        windowData->textureContainer.header.info.format = SwapchainCompositionToFormat[swapchainComposition];

        return true;
    }
}

static bool METAL_SetAllowedFramesInFlight(
    SDL_GPURenderer *driverData,
    Uint32 allowedFramesInFlight)
{
    @autoreleasepool {
        MetalRenderer *renderer = (MetalRenderer *)driverData;

        if (!METAL_Wait(driverData)) {
            return false;
        }

        renderer->allowedFramesInFlight = allowedFramesInFlight;
        return true;
    }
}

// Submission

static bool METAL_Submit(
    SDL_GPUCommandBuffer *commandBuffer)
{
    @autoreleasepool {
        MetalCommandBuffer *metalCommandBuffer = (MetalCommandBuffer *)commandBuffer;
        MetalRenderer *renderer = metalCommandBuffer->renderer;

        SDL_LockMutex(renderer->submitLock);

        if (!METAL_INTERNAL_AcquireFence(renderer, metalCommandBuffer)) {
            SDL_UnlockMutex(renderer->submitLock);
            return false;
        }

        // Enqueue present requests, if applicable
        for (Uint32 i = 0; i < metalCommandBuffer->windowDataCount; i += 1) {
            MetalWindowData *windowData = metalCommandBuffer->windowDatas[i];
            [metalCommandBuffer->handle presentDrawable:windowData->drawable];
            windowData->drawable = nil;

            windowData->inFlightFences[windowData->frameCounter] = (SDL_GPUFence *)metalCommandBuffer->fence;

            (void)SDL_AtomicIncRef(&metalCommandBuffer->fence->referenceCount);

            windowData->frameCounter = (windowData->frameCounter + 1) % renderer->allowedFramesInFlight;
        }

        // Notify the fence when the command buffer has completed
        [metalCommandBuffer->handle addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
          SDL_AtomicIncRef(&metalCommandBuffer->fence->complete);
        }];

        // Submit the command buffer
        [metalCommandBuffer->handle commit];
        metalCommandBuffer->handle = nil;

        // Mark the command buffer as submitted
        if (renderer->submittedCommandBufferCount >= renderer->submittedCommandBufferCapacity) {
            renderer->submittedCommandBufferCapacity = renderer->submittedCommandBufferCount + 1;

            renderer->submittedCommandBuffers = SDL_realloc(
                renderer->submittedCommandBuffers,
                sizeof(MetalCommandBuffer *) * renderer->submittedCommandBufferCapacity);
        }
        renderer->submittedCommandBuffers[renderer->submittedCommandBufferCount] = metalCommandBuffer;
        renderer->submittedCommandBufferCount += 1;

        // Check if we can perform any cleanups
        for (Sint32 i = renderer->submittedCommandBufferCount - 1; i >= 0; i -= 1) {
            if (SDL_GetAtomicInt(&renderer->submittedCommandBuffers[i]->fence->complete)) {
                METAL_INTERNAL_CleanCommandBuffer(
                    renderer,
                    renderer->submittedCommandBuffers[i],
                    false);
            }
        }

        METAL_INTERNAL_PerformPendingDestroys(renderer);

        SDL_UnlockMutex(renderer->submitLock);

        return true;
    }
}

static SDL_GPUFence *METAL_SubmitAndAcquireFence(
    SDL_GPUCommandBuffer *commandBuffer)
{
    MetalCommandBuffer *metalCommandBuffer = (MetalCommandBuffer *)commandBuffer;
    metalCommandBuffer->autoReleaseFence = false;
    if (!METAL_Submit(commandBuffer)) {
        return NULL;
    }
    return (SDL_GPUFence *)metalCommandBuffer->fence;
}

static bool METAL_Cancel(
    SDL_GPUCommandBuffer *commandBuffer)
{
    MetalCommandBuffer *metalCommandBuffer = (MetalCommandBuffer *)commandBuffer;
    MetalRenderer *renderer = metalCommandBuffer->renderer;

    metalCommandBuffer->autoReleaseFence = false;
    SDL_LockMutex(renderer->submitLock);
    METAL_INTERNAL_CleanCommandBuffer(renderer, metalCommandBuffer, true);
    SDL_UnlockMutex(renderer->submitLock);

    return true;
}

static bool METAL_Wait(
    SDL_GPURenderer *driverData)
{
    @autoreleasepool {
        MetalRenderer *renderer = (MetalRenderer *)driverData;
        MetalCommandBuffer *commandBuffer;

        /*
         * Wait for all submitted command buffers to complete.
         * Sort of equivalent to vkDeviceWaitIdle.
         */
        for (Uint32 i = 0; i < renderer->submittedCommandBufferCount; i += 1) {
            while (!SDL_GetAtomicInt(&renderer->submittedCommandBuffers[i]->fence->complete)) {
                // Spin!
            }
        }

        SDL_LockMutex(renderer->submitLock);

        for (Sint32 i = renderer->submittedCommandBufferCount - 1; i >= 0; i -= 1) {
            commandBuffer = renderer->submittedCommandBuffers[i];
            METAL_INTERNAL_CleanCommandBuffer(renderer, commandBuffer, false);
        }

        METAL_INTERNAL_PerformPendingDestroys(renderer);

        SDL_UnlockMutex(renderer->submitLock);

        return true;
    }
}

// Format Info

// FIXME: Check simultaneous read-write support
static bool METAL_SupportsTextureFormat(
    SDL_GPURenderer *driverData,
    SDL_GPUTextureFormat format,
    SDL_GPUTextureType type,
    SDL_GPUTextureUsageFlags usage)
{
    @autoreleasepool {
        MetalRenderer *renderer = (MetalRenderer *)driverData;

        // Only depth textures can be used as... depth textures
        if ((usage & SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET)) {
            if (!IsDepthFormat(format)) {
                return false;
            }
        }

        // Cube arrays are not supported on older iOS devices
        if (type == SDL_GPU_TEXTURETYPE_CUBE_ARRAY) {
#ifdef SDL_PLATFORM_MACOS
            return true;
#else
            if (@available(iOS 13.0, tvOS 13.0, *)) {
                if (!([renderer->device supportsFamily:MTLGPUFamilyCommon2] ||
                      [renderer->device supportsFamily:MTLGPUFamilyApple4])) {
                    return false;
                }
            } else {
                return false;
            }
#endif
        }

        switch (format) {
        // Apple GPU exclusive
        case SDL_GPU_TEXTUREFORMAT_B5G6R5_UNORM:
        case SDL_GPU_TEXTUREFORMAT_B5G5R5A1_UNORM:
        case SDL_GPU_TEXTUREFORMAT_B4G4R4A4_UNORM:
            if (@available(macOS 10.15, iOS 13.0, tvOS 13.0, *)) {
                return [renderer->device supportsFamily:MTLGPUFamilyApple1];
            } else {
                return false;
            }

        // Requires BC compression support
        case SDL_GPU_TEXTUREFORMAT_BC1_RGBA_UNORM:
        case SDL_GPU_TEXTUREFORMAT_BC2_RGBA_UNORM:
        case SDL_GPU_TEXTUREFORMAT_BC3_RGBA_UNORM:
        case SDL_GPU_TEXTUREFORMAT_BC4_R_UNORM:
        case SDL_GPU_TEXTUREFORMAT_BC5_RG_UNORM:
        case SDL_GPU_TEXTUREFORMAT_BC7_RGBA_UNORM:
        case SDL_GPU_TEXTUREFORMAT_BC6H_RGB_FLOAT:
        case SDL_GPU_TEXTUREFORMAT_BC6H_RGB_UFLOAT:
        case SDL_GPU_TEXTUREFORMAT_BC1_RGBA_UNORM_SRGB:
        case SDL_GPU_TEXTUREFORMAT_BC2_RGBA_UNORM_SRGB:
        case SDL_GPU_TEXTUREFORMAT_BC3_RGBA_UNORM_SRGB:
        case SDL_GPU_TEXTUREFORMAT_BC7_RGBA_UNORM_SRGB:
            if (@available(iOS 16.4, tvOS 16.4, *)) {
                if (usage & SDL_GPU_TEXTUREUSAGE_COLOR_TARGET) {
                    return false;
                }
                if (@available(macOS 11.0, *)) {
                    return [renderer->device supportsBCTextureCompression];
                } else {
                    return true;
                }
            } else {
                return false;
            }

        // Requires D24S8 support
        case SDL_GPU_TEXTUREFORMAT_D24_UNORM:
        case SDL_GPU_TEXTUREFORMAT_D24_UNORM_S8_UINT:
#ifdef SDL_PLATFORM_MACOS
            return [renderer->device isDepth24Stencil8PixelFormatSupported];
#else
            return false;
#endif

        case SDL_GPU_TEXTUREFORMAT_D16_UNORM:
            if (@available(macOS 10.12, iOS 13.0, tvOS 13.0, *)) {
                return true;
            } else {
                return false;
            }

        case SDL_GPU_TEXTUREFORMAT_ASTC_4x4_UNORM:
        case SDL_GPU_TEXTUREFORMAT_ASTC_5x4_UNORM:
        case SDL_GPU_TEXTUREFORMAT_ASTC_5x5_UNORM:
        case SDL_GPU_TEXTUREFORMAT_ASTC_6x5_UNORM:
        case SDL_GPU_TEXTUREFORMAT_ASTC_6x6_UNORM:
        case SDL_GPU_TEXTUREFORMAT_ASTC_8x5_UNORM:
        case SDL_GPU_TEXTUREFORMAT_ASTC_8x6_UNORM:
        case SDL_GPU_TEXTUREFORMAT_ASTC_8x8_UNORM:
        case SDL_GPU_TEXTUREFORMAT_ASTC_10x5_UNORM:
        case SDL_GPU_TEXTUREFORMAT_ASTC_10x6_UNORM:
        case SDL_GPU_TEXTUREFORMAT_ASTC_10x8_UNORM:
        case SDL_GPU_TEXTUREFORMAT_ASTC_10x10_UNORM:
        case SDL_GPU_TEXTUREFORMAT_ASTC_12x10_UNORM:
        case SDL_GPU_TEXTUREFORMAT_ASTC_12x12_UNORM:
        case SDL_GPU_TEXTUREFORMAT_ASTC_4x4_UNORM_SRGB:
        case SDL_GPU_TEXTUREFORMAT_ASTC_5x4_UNORM_SRGB:
        case SDL_GPU_TEXTUREFORMAT_ASTC_5x5_UNORM_SRGB:
        case SDL_GPU_TEXTUREFORMAT_ASTC_6x5_UNORM_SRGB:
        case SDL_GPU_TEXTUREFORMAT_ASTC_6x6_UNORM_SRGB:
        case SDL_GPU_TEXTUREFORMAT_ASTC_8x5_UNORM_SRGB:
        case SDL_GPU_TEXTUREFORMAT_ASTC_8x6_UNORM_SRGB:
        case SDL_GPU_TEXTUREFORMAT_ASTC_8x8_UNORM_SRGB:
        case SDL_GPU_TEXTUREFORMAT_ASTC_10x5_UNORM_SRGB:
        case SDL_GPU_TEXTUREFORMAT_ASTC_10x6_UNORM_SRGB:
        case SDL_GPU_TEXTUREFORMAT_ASTC_10x8_UNORM_SRGB:
        case SDL_GPU_TEXTUREFORMAT_ASTC_10x10_UNORM_SRGB:
        case SDL_GPU_TEXTUREFORMAT_ASTC_12x10_UNORM_SRGB:
        case SDL_GPU_TEXTUREFORMAT_ASTC_12x12_UNORM_SRGB:
#ifdef SDL_PLATFORM_MACOS
            if (@available(macOS 11.0, *)) {
                return [renderer->device supportsFamily:MTLGPUFamilyApple7];
            } else {
                return false;
            }
#else
            return true;
#endif
        case SDL_GPU_TEXTUREFORMAT_ASTC_4x4_FLOAT:
        case SDL_GPU_TEXTUREFORMAT_ASTC_5x4_FLOAT:
        case SDL_GPU_TEXTUREFORMAT_ASTC_5x5_FLOAT:
        case SDL_GPU_TEXTUREFORMAT_ASTC_6x5_FLOAT:
        case SDL_GPU_TEXTUREFORMAT_ASTC_6x6_FLOAT:
        case SDL_GPU_TEXTUREFORMAT_ASTC_8x5_FLOAT:
        case SDL_GPU_TEXTUREFORMAT_ASTC_8x6_FLOAT:
        case SDL_GPU_TEXTUREFORMAT_ASTC_8x8_FLOAT:
        case SDL_GPU_TEXTUREFORMAT_ASTC_10x5_FLOAT:
        case SDL_GPU_TEXTUREFORMAT_ASTC_10x6_FLOAT:
        case SDL_GPU_TEXTUREFORMAT_ASTC_10x8_FLOAT:
        case SDL_GPU_TEXTUREFORMAT_ASTC_10x10_FLOAT:
        case SDL_GPU_TEXTUREFORMAT_ASTC_12x10_FLOAT:
        case SDL_GPU_TEXTUREFORMAT_ASTC_12x12_FLOAT:
#ifdef SDL_PLATFORM_MACOS
            if (@available(macOS 11.0, *)) {
                return [renderer->device supportsFamily:MTLGPUFamilyApple7];
            } else {
                return false;
            }
#else
            if (@available(iOS 13.0, tvOS 13.0, *)) {
                return [renderer->device supportsFamily:MTLGPUFamilyApple6];
            } else {
                return false;
            }
#endif
        default:
            return true;
        }
    }
}

// Device Creation

static bool METAL_PrepareDriver(SDL_VideoDevice *this)
{
    if (@available(macOS 10.14, iOS 13.0, tvOS 13.0, *)) {
        return (this->Metal_CreateView != NULL);
    }
    return false;
}

static void METAL_INTERNAL_InitBlitResources(
    MetalRenderer *renderer)
{
    SDL_GPUShaderCreateInfo shaderModuleCreateInfo;
    SDL_GPUSamplerCreateInfo createinfo;

    // Allocate the dynamic blit pipeline list
    renderer->blitPipelineCapacity = 2;
    renderer->blitPipelineCount = 0;
    renderer->blitPipelines = SDL_calloc(
        renderer->blitPipelineCapacity, sizeof(BlitPipelineCacheEntry));

    // Fullscreen vertex shader
    SDL_zero(shaderModuleCreateInfo);
    shaderModuleCreateInfo.code = FullscreenVert_metallib;
    shaderModuleCreateInfo.code_size = FullscreenVert_metallib_len;
    shaderModuleCreateInfo.stage = SDL_GPU_SHADERSTAGE_VERTEX;
    shaderModuleCreateInfo.format = SDL_GPU_SHADERFORMAT_METALLIB;
    shaderModuleCreateInfo.entrypoint = "FullscreenVert";

    renderer->blitVertexShader = METAL_CreateShader(
        (SDL_GPURenderer *)renderer,
        &shaderModuleCreateInfo);

    if (renderer->blitVertexShader == NULL) {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Failed to compile vertex shader for blit!");
    }

    // BlitFrom2D fragment shader
    shaderModuleCreateInfo.code = BlitFrom2D_metallib;
    shaderModuleCreateInfo.code_size = BlitFrom2D_metallib_len;
    shaderModuleCreateInfo.stage = SDL_GPU_SHADERSTAGE_FRAGMENT;
    shaderModuleCreateInfo.entrypoint = "BlitFrom2D";
    shaderModuleCreateInfo.num_samplers = 1;
    shaderModuleCreateInfo.num_uniform_buffers = 1;

    renderer->blitFrom2DShader = METAL_CreateShader(
        (SDL_GPURenderer *)renderer,
        &shaderModuleCreateInfo);

    if (renderer->blitFrom2DShader == NULL) {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Failed to compile BlitFrom2D fragment shader!");
    }

    // BlitFrom2DArray fragment shader
    shaderModuleCreateInfo.code = BlitFrom2DArray_metallib;
    shaderModuleCreateInfo.code_size = BlitFrom2DArray_metallib_len;
    shaderModuleCreateInfo.entrypoint = "BlitFrom2DArray";

    renderer->blitFrom2DArrayShader = METAL_CreateShader(
        (SDL_GPURenderer *)renderer,
        &shaderModuleCreateInfo);

    if (renderer->blitFrom2DArrayShader == NULL) {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Failed to compile BlitFrom2DArray fragment shader!");
    }

    // BlitFrom3D fragment shader
    shaderModuleCreateInfo.code = BlitFrom3D_metallib;
    shaderModuleCreateInfo.code_size = BlitFrom3D_metallib_len;
    shaderModuleCreateInfo.entrypoint = "BlitFrom3D";

    renderer->blitFrom3DShader = METAL_CreateShader(
        (SDL_GPURenderer *)renderer,
        &shaderModuleCreateInfo);

    if (renderer->blitFrom3DShader == NULL) {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Failed to compile BlitFrom3D fragment shader!");
    }

    // BlitFromCube fragment shader
    shaderModuleCreateInfo.code = BlitFromCube_metallib;
    shaderModuleCreateInfo.code_size = BlitFromCube_metallib_len;
    shaderModuleCreateInfo.entrypoint = "BlitFromCube";

    renderer->blitFromCubeShader = METAL_CreateShader(
        (SDL_GPURenderer *)renderer,
        &shaderModuleCreateInfo);

    if (renderer->blitFromCubeShader == NULL) {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Failed to compile BlitFromCube fragment shader!");
    }

    // BlitFromCubeArray fragment shader
    shaderModuleCreateInfo.code = BlitFromCubeArray_metallib;
    shaderModuleCreateInfo.code_size = BlitFromCubeArray_metallib_len;
    shaderModuleCreateInfo.entrypoint = "BlitFromCubeArray";

    renderer->blitFromCubeArrayShader = METAL_CreateShader(
        (SDL_GPURenderer *)renderer,
        &shaderModuleCreateInfo);

    if (renderer->blitFromCubeArrayShader == NULL) {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Failed to compile BlitFromCubeArray fragment shader!");
    }

    // Create samplers
    createinfo.address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    createinfo.address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    createinfo.address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    createinfo.enable_anisotropy = 0;
    createinfo.enable_compare = 0;
    createinfo.mag_filter = SDL_GPU_FILTER_NEAREST;
    createinfo.min_filter = SDL_GPU_FILTER_NEAREST;
    createinfo.mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_NEAREST;
    createinfo.mip_lod_bias = 0.0f;
    createinfo.min_lod = 0;
    createinfo.max_lod = 1000;
    createinfo.max_anisotropy = 1.0f;
    createinfo.compare_op = SDL_GPU_COMPAREOP_ALWAYS;

    renderer->blitNearestSampler = METAL_CreateSampler(
        (SDL_GPURenderer *)renderer,
        &createinfo);

    if (renderer->blitNearestSampler == NULL) {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Failed to create blit nearest sampler!");
    }

    createinfo.mag_filter = SDL_GPU_FILTER_LINEAR;
    createinfo.min_filter = SDL_GPU_FILTER_LINEAR;
    createinfo.mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_LINEAR;

    renderer->blitLinearSampler = METAL_CreateSampler(
        (SDL_GPURenderer *)renderer,
        &createinfo);

    if (renderer->blitLinearSampler == NULL) {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Failed to create blit linear sampler!");
    }
}

static void METAL_INTERNAL_DestroyBlitResources(
    SDL_GPURenderer *driverData)
{
    MetalRenderer *renderer = (MetalRenderer *)driverData;
    METAL_ReleaseSampler(driverData, renderer->blitLinearSampler);
    METAL_ReleaseSampler(driverData, renderer->blitNearestSampler);
    METAL_ReleaseShader(driverData, renderer->blitVertexShader);
    METAL_ReleaseShader(driverData, renderer->blitFrom2DShader);
    METAL_ReleaseShader(driverData, renderer->blitFrom2DArrayShader);
    METAL_ReleaseShader(driverData, renderer->blitFrom3DShader);
    METAL_ReleaseShader(driverData, renderer->blitFromCubeShader);
    METAL_ReleaseShader(driverData, renderer->blitFromCubeArrayShader);

    for (Uint32 i = 0; i < renderer->blitPipelineCount; i += 1) {
        METAL_ReleaseGraphicsPipeline(driverData, renderer->blitPipelines[i].pipeline);
    }
    SDL_free(renderer->blitPipelines);
}

static SDL_GPUDevice *METAL_CreateDevice(bool debugMode, bool preferLowPower, SDL_PropertiesID props)
{
    @autoreleasepool {
        MetalRenderer *renderer;
        id<MTLDevice> device = NULL;
        bool hasHardwareSupport = false;

        bool verboseLogs = SDL_GetBooleanProperty(
            props,
            SDL_PROP_GPU_DEVICE_CREATE_VERBOSE_BOOLEAN,
            true);

        if (debugMode) {
            /* Due to a Metal driver quirk, once a MTLDevice has been created
             * with this environment variable set, the Metal validation layers
             * will remain enabled for the rest of the application's lifespan,
             * even if the device is destroyed and recreated.
             */
            SDL_setenv_unsafe("MTL_DEBUG_LAYER", "1", 0);
        }

        // Create the Metal device and command queue
#ifdef SDL_PLATFORM_MACOS
        if (preferLowPower) {
            NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
            for (id<MTLDevice> candidate in devices) {
                if (candidate.isLowPower) {
                    device = candidate;
                    break;
                }
            }
        }
#endif
        if (device == NULL) {
            device = MTLCreateSystemDefaultDevice();
            if (device == NULL) {
                SDL_SetError("Failed to create Metal device");
                return NULL;
            }
        }

#ifdef SDL_PLATFORM_MACOS
        hasHardwareSupport = true;
        if (@available(macOS 10.15, *)) {
            hasHardwareSupport = [device supportsFamily:MTLGPUFamilyMac2];
        } else if (@available(macOS 10.14, *)) {
            hasHardwareSupport = [device supportsFeatureSet:MTLFeatureSet_macOS_GPUFamily2_v1];
        }
#else
        if (@available(iOS 13.0, tvOS 13.0, *)) {
            hasHardwareSupport = [device supportsFamily:MTLGPUFamilyApple3];
        }
#endif

        if (!hasHardwareSupport) {
            SDL_SetError("Device does not meet the hardware requirements for SDL_GPU Metal");
            return NULL;
        }

        // Allocate and zero out the renderer
        renderer = (MetalRenderer *)SDL_calloc(1, sizeof(MetalRenderer));

        renderer->device = device;
        renderer->queue = [device newCommandQueue];

        renderer->props = SDL_CreateProperties();
        if (verboseLogs) {
            SDL_LogInfo(SDL_LOG_CATEGORY_GPU, "SDL_GPU Driver: Metal");
        }

        // Record device name
        const char *deviceName = [device.name UTF8String];
        SDL_SetStringProperty(
            renderer->props,
            SDL_PROP_GPU_DEVICE_NAME_STRING,
            deviceName);
        if (verboseLogs) {
            SDL_LogInfo(SDL_LOG_CATEGORY_GPU, "Metal Device: %s", deviceName);
        }

        // Remember debug mode
        renderer->debugMode = debugMode;
        renderer->allowedFramesInFlight = 2;

        // Set up colorspace array
        SwapchainCompositionToColorSpace[0] = kCGColorSpaceSRGB;
        SwapchainCompositionToColorSpace[1] = kCGColorSpaceSRGB;
        SwapchainCompositionToColorSpace[2] = kCGColorSpaceExtendedLinearSRGB;
        if (@available(macOS 11.0, iOS 14.0, tvOS 14.0, *)) {
            SwapchainCompositionToColorSpace[3] = kCGColorSpaceITUR_2100_PQ;
        } else {
            SwapchainCompositionToColorSpace[3] = NULL;
        }

        // Create mutexes
        renderer->submitLock = SDL_CreateMutex();
        renderer->acquireCommandBufferLock = SDL_CreateMutex();
        renderer->acquireUniformBufferLock = SDL_CreateMutex();
        renderer->disposeLock = SDL_CreateMutex();
        renderer->fenceLock = SDL_CreateMutex();
        renderer->windowLock = SDL_CreateMutex();

        // Create command buffer pool
        METAL_INTERNAL_AllocateCommandBuffers(renderer, 2);

        // Create fence pool
        renderer->availableFenceCapacity = 2;
        renderer->availableFences = SDL_calloc(
            renderer->availableFenceCapacity, sizeof(MetalFence *));

        // Create uniform buffer pool
        renderer->uniformBufferPoolCapacity = 32;
        renderer->uniformBufferPoolCount = 32;
        renderer->uniformBufferPool = SDL_calloc(
            renderer->uniformBufferPoolCapacity, sizeof(MetalUniformBuffer *));

        for (Uint32 i = 0; i < renderer->uniformBufferPoolCount; i += 1) {
            renderer->uniformBufferPool[i] = METAL_INTERNAL_CreateUniformBuffer(
                renderer,
                UNIFORM_BUFFER_SIZE);
        }

        // Create deferred destroy arrays
        renderer->bufferContainersToDestroyCapacity = 2;
        renderer->bufferContainersToDestroyCount = 0;
        renderer->bufferContainersToDestroy = SDL_calloc(
            renderer->bufferContainersToDestroyCapacity, sizeof(MetalBufferContainer *));

        renderer->textureContainersToDestroyCapacity = 2;
        renderer->textureContainersToDestroyCount = 0;
        renderer->textureContainersToDestroy = SDL_calloc(
            renderer->textureContainersToDestroyCapacity, sizeof(MetalTextureContainer *));

        // Create claimed window list
        renderer->claimedWindowCapacity = 1;
        renderer->claimedWindows = SDL_calloc(
            renderer->claimedWindowCapacity, sizeof(MetalWindowData *));

        // Initialize blit resources
        METAL_INTERNAL_InitBlitResources(renderer);

        SDL_GPUDevice *result = SDL_calloc(1, sizeof(SDL_GPUDevice));
        ASSIGN_DRIVER(METAL)
        result->driverData = (SDL_GPURenderer *)renderer;
        renderer->sdlGPUDevice = result;

        return result;
    }
}

SDL_GPUBootstrap MetalDriver = {
    "metal",
    SDL_GPU_SHADERFORMAT_MSL | SDL_GPU_SHADERFORMAT_METALLIB,
    METAL_PrepareDriver,
    METAL_CreateDevice
};

#endif // SDL_GPU_METAL
