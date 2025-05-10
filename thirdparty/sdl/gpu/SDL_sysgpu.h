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
#include "../video/SDL_sysvideo.h"
#include "SDL_internal.h"

#ifndef SDL_GPU_DRIVER_H
#define SDL_GPU_DRIVER_H

// GraphicsDevice Limits

#define MAX_TEXTURE_SAMPLERS_PER_STAGE 16
#define MAX_STORAGE_TEXTURES_PER_STAGE 8
#define MAX_STORAGE_BUFFERS_PER_STAGE  8
#define MAX_UNIFORM_BUFFERS_PER_STAGE  4
#define MAX_COMPUTE_WRITE_TEXTURES     8
#define MAX_COMPUTE_WRITE_BUFFERS      8
#define UNIFORM_BUFFER_SIZE            32768
#define MAX_VERTEX_BUFFERS             16
#define MAX_VERTEX_ATTRIBUTES          16
#define MAX_COLOR_TARGET_BINDINGS      4
#define MAX_PRESENT_COUNT              16
#define MAX_FRAMES_IN_FLIGHT           3

// Common Structs

typedef struct Pass
{
    SDL_GPUCommandBuffer *command_buffer;
    bool in_progress;
} Pass;

typedef struct RenderPass
{
    SDL_GPUCommandBuffer *command_buffer;
    bool in_progress;
    SDL_GPUTexture *color_targets[MAX_COLOR_TARGET_BINDINGS];
    Uint32 num_color_targets;
    SDL_GPUTexture *depth_stencil_target;
} RenderPass;

typedef struct CommandBufferCommonHeader
{
    SDL_GPUDevice *device;
    RenderPass render_pass;
    bool graphics_pipeline_bound;
    Pass compute_pass;
    bool compute_pipeline_bound;
    Pass copy_pass;
    bool swapchain_texture_acquired;
    bool submitted;
} CommandBufferCommonHeader;

typedef struct TextureCommonHeader
{
    SDL_GPUTextureCreateInfo info;
} TextureCommonHeader;

typedef struct BlitFragmentUniforms
{
    // texcoord space
    float left;
    float top;
    float width;
    float height;

    Uint32 mip_level;
    float layer_or_depth;
} BlitFragmentUniforms;

typedef struct BlitPipelineCacheEntry
{
    SDL_GPUTextureType type;
    SDL_GPUTextureFormat format;
    SDL_GPUGraphicsPipeline *pipeline;
} BlitPipelineCacheEntry;

// Internal Helper Utilities

#define SDL_GPU_TEXTUREFORMAT_MAX_ENUM_VALUE        (SDL_GPU_TEXTUREFORMAT_ASTC_12x12_FLOAT + 1)
#define SDL_GPU_VERTEXELEMENTFORMAT_MAX_ENUM_VALUE  (SDL_GPU_VERTEXELEMENTFORMAT_HALF4 + 1)
#define SDL_GPU_COMPAREOP_MAX_ENUM_VALUE            (SDL_GPU_COMPAREOP_ALWAYS + 1)
#define SDL_GPU_STENCILOP_MAX_ENUM_VALUE            (SDL_GPU_STENCILOP_DECREMENT_AND_WRAP + 1)
#define SDL_GPU_BLENDOP_MAX_ENUM_VALUE              (SDL_GPU_BLENDOP_MAX + 1)
#define SDL_GPU_BLENDFACTOR_MAX_ENUM_VALUE          (SDL_GPU_BLENDFACTOR_SRC_ALPHA_SATURATE + 1)
#define SDL_GPU_SWAPCHAINCOMPOSITION_MAX_ENUM_VALUE (SDL_GPU_SWAPCHAINCOMPOSITION_HDR10_ST2084 + 1)
#define SDL_GPU_PRESENTMODE_MAX_ENUM_VALUE          (SDL_GPU_PRESENTMODE_MAILBOX + 1)

static inline Sint32 Texture_GetBlockWidth(
    SDL_GPUTextureFormat format)
{
    switch (format) {
    case SDL_GPU_TEXTUREFORMAT_ASTC_12x10_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_12x12_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_12x10_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_12x12_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_12x10_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_ASTC_12x12_FLOAT:
        return 12;
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x5_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x6_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x8_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x10_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x5_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x6_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x8_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x10_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x5_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x6_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x8_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x10_FLOAT:
        return 10;
    case SDL_GPU_TEXTUREFORMAT_ASTC_8x5_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_8x6_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_8x8_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_8x5_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_8x6_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_8x8_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_8x5_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_ASTC_8x6_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_ASTC_8x8_FLOAT:
        return 8;
    case SDL_GPU_TEXTUREFORMAT_ASTC_6x5_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_6x6_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_6x5_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_6x6_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_6x5_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_ASTC_6x6_FLOAT:
        return 6;
    case SDL_GPU_TEXTUREFORMAT_ASTC_5x4_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_5x5_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_5x4_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_5x5_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_5x4_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_ASTC_5x5_FLOAT:
        return 5;
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
    case SDL_GPU_TEXTUREFORMAT_ASTC_4x4_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_4x4_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_4x4_FLOAT:
        return 4;
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM:
    case SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM:
    case SDL_GPU_TEXTUREFORMAT_B5G6R5_UNORM:
    case SDL_GPU_TEXTUREFORMAT_B5G5R5A1_UNORM:
    case SDL_GPU_TEXTUREFORMAT_B4G4R4A4_UNORM:
    case SDL_GPU_TEXTUREFORMAT_R10G10B10A2_UNORM:
    case SDL_GPU_TEXTUREFORMAT_R8G8_UNORM:
    case SDL_GPU_TEXTUREFORMAT_R16G16_UNORM:
    case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_UNORM:
    case SDL_GPU_TEXTUREFORMAT_R8_UNORM:
    case SDL_GPU_TEXTUREFORMAT_R16_UNORM:
    case SDL_GPU_TEXTUREFORMAT_A8_UNORM:
    case SDL_GPU_TEXTUREFORMAT_R8_SNORM:
    case SDL_GPU_TEXTUREFORMAT_R8G8_SNORM:
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_SNORM:
    case SDL_GPU_TEXTUREFORMAT_R16_SNORM:
    case SDL_GPU_TEXTUREFORMAT_R16G16_SNORM:
    case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_SNORM:
    case SDL_GPU_TEXTUREFORMAT_R16_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_R16G16_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_R32_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_R32G32_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_R32G32B32A32_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_R11G11B10_UFLOAT:
    case SDL_GPU_TEXTUREFORMAT_R8_UINT:
    case SDL_GPU_TEXTUREFORMAT_R8G8_UINT:
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UINT:
    case SDL_GPU_TEXTUREFORMAT_R16_UINT:
    case SDL_GPU_TEXTUREFORMAT_R16G16_UINT:
    case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_UINT:
    case SDL_GPU_TEXTUREFORMAT_R32_UINT:
    case SDL_GPU_TEXTUREFORMAT_R32G32_UINT:
    case SDL_GPU_TEXTUREFORMAT_R32G32B32A32_UINT:
    case SDL_GPU_TEXTUREFORMAT_R8_INT:
    case SDL_GPU_TEXTUREFORMAT_R8G8_INT:
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_INT:
    case SDL_GPU_TEXTUREFORMAT_R16_INT:
    case SDL_GPU_TEXTUREFORMAT_R16G16_INT:
    case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_INT:
    case SDL_GPU_TEXTUREFORMAT_R32_INT:
    case SDL_GPU_TEXTUREFORMAT_R32G32_INT:
    case SDL_GPU_TEXTUREFORMAT_R32G32B32A32_INT:
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_D16_UNORM:
    case SDL_GPU_TEXTUREFORMAT_D24_UNORM:
    case SDL_GPU_TEXTUREFORMAT_D32_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_D24_UNORM_S8_UINT:
    case SDL_GPU_TEXTUREFORMAT_D32_FLOAT_S8_UINT:
        return 1;
    default:
        SDL_assert_release(!"Unrecognized TextureFormat!");
        return 0;
    }
}

static inline Sint32 Texture_GetBlockHeight(
    SDL_GPUTextureFormat format)
{
    switch (format) {
    case SDL_GPU_TEXTUREFORMAT_ASTC_12x12_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_12x12_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_12x12_FLOAT:
        return 12;
    case SDL_GPU_TEXTUREFORMAT_ASTC_12x10_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_12x10_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_12x10_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x10_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x10_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x10_FLOAT:
        return 10;
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x8_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x8_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x8_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_ASTC_8x8_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_8x8_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_8x8_FLOAT:
        return 8;
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x6_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x6_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x6_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_ASTC_8x6_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_8x6_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_8x6_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_ASTC_6x6_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_6x6_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_6x6_FLOAT:
        return 6;
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x5_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x5_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x5_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_ASTC_8x5_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_8x5_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_8x5_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_ASTC_6x5_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_6x5_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_6x5_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_ASTC_5x5_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_5x5_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_5x5_FLOAT:
        return 5;
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
    case SDL_GPU_TEXTUREFORMAT_ASTC_5x4_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_5x4_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_5x4_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_ASTC_4x4_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_4x4_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_4x4_FLOAT:
        return 4;
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM:
    case SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM:
    case SDL_GPU_TEXTUREFORMAT_B5G6R5_UNORM:
    case SDL_GPU_TEXTUREFORMAT_B5G5R5A1_UNORM:
    case SDL_GPU_TEXTUREFORMAT_B4G4R4A4_UNORM:
    case SDL_GPU_TEXTUREFORMAT_R10G10B10A2_UNORM:
    case SDL_GPU_TEXTUREFORMAT_R8G8_UNORM:
    case SDL_GPU_TEXTUREFORMAT_R16G16_UNORM:
    case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_UNORM:
    case SDL_GPU_TEXTUREFORMAT_R8_UNORM:
    case SDL_GPU_TEXTUREFORMAT_R16_UNORM:
    case SDL_GPU_TEXTUREFORMAT_A8_UNORM:
    case SDL_GPU_TEXTUREFORMAT_R8_SNORM:
    case SDL_GPU_TEXTUREFORMAT_R8G8_SNORM:
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_SNORM:
    case SDL_GPU_TEXTUREFORMAT_R16_SNORM:
    case SDL_GPU_TEXTUREFORMAT_R16G16_SNORM:
    case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_SNORM:
    case SDL_GPU_TEXTUREFORMAT_R16_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_R16G16_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_R32_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_R32G32_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_R32G32B32A32_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_R11G11B10_UFLOAT:
    case SDL_GPU_TEXTUREFORMAT_R8_UINT:
    case SDL_GPU_TEXTUREFORMAT_R8G8_UINT:
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UINT:
    case SDL_GPU_TEXTUREFORMAT_R16_UINT:
    case SDL_GPU_TEXTUREFORMAT_R16G16_UINT:
    case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_UINT:
    case SDL_GPU_TEXTUREFORMAT_R32_UINT:
    case SDL_GPU_TEXTUREFORMAT_R32G32_UINT:
    case SDL_GPU_TEXTUREFORMAT_R32G32B32A32_UINT:
    case SDL_GPU_TEXTUREFORMAT_R8_INT:
    case SDL_GPU_TEXTUREFORMAT_R8G8_INT:
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_INT:
    case SDL_GPU_TEXTUREFORMAT_R16_INT:
    case SDL_GPU_TEXTUREFORMAT_R16G16_INT:
    case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_INT:
    case SDL_GPU_TEXTUREFORMAT_R32_INT:
    case SDL_GPU_TEXTUREFORMAT_R32G32_INT:
    case SDL_GPU_TEXTUREFORMAT_R32G32B32A32_INT:
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_D16_UNORM:
    case SDL_GPU_TEXTUREFORMAT_D24_UNORM:
    case SDL_GPU_TEXTUREFORMAT_D32_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_D24_UNORM_S8_UINT:
    case SDL_GPU_TEXTUREFORMAT_D32_FLOAT_S8_UINT:
        return 1;
    default:
        SDL_assert_release(!"Unrecognized TextureFormat!");
        return 0;
    }
}

static inline bool IsDepthFormat(
    SDL_GPUTextureFormat format)
{
    switch (format) {
    case SDL_GPU_TEXTUREFORMAT_D16_UNORM:
    case SDL_GPU_TEXTUREFORMAT_D24_UNORM:
    case SDL_GPU_TEXTUREFORMAT_D32_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_D24_UNORM_S8_UINT:
    case SDL_GPU_TEXTUREFORMAT_D32_FLOAT_S8_UINT:
        return true;

    default:
        return false;
    }
}

static inline bool IsStencilFormat(
    SDL_GPUTextureFormat format)
{
    switch (format) {
    case SDL_GPU_TEXTUREFORMAT_D24_UNORM_S8_UINT:
    case SDL_GPU_TEXTUREFORMAT_D32_FLOAT_S8_UINT:
        return true;

    default:
        return false;
    }
}

static inline bool IsIntegerFormat(
    SDL_GPUTextureFormat format)
{
    switch (format) {
    case SDL_GPU_TEXTUREFORMAT_R8_UINT:
    case SDL_GPU_TEXTUREFORMAT_R8G8_UINT:
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UINT:
    case SDL_GPU_TEXTUREFORMAT_R16_UINT:
    case SDL_GPU_TEXTUREFORMAT_R16G16_UINT:
    case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_UINT:
    case SDL_GPU_TEXTUREFORMAT_R8_INT:
    case SDL_GPU_TEXTUREFORMAT_R8G8_INT:
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_INT:
    case SDL_GPU_TEXTUREFORMAT_R16_INT:
    case SDL_GPU_TEXTUREFORMAT_R16G16_INT:
    case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_INT:
        return true;

    default:
        return false;
    }
}

static inline bool IsCompressedFormat(
    SDL_GPUTextureFormat format)
{
    switch (format) {
    case SDL_GPU_TEXTUREFORMAT_BC1_RGBA_UNORM:
    case SDL_GPU_TEXTUREFORMAT_BC1_RGBA_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_BC2_RGBA_UNORM:
    case SDL_GPU_TEXTUREFORMAT_BC2_RGBA_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_BC3_RGBA_UNORM:
    case SDL_GPU_TEXTUREFORMAT_BC3_RGBA_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_BC4_R_UNORM:
    case SDL_GPU_TEXTUREFORMAT_BC5_RG_UNORM:
    case SDL_GPU_TEXTUREFORMAT_BC6H_RGB_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_BC6H_RGB_UFLOAT:
    case SDL_GPU_TEXTUREFORMAT_BC7_RGBA_UNORM:
    case SDL_GPU_TEXTUREFORMAT_BC7_RGBA_UNORM_SRGB:
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
        return true;

    default:
        return false;
    }
}

static inline bool FormatHasAlpha(
    SDL_GPUTextureFormat format)
{
    switch (format) {
        case SDL_GPU_TEXTUREFORMAT_ASTC_12x10_UNORM:
        case SDL_GPU_TEXTUREFORMAT_ASTC_12x12_UNORM:
        case SDL_GPU_TEXTUREFORMAT_ASTC_12x10_UNORM_SRGB:
        case SDL_GPU_TEXTUREFORMAT_ASTC_12x12_UNORM_SRGB:
        case SDL_GPU_TEXTUREFORMAT_ASTC_12x10_FLOAT:
        case SDL_GPU_TEXTUREFORMAT_ASTC_12x12_FLOAT:
        case SDL_GPU_TEXTUREFORMAT_ASTC_10x5_UNORM:
        case SDL_GPU_TEXTUREFORMAT_ASTC_10x6_UNORM:
        case SDL_GPU_TEXTUREFORMAT_ASTC_10x8_UNORM:
        case SDL_GPU_TEXTUREFORMAT_ASTC_10x10_UNORM:
        case SDL_GPU_TEXTUREFORMAT_ASTC_10x5_UNORM_SRGB:
        case SDL_GPU_TEXTUREFORMAT_ASTC_10x6_UNORM_SRGB:
        case SDL_GPU_TEXTUREFORMAT_ASTC_10x8_UNORM_SRGB:
        case SDL_GPU_TEXTUREFORMAT_ASTC_10x10_UNORM_SRGB:
        case SDL_GPU_TEXTUREFORMAT_ASTC_10x5_FLOAT:
        case SDL_GPU_TEXTUREFORMAT_ASTC_10x6_FLOAT:
        case SDL_GPU_TEXTUREFORMAT_ASTC_10x8_FLOAT:
        case SDL_GPU_TEXTUREFORMAT_ASTC_10x10_FLOAT:
        case SDL_GPU_TEXTUREFORMAT_ASTC_8x5_UNORM:
        case SDL_GPU_TEXTUREFORMAT_ASTC_8x6_UNORM:
        case SDL_GPU_TEXTUREFORMAT_ASTC_8x8_UNORM:
        case SDL_GPU_TEXTUREFORMAT_ASTC_8x5_UNORM_SRGB:
        case SDL_GPU_TEXTUREFORMAT_ASTC_8x6_UNORM_SRGB:
        case SDL_GPU_TEXTUREFORMAT_ASTC_8x8_UNORM_SRGB:
        case SDL_GPU_TEXTUREFORMAT_ASTC_8x5_FLOAT:
        case SDL_GPU_TEXTUREFORMAT_ASTC_8x6_FLOAT:
        case SDL_GPU_TEXTUREFORMAT_ASTC_8x8_FLOAT:
        case SDL_GPU_TEXTUREFORMAT_ASTC_6x5_UNORM:
        case SDL_GPU_TEXTUREFORMAT_ASTC_6x6_UNORM:
        case SDL_GPU_TEXTUREFORMAT_ASTC_6x5_UNORM_SRGB:
        case SDL_GPU_TEXTUREFORMAT_ASTC_6x6_UNORM_SRGB:
        case SDL_GPU_TEXTUREFORMAT_ASTC_6x5_FLOAT:
        case SDL_GPU_TEXTUREFORMAT_ASTC_6x6_FLOAT:
        case SDL_GPU_TEXTUREFORMAT_ASTC_5x4_UNORM:
        case SDL_GPU_TEXTUREFORMAT_ASTC_5x5_UNORM:
        case SDL_GPU_TEXTUREFORMAT_ASTC_5x4_UNORM_SRGB:
        case SDL_GPU_TEXTUREFORMAT_ASTC_5x5_UNORM_SRGB:
        case SDL_GPU_TEXTUREFORMAT_ASTC_5x4_FLOAT:
        case SDL_GPU_TEXTUREFORMAT_ASTC_5x5_FLOAT:
        case SDL_GPU_TEXTUREFORMAT_ASTC_4x4_UNORM:
        case SDL_GPU_TEXTUREFORMAT_ASTC_4x4_UNORM_SRGB:
        case SDL_GPU_TEXTUREFORMAT_ASTC_4x4_FLOAT:
            // ASTC textures may or may not have alpha; return true as this is mainly intended for validation
            return true;

        case SDL_GPU_TEXTUREFORMAT_BC1_RGBA_UNORM:
        case SDL_GPU_TEXTUREFORMAT_BC2_RGBA_UNORM:
        case SDL_GPU_TEXTUREFORMAT_BC3_RGBA_UNORM:
        case SDL_GPU_TEXTUREFORMAT_BC7_RGBA_UNORM:
        case SDL_GPU_TEXTUREFORMAT_BC1_RGBA_UNORM_SRGB:
        case SDL_GPU_TEXTUREFORMAT_BC2_RGBA_UNORM_SRGB:
        case SDL_GPU_TEXTUREFORMAT_BC3_RGBA_UNORM_SRGB:
        case SDL_GPU_TEXTUREFORMAT_BC7_RGBA_UNORM_SRGB:
        case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM:
        case SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM:
        case SDL_GPU_TEXTUREFORMAT_B5G5R5A1_UNORM:
        case SDL_GPU_TEXTUREFORMAT_B4G4R4A4_UNORM:
        case SDL_GPU_TEXTUREFORMAT_R10G10B10A2_UNORM:
        case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_UNORM:
        case SDL_GPU_TEXTUREFORMAT_A8_UNORM:
        case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_SNORM:
        case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_SNORM:
        case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT:
        case SDL_GPU_TEXTUREFORMAT_R32G32B32A32_FLOAT:
        case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UINT:
        case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_UINT:
        case SDL_GPU_TEXTUREFORMAT_R32G32B32A32_UINT:
        case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_INT:
        case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_INT:
        case SDL_GPU_TEXTUREFORMAT_R32G32B32A32_INT:
        case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM_SRGB:
        case SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM_SRGB:
            return true;

        default:
            return false;
    }
}

static inline Uint32 IndexSize(SDL_GPUIndexElementSize size)
{
    return (size == SDL_GPU_INDEXELEMENTSIZE_16BIT) ? 2 : 4;
}

static inline Uint32 BytesPerRow(
    Sint32 width,
    SDL_GPUTextureFormat format)
{
    Uint32 blockWidth = Texture_GetBlockWidth(format);
    Uint32 blocksPerRow = (width + blockWidth - 1) / blockWidth;
    return blocksPerRow * SDL_GPUTextureFormatTexelBlockSize(format);
}

// Internal Macros

#define EXPAND_ARRAY_IF_NEEDED(arr, elementType, newCount, capacity, newCapacity) \
    do {                                                                          \
        if ((newCount) >= (capacity)) {                                           \
            (capacity) = (newCapacity);                                           \
            (arr) = (elementType *)SDL_realloc(                                   \
                (arr),                                                            \
                sizeof(elementType) * (capacity));                                \
        }                                                                         \
    } while (0)

// Internal Declarations

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

SDL_GPUGraphicsPipeline *SDL_GPU_FetchBlitPipeline(
    SDL_GPUDevice *device,
    SDL_GPUTextureType sourceTextureType,
    SDL_GPUTextureFormat destinationFormat,
    SDL_GPUShader *blitVertexShader,
    SDL_GPUShader *blitFrom2DShader,
    SDL_GPUShader *blitFrom2DArrayShader,
    SDL_GPUShader *blitFrom3DShader,
    SDL_GPUShader *blitFromCubeShader,
    SDL_GPUShader *blitFromCubeArrayShader,
    BlitPipelineCacheEntry **blitPipelines,
    Uint32 *blitPipelineCount,
    Uint32 *blitPipelineCapacity);

void SDL_GPU_BlitCommon(
    SDL_GPUCommandBuffer *commandBuffer,
    const SDL_GPUBlitInfo *info,
    SDL_GPUSampler *blitLinearSampler,
    SDL_GPUSampler *blitNearestSampler,
    SDL_GPUShader *blitVertexShader,
    SDL_GPUShader *blitFrom2DShader,
    SDL_GPUShader *blitFrom2DArrayShader,
    SDL_GPUShader *blitFrom3DShader,
    SDL_GPUShader *blitFromCubeShader,
    SDL_GPUShader *blitFromCubeArrayShader,
    BlitPipelineCacheEntry **blitPipelines,
    Uint32 *blitPipelineCount,
    Uint32 *blitPipelineCapacity);

#ifdef __cplusplus
}
#endif // __cplusplus

// SDL_GPUDevice Definition

typedef struct SDL_GPURenderer SDL_GPURenderer;

struct SDL_GPUDevice
{
    // Device

    void (*DestroyDevice)(SDL_GPUDevice *device);

    SDL_PropertiesID (*GetDeviceProperties)(SDL_GPUDevice *device);

    // State Creation

    SDL_GPUComputePipeline *(*CreateComputePipeline)(
        SDL_GPURenderer *driverData,
        const SDL_GPUComputePipelineCreateInfo *createinfo);

    SDL_GPUGraphicsPipeline *(*CreateGraphicsPipeline)(
        SDL_GPURenderer *driverData,
        const SDL_GPUGraphicsPipelineCreateInfo *createinfo);

    SDL_GPUSampler *(*CreateSampler)(
        SDL_GPURenderer *driverData,
        const SDL_GPUSamplerCreateInfo *createinfo);

    SDL_GPUShader *(*CreateShader)(
        SDL_GPURenderer *driverData,
        const SDL_GPUShaderCreateInfo *createinfo);

    SDL_GPUTexture *(*CreateTexture)(
        SDL_GPURenderer *driverData,
        const SDL_GPUTextureCreateInfo *createinfo);

    SDL_GPUBuffer *(*CreateBuffer)(
        SDL_GPURenderer *driverData,
        SDL_GPUBufferUsageFlags usageFlags,
        Uint32 size,
        const char *debugName);

    SDL_GPUTransferBuffer *(*CreateTransferBuffer)(
        SDL_GPURenderer *driverData,
        SDL_GPUTransferBufferUsage usage,
        Uint32 size,
        const char *debugName);

    // Debug Naming

    void (*SetBufferName)(
        SDL_GPURenderer *driverData,
        SDL_GPUBuffer *buffer,
        const char *text);

    void (*SetTextureName)(
        SDL_GPURenderer *driverData,
        SDL_GPUTexture *texture,
        const char *text);

    void (*InsertDebugLabel)(
        SDL_GPUCommandBuffer *commandBuffer,
        const char *text);

    void (*PushDebugGroup)(
        SDL_GPUCommandBuffer *commandBuffer,
        const char *name);

    void (*PopDebugGroup)(
        SDL_GPUCommandBuffer *commandBuffer);

    // Disposal

    void (*ReleaseTexture)(
        SDL_GPURenderer *driverData,
        SDL_GPUTexture *texture);

    void (*ReleaseSampler)(
        SDL_GPURenderer *driverData,
        SDL_GPUSampler *sampler);

    void (*ReleaseBuffer)(
        SDL_GPURenderer *driverData,
        SDL_GPUBuffer *buffer);

    void (*ReleaseTransferBuffer)(
        SDL_GPURenderer *driverData,
        SDL_GPUTransferBuffer *transferBuffer);

    void (*ReleaseShader)(
        SDL_GPURenderer *driverData,
        SDL_GPUShader *shader);

    void (*ReleaseComputePipeline)(
        SDL_GPURenderer *driverData,
        SDL_GPUComputePipeline *computePipeline);

    void (*ReleaseGraphicsPipeline)(
        SDL_GPURenderer *driverData,
        SDL_GPUGraphicsPipeline *graphicsPipeline);

    // Render Pass

    void (*BeginRenderPass)(
        SDL_GPUCommandBuffer *commandBuffer,
        const SDL_GPUColorTargetInfo *colorTargetInfos,
        Uint32 numColorTargets,
        const SDL_GPUDepthStencilTargetInfo *depthStencilTargetInfo);

    void (*BindGraphicsPipeline)(
        SDL_GPUCommandBuffer *commandBuffer,
        SDL_GPUGraphicsPipeline *graphicsPipeline);

    void (*SetViewport)(
        SDL_GPUCommandBuffer *commandBuffer,
        const SDL_GPUViewport *viewport);

    void (*SetScissor)(
        SDL_GPUCommandBuffer *commandBuffer,
        const SDL_Rect *scissor);

    void (*SetBlendConstants)(
        SDL_GPUCommandBuffer *commandBuffer,
        SDL_FColor blendConstants);

    void (*SetStencilReference)(
        SDL_GPUCommandBuffer *commandBuffer,
        Uint8 reference);

    void (*BindVertexBuffers)(
        SDL_GPUCommandBuffer *commandBuffer,
        Uint32 firstSlot,
        const SDL_GPUBufferBinding *bindings,
        Uint32 numBindings);

    void (*BindIndexBuffer)(
        SDL_GPUCommandBuffer *commandBuffer,
        const SDL_GPUBufferBinding *binding,
        SDL_GPUIndexElementSize indexElementSize);

    void (*BindVertexSamplers)(
        SDL_GPUCommandBuffer *commandBuffer,
        Uint32 firstSlot,
        const SDL_GPUTextureSamplerBinding *textureSamplerBindings,
        Uint32 numBindings);

    void (*BindVertexStorageTextures)(
        SDL_GPUCommandBuffer *commandBuffer,
        Uint32 firstSlot,
        SDL_GPUTexture *const *storageTextures,
        Uint32 numBindings);

    void (*BindVertexStorageBuffers)(
        SDL_GPUCommandBuffer *commandBuffer,
        Uint32 firstSlot,
        SDL_GPUBuffer *const *storageBuffers,
        Uint32 numBindings);

    void (*BindFragmentSamplers)(
        SDL_GPUCommandBuffer *commandBuffer,
        Uint32 firstSlot,
        const SDL_GPUTextureSamplerBinding *textureSamplerBindings,
        Uint32 numBindings);

    void (*BindFragmentStorageTextures)(
        SDL_GPUCommandBuffer *commandBuffer,
        Uint32 firstSlot,
        SDL_GPUTexture *const *storageTextures,
        Uint32 numBindings);

    void (*BindFragmentStorageBuffers)(
        SDL_GPUCommandBuffer *commandBuffer,
        Uint32 firstSlot,
        SDL_GPUBuffer *const *storageBuffers,
        Uint32 numBindings);

    void (*PushVertexUniformData)(
        SDL_GPUCommandBuffer *commandBuffer,
        Uint32 slotIndex,
        const void *data,
        Uint32 length);

    void (*PushFragmentUniformData)(
        SDL_GPUCommandBuffer *commandBuffer,
        Uint32 slotIndex,
        const void *data,
        Uint32 length);

    void (*DrawIndexedPrimitives)(
        SDL_GPUCommandBuffer *commandBuffer,
        Uint32 numIndices,
        Uint32 numInstances,
        Uint32 firstIndex,
        Sint32 vertexOffset,
        Uint32 firstInstance);

    void (*DrawPrimitives)(
        SDL_GPUCommandBuffer *commandBuffer,
        Uint32 numVertices,
        Uint32 numInstances,
        Uint32 firstVertex,
        Uint32 firstInstance);

    void (*DrawPrimitivesIndirect)(
        SDL_GPUCommandBuffer *commandBuffer,
        SDL_GPUBuffer *buffer,
        Uint32 offset,
        Uint32 drawCount);

    void (*DrawIndexedPrimitivesIndirect)(
        SDL_GPUCommandBuffer *commandBuffer,
        SDL_GPUBuffer *buffer,
        Uint32 offset,
        Uint32 drawCount);

    void (*EndRenderPass)(
        SDL_GPUCommandBuffer *commandBuffer);

    // Compute Pass

    void (*BeginComputePass)(
        SDL_GPUCommandBuffer *commandBuffer,
        const SDL_GPUStorageTextureReadWriteBinding *storageTextureBindings,
        Uint32 numStorageTextureBindings,
        const SDL_GPUStorageBufferReadWriteBinding *storageBufferBindings,
        Uint32 numStorageBufferBindings);

    void (*BindComputePipeline)(
        SDL_GPUCommandBuffer *commandBuffer,
        SDL_GPUComputePipeline *computePipeline);

    void (*BindComputeSamplers)(
        SDL_GPUCommandBuffer *commandBuffer,
        Uint32 firstSlot,
        const SDL_GPUTextureSamplerBinding *textureSamplerBindings,
        Uint32 numBindings);

    void (*BindComputeStorageTextures)(
        SDL_GPUCommandBuffer *commandBuffer,
        Uint32 firstSlot,
        SDL_GPUTexture *const *storageTextures,
        Uint32 numBindings);

    void (*BindComputeStorageBuffers)(
        SDL_GPUCommandBuffer *commandBuffer,
        Uint32 firstSlot,
        SDL_GPUBuffer *const *storageBuffers,
        Uint32 numBindings);

    void (*PushComputeUniformData)(
        SDL_GPUCommandBuffer *commandBuffer,
        Uint32 slotIndex,
        const void *data,
        Uint32 length);

    void (*DispatchCompute)(
        SDL_GPUCommandBuffer *commandBuffer,
        Uint32 groupcountX,
        Uint32 groupcountY,
        Uint32 groupcountZ);

    void (*DispatchComputeIndirect)(
        SDL_GPUCommandBuffer *commandBuffer,
        SDL_GPUBuffer *buffer,
        Uint32 offset);

    void (*EndComputePass)(
        SDL_GPUCommandBuffer *commandBuffer);

    // TransferBuffer Data

    void *(*MapTransferBuffer)(
        SDL_GPURenderer *device,
        SDL_GPUTransferBuffer *transferBuffer,
        bool cycle);

    void (*UnmapTransferBuffer)(
        SDL_GPURenderer *device,
        SDL_GPUTransferBuffer *transferBuffer);

    // Copy Pass

    void (*BeginCopyPass)(
        SDL_GPUCommandBuffer *commandBuffer);

    void (*UploadToTexture)(
        SDL_GPUCommandBuffer *commandBuffer,
        const SDL_GPUTextureTransferInfo *source,
        const SDL_GPUTextureRegion *destination,
        bool cycle);

    void (*UploadToBuffer)(
        SDL_GPUCommandBuffer *commandBuffer,
        const SDL_GPUTransferBufferLocation *source,
        const SDL_GPUBufferRegion *destination,
        bool cycle);

    void (*CopyTextureToTexture)(
        SDL_GPUCommandBuffer *commandBuffer,
        const SDL_GPUTextureLocation *source,
        const SDL_GPUTextureLocation *destination,
        Uint32 w,
        Uint32 h,
        Uint32 d,
        bool cycle);

    void (*CopyBufferToBuffer)(
        SDL_GPUCommandBuffer *commandBuffer,
        const SDL_GPUBufferLocation *source,
        const SDL_GPUBufferLocation *destination,
        Uint32 size,
        bool cycle);

    void (*GenerateMipmaps)(
        SDL_GPUCommandBuffer *commandBuffer,
        SDL_GPUTexture *texture);

    void (*DownloadFromTexture)(
        SDL_GPUCommandBuffer *commandBuffer,
        const SDL_GPUTextureRegion *source,
        const SDL_GPUTextureTransferInfo *destination);

    void (*DownloadFromBuffer)(
        SDL_GPUCommandBuffer *commandBuffer,
        const SDL_GPUBufferRegion *source,
        const SDL_GPUTransferBufferLocation *destination);

    void (*EndCopyPass)(
        SDL_GPUCommandBuffer *commandBuffer);

    void (*Blit)(
        SDL_GPUCommandBuffer *commandBuffer,
        const SDL_GPUBlitInfo *info);

    // Submission/Presentation

    bool (*SupportsSwapchainComposition)(
        SDL_GPURenderer *driverData,
        SDL_Window *window,
        SDL_GPUSwapchainComposition swapchainComposition);

    bool (*SupportsPresentMode)(
        SDL_GPURenderer *driverData,
        SDL_Window *window,
        SDL_GPUPresentMode presentMode);

    bool (*ClaimWindow)(
        SDL_GPURenderer *driverData,
        SDL_Window *window);

    void (*ReleaseWindow)(
        SDL_GPURenderer *driverData,
        SDL_Window *window);

    bool (*SetSwapchainParameters)(
        SDL_GPURenderer *driverData,
        SDL_Window *window,
        SDL_GPUSwapchainComposition swapchainComposition,
        SDL_GPUPresentMode presentMode);

    bool (*SetAllowedFramesInFlight)(
        SDL_GPURenderer *driverData,
        Uint32 allowedFramesInFlight);

    SDL_GPUTextureFormat (*GetSwapchainTextureFormat)(
        SDL_GPURenderer *driverData,
        SDL_Window *window);

    SDL_GPUCommandBuffer *(*AcquireCommandBuffer)(
        SDL_GPURenderer *driverData);

    bool (*AcquireSwapchainTexture)(
        SDL_GPUCommandBuffer *commandBuffer,
        SDL_Window *window,
        SDL_GPUTexture **swapchainTexture,
        Uint32 *swapchainTextureWidth,
        Uint32 *swapchainTextureHeight);

    bool (*WaitForSwapchain)(
        SDL_GPURenderer *driverData,
        SDL_Window *window);

    bool (*WaitAndAcquireSwapchainTexture)(
        SDL_GPUCommandBuffer *commandBuffer,
        SDL_Window *window,
        SDL_GPUTexture **swapchainTexture,
        Uint32 *swapchainTextureWidth,
        Uint32 *swapchainTextureHeight);

    bool (*Submit)(
        SDL_GPUCommandBuffer *commandBuffer);

    SDL_GPUFence *(*SubmitAndAcquireFence)(
        SDL_GPUCommandBuffer *commandBuffer);

    bool (*Cancel)(
        SDL_GPUCommandBuffer *commandBuffer);

    bool (*Wait)(
        SDL_GPURenderer *driverData);

    bool (*WaitForFences)(
        SDL_GPURenderer *driverData,
        bool waitAll,
        SDL_GPUFence *const *fences,
        Uint32 numFences);

    bool (*QueryFence)(
        SDL_GPURenderer *driverData,
        SDL_GPUFence *fence);

    void (*ReleaseFence)(
        SDL_GPURenderer *driverData,
        SDL_GPUFence *fence);

    // Feature Queries

    bool (*SupportsTextureFormat)(
        SDL_GPURenderer *driverData,
        SDL_GPUTextureFormat format,
        SDL_GPUTextureType type,
        SDL_GPUTextureUsageFlags usage);

    bool (*SupportsSampleCount)(
        SDL_GPURenderer *driverData,
        SDL_GPUTextureFormat format,
        SDL_GPUSampleCount desiredSampleCount);

    // Opaque pointer for the Driver
    SDL_GPURenderer *driverData;

    // Store this for SDL_GetGPUDeviceDriver()
    const char *backend;

    // Store this for SDL_GetGPUShaderFormats()
    SDL_GPUShaderFormat shader_formats;

    // Store this for SDL_gpu.c's debug layer
    bool debug_mode;
};

#define ASSIGN_DRIVER_FUNC(func, name) \
    result->func = name##_##func;
#define ASSIGN_DRIVER(name)                                 \
    ASSIGN_DRIVER_FUNC(DestroyDevice, name)                 \
    ASSIGN_DRIVER_FUNC(GetDeviceProperties, name)      \
    ASSIGN_DRIVER_FUNC(CreateComputePipeline, name)         \
    ASSIGN_DRIVER_FUNC(CreateGraphicsPipeline, name)        \
    ASSIGN_DRIVER_FUNC(CreateSampler, name)                 \
    ASSIGN_DRIVER_FUNC(CreateShader, name)                  \
    ASSIGN_DRIVER_FUNC(CreateTexture, name)                 \
    ASSIGN_DRIVER_FUNC(CreateBuffer, name)                  \
    ASSIGN_DRIVER_FUNC(CreateTransferBuffer, name)          \
    ASSIGN_DRIVER_FUNC(SetBufferName, name)                 \
    ASSIGN_DRIVER_FUNC(SetTextureName, name)                \
    ASSIGN_DRIVER_FUNC(InsertDebugLabel, name)              \
    ASSIGN_DRIVER_FUNC(PushDebugGroup, name)                \
    ASSIGN_DRIVER_FUNC(PopDebugGroup, name)                 \
    ASSIGN_DRIVER_FUNC(ReleaseTexture, name)                \
    ASSIGN_DRIVER_FUNC(ReleaseSampler, name)                \
    ASSIGN_DRIVER_FUNC(ReleaseBuffer, name)                 \
    ASSIGN_DRIVER_FUNC(ReleaseTransferBuffer, name)         \
    ASSIGN_DRIVER_FUNC(ReleaseShader, name)                 \
    ASSIGN_DRIVER_FUNC(ReleaseComputePipeline, name)        \
    ASSIGN_DRIVER_FUNC(ReleaseGraphicsPipeline, name)       \
    ASSIGN_DRIVER_FUNC(BeginRenderPass, name)               \
    ASSIGN_DRIVER_FUNC(BindGraphicsPipeline, name)          \
    ASSIGN_DRIVER_FUNC(SetViewport, name)                   \
    ASSIGN_DRIVER_FUNC(SetScissor, name)                    \
    ASSIGN_DRIVER_FUNC(SetBlendConstants, name)             \
    ASSIGN_DRIVER_FUNC(SetStencilReference, name)           \
    ASSIGN_DRIVER_FUNC(BindVertexBuffers, name)             \
    ASSIGN_DRIVER_FUNC(BindIndexBuffer, name)               \
    ASSIGN_DRIVER_FUNC(BindVertexSamplers, name)            \
    ASSIGN_DRIVER_FUNC(BindVertexStorageTextures, name)     \
    ASSIGN_DRIVER_FUNC(BindVertexStorageBuffers, name)      \
    ASSIGN_DRIVER_FUNC(BindFragmentSamplers, name)          \
    ASSIGN_DRIVER_FUNC(BindFragmentStorageTextures, name)   \
    ASSIGN_DRIVER_FUNC(BindFragmentStorageBuffers, name)    \
    ASSIGN_DRIVER_FUNC(PushVertexUniformData, name)         \
    ASSIGN_DRIVER_FUNC(PushFragmentUniformData, name)       \
    ASSIGN_DRIVER_FUNC(DrawIndexedPrimitives, name)         \
    ASSIGN_DRIVER_FUNC(DrawPrimitives, name)                \
    ASSIGN_DRIVER_FUNC(DrawPrimitivesIndirect, name)        \
    ASSIGN_DRIVER_FUNC(DrawIndexedPrimitivesIndirect, name) \
    ASSIGN_DRIVER_FUNC(EndRenderPass, name)                 \
    ASSIGN_DRIVER_FUNC(BeginComputePass, name)              \
    ASSIGN_DRIVER_FUNC(BindComputePipeline, name)           \
    ASSIGN_DRIVER_FUNC(BindComputeSamplers, name)           \
    ASSIGN_DRIVER_FUNC(BindComputeStorageTextures, name)    \
    ASSIGN_DRIVER_FUNC(BindComputeStorageBuffers, name)     \
    ASSIGN_DRIVER_FUNC(PushComputeUniformData, name)        \
    ASSIGN_DRIVER_FUNC(DispatchCompute, name)               \
    ASSIGN_DRIVER_FUNC(DispatchComputeIndirect, name)       \
    ASSIGN_DRIVER_FUNC(EndComputePass, name)                \
    ASSIGN_DRIVER_FUNC(MapTransferBuffer, name)             \
    ASSIGN_DRIVER_FUNC(UnmapTransferBuffer, name)           \
    ASSIGN_DRIVER_FUNC(BeginCopyPass, name)                 \
    ASSIGN_DRIVER_FUNC(UploadToTexture, name)               \
    ASSIGN_DRIVER_FUNC(UploadToBuffer, name)                \
    ASSIGN_DRIVER_FUNC(DownloadFromTexture, name)           \
    ASSIGN_DRIVER_FUNC(DownloadFromBuffer, name)            \
    ASSIGN_DRIVER_FUNC(CopyTextureToTexture, name)          \
    ASSIGN_DRIVER_FUNC(CopyBufferToBuffer, name)            \
    ASSIGN_DRIVER_FUNC(GenerateMipmaps, name)               \
    ASSIGN_DRIVER_FUNC(EndCopyPass, name)                   \
    ASSIGN_DRIVER_FUNC(Blit, name)                          \
    ASSIGN_DRIVER_FUNC(SupportsSwapchainComposition, name)  \
    ASSIGN_DRIVER_FUNC(SupportsPresentMode, name)           \
    ASSIGN_DRIVER_FUNC(ClaimWindow, name)                   \
    ASSIGN_DRIVER_FUNC(ReleaseWindow, name)                 \
    ASSIGN_DRIVER_FUNC(SetSwapchainParameters, name)        \
    ASSIGN_DRIVER_FUNC(SetAllowedFramesInFlight, name)      \
    ASSIGN_DRIVER_FUNC(GetSwapchainTextureFormat, name)     \
    ASSIGN_DRIVER_FUNC(AcquireCommandBuffer, name)          \
    ASSIGN_DRIVER_FUNC(AcquireSwapchainTexture, name)       \
    ASSIGN_DRIVER_FUNC(WaitForSwapchain, name)              \
    ASSIGN_DRIVER_FUNC(WaitAndAcquireSwapchainTexture, name)\
    ASSIGN_DRIVER_FUNC(Submit, name)                        \
    ASSIGN_DRIVER_FUNC(SubmitAndAcquireFence, name)         \
    ASSIGN_DRIVER_FUNC(Cancel, name)                        \
    ASSIGN_DRIVER_FUNC(Wait, name)                          \
    ASSIGN_DRIVER_FUNC(WaitForFences, name)                 \
    ASSIGN_DRIVER_FUNC(QueryFence, name)                    \
    ASSIGN_DRIVER_FUNC(ReleaseFence, name)                  \
    ASSIGN_DRIVER_FUNC(SupportsTextureFormat, name)         \
    ASSIGN_DRIVER_FUNC(SupportsSampleCount, name)

typedef struct SDL_GPUBootstrap
{
    const char *name;
    const SDL_GPUShaderFormat shader_formats;
    bool (*PrepareDriver)(SDL_VideoDevice *_this);
    SDL_GPUDevice *(*CreateDevice)(bool debug_mode, bool prefer_low_power, SDL_PropertiesID props);
} SDL_GPUBootstrap;

#ifdef __cplusplus
extern "C" {
#endif

extern SDL_GPUBootstrap VulkanDriver;
extern SDL_GPUBootstrap D3D12Driver;
extern SDL_GPUBootstrap MetalDriver;
extern SDL_GPUBootstrap PrivateGPUDriver;

#ifdef __cplusplus
}
#endif

#endif // SDL_GPU_DRIVER_H
