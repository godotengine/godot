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

#ifdef SDL_GPU_VULKAN

// Needed for VK_KHR_portability_subset
#define VK_ENABLE_BETA_EXTENSIONS

#define VK_NO_PROTOTYPES
#include "../../video/khronos/vulkan/vulkan.h"

#include <SDL3/SDL_vulkan.h>

#include "../SDL_sysgpu.h"

// Global Vulkan Loader Entry Points

static PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = NULL;

#define VULKAN_GLOBAL_FUNCTION(name) \
    static PFN_##name name = NULL;
#include "SDL_gpu_vulkan_vkfuncs.h"

typedef struct VulkanExtensions
{
    // These extensions are required!

    // Globally supported
    Uint8 KHR_swapchain;
    // Core since 1.1, needed for negative VkViewport::height
    Uint8 KHR_maintenance1;

    // These extensions are optional!

    // Core since 1.2, but requires annoying paperwork to implement
    Uint8 KHR_driver_properties;
    // Only required for special implementations (i.e. MoltenVK)
    Uint8 KHR_portability_subset;
    // Only required for decoding HDR ASTC textures
    Uint8 EXT_texture_compression_astc_hdr;
} VulkanExtensions;

// Defines

#define SMALL_ALLOCATION_THRESHOLD    2097152  // 2   MiB
#define SMALL_ALLOCATION_SIZE         16777216 // 16  MiB
#define LARGE_ALLOCATION_INCREMENT    67108864 // 64  MiB
#define MAX_UBO_SECTION_SIZE          4096     // 4   KiB
#define DESCRIPTOR_POOL_SIZE          128
#define WINDOW_PROPERTY_DATA          "SDL_GPUVulkanWindowPropertyData"

#define IDENTITY_SWIZZLE               \
    {                                  \
        VK_COMPONENT_SWIZZLE_IDENTITY, \
        VK_COMPONENT_SWIZZLE_IDENTITY, \
        VK_COMPONENT_SWIZZLE_IDENTITY, \
        VK_COMPONENT_SWIZZLE_IDENTITY  \
    }

// Conversions

static const Uint8 DEVICE_PRIORITY_HIGHPERFORMANCE[] = {
    0, // VK_PHYSICAL_DEVICE_TYPE_OTHER
    3, // VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
    4, // VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
    2, // VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU
    1  // VK_PHYSICAL_DEVICE_TYPE_CPU
};

static const Uint8 DEVICE_PRIORITY_LOWPOWER[] = {
    0, // VK_PHYSICAL_DEVICE_TYPE_OTHER
    4, // VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
    3, // VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
    2, // VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU
    1  // VK_PHYSICAL_DEVICE_TYPE_CPU
};

static VkPresentModeKHR SDLToVK_PresentMode[] = {
    VK_PRESENT_MODE_FIFO_KHR,
    VK_PRESENT_MODE_IMMEDIATE_KHR,
    VK_PRESENT_MODE_MAILBOX_KHR
};

static VkFormat SDLToVK_TextureFormat[] = {
    VK_FORMAT_UNDEFINED,                   // INVALID
    VK_FORMAT_R8_UNORM,                    // A8_UNORM
    VK_FORMAT_R8_UNORM,                    // R8_UNORM
    VK_FORMAT_R8G8_UNORM,                  // R8G8_UNORM
    VK_FORMAT_R8G8B8A8_UNORM,              // R8G8B8A8_UNORM
    VK_FORMAT_R16_UNORM,                   // R16_UNORM
    VK_FORMAT_R16G16_UNORM,                // R16G16_UNORM
    VK_FORMAT_R16G16B16A16_UNORM,          // R16G16B16A16_UNORM
    VK_FORMAT_A2B10G10R10_UNORM_PACK32,    // R10G10B10A2_UNORM
    VK_FORMAT_R5G6B5_UNORM_PACK16,         // B5G6R5_UNORM
    VK_FORMAT_A1R5G5B5_UNORM_PACK16,       // B5G5R5A1_UNORM
    VK_FORMAT_B4G4R4A4_UNORM_PACK16,       // B4G4R4A4_UNORM
    VK_FORMAT_B8G8R8A8_UNORM,              // B8G8R8A8_UNORM
    VK_FORMAT_BC1_RGBA_UNORM_BLOCK,        // BC1_UNORM
    VK_FORMAT_BC2_UNORM_BLOCK,             // BC2_UNORM
    VK_FORMAT_BC3_UNORM_BLOCK,             // BC3_UNORM
    VK_FORMAT_BC4_UNORM_BLOCK,             // BC4_UNORM
    VK_FORMAT_BC5_UNORM_BLOCK,             // BC5_UNORM
    VK_FORMAT_BC7_UNORM_BLOCK,             // BC7_UNORM
    VK_FORMAT_BC6H_SFLOAT_BLOCK,           // BC6H_FLOAT
    VK_FORMAT_BC6H_UFLOAT_BLOCK,           // BC6H_UFLOAT
    VK_FORMAT_R8_SNORM,                    // R8_SNORM
    VK_FORMAT_R8G8_SNORM,                  // R8G8_SNORM
    VK_FORMAT_R8G8B8A8_SNORM,              // R8G8B8A8_SNORM
    VK_FORMAT_R16_SNORM,                   // R16_SNORM
    VK_FORMAT_R16G16_SNORM,                // R16G16_SNORM
    VK_FORMAT_R16G16B16A16_SNORM,          // R16G16B16A16_SNORM
    VK_FORMAT_R16_SFLOAT,                  // R16_FLOAT
    VK_FORMAT_R16G16_SFLOAT,               // R16G16_FLOAT
    VK_FORMAT_R16G16B16A16_SFLOAT,         // R16G16B16A16_FLOAT
    VK_FORMAT_R32_SFLOAT,                  // R32_FLOAT
    VK_FORMAT_R32G32_SFLOAT,               // R32G32_FLOAT
    VK_FORMAT_R32G32B32A32_SFLOAT,         // R32G32B32A32_FLOAT
    VK_FORMAT_B10G11R11_UFLOAT_PACK32,     // R11G11B10_UFLOAT
    VK_FORMAT_R8_UINT,                     // R8_UINT
    VK_FORMAT_R8G8_UINT,                   // R8G8_UINT
    VK_FORMAT_R8G8B8A8_UINT,               // R8G8B8A8_UINT
    VK_FORMAT_R16_UINT,                    // R16_UINT
    VK_FORMAT_R16G16_UINT,                 // R16G16_UINT
    VK_FORMAT_R16G16B16A16_UINT,           // R16G16B16A16_UINT
    VK_FORMAT_R32_UINT,                    // R32_UINT
    VK_FORMAT_R32G32_UINT,                 // R32G32_UINT
    VK_FORMAT_R32G32B32A32_UINT,           // R32G32B32A32_UINT
    VK_FORMAT_R8_SINT,                     // R8_INT
    VK_FORMAT_R8G8_SINT,                   // R8G8_INT
    VK_FORMAT_R8G8B8A8_SINT,               // R8G8B8A8_INT
    VK_FORMAT_R16_SINT,                    // R16_INT
    VK_FORMAT_R16G16_SINT,                 // R16G16_INT
    VK_FORMAT_R16G16B16A16_SINT,           // R16G16B16A16_INT
    VK_FORMAT_R32_SINT,                    // R32_INT
    VK_FORMAT_R32G32_SINT,                 // R32G32_INT
    VK_FORMAT_R32G32B32A32_SINT,           // R32G32B32A32_INT
    VK_FORMAT_R8G8B8A8_SRGB,               // R8G8B8A8_UNORM_SRGB
    VK_FORMAT_B8G8R8A8_SRGB,               // B8G8R8A8_UNORM_SRGB
    VK_FORMAT_BC1_RGBA_SRGB_BLOCK,         // BC1_UNORM_SRGB
    VK_FORMAT_BC2_SRGB_BLOCK,              // BC3_UNORM_SRGB
    VK_FORMAT_BC3_SRGB_BLOCK,              // BC3_UNORM_SRGB
    VK_FORMAT_BC7_SRGB_BLOCK,              // BC7_UNORM_SRGB
    VK_FORMAT_D16_UNORM,                   // D16_UNORM
    VK_FORMAT_X8_D24_UNORM_PACK32,         // D24_UNORM
    VK_FORMAT_D32_SFLOAT,                  // D32_FLOAT
    VK_FORMAT_D24_UNORM_S8_UINT,           // D24_UNORM_S8_UINT
    VK_FORMAT_D32_SFLOAT_S8_UINT,          // D32_FLOAT_S8_UINT
    VK_FORMAT_ASTC_4x4_UNORM_BLOCK,        // ASTC_4x4_UNORM
    VK_FORMAT_ASTC_5x4_UNORM_BLOCK,        // ASTC_5x4_UNORM
    VK_FORMAT_ASTC_5x5_UNORM_BLOCK,        // ASTC_5x5_UNORM
    VK_FORMAT_ASTC_6x5_UNORM_BLOCK,        // ASTC_6x5_UNORM
    VK_FORMAT_ASTC_6x6_UNORM_BLOCK,        // ASTC_6x6_UNORM
    VK_FORMAT_ASTC_8x5_UNORM_BLOCK,        // ASTC_8x5_UNORM
    VK_FORMAT_ASTC_8x6_UNORM_BLOCK,        // ASTC_8x6_UNORM
    VK_FORMAT_ASTC_8x8_UNORM_BLOCK,        // ASTC_8x8_UNORM
    VK_FORMAT_ASTC_10x5_UNORM_BLOCK,       // ASTC_10x5_UNORM
    VK_FORMAT_ASTC_10x6_UNORM_BLOCK,       // ASTC_10x6_UNORM
    VK_FORMAT_ASTC_10x8_UNORM_BLOCK,       // ASTC_10x8_UNORM
    VK_FORMAT_ASTC_10x10_UNORM_BLOCK,      // ASTC_10x10_UNORM
    VK_FORMAT_ASTC_12x10_UNORM_BLOCK,      // ASTC_12x10_UNORM
    VK_FORMAT_ASTC_12x12_UNORM_BLOCK,      // ASTC_12x12_UNORM
    VK_FORMAT_ASTC_4x4_SRGB_BLOCK,         // ASTC_4x4_UNORM_SRGB
    VK_FORMAT_ASTC_5x4_SRGB_BLOCK,         // ASTC_5x4_UNORM_SRGB
    VK_FORMAT_ASTC_5x5_SRGB_BLOCK,         // ASTC_5x5_UNORM_SRGB
    VK_FORMAT_ASTC_6x5_SRGB_BLOCK,         // ASTC_6x5_UNORM_SRGB
    VK_FORMAT_ASTC_6x6_SRGB_BLOCK,         // ASTC_6x6_UNORM_SRGB
    VK_FORMAT_ASTC_8x5_SRGB_BLOCK,         // ASTC_8x5_UNORM_SRGB
    VK_FORMAT_ASTC_8x6_SRGB_BLOCK,         // ASTC_8x6_UNORM_SRGB
    VK_FORMAT_ASTC_8x8_SRGB_BLOCK,         // ASTC_8x8_UNORM_SRGB
    VK_FORMAT_ASTC_10x5_SRGB_BLOCK,        // ASTC_10x5_UNORM_SRGB
    VK_FORMAT_ASTC_10x6_SRGB_BLOCK,        // ASTC_10x6_UNORM_SRGB
    VK_FORMAT_ASTC_10x8_SRGB_BLOCK,        // ASTC_10x8_UNORM_SRGB
    VK_FORMAT_ASTC_10x10_SRGB_BLOCK,       // ASTC_10x10_UNORM_SRGB
    VK_FORMAT_ASTC_12x10_SRGB_BLOCK,       // ASTC_12x10_UNORM_SRGB
    VK_FORMAT_ASTC_12x12_SRGB_BLOCK,       // ASTC_12x12_UNORM_SRGB
    VK_FORMAT_ASTC_4x4_SFLOAT_BLOCK_EXT,   // ASTC_4x4_FLOAT
    VK_FORMAT_ASTC_5x4_SFLOAT_BLOCK_EXT,   // ASTC_5x4_FLOAT
    VK_FORMAT_ASTC_5x5_SFLOAT_BLOCK_EXT,   // ASTC_5x5_FLOAT
    VK_FORMAT_ASTC_6x5_SFLOAT_BLOCK_EXT,   // ASTC_6x5_FLOAT
    VK_FORMAT_ASTC_6x6_SFLOAT_BLOCK_EXT,   // ASTC_6x6_FLOAT
    VK_FORMAT_ASTC_8x5_SFLOAT_BLOCK_EXT,   // ASTC_8x5_FLOAT
    VK_FORMAT_ASTC_8x6_SFLOAT_BLOCK_EXT,   // ASTC_8x6_FLOAT
    VK_FORMAT_ASTC_8x8_SFLOAT_BLOCK_EXT,   // ASTC_8x8_FLOAT
    VK_FORMAT_ASTC_10x5_SFLOAT_BLOCK_EXT,  // ASTC_10x5_FLOAT
    VK_FORMAT_ASTC_10x6_SFLOAT_BLOCK_EXT,  // ASTC_10x6_FLOAT
    VK_FORMAT_ASTC_10x8_SFLOAT_BLOCK_EXT,  // ASTC_10x8_FLOAT
    VK_FORMAT_ASTC_10x10_SFLOAT_BLOCK_EXT, // ASTC_10x10_FLOAT
    VK_FORMAT_ASTC_12x10_SFLOAT_BLOCK_EXT, // ASTC_12x10_FLOAT
    VK_FORMAT_ASTC_12x12_SFLOAT_BLOCK      // ASTC_12x12_FLOAT
};
SDL_COMPILE_TIME_ASSERT(SDLToVK_TextureFormat, SDL_arraysize(SDLToVK_TextureFormat) == SDL_GPU_TEXTUREFORMAT_MAX_ENUM_VALUE);

static VkComponentMapping SwizzleForSDLFormat(SDL_GPUTextureFormat format)
{
    if (format == SDL_GPU_TEXTUREFORMAT_A8_UNORM) {
        // TODO: use VK_FORMAT_A8_UNORM_KHR from VK_KHR_maintenance5 when available
        return (VkComponentMapping){
            VK_COMPONENT_SWIZZLE_ZERO,
            VK_COMPONENT_SWIZZLE_ZERO,
            VK_COMPONENT_SWIZZLE_ZERO,
            VK_COMPONENT_SWIZZLE_R,
        };
    }

    if (format == SDL_GPU_TEXTUREFORMAT_B4G4R4A4_UNORM) {
        // ARGB -> BGRA
        // TODO: use VK_FORMAT_A4R4G4B4_UNORM_PACK16_EXT from VK_EXT_4444_formats when available
        return (VkComponentMapping){
            VK_COMPONENT_SWIZZLE_G,
            VK_COMPONENT_SWIZZLE_R,
            VK_COMPONENT_SWIZZLE_A,
            VK_COMPONENT_SWIZZLE_B,
        };
    }

    return (VkComponentMapping)IDENTITY_SWIZZLE;
}

static VkFormat SwapchainCompositionToFormat[] = {
    VK_FORMAT_B8G8R8A8_UNORM,          // SDR
    VK_FORMAT_B8G8R8A8_SRGB,           // SDR_LINEAR
    VK_FORMAT_R16G16B16A16_SFLOAT,     // HDR_EXTENDED_LINEAR
    VK_FORMAT_A2B10G10R10_UNORM_PACK32 // HDR10_ST2084
};

static VkFormat SwapchainCompositionToFallbackFormat[] = {
    VK_FORMAT_R8G8B8A8_UNORM, // SDR
    VK_FORMAT_R8G8B8A8_SRGB,  // SDR_LINEAR
    VK_FORMAT_UNDEFINED,      // HDR_EXTENDED_LINEAR (no fallback)
    VK_FORMAT_UNDEFINED       // HDR10_ST2084 (no fallback)
};

static SDL_GPUTextureFormat SwapchainCompositionToSDLFormat(
    SDL_GPUSwapchainComposition composition,
    bool usingFallback)
{
    switch (composition) {
    case SDL_GPU_SWAPCHAINCOMPOSITION_SDR:
        return usingFallback ? SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM : SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM;
    case SDL_GPU_SWAPCHAINCOMPOSITION_SDR_LINEAR:
        return usingFallback ? SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM_SRGB : SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM_SRGB;
    case SDL_GPU_SWAPCHAINCOMPOSITION_HDR_EXTENDED_LINEAR:
        return SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT;
    case SDL_GPU_SWAPCHAINCOMPOSITION_HDR10_ST2084:
        return SDL_GPU_TEXTUREFORMAT_R10G10B10A2_UNORM;
    default:
        return SDL_GPU_TEXTUREFORMAT_INVALID;
    }
}

static VkColorSpaceKHR SwapchainCompositionToColorSpace[] = {
    VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,       // SDR
    VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,       // SDR_LINEAR
    VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT, // HDR_EXTENDED_LINEAR
    VK_COLOR_SPACE_HDR10_ST2084_EXT          // HDR10_ST2084
};

static VkComponentMapping SwapchainCompositionSwizzle[] = {
    IDENTITY_SWIZZLE, // SDR
    IDENTITY_SWIZZLE, // SDR_LINEAR
    IDENTITY_SWIZZLE, // HDR_EXTENDED_LINEAR
    {
        // HDR10_ST2084
        VK_COMPONENT_SWIZZLE_R,
        VK_COMPONENT_SWIZZLE_G,
        VK_COMPONENT_SWIZZLE_B,
        VK_COMPONENT_SWIZZLE_A,
    }
};

static VkFormat SDLToVK_VertexFormat[] = {
    VK_FORMAT_UNDEFINED,           // INVALID
    VK_FORMAT_R32_SINT,            // INT
    VK_FORMAT_R32G32_SINT,         // INT2
    VK_FORMAT_R32G32B32_SINT,      // INT3
    VK_FORMAT_R32G32B32A32_SINT,   // INT4
    VK_FORMAT_R32_UINT,            // UINT
    VK_FORMAT_R32G32_UINT,         // UINT2
    VK_FORMAT_R32G32B32_UINT,      // UINT3
    VK_FORMAT_R32G32B32A32_UINT,   // UINT4
    VK_FORMAT_R32_SFLOAT,          // FLOAT
    VK_FORMAT_R32G32_SFLOAT,       // FLOAT2
    VK_FORMAT_R32G32B32_SFLOAT,    // FLOAT3
    VK_FORMAT_R32G32B32A32_SFLOAT, // FLOAT4
    VK_FORMAT_R8G8_SINT,           // BYTE2
    VK_FORMAT_R8G8B8A8_SINT,       // BYTE4
    VK_FORMAT_R8G8_UINT,           // UBYTE2
    VK_FORMAT_R8G8B8A8_UINT,       // UBYTE4
    VK_FORMAT_R8G8_SNORM,          // BYTE2_NORM
    VK_FORMAT_R8G8B8A8_SNORM,      // BYTE4_NORM
    VK_FORMAT_R8G8_UNORM,          // UBYTE2_NORM
    VK_FORMAT_R8G8B8A8_UNORM,      // UBYTE4_NORM
    VK_FORMAT_R16G16_SINT,         // SHORT2
    VK_FORMAT_R16G16B16A16_SINT,   // SHORT4
    VK_FORMAT_R16G16_UINT,         // USHORT2
    VK_FORMAT_R16G16B16A16_UINT,   // USHORT4
    VK_FORMAT_R16G16_SNORM,        // SHORT2_NORM
    VK_FORMAT_R16G16B16A16_SNORM,  // SHORT4_NORM
    VK_FORMAT_R16G16_UNORM,        // USHORT2_NORM
    VK_FORMAT_R16G16B16A16_UNORM,  // USHORT4_NORM
    VK_FORMAT_R16G16_SFLOAT,       // HALF2
    VK_FORMAT_R16G16B16A16_SFLOAT  // HALF4
};
SDL_COMPILE_TIME_ASSERT(SDLToVK_VertexFormat, SDL_arraysize(SDLToVK_VertexFormat) == SDL_GPU_VERTEXELEMENTFORMAT_MAX_ENUM_VALUE);

static VkIndexType SDLToVK_IndexType[] = {
    VK_INDEX_TYPE_UINT16,
    VK_INDEX_TYPE_UINT32
};

static VkPrimitiveTopology SDLToVK_PrimitiveType[] = {
    VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
    VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
    VK_PRIMITIVE_TOPOLOGY_LINE_STRIP,
    VK_PRIMITIVE_TOPOLOGY_POINT_LIST
};

static VkCullModeFlags SDLToVK_CullMode[] = {
    VK_CULL_MODE_NONE,
    VK_CULL_MODE_FRONT_BIT,
    VK_CULL_MODE_BACK_BIT,
    VK_CULL_MODE_FRONT_AND_BACK
};

static VkFrontFace SDLToVK_FrontFace[] = {
    VK_FRONT_FACE_COUNTER_CLOCKWISE,
    VK_FRONT_FACE_CLOCKWISE
};

static VkBlendFactor SDLToVK_BlendFactor[] = {
    VK_BLEND_FACTOR_ZERO, // INVALID
    VK_BLEND_FACTOR_ZERO,
    VK_BLEND_FACTOR_ONE,
    VK_BLEND_FACTOR_SRC_COLOR,
    VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR,
    VK_BLEND_FACTOR_DST_COLOR,
    VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR,
    VK_BLEND_FACTOR_SRC_ALPHA,
    VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
    VK_BLEND_FACTOR_DST_ALPHA,
    VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA,
    VK_BLEND_FACTOR_CONSTANT_COLOR,
    VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR,
    VK_BLEND_FACTOR_SRC_ALPHA_SATURATE
};
SDL_COMPILE_TIME_ASSERT(SDLToVK_BlendFactor, SDL_arraysize(SDLToVK_BlendFactor) == SDL_GPU_BLENDFACTOR_MAX_ENUM_VALUE);

static VkBlendOp SDLToVK_BlendOp[] = {
    VK_BLEND_OP_ADD, // INVALID
    VK_BLEND_OP_ADD,
    VK_BLEND_OP_SUBTRACT,
    VK_BLEND_OP_REVERSE_SUBTRACT,
    VK_BLEND_OP_MIN,
    VK_BLEND_OP_MAX
};
SDL_COMPILE_TIME_ASSERT(SDLToVK_BlendOp, SDL_arraysize(SDLToVK_BlendOp) == SDL_GPU_BLENDOP_MAX_ENUM_VALUE);

static VkCompareOp SDLToVK_CompareOp[] = {
    VK_COMPARE_OP_NEVER, // INVALID
    VK_COMPARE_OP_NEVER,
    VK_COMPARE_OP_LESS,
    VK_COMPARE_OP_EQUAL,
    VK_COMPARE_OP_LESS_OR_EQUAL,
    VK_COMPARE_OP_GREATER,
    VK_COMPARE_OP_NOT_EQUAL,
    VK_COMPARE_OP_GREATER_OR_EQUAL,
    VK_COMPARE_OP_ALWAYS
};
SDL_COMPILE_TIME_ASSERT(SDLToVK_CompareOp, SDL_arraysize(SDLToVK_CompareOp) == SDL_GPU_COMPAREOP_MAX_ENUM_VALUE);

static VkStencilOp SDLToVK_StencilOp[] = {
    VK_STENCIL_OP_KEEP, // INVALID
    VK_STENCIL_OP_KEEP,
    VK_STENCIL_OP_ZERO,
    VK_STENCIL_OP_REPLACE,
    VK_STENCIL_OP_INCREMENT_AND_CLAMP,
    VK_STENCIL_OP_DECREMENT_AND_CLAMP,
    VK_STENCIL_OP_INVERT,
    VK_STENCIL_OP_INCREMENT_AND_WRAP,
    VK_STENCIL_OP_DECREMENT_AND_WRAP
};
SDL_COMPILE_TIME_ASSERT(SDLToVK_StencilOp, SDL_arraysize(SDLToVK_StencilOp) == SDL_GPU_STENCILOP_MAX_ENUM_VALUE);

static VkAttachmentLoadOp SDLToVK_LoadOp[] = {
    VK_ATTACHMENT_LOAD_OP_LOAD,
    VK_ATTACHMENT_LOAD_OP_CLEAR,
    VK_ATTACHMENT_LOAD_OP_DONT_CARE
};

static VkAttachmentStoreOp SDLToVK_StoreOp[] = {
    VK_ATTACHMENT_STORE_OP_STORE,
    VK_ATTACHMENT_STORE_OP_DONT_CARE,
    VK_ATTACHMENT_STORE_OP_DONT_CARE,
    VK_ATTACHMENT_STORE_OP_STORE
};

static VkSampleCountFlagBits SDLToVK_SampleCount[] = {
    VK_SAMPLE_COUNT_1_BIT,
    VK_SAMPLE_COUNT_2_BIT,
    VK_SAMPLE_COUNT_4_BIT,
    VK_SAMPLE_COUNT_8_BIT
};

static VkVertexInputRate SDLToVK_VertexInputRate[] = {
    VK_VERTEX_INPUT_RATE_VERTEX,
    VK_VERTEX_INPUT_RATE_INSTANCE
};

static VkFilter SDLToVK_Filter[] = {
    VK_FILTER_NEAREST,
    VK_FILTER_LINEAR
};

static VkSamplerMipmapMode SDLToVK_SamplerMipmapMode[] = {
    VK_SAMPLER_MIPMAP_MODE_NEAREST,
    VK_SAMPLER_MIPMAP_MODE_LINEAR
};

static VkSamplerAddressMode SDLToVK_SamplerAddressMode[] = {
    VK_SAMPLER_ADDRESS_MODE_REPEAT,
    VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
    VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE
};

// Structures

typedef struct VulkanMemoryAllocation VulkanMemoryAllocation;
typedef struct VulkanBuffer VulkanBuffer;
typedef struct VulkanBufferContainer VulkanBufferContainer;
typedef struct VulkanUniformBuffer VulkanUniformBuffer;
typedef struct VulkanTexture VulkanTexture;
typedef struct VulkanTextureContainer VulkanTextureContainer;

typedef struct VulkanFenceHandle
{
    VkFence fence;
    SDL_AtomicInt referenceCount;
} VulkanFenceHandle;

// Memory Allocation

typedef struct VulkanMemoryFreeRegion
{
    VulkanMemoryAllocation *allocation;
    VkDeviceSize offset;
    VkDeviceSize size;
    Uint32 allocationIndex;
    Uint32 sortedIndex;
} VulkanMemoryFreeRegion;

typedef struct VulkanMemoryUsedRegion
{
    VulkanMemoryAllocation *allocation;
    VkDeviceSize offset;
    VkDeviceSize size;
    VkDeviceSize resourceOffset; // differs from offset based on alignment
    VkDeviceSize resourceSize;   // differs from size based on alignment
    VkDeviceSize alignment;
    Uint8 isBuffer;
    union
    {
        VulkanBuffer *vulkanBuffer;
        VulkanTexture *vulkanTexture;
    };
} VulkanMemoryUsedRegion;

typedef struct VulkanMemorySubAllocator
{
    Uint32 memoryTypeIndex;
    VulkanMemoryAllocation **allocations;
    Uint32 allocationCount;
    VulkanMemoryFreeRegion **sortedFreeRegions;
    Uint32 sortedFreeRegionCount;
    Uint32 sortedFreeRegionCapacity;
} VulkanMemorySubAllocator;

struct VulkanMemoryAllocation
{
    VulkanMemorySubAllocator *allocator;
    VkDeviceMemory memory;
    VkDeviceSize size;
    VulkanMemoryUsedRegion **usedRegions;
    Uint32 usedRegionCount;
    Uint32 usedRegionCapacity;
    VulkanMemoryFreeRegion **freeRegions;
    Uint32 freeRegionCount;
    Uint32 freeRegionCapacity;
    Uint8 availableForAllocation;
    VkDeviceSize freeSpace;
    VkDeviceSize usedSpace;
    Uint8 *mapPointer;
    SDL_Mutex *memoryLock;
};

typedef struct VulkanMemoryAllocator
{
    VulkanMemorySubAllocator subAllocators[VK_MAX_MEMORY_TYPES];
} VulkanMemoryAllocator;

// Memory structures

typedef enum VulkanBufferType
{
    VULKAN_BUFFER_TYPE_GPU,
    VULKAN_BUFFER_TYPE_UNIFORM,
    VULKAN_BUFFER_TYPE_TRANSFER
} VulkanBufferType;

struct VulkanBuffer
{
    VulkanBufferContainer *container;
    Uint32 containerIndex;

    VkBuffer buffer;
    VulkanMemoryUsedRegion *usedRegion;

    // Needed for uniforms and defrag
    VulkanBufferType type;
    SDL_GPUBufferUsageFlags usage;
    VkDeviceSize size;

    SDL_AtomicInt referenceCount;
    bool transitioned;
    bool markedForDestroy; // so that defrag doesn't double-free
    VulkanUniformBuffer *uniformBufferForDefrag;
};

struct VulkanBufferContainer
{
    VulkanBuffer *activeBuffer;

    VulkanBuffer **buffers;
    Uint32 bufferCapacity;
    Uint32 bufferCount;

    bool dedicated;
    char *debugName;
};

// Renderer Structure

typedef struct QueueFamilyIndices
{
    Uint32 graphicsFamily;
    Uint32 presentFamily;
    Uint32 computeFamily;
    Uint32 transferFamily;
} QueueFamilyIndices;

typedef struct VulkanSampler
{
    VkSampler sampler;
    SDL_AtomicInt referenceCount;
} VulkanSampler;

typedef struct VulkanShader
{
    VkShaderModule shaderModule;
    char *entrypointName;
    SDL_GPUShaderStage stage;
    Uint32 numSamplers;
    Uint32 numStorageTextures;
    Uint32 numStorageBuffers;
    Uint32 numUniformBuffers;
    SDL_AtomicInt referenceCount;
} VulkanShader;

/* Textures are made up of individual subresources.
 * This helps us barrier the resource efficiently.
 */
typedef struct VulkanTextureSubresource
{
    VulkanTexture *parent;
    Uint32 layer;
    Uint32 level;

    VkImageView *renderTargetViews; // One render target view per depth slice
    VkImageView computeWriteView;
    VkImageView depthStencilView;
} VulkanTextureSubresource;

struct VulkanTexture
{
    VulkanTextureContainer *container;
    Uint32 containerIndex;

    VulkanMemoryUsedRegion *usedRegion;

    VkImage image;
    VkImageView fullView; // used for samplers and storage reads
    VkComponentMapping swizzle;
    VkImageAspectFlags aspectFlags;
    Uint32 depth; // used for cleanup only

    // FIXME: It'd be nice if we didn't have to have this on the texture...
    SDL_GPUTextureUsageFlags usage; // used for defrag transitions only.

    Uint32 subresourceCount;
    VulkanTextureSubresource *subresources;

    bool markedForDestroy; // so that defrag doesn't double-free
    SDL_AtomicInt referenceCount;
};

struct VulkanTextureContainer
{
    TextureCommonHeader header;

    VulkanTexture *activeTexture;

    Uint32 textureCapacity;
    Uint32 textureCount;
    VulkanTexture **textures;

    char *debugName;
    bool canBeCycled;
};

typedef enum VulkanBufferUsageMode
{
    VULKAN_BUFFER_USAGE_MODE_COPY_SOURCE,
    VULKAN_BUFFER_USAGE_MODE_COPY_DESTINATION,
    VULKAN_BUFFER_USAGE_MODE_VERTEX_READ,
    VULKAN_BUFFER_USAGE_MODE_INDEX_READ,
    VULKAN_BUFFER_USAGE_MODE_INDIRECT,
    VULKAN_BUFFER_USAGE_MODE_GRAPHICS_STORAGE_READ,
    VULKAN_BUFFER_USAGE_MODE_COMPUTE_STORAGE_READ,
    VULKAN_BUFFER_USAGE_MODE_COMPUTE_STORAGE_READ_WRITE,
} VulkanBufferUsageMode;

typedef enum VulkanTextureUsageMode
{
    VULKAN_TEXTURE_USAGE_MODE_UNINITIALIZED,
    VULKAN_TEXTURE_USAGE_MODE_COPY_SOURCE,
    VULKAN_TEXTURE_USAGE_MODE_COPY_DESTINATION,
    VULKAN_TEXTURE_USAGE_MODE_SAMPLER,
    VULKAN_TEXTURE_USAGE_MODE_GRAPHICS_STORAGE_READ,
    VULKAN_TEXTURE_USAGE_MODE_COMPUTE_STORAGE_READ,
    VULKAN_TEXTURE_USAGE_MODE_COMPUTE_STORAGE_READ_WRITE,
    VULKAN_TEXTURE_USAGE_MODE_COLOR_ATTACHMENT,
    VULKAN_TEXTURE_USAGE_MODE_DEPTH_STENCIL_ATTACHMENT,
    VULKAN_TEXTURE_USAGE_MODE_PRESENT
} VulkanTextureUsageMode;

typedef enum VulkanUniformBufferStage
{
    VULKAN_UNIFORM_BUFFER_STAGE_VERTEX,
    VULKAN_UNIFORM_BUFFER_STAGE_FRAGMENT,
    VULKAN_UNIFORM_BUFFER_STAGE_COMPUTE
} VulkanUniformBufferStage;

typedef struct VulkanFramebuffer
{
    VkFramebuffer framebuffer;
    SDL_AtomicInt referenceCount;
} VulkanFramebuffer;

typedef struct WindowData
{
    SDL_Window *window;
    SDL_GPUSwapchainComposition swapchainComposition;
    SDL_GPUPresentMode presentMode;
    bool needsSwapchainRecreate;
    Uint32 swapchainCreateWidth;
    Uint32 swapchainCreateHeight;

    // Window surface
    VkSurfaceKHR surface;

    // Swapchain for window surface
    VkSwapchainKHR swapchain;
    VkFormat format;
    VkColorSpaceKHR colorSpace;
    VkComponentMapping swapchainSwizzle;
    bool usingFallbackFormat;

    // Swapchain images
    VulkanTextureContainer *textureContainers; // use containers so that swapchain textures can use the same API as other textures
    Uint32 imageCount;
    Uint32 width;
    Uint32 height;

    // Synchronization primitives
    VkSemaphore imageAvailableSemaphore[MAX_FRAMES_IN_FLIGHT];
    VkSemaphore renderFinishedSemaphore[MAX_FRAMES_IN_FLIGHT];
    SDL_GPUFence *inFlightFences[MAX_FRAMES_IN_FLIGHT];

    Uint32 frameCounter;
} WindowData;

typedef struct SwapchainSupportDetails
{
    VkSurfaceCapabilitiesKHR capabilities;
    VkSurfaceFormatKHR *formats;
    Uint32 formatsLength;
    VkPresentModeKHR *presentModes;
    Uint32 presentModesLength;
} SwapchainSupportDetails;

typedef struct VulkanPresentData
{
    WindowData *windowData;
    Uint32 swapchainImageIndex;
} VulkanPresentData;

struct VulkanUniformBuffer
{
    VulkanBuffer *buffer;
    Uint32 drawOffset;
    Uint32 writeOffset;
};

typedef struct VulkanDescriptorInfo
{
    VkDescriptorType descriptorType;
    VkShaderStageFlagBits stageFlag;
} VulkanDescriptorInfo;

typedef struct DescriptorSetPool
{
    // It's a pool... of pools!!!
    Uint32 poolCount;
    VkDescriptorPool *descriptorPools;

    // We'll just manage the descriptor sets ourselves instead of freeing the sets
    VkDescriptorSet *descriptorSets;
    Uint32 descriptorSetCount;
    Uint32 descriptorSetIndex;
} DescriptorSetPool;

// A command buffer acquires a cache at command buffer acquisition time
typedef struct DescriptorSetCache
{
    // Pools are indexed by DescriptorSetLayoutID which increases monotonically
    // There's only a certain number of maximum layouts possible since we de-duplicate them.
    DescriptorSetPool *pools;
    Uint32 poolCount;
} DescriptorSetCache;

typedef struct DescriptorSetLayoutHashTableKey
{
    VkShaderStageFlagBits shaderStage;
    // Category 1: read resources
    Uint32 samplerCount;
    Uint32 storageBufferCount;
    Uint32 storageTextureCount;
    // Category 2: write resources
    Uint32 writeStorageBufferCount;
    Uint32 writeStorageTextureCount;
    // Category 3: uniform buffers
    Uint32 uniformBufferCount;
} DescriptorSetLayoutHashTableKey;

typedef uint32_t DescriptorSetLayoutID;

typedef struct DescriptorSetLayout
{
    DescriptorSetLayoutID ID;
    VkDescriptorSetLayout descriptorSetLayout;

    // Category 1: read resources
    Uint32 samplerCount;
    Uint32 storageBufferCount;
    Uint32 storageTextureCount;
    // Category 2: write resources
    Uint32 writeStorageBufferCount;
    Uint32 writeStorageTextureCount;
    // Category 3: uniform buffers
    Uint32 uniformBufferCount;
} DescriptorSetLayout;

typedef struct GraphicsPipelineResourceLayoutHashTableKey
{
    Uint32 vertexSamplerCount;
    Uint32 vertexStorageBufferCount;
    Uint32 vertexStorageTextureCount;
    Uint32 vertexUniformBufferCount;

    Uint32 fragmentSamplerCount;
    Uint32 fragmentStorageBufferCount;
    Uint32 fragmentStorageTextureCount;
    Uint32 fragmentUniformBufferCount;
} GraphicsPipelineResourceLayoutHashTableKey;

typedef struct VulkanGraphicsPipelineResourceLayout
{
    VkPipelineLayout pipelineLayout;

    /*
     * Descriptor set layout is as follows:
     * 0: vertex resources
     * 1: vertex uniform buffers
     * 2: fragment resources
     * 3: fragment uniform buffers
     */
    DescriptorSetLayout *descriptorSetLayouts[4];

    Uint32 vertexSamplerCount;
    Uint32 vertexStorageBufferCount;
    Uint32 vertexStorageTextureCount;
    Uint32 vertexUniformBufferCount;

    Uint32 fragmentSamplerCount;
    Uint32 fragmentStorageBufferCount;
    Uint32 fragmentStorageTextureCount;
    Uint32 fragmentUniformBufferCount;
} VulkanGraphicsPipelineResourceLayout;

typedef struct VulkanGraphicsPipeline
{
    VkPipeline pipeline;
    SDL_GPUPrimitiveType primitiveType;

    VulkanGraphicsPipelineResourceLayout *resourceLayout;

    VulkanShader *vertexShader;
    VulkanShader *fragmentShader;

    SDL_AtomicInt referenceCount;
} VulkanGraphicsPipeline;

typedef struct ComputePipelineResourceLayoutHashTableKey
{
    Uint32 samplerCount;
    Uint32 readonlyStorageTextureCount;
    Uint32 readonlyStorageBufferCount;
    Uint32 readWriteStorageTextureCount;
    Uint32 readWriteStorageBufferCount;
    Uint32 uniformBufferCount;
} ComputePipelineResourceLayoutHashTableKey;

typedef struct VulkanComputePipelineResourceLayout
{
    VkPipelineLayout pipelineLayout;

    /*
     * Descriptor set layout is as follows:
     * 0: samplers, then read-only textures, then read-only buffers
     * 1: write-only textures, then write-only buffers
     * 2: uniform buffers
     */
    DescriptorSetLayout *descriptorSetLayouts[3];

    Uint32 numSamplers;
    Uint32 numReadonlyStorageTextures;
    Uint32 numReadonlyStorageBuffers;
    Uint32 numReadWriteStorageTextures;
    Uint32 numReadWriteStorageBuffers;
    Uint32 numUniformBuffers;
} VulkanComputePipelineResourceLayout;

typedef struct VulkanComputePipeline
{
    VkShaderModule shaderModule;
    VkPipeline pipeline;
    VulkanComputePipelineResourceLayout *resourceLayout;
    SDL_AtomicInt referenceCount;
} VulkanComputePipeline;

typedef struct RenderPassColorTargetDescription
{
    VkFormat format;
    SDL_GPULoadOp loadOp;
    SDL_GPUStoreOp storeOp;
} RenderPassColorTargetDescription;

typedef struct RenderPassDepthStencilTargetDescription
{
    VkFormat format;
    SDL_GPULoadOp loadOp;
    SDL_GPUStoreOp storeOp;
    SDL_GPULoadOp stencilLoadOp;
    SDL_GPUStoreOp stencilStoreOp;
} RenderPassDepthStencilTargetDescription;

typedef struct CommandPoolHashTableKey
{
    SDL_ThreadID threadID;
} CommandPoolHashTableKey;

typedef struct RenderPassHashTableKey
{
    RenderPassColorTargetDescription colorTargetDescriptions[MAX_COLOR_TARGET_BINDINGS];
    Uint32 numColorTargets;
    VkFormat resolveTargetFormats[MAX_COLOR_TARGET_BINDINGS];
    Uint32 numResolveTargets;
    RenderPassDepthStencilTargetDescription depthStencilTargetDescription;
    VkSampleCountFlagBits sampleCount;
} RenderPassHashTableKey;

typedef struct VulkanRenderPassHashTableValue
{
    VkRenderPass handle;
} VulkanRenderPassHashTableValue;

typedef struct FramebufferHashTableKey
{
    VkImageView colorAttachmentViews[MAX_COLOR_TARGET_BINDINGS];
    Uint32 numColorTargets;
    VkImageView resolveAttachmentViews[MAX_COLOR_TARGET_BINDINGS];
    Uint32 numResolveAttachments;
    VkImageView depthStencilAttachmentView;
    Uint32 width;
    Uint32 height;
} FramebufferHashTableKey;

// Command structures

typedef struct VulkanFencePool
{
    SDL_Mutex *lock;

    VulkanFenceHandle **availableFences;
    Uint32 availableFenceCount;
    Uint32 availableFenceCapacity;
} VulkanFencePool;

typedef struct VulkanCommandPool VulkanCommandPool;

typedef struct VulkanRenderer VulkanRenderer;

typedef struct VulkanCommandBuffer
{
    CommandBufferCommonHeader common;
    VulkanRenderer *renderer;

    VkCommandBuffer commandBuffer;
    VulkanCommandPool *commandPool;

    VulkanPresentData *presentDatas;
    Uint32 presentDataCount;
    Uint32 presentDataCapacity;

    VkSemaphore *waitSemaphores;
    Uint32 waitSemaphoreCount;
    Uint32 waitSemaphoreCapacity;

    VkSemaphore *signalSemaphores;
    Uint32 signalSemaphoreCount;
    Uint32 signalSemaphoreCapacity;

    VulkanComputePipeline *currentComputePipeline;
    VulkanGraphicsPipeline *currentGraphicsPipeline;

    // Keep track of resources transitioned away from their default state to barrier them on pass end

    VulkanTextureSubresource *colorAttachmentSubresources[MAX_COLOR_TARGET_BINDINGS];
    Uint32 colorAttachmentSubresourceCount;
    VulkanTextureSubresource *resolveAttachmentSubresources[MAX_COLOR_TARGET_BINDINGS];
    Uint32 resolveAttachmentSubresourceCount;

    VulkanTextureSubresource *depthStencilAttachmentSubresource; // may be NULL

    // Dynamic state

    VkViewport currentViewport;
    VkRect2D currentScissor;
    float blendConstants[4];
    Uint8 stencilRef;

    // Resource bind state

    DescriptorSetCache *descriptorSetCache; // acquired when command buffer is acquired

    bool needNewVertexResourceDescriptorSet;
    bool needNewVertexUniformDescriptorSet;
    bool needNewVertexUniformOffsets;
    bool needNewFragmentResourceDescriptorSet;
    bool needNewFragmentUniformDescriptorSet;
    bool needNewFragmentUniformOffsets;

    bool needNewComputeReadOnlyDescriptorSet;
    bool needNewComputeReadWriteDescriptorSet;
    bool needNewComputeUniformDescriptorSet;
    bool needNewComputeUniformOffsets;

    VkDescriptorSet vertexResourceDescriptorSet;
    VkDescriptorSet vertexUniformDescriptorSet;
    VkDescriptorSet fragmentResourceDescriptorSet;
    VkDescriptorSet fragmentUniformDescriptorSet;

    VkDescriptorSet computeReadOnlyDescriptorSet;
    VkDescriptorSet computeReadWriteDescriptorSet;
    VkDescriptorSet computeUniformDescriptorSet;

    VkBuffer vertexBuffers[MAX_VERTEX_BUFFERS];
    VkDeviceSize vertexBufferOffsets[MAX_VERTEX_BUFFERS];
    Uint32 vertexBufferCount;
    bool needVertexBufferBind;

    VulkanTexture *vertexSamplerTextures[MAX_TEXTURE_SAMPLERS_PER_STAGE];
    VulkanSampler *vertexSamplers[MAX_TEXTURE_SAMPLERS_PER_STAGE];
    VulkanTexture *vertexStorageTextures[MAX_STORAGE_TEXTURES_PER_STAGE];
    VulkanBuffer *vertexStorageBuffers[MAX_STORAGE_BUFFERS_PER_STAGE];

    VulkanTexture *fragmentSamplerTextures[MAX_TEXTURE_SAMPLERS_PER_STAGE];
    VulkanSampler *fragmentSamplers[MAX_TEXTURE_SAMPLERS_PER_STAGE];
    VulkanTexture *fragmentStorageTextures[MAX_STORAGE_TEXTURES_PER_STAGE];
    VulkanBuffer *fragmentStorageBuffers[MAX_STORAGE_BUFFERS_PER_STAGE];

    VulkanTextureSubresource *readWriteComputeStorageTextureSubresources[MAX_COMPUTE_WRITE_TEXTURES];
    Uint32 readWriteComputeStorageTextureSubresourceCount;
    VulkanBuffer *readWriteComputeStorageBuffers[MAX_COMPUTE_WRITE_BUFFERS];

    VulkanTexture *computeSamplerTextures[MAX_TEXTURE_SAMPLERS_PER_STAGE];
    VulkanSampler *computeSamplers[MAX_TEXTURE_SAMPLERS_PER_STAGE];
    VulkanTexture *readOnlyComputeStorageTextures[MAX_STORAGE_TEXTURES_PER_STAGE];
    VulkanBuffer *readOnlyComputeStorageBuffers[MAX_STORAGE_BUFFERS_PER_STAGE];

    // Uniform buffers

    VulkanUniformBuffer *vertexUniformBuffers[MAX_UNIFORM_BUFFERS_PER_STAGE];
    VulkanUniformBuffer *fragmentUniformBuffers[MAX_UNIFORM_BUFFERS_PER_STAGE];
    VulkanUniformBuffer *computeUniformBuffers[MAX_UNIFORM_BUFFERS_PER_STAGE];

    // Track used resources

    VulkanBuffer **usedBuffers;
    Sint32 usedBufferCount;
    Sint32 usedBufferCapacity;

    VulkanTexture **usedTextures;
    Sint32 usedTextureCount;
    Sint32 usedTextureCapacity;

    VulkanSampler **usedSamplers;
    Sint32 usedSamplerCount;
    Sint32 usedSamplerCapacity;

    VulkanGraphicsPipeline **usedGraphicsPipelines;
    Sint32 usedGraphicsPipelineCount;
    Sint32 usedGraphicsPipelineCapacity;

    VulkanComputePipeline **usedComputePipelines;
    Sint32 usedComputePipelineCount;
    Sint32 usedComputePipelineCapacity;

    VulkanFramebuffer **usedFramebuffers;
    Sint32 usedFramebufferCount;
    Sint32 usedFramebufferCapacity;

    VulkanUniformBuffer **usedUniformBuffers;
    Sint32 usedUniformBufferCount;
    Sint32 usedUniformBufferCapacity;

    VulkanFenceHandle *inFlightFence;
    bool autoReleaseFence;

    bool isDefrag; // Whether this CB was created for defragging
} VulkanCommandBuffer;

struct VulkanCommandPool
{
    SDL_ThreadID threadID;
    VkCommandPool commandPool;

    VulkanCommandBuffer **inactiveCommandBuffers;
    Uint32 inactiveCommandBufferCapacity;
    Uint32 inactiveCommandBufferCount;
};

// Context

struct VulkanRenderer
{
    VkInstance instance;
    VkPhysicalDevice physicalDevice;
    VkPhysicalDeviceProperties2KHR physicalDeviceProperties;
    VkPhysicalDeviceDriverPropertiesKHR physicalDeviceDriverProperties;
    VkDevice logicalDevice;
    Uint8 integratedMemoryNotification;
    Uint8 outOfDeviceLocalMemoryWarning;
    Uint8 outofBARMemoryWarning;
    Uint8 fillModeOnlyWarning;

    bool debugMode;
    bool preferLowPower;
    SDL_PropertiesID props;
    Uint32 allowedFramesInFlight;

    VulkanExtensions supports;
    bool supportsDebugUtils;
    bool supportsColorspace;
    bool supportsFillModeNonSolid;
    bool supportsMultiDrawIndirect;

    VulkanMemoryAllocator *memoryAllocator;
    VkPhysicalDeviceMemoryProperties memoryProperties;
    bool checkEmptyAllocations;

    WindowData **claimedWindows;
    Uint32 claimedWindowCount;
    Uint32 claimedWindowCapacity;

    Uint32 queueFamilyIndex;
    VkQueue unifiedQueue;

    VulkanCommandBuffer **submittedCommandBuffers;
    Uint32 submittedCommandBufferCount;
    Uint32 submittedCommandBufferCapacity;

    VulkanFencePool fencePool;

    SDL_HashTable *commandPoolHashTable;
    SDL_HashTable *renderPassHashTable;
    SDL_HashTable *framebufferHashTable;
    SDL_HashTable *graphicsPipelineResourceLayoutHashTable;
    SDL_HashTable *computePipelineResourceLayoutHashTable;
    SDL_HashTable *descriptorSetLayoutHashTable;

    VulkanUniformBuffer **uniformBufferPool;
    Uint32 uniformBufferPoolCount;
    Uint32 uniformBufferPoolCapacity;

    DescriptorSetCache **descriptorSetCachePool;
    Uint32 descriptorSetCachePoolCount;
    Uint32 descriptorSetCachePoolCapacity;

    SDL_AtomicInt layoutResourceID;

    Uint32 minUBOAlignment;

    // Deferred resource destruction

    VulkanTexture **texturesToDestroy;
    Uint32 texturesToDestroyCount;
    Uint32 texturesToDestroyCapacity;

    VulkanBuffer **buffersToDestroy;
    Uint32 buffersToDestroyCount;
    Uint32 buffersToDestroyCapacity;

    VulkanSampler **samplersToDestroy;
    Uint32 samplersToDestroyCount;
    Uint32 samplersToDestroyCapacity;

    VulkanGraphicsPipeline **graphicsPipelinesToDestroy;
    Uint32 graphicsPipelinesToDestroyCount;
    Uint32 graphicsPipelinesToDestroyCapacity;

    VulkanComputePipeline **computePipelinesToDestroy;
    Uint32 computePipelinesToDestroyCount;
    Uint32 computePipelinesToDestroyCapacity;

    VulkanShader **shadersToDestroy;
    Uint32 shadersToDestroyCount;
    Uint32 shadersToDestroyCapacity;

    VulkanFramebuffer **framebuffersToDestroy;
    Uint32 framebuffersToDestroyCount;
    Uint32 framebuffersToDestroyCapacity;

    SDL_Mutex *allocatorLock;
    SDL_Mutex *disposeLock;
    SDL_Mutex *submitLock;
    SDL_Mutex *acquireCommandBufferLock;
    SDL_Mutex *acquireUniformBufferLock;
    SDL_Mutex *renderPassFetchLock;
    SDL_Mutex *framebufferFetchLock;
    SDL_Mutex *graphicsPipelineLayoutFetchLock;
    SDL_Mutex *computePipelineLayoutFetchLock;
    SDL_Mutex *descriptorSetLayoutFetchLock;
    SDL_Mutex *windowLock;

    Uint8 defragInProgress;

    VulkanMemoryAllocation **allocationsToDefrag;
    Uint32 allocationsToDefragCount;
    Uint32 allocationsToDefragCapacity;

#define VULKAN_INSTANCE_FUNCTION(func) \
    PFN_##func func;
#define VULKAN_DEVICE_FUNCTION(func) \
    PFN_##func func;
#include "SDL_gpu_vulkan_vkfuncs.h"
};

// Forward declarations

static bool VULKAN_INTERNAL_DefragmentMemory(VulkanRenderer *renderer, VulkanCommandBuffer *commandBuffer);
static bool VULKAN_INTERNAL_BeginCommandBuffer(VulkanRenderer *renderer, VulkanCommandBuffer *commandBuffer);
static void VULKAN_ReleaseWindow(SDL_GPURenderer *driverData, SDL_Window *window);
static bool VULKAN_Wait(SDL_GPURenderer *driverData);
static bool VULKAN_WaitForFences(SDL_GPURenderer *driverData, bool waitAll, SDL_GPUFence *const *fences, Uint32 numFences);
static bool VULKAN_Submit(SDL_GPUCommandBuffer *commandBuffer);
static SDL_GPUCommandBuffer *VULKAN_AcquireCommandBuffer(SDL_GPURenderer *driverData);

// Error Handling

static inline const char *VkErrorMessages(VkResult code)
{
#define ERR_TO_STR(e) \
    case e:           \
        return #e;
    switch (code) {
        ERR_TO_STR(VK_ERROR_OUT_OF_HOST_MEMORY)
        ERR_TO_STR(VK_ERROR_OUT_OF_DEVICE_MEMORY)
        ERR_TO_STR(VK_ERROR_FRAGMENTED_POOL)
        ERR_TO_STR(VK_ERROR_OUT_OF_POOL_MEMORY)
        ERR_TO_STR(VK_ERROR_INITIALIZATION_FAILED)
        ERR_TO_STR(VK_ERROR_LAYER_NOT_PRESENT)
        ERR_TO_STR(VK_ERROR_EXTENSION_NOT_PRESENT)
        ERR_TO_STR(VK_ERROR_FEATURE_NOT_PRESENT)
        ERR_TO_STR(VK_ERROR_TOO_MANY_OBJECTS)
        ERR_TO_STR(VK_ERROR_DEVICE_LOST)
        ERR_TO_STR(VK_ERROR_INCOMPATIBLE_DRIVER)
        ERR_TO_STR(VK_ERROR_OUT_OF_DATE_KHR)
        ERR_TO_STR(VK_ERROR_SURFACE_LOST_KHR)
        ERR_TO_STR(VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT)
        ERR_TO_STR(VK_SUBOPTIMAL_KHR)
        ERR_TO_STR(VK_ERROR_NATIVE_WINDOW_IN_USE_KHR)
        ERR_TO_STR(VK_ERROR_INVALID_SHADER_NV)
    default:
        return "Unhandled VkResult!";
    }
#undef ERR_TO_STR
}

#define SET_ERROR_AND_RETURN(fmt, msg, ret)               \
    do {                                                  \
        if (renderer->debugMode) {                        \
            SDL_LogError(SDL_LOG_CATEGORY_GPU, fmt, msg); \
        }                                                 \
        SDL_SetError((fmt), (msg));                       \
        return ret;                                       \
    } while (0)

#define SET_STRING_ERROR_AND_RETURN(msg, ret) SET_ERROR_AND_RETURN("%s", msg, ret)

#define CHECK_VULKAN_ERROR_AND_RETURN(res, fn, ret)                                     \
    do {                                                                                \
        if ((res) != VK_SUCCESS) {                                                      \
            if (renderer->debugMode) {                                                  \
                SDL_LogError(SDL_LOG_CATEGORY_GPU, "%s %s", #fn, VkErrorMessages(res)); \
            }                                                                           \
            SDL_SetError("%s %s", #fn, VkErrorMessages(res));                           \
            return (ret);                                                               \
        }                                                                               \
    } while (0)

// Utility

static inline VkPolygonMode SDLToVK_PolygonMode(
    VulkanRenderer *renderer,
    SDL_GPUFillMode mode)
{
    if (mode == SDL_GPU_FILLMODE_FILL) {
        return VK_POLYGON_MODE_FILL; // always available!
    }

    if (renderer->supportsFillModeNonSolid && mode == SDL_GPU_FILLMODE_LINE) {
        return VK_POLYGON_MODE_LINE;
    }

    if (!renderer->fillModeOnlyWarning) {
        SDL_LogWarn(SDL_LOG_CATEGORY_GPU, "Unsupported fill mode requested, using FILL!");
        renderer->fillModeOnlyWarning = 1;
    }
    return VK_POLYGON_MODE_FILL;
}

// Memory Management

// Vulkan: Memory Allocation

static inline VkDeviceSize VULKAN_INTERNAL_NextHighestAlignment(
    VkDeviceSize n,
    VkDeviceSize align)
{
    return align * ((n + align - 1) / align);
}

static inline Uint32 VULKAN_INTERNAL_NextHighestAlignment32(
    Uint32 n,
    Uint32 align)
{
    return align * ((n + align - 1) / align);
}

static void VULKAN_INTERNAL_MakeMemoryUnavailable(
    VulkanMemoryAllocation *allocation)
{
    Uint32 i, j;
    VulkanMemoryFreeRegion *freeRegion;

    allocation->availableForAllocation = 0;

    for (i = 0; i < allocation->freeRegionCount; i += 1) {
        freeRegion = allocation->freeRegions[i];

        // close the gap in the sorted list
        if (allocation->allocator->sortedFreeRegionCount > 1) {
            for (j = freeRegion->sortedIndex; j < allocation->allocator->sortedFreeRegionCount - 1; j += 1) {
                allocation->allocator->sortedFreeRegions[j] =
                    allocation->allocator->sortedFreeRegions[j + 1];

                allocation->allocator->sortedFreeRegions[j]->sortedIndex = j;
            }
        }

        allocation->allocator->sortedFreeRegionCount -= 1;
    }
}

static void VULKAN_INTERNAL_MarkAllocationsForDefrag(
    VulkanRenderer *renderer)
{
    Uint32 memoryType, allocationIndex;
    VulkanMemorySubAllocator *currentAllocator;

    for (memoryType = 0; memoryType < VK_MAX_MEMORY_TYPES; memoryType += 1) {
        currentAllocator = &renderer->memoryAllocator->subAllocators[memoryType];

        for (allocationIndex = 0; allocationIndex < currentAllocator->allocationCount; allocationIndex += 1) {
            if (currentAllocator->allocations[allocationIndex]->availableForAllocation == 1) {
                if (currentAllocator->allocations[allocationIndex]->freeRegionCount > 1) {
                    EXPAND_ARRAY_IF_NEEDED(
                        renderer->allocationsToDefrag,
                        VulkanMemoryAllocation *,
                        renderer->allocationsToDefragCount + 1,
                        renderer->allocationsToDefragCapacity,
                        renderer->allocationsToDefragCapacity * 2);

                    renderer->allocationsToDefrag[renderer->allocationsToDefragCount] =
                        currentAllocator->allocations[allocationIndex];

                    renderer->allocationsToDefragCount += 1;

                    VULKAN_INTERNAL_MakeMemoryUnavailable(
                        currentAllocator->allocations[allocationIndex]);
                }
            }
        }
    }
}

static void VULKAN_INTERNAL_RemoveMemoryFreeRegion(
    VulkanRenderer *renderer,
    VulkanMemoryFreeRegion *freeRegion)
{
    Uint32 i;

    SDL_LockMutex(renderer->allocatorLock);

    if (freeRegion->allocation->availableForAllocation) {
        // close the gap in the sorted list
        if (freeRegion->allocation->allocator->sortedFreeRegionCount > 1) {
            for (i = freeRegion->sortedIndex; i < freeRegion->allocation->allocator->sortedFreeRegionCount - 1; i += 1) {
                freeRegion->allocation->allocator->sortedFreeRegions[i] =
                    freeRegion->allocation->allocator->sortedFreeRegions[i + 1];

                freeRegion->allocation->allocator->sortedFreeRegions[i]->sortedIndex = i;
            }
        }

        freeRegion->allocation->allocator->sortedFreeRegionCount -= 1;
    }

    // close the gap in the buffer list
    if (freeRegion->allocation->freeRegionCount > 1 && freeRegion->allocationIndex != freeRegion->allocation->freeRegionCount - 1) {
        freeRegion->allocation->freeRegions[freeRegion->allocationIndex] =
            freeRegion->allocation->freeRegions[freeRegion->allocation->freeRegionCount - 1];

        freeRegion->allocation->freeRegions[freeRegion->allocationIndex]->allocationIndex =
            freeRegion->allocationIndex;
    }

    freeRegion->allocation->freeRegionCount -= 1;

    freeRegion->allocation->freeSpace -= freeRegion->size;

    SDL_free(freeRegion);

    SDL_UnlockMutex(renderer->allocatorLock);
}

static void VULKAN_INTERNAL_NewMemoryFreeRegion(
    VulkanRenderer *renderer,
    VulkanMemoryAllocation *allocation,
    VkDeviceSize offset,
    VkDeviceSize size)
{
    VulkanMemoryFreeRegion *newFreeRegion;
    VkDeviceSize newOffset, newSize;
    Sint32 insertionIndex = 0;

    SDL_LockMutex(renderer->allocatorLock);

    // look for an adjacent region to merge
    for (Sint32 i = allocation->freeRegionCount - 1; i >= 0; i -= 1) {
        // check left side
        if (allocation->freeRegions[i]->offset + allocation->freeRegions[i]->size == offset) {
            newOffset = allocation->freeRegions[i]->offset;
            newSize = allocation->freeRegions[i]->size + size;

            VULKAN_INTERNAL_RemoveMemoryFreeRegion(renderer, allocation->freeRegions[i]);
            VULKAN_INTERNAL_NewMemoryFreeRegion(renderer, allocation, newOffset, newSize);

            SDL_UnlockMutex(renderer->allocatorLock);
            return;
        }

        // check right side
        if (allocation->freeRegions[i]->offset == offset + size) {
            newOffset = offset;
            newSize = allocation->freeRegions[i]->size + size;

            VULKAN_INTERNAL_RemoveMemoryFreeRegion(renderer, allocation->freeRegions[i]);
            VULKAN_INTERNAL_NewMemoryFreeRegion(renderer, allocation, newOffset, newSize);

            SDL_UnlockMutex(renderer->allocatorLock);
            return;
        }
    }

    // region is not contiguous with another free region, make a new one
    allocation->freeRegionCount += 1;
    if (allocation->freeRegionCount > allocation->freeRegionCapacity) {
        allocation->freeRegionCapacity *= 2;
        allocation->freeRegions = SDL_realloc(
            allocation->freeRegions,
            sizeof(VulkanMemoryFreeRegion *) * allocation->freeRegionCapacity);
    }

    newFreeRegion = SDL_malloc(sizeof(VulkanMemoryFreeRegion));
    newFreeRegion->offset = offset;
    newFreeRegion->size = size;
    newFreeRegion->allocation = allocation;

    allocation->freeSpace += size;

    allocation->freeRegions[allocation->freeRegionCount - 1] = newFreeRegion;
    newFreeRegion->allocationIndex = allocation->freeRegionCount - 1;

    if (allocation->availableForAllocation) {
        for (Uint32 i = 0; i < allocation->allocator->sortedFreeRegionCount; i += 1) {
            if (allocation->allocator->sortedFreeRegions[i]->size < size) {
                // this is where the new region should go
                break;
            }

            insertionIndex += 1;
        }

        if (allocation->allocator->sortedFreeRegionCount + 1 > allocation->allocator->sortedFreeRegionCapacity) {
            allocation->allocator->sortedFreeRegionCapacity *= 2;
            allocation->allocator->sortedFreeRegions = SDL_realloc(
                allocation->allocator->sortedFreeRegions,
                sizeof(VulkanMemoryFreeRegion *) * allocation->allocator->sortedFreeRegionCapacity);
        }

        // perform insertion sort
        if (allocation->allocator->sortedFreeRegionCount > 0 && (Uint32)insertionIndex != allocation->allocator->sortedFreeRegionCount) {
            for (Sint32 i = allocation->allocator->sortedFreeRegionCount; i > insertionIndex && i > 0; i -= 1) {
                allocation->allocator->sortedFreeRegions[i] = allocation->allocator->sortedFreeRegions[i - 1];
                allocation->allocator->sortedFreeRegions[i]->sortedIndex = i;
            }
        }

        allocation->allocator->sortedFreeRegionCount += 1;
        allocation->allocator->sortedFreeRegions[insertionIndex] = newFreeRegion;
        newFreeRegion->sortedIndex = insertionIndex;
    }

    SDL_UnlockMutex(renderer->allocatorLock);
}

static VulkanMemoryUsedRegion *VULKAN_INTERNAL_NewMemoryUsedRegion(
    VulkanRenderer *renderer,
    VulkanMemoryAllocation *allocation,
    VkDeviceSize offset,
    VkDeviceSize size,
    VkDeviceSize resourceOffset,
    VkDeviceSize resourceSize,
    VkDeviceSize alignment)
{
    VulkanMemoryUsedRegion *memoryUsedRegion;

    SDL_LockMutex(renderer->allocatorLock);

    if (allocation->usedRegionCount == allocation->usedRegionCapacity) {
        allocation->usedRegionCapacity *= 2;
        allocation->usedRegions = SDL_realloc(
            allocation->usedRegions,
            allocation->usedRegionCapacity * sizeof(VulkanMemoryUsedRegion *));
    }

    memoryUsedRegion = SDL_malloc(sizeof(VulkanMemoryUsedRegion));
    memoryUsedRegion->allocation = allocation;
    memoryUsedRegion->offset = offset;
    memoryUsedRegion->size = size;
    memoryUsedRegion->resourceOffset = resourceOffset;
    memoryUsedRegion->resourceSize = resourceSize;
    memoryUsedRegion->alignment = alignment;

    allocation->usedSpace += size;

    allocation->usedRegions[allocation->usedRegionCount] = memoryUsedRegion;
    allocation->usedRegionCount += 1;

    SDL_UnlockMutex(renderer->allocatorLock);

    return memoryUsedRegion;
}

static void VULKAN_INTERNAL_RemoveMemoryUsedRegion(
    VulkanRenderer *renderer,
    VulkanMemoryUsedRegion *usedRegion)
{
    Uint32 i;

    SDL_LockMutex(renderer->allocatorLock);

    for (i = 0; i < usedRegion->allocation->usedRegionCount; i += 1) {
        if (usedRegion->allocation->usedRegions[i] == usedRegion) {
            // plug the hole
            if (i != usedRegion->allocation->usedRegionCount - 1) {
                usedRegion->allocation->usedRegions[i] = usedRegion->allocation->usedRegions[usedRegion->allocation->usedRegionCount - 1];
            }

            break;
        }
    }

    usedRegion->allocation->usedSpace -= usedRegion->size;

    usedRegion->allocation->usedRegionCount -= 1;

    VULKAN_INTERNAL_NewMemoryFreeRegion(
        renderer,
        usedRegion->allocation,
        usedRegion->offset,
        usedRegion->size);

    if (usedRegion->allocation->usedRegionCount == 0) {
        renderer->checkEmptyAllocations = true;
    }

    SDL_free(usedRegion);

    SDL_UnlockMutex(renderer->allocatorLock);
}

static bool VULKAN_INTERNAL_CheckMemoryTypeArrayUnique(
    Uint32 memoryTypeIndex,
    const Uint32 *memoryTypeIndexArray,
    Uint32 count)
{
    Uint32 i = 0;

    for (i = 0; i < count; i += 1) {
        if (memoryTypeIndexArray[i] == memoryTypeIndex) {
            return false;
        }
    }

    return true;
}

/* Returns an array of memory type indices in order of preference.
 * Memory types are requested with the following three guidelines:
 *
 * Required: Absolutely necessary
 * Preferred: Nice to have, but not necessary
 * Tolerable: Can be allowed if there are no other options
 *
 * We return memory types in this order:
 * 1. Required and preferred. This is the best category.
 * 2. Required only.
 * 3. Required, preferred, and tolerable.
 * 4. Required and tolerable. This is the worst category.
 */
static Uint32 *VULKAN_INTERNAL_FindBestMemoryTypes(
    VulkanRenderer *renderer,
    Uint32 typeFilter,
    VkMemoryPropertyFlags requiredProperties,
    VkMemoryPropertyFlags preferredProperties,
    VkMemoryPropertyFlags tolerableProperties,
    Uint32 *pCount)
{
    Uint32 i;
    Uint32 index = 0;
    Uint32 *result = SDL_malloc(sizeof(Uint32) * renderer->memoryProperties.memoryTypeCount);

    // required + preferred + !tolerable
    for (i = 0; i < renderer->memoryProperties.memoryTypeCount; i += 1) {
        if ((typeFilter & (1 << i)) &&
            (renderer->memoryProperties.memoryTypes[i].propertyFlags & requiredProperties) == requiredProperties &&
            (renderer->memoryProperties.memoryTypes[i].propertyFlags & preferredProperties) == preferredProperties &&
            (renderer->memoryProperties.memoryTypes[i].propertyFlags & tolerableProperties) == 0) {
            if (VULKAN_INTERNAL_CheckMemoryTypeArrayUnique(
                    i,
                    result,
                    index)) {
                result[index] = i;
                index += 1;
            }
        }
    }

    // required + !preferred + !tolerable
    for (i = 0; i < renderer->memoryProperties.memoryTypeCount; i += 1) {
        if ((typeFilter & (1 << i)) &&
            (renderer->memoryProperties.memoryTypes[i].propertyFlags & requiredProperties) == requiredProperties &&
            (renderer->memoryProperties.memoryTypes[i].propertyFlags & preferredProperties) == 0 &&
            (renderer->memoryProperties.memoryTypes[i].propertyFlags & tolerableProperties) == 0) {
            if (VULKAN_INTERNAL_CheckMemoryTypeArrayUnique(
                    i,
                    result,
                    index)) {
                result[index] = i;
                index += 1;
            }
        }
    }

    // required + preferred + tolerable
    for (i = 0; i < renderer->memoryProperties.memoryTypeCount; i += 1) {
        if ((typeFilter & (1 << i)) &&
            (renderer->memoryProperties.memoryTypes[i].propertyFlags & requiredProperties) == requiredProperties &&
            (renderer->memoryProperties.memoryTypes[i].propertyFlags & preferredProperties) == preferredProperties &&
            (renderer->memoryProperties.memoryTypes[i].propertyFlags & tolerableProperties) == tolerableProperties) {
            if (VULKAN_INTERNAL_CheckMemoryTypeArrayUnique(
                    i,
                    result,
                    index)) {
                result[index] = i;
                index += 1;
            }
        }
    }

    // required + !preferred + tolerable
    for (i = 0; i < renderer->memoryProperties.memoryTypeCount; i += 1) {
        if ((typeFilter & (1 << i)) &&
            (renderer->memoryProperties.memoryTypes[i].propertyFlags & requiredProperties) == requiredProperties &&
            (renderer->memoryProperties.memoryTypes[i].propertyFlags & preferredProperties) == 0 &&
            (renderer->memoryProperties.memoryTypes[i].propertyFlags & tolerableProperties) == tolerableProperties) {
            if (VULKAN_INTERNAL_CheckMemoryTypeArrayUnique(
                    i,
                    result,
                    index)) {
                result[index] = i;
                index += 1;
            }
        }
    }

    *pCount = index;
    return result;
}

static Uint32 *VULKAN_INTERNAL_FindBestBufferMemoryTypes(
    VulkanRenderer *renderer,
    VkBuffer buffer,
    VkMemoryPropertyFlags requiredMemoryProperties,
    VkMemoryPropertyFlags preferredMemoryProperties,
    VkMemoryPropertyFlags tolerableMemoryProperties,
    VkMemoryRequirements *pMemoryRequirements,
    Uint32 *pCount)
{
    renderer->vkGetBufferMemoryRequirements(
        renderer->logicalDevice,
        buffer,
        pMemoryRequirements);

    return VULKAN_INTERNAL_FindBestMemoryTypes(
        renderer,
        pMemoryRequirements->memoryTypeBits,
        requiredMemoryProperties,
        preferredMemoryProperties,
        tolerableMemoryProperties,
        pCount);
}

static Uint32 *VULKAN_INTERNAL_FindBestImageMemoryTypes(
    VulkanRenderer *renderer,
    VkImage image,
    VkMemoryPropertyFlags preferredMemoryPropertyFlags,
    VkMemoryRequirements *pMemoryRequirements,
    Uint32 *pCount)
{
    renderer->vkGetImageMemoryRequirements(
        renderer->logicalDevice,
        image,
        pMemoryRequirements);

    return VULKAN_INTERNAL_FindBestMemoryTypes(
        renderer,
        pMemoryRequirements->memoryTypeBits,
        0,
        preferredMemoryPropertyFlags,
        0,
        pCount);
}

static void VULKAN_INTERNAL_DeallocateMemory(
    VulkanRenderer *renderer,
    VulkanMemorySubAllocator *allocator,
    Uint32 allocationIndex)
{
    Uint32 i;

    VulkanMemoryAllocation *allocation = allocator->allocations[allocationIndex];

    SDL_LockMutex(renderer->allocatorLock);

    // If this allocation was marked for defrag, cancel that
    for (i = 0; i < renderer->allocationsToDefragCount; i += 1) {
        if (allocation == renderer->allocationsToDefrag[i]) {
            renderer->allocationsToDefrag[i] = renderer->allocationsToDefrag[renderer->allocationsToDefragCount - 1];
            renderer->allocationsToDefragCount -= 1;

            break;
        }
    }

    for (i = 0; i < allocation->freeRegionCount; i += 1) {
        VULKAN_INTERNAL_RemoveMemoryFreeRegion(
            renderer,
            allocation->freeRegions[i]);
    }
    SDL_free(allocation->freeRegions);

    /* no need to iterate used regions because deallocate
     * only happens when there are 0 used regions
     */
    SDL_free(allocation->usedRegions);

    renderer->vkFreeMemory(
        renderer->logicalDevice,
        allocation->memory,
        NULL);

    SDL_DestroyMutex(allocation->memoryLock);
    SDL_free(allocation);

    if (allocationIndex != allocator->allocationCount - 1) {
        allocator->allocations[allocationIndex] = allocator->allocations[allocator->allocationCount - 1];
    }

    allocator->allocationCount -= 1;

    SDL_UnlockMutex(renderer->allocatorLock);
}

static Uint8 VULKAN_INTERNAL_AllocateMemory(
    VulkanRenderer *renderer,
    Uint32 memoryTypeIndex,
    VkDeviceSize allocationSize,
    Uint8 isHostVisible,
    VulkanMemoryAllocation **pMemoryAllocation)
{
    VulkanMemoryAllocation *allocation;
    VulkanMemorySubAllocator *allocator = &renderer->memoryAllocator->subAllocators[memoryTypeIndex];
    VkMemoryAllocateInfo allocInfo;
    VkResult result;

    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.pNext = NULL;
    allocInfo.memoryTypeIndex = memoryTypeIndex;
    allocInfo.allocationSize = allocationSize;

    allocation = SDL_malloc(sizeof(VulkanMemoryAllocation));
    allocation->size = allocationSize;
    allocation->freeSpace = 0; // added by FreeRegions
    allocation->usedSpace = 0; // added by UsedRegions
    allocation->memoryLock = SDL_CreateMutex();

    allocator->allocationCount += 1;
    allocator->allocations = SDL_realloc(
        allocator->allocations,
        sizeof(VulkanMemoryAllocation *) * allocator->allocationCount);

    allocator->allocations[allocator->allocationCount - 1] = allocation;

    allocInfo.pNext = NULL;
    allocation->availableForAllocation = 1;

    allocation->usedRegions = SDL_malloc(sizeof(VulkanMemoryUsedRegion *));
    allocation->usedRegionCount = 0;
    allocation->usedRegionCapacity = 1;

    allocation->freeRegions = SDL_malloc(sizeof(VulkanMemoryFreeRegion *));
    allocation->freeRegionCount = 0;
    allocation->freeRegionCapacity = 1;

    allocation->allocator = allocator;

    result = renderer->vkAllocateMemory(
        renderer->logicalDevice,
        &allocInfo,
        NULL,
        &allocation->memory);

    if (result != VK_SUCCESS) {
        // Uh oh, we couldn't allocate, time to clean up
        SDL_free(allocation->freeRegions);

        allocator->allocationCount -= 1;
        allocator->allocations = SDL_realloc(
            allocator->allocations,
            sizeof(VulkanMemoryAllocation *) * allocator->allocationCount);

        SDL_free(allocation);

        return 0;
    }

    // Persistent mapping for host-visible memory
    if (isHostVisible) {
        result = renderer->vkMapMemory(
            renderer->logicalDevice,
            allocation->memory,
            0,
            VK_WHOLE_SIZE,
            0,
            (void **)&allocation->mapPointer);
        CHECK_VULKAN_ERROR_AND_RETURN(result, vkMapMemory, 0);
    } else {
        allocation->mapPointer = NULL;
    }

    VULKAN_INTERNAL_NewMemoryFreeRegion(
        renderer,
        allocation,
        0,
        allocation->size);

    *pMemoryAllocation = allocation;
    return 1;
}

static Uint8 VULKAN_INTERNAL_BindBufferMemory(
    VulkanRenderer *renderer,
    VulkanMemoryUsedRegion *usedRegion,
    VkDeviceSize alignedOffset,
    VkBuffer buffer)
{
    VkResult vulkanResult;

    SDL_LockMutex(usedRegion->allocation->memoryLock);

    vulkanResult = renderer->vkBindBufferMemory(
        renderer->logicalDevice,
        buffer,
        usedRegion->allocation->memory,
        alignedOffset);

    SDL_UnlockMutex(usedRegion->allocation->memoryLock);

    CHECK_VULKAN_ERROR_AND_RETURN(vulkanResult, vkBindBufferMemory, 0);

    return 1;
}

static Uint8 VULKAN_INTERNAL_BindImageMemory(
    VulkanRenderer *renderer,
    VulkanMemoryUsedRegion *usedRegion,
    VkDeviceSize alignedOffset,
    VkImage image)
{
    VkResult vulkanResult;

    SDL_LockMutex(usedRegion->allocation->memoryLock);

    vulkanResult = renderer->vkBindImageMemory(
        renderer->logicalDevice,
        image,
        usedRegion->allocation->memory,
        alignedOffset);

    SDL_UnlockMutex(usedRegion->allocation->memoryLock);

    CHECK_VULKAN_ERROR_AND_RETURN(vulkanResult, vkBindImageMemory, 0);

    return 1;
}

static Uint8 VULKAN_INTERNAL_BindResourceMemory(
    VulkanRenderer *renderer,
    Uint32 memoryTypeIndex,
    VkMemoryRequirements *memoryRequirements,
    VkDeviceSize resourceSize, // may be different from requirements size!
    bool dedicated,            // the entire memory allocation should be used for this resource
    VkBuffer buffer,           // may be VK_NULL_HANDLE
    VkImage image,             // may be VK_NULL_HANDLE
    VulkanMemoryUsedRegion **pMemoryUsedRegion)
{
    VulkanMemoryAllocation *allocation;
    VulkanMemorySubAllocator *allocator;
    VulkanMemoryFreeRegion *region;
    VulkanMemoryFreeRegion *selectedRegion;
    VulkanMemoryUsedRegion *usedRegion;

    VkDeviceSize requiredSize, allocationSize;
    VkDeviceSize alignedOffset = 0;
    VkDeviceSize newRegionSize, newRegionOffset;
    Uint8 isHostVisible, smallAllocation, allocationResult;
    Sint32 i;

    isHostVisible =
        (renderer->memoryProperties.memoryTypes[memoryTypeIndex].propertyFlags &
         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;

    allocator = &renderer->memoryAllocator->subAllocators[memoryTypeIndex];
    requiredSize = memoryRequirements->size;
    smallAllocation = requiredSize <= SMALL_ALLOCATION_THRESHOLD;

    if ((buffer == VK_NULL_HANDLE && image == VK_NULL_HANDLE) ||
        (buffer != VK_NULL_HANDLE && image != VK_NULL_HANDLE)) {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "BindResourceMemory must be given either a VulkanBuffer or a VulkanTexture");
        return 0;
    }

    SDL_LockMutex(renderer->allocatorLock);

    selectedRegion = NULL;

    if (dedicated) {
        // Force an allocation
        allocationSize = requiredSize;
    } else {
        // Search for a suitable existing free region
        for (i = allocator->sortedFreeRegionCount - 1; i >= 0; i -= 1) {
            region = allocator->sortedFreeRegions[i];

            if (smallAllocation && region->allocation->size != SMALL_ALLOCATION_SIZE) {
                // region is not in a small allocation
                continue;
            }

            if (!smallAllocation && region->allocation->size == SMALL_ALLOCATION_SIZE) {
                // allocation is not small and current region is in a small allocation
                continue;
            }

            alignedOffset = VULKAN_INTERNAL_NextHighestAlignment(
                region->offset,
                memoryRequirements->alignment);

            if (alignedOffset + requiredSize <= region->offset + region->size) {
                selectedRegion = region;
                break;
            }
        }

        if (selectedRegion != NULL) {
            region = selectedRegion;
            allocation = region->allocation;

            usedRegion = VULKAN_INTERNAL_NewMemoryUsedRegion(
                renderer,
                allocation,
                region->offset,
                requiredSize + (alignedOffset - region->offset),
                alignedOffset,
                resourceSize,
                memoryRequirements->alignment);

            usedRegion->isBuffer = buffer != VK_NULL_HANDLE;

            newRegionSize = region->size - ((alignedOffset - region->offset) + requiredSize);
            newRegionOffset = alignedOffset + requiredSize;

            // remove and add modified region to re-sort
            VULKAN_INTERNAL_RemoveMemoryFreeRegion(renderer, region);

            // if size is 0, no need to re-insert
            if (newRegionSize != 0) {
                VULKAN_INTERNAL_NewMemoryFreeRegion(
                    renderer,
                    allocation,
                    newRegionOffset,
                    newRegionSize);
            }

            SDL_UnlockMutex(renderer->allocatorLock);

            if (buffer != VK_NULL_HANDLE) {
                if (!VULKAN_INTERNAL_BindBufferMemory(
                        renderer,
                        usedRegion,
                        alignedOffset,
                        buffer)) {
                    VULKAN_INTERNAL_RemoveMemoryUsedRegion(
                        renderer,
                        usedRegion);

                    return 0;
                }
            } else if (image != VK_NULL_HANDLE) {
                if (!VULKAN_INTERNAL_BindImageMemory(
                        renderer,
                        usedRegion,
                        alignedOffset,
                        image)) {
                    VULKAN_INTERNAL_RemoveMemoryUsedRegion(
                        renderer,
                        usedRegion);

                    return 0;
                }
            }

            *pMemoryUsedRegion = usedRegion;
            return 1;
        }

        // No suitable free regions exist, allocate a new memory region
        if (
            renderer->allocationsToDefragCount == 0 &&
            !renderer->defragInProgress) {
            // Mark currently fragmented allocations for defrag
            VULKAN_INTERNAL_MarkAllocationsForDefrag(renderer);
        }

        if (requiredSize > SMALL_ALLOCATION_THRESHOLD) {
            // allocate a page of required size aligned to LARGE_ALLOCATION_INCREMENT increments
            allocationSize =
                VULKAN_INTERNAL_NextHighestAlignment(requiredSize, LARGE_ALLOCATION_INCREMENT);
        } else {
            allocationSize = SMALL_ALLOCATION_SIZE;
        }
    }

    allocationResult = VULKAN_INTERNAL_AllocateMemory(
        renderer,
        memoryTypeIndex,
        allocationSize,
        isHostVisible,
        &allocation);

    // Uh oh, we're out of memory
    if (allocationResult == 0) {
        SDL_UnlockMutex(renderer->allocatorLock);

        // Responsibility of the caller to handle being out of memory
        return 2;
    }

    usedRegion = VULKAN_INTERNAL_NewMemoryUsedRegion(
        renderer,
        allocation,
        0,
        requiredSize,
        0,
        resourceSize,
        memoryRequirements->alignment);

    usedRegion->isBuffer = buffer != VK_NULL_HANDLE;

    region = allocation->freeRegions[0];

    newRegionOffset = region->offset + requiredSize;
    newRegionSize = region->size - requiredSize;

    VULKAN_INTERNAL_RemoveMemoryFreeRegion(renderer, region);

    if (newRegionSize != 0) {
        VULKAN_INTERNAL_NewMemoryFreeRegion(
            renderer,
            allocation,
            newRegionOffset,
            newRegionSize);
    }

    SDL_UnlockMutex(renderer->allocatorLock);

    if (buffer != VK_NULL_HANDLE) {
        if (!VULKAN_INTERNAL_BindBufferMemory(
                renderer,
                usedRegion,
                0,
                buffer)) {
            VULKAN_INTERNAL_RemoveMemoryUsedRegion(
                renderer,
                usedRegion);

            return 0;
        }
    } else if (image != VK_NULL_HANDLE) {
        if (!VULKAN_INTERNAL_BindImageMemory(
                renderer,
                usedRegion,
                0,
                image)) {
            VULKAN_INTERNAL_RemoveMemoryUsedRegion(
                renderer,
                usedRegion);

            return 0;
        }
    }

    *pMemoryUsedRegion = usedRegion;
    return 1;
}

static Uint8 VULKAN_INTERNAL_BindMemoryForImage(
    VulkanRenderer *renderer,
    VkImage image,
    VulkanMemoryUsedRegion **usedRegion)
{
    Uint8 bindResult = 0;
    Uint32 memoryTypeCount = 0;
    Uint32 *memoryTypesToTry = NULL;
    Uint32 selectedMemoryTypeIndex = 0;
    Uint32 i;
    VkMemoryPropertyFlags preferredMemoryPropertyFlags;
    VkMemoryRequirements memoryRequirements;

    /* Vulkan memory types have several memory properties.
     *
     * Unlike buffers, images are always optimally stored device-local,
     * so that is the only property we prefer here.
     *
     * If memory is constrained, it is fine for the texture to not
     * be device-local.
     */
    preferredMemoryPropertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    memoryTypesToTry = VULKAN_INTERNAL_FindBestImageMemoryTypes(
        renderer,
        image,
        preferredMemoryPropertyFlags,
        &memoryRequirements,
        &memoryTypeCount);

    for (i = 0; i < memoryTypeCount; i += 1) {
        bindResult = VULKAN_INTERNAL_BindResourceMemory(
            renderer,
            memoryTypesToTry[i],
            &memoryRequirements,
            memoryRequirements.size,
            false,
            VK_NULL_HANDLE,
            image,
            usedRegion);

        if (bindResult == 1) {
            selectedMemoryTypeIndex = memoryTypesToTry[i];
            break;
        }
    }

    SDL_free(memoryTypesToTry);

    // Check for warnings on success
    if (bindResult == 1) {
        if (!renderer->outOfDeviceLocalMemoryWarning) {
            if ((renderer->memoryProperties.memoryTypes[selectedMemoryTypeIndex].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) == 0) {
                SDL_LogWarn(SDL_LOG_CATEGORY_GPU, "Out of device-local memory, allocating textures on host-local memory!");
                renderer->outOfDeviceLocalMemoryWarning = 1;
            }
        }
    }

    return bindResult;
}

static Uint8 VULKAN_INTERNAL_BindMemoryForBuffer(
    VulkanRenderer *renderer,
    VkBuffer buffer,
    VkDeviceSize size,
    VulkanBufferType type,
    bool dedicated,
    VulkanMemoryUsedRegion **usedRegion)
{
    Uint8 bindResult = 0;
    Uint32 memoryTypeCount = 0;
    Uint32 *memoryTypesToTry = NULL;
    Uint32 selectedMemoryTypeIndex = 0;
    Uint32 i;
    VkMemoryPropertyFlags requiredMemoryPropertyFlags = 0;
    VkMemoryPropertyFlags preferredMemoryPropertyFlags = 0;
    VkMemoryPropertyFlags tolerableMemoryPropertyFlags = 0;
    VkMemoryRequirements memoryRequirements;

    /* Buffers need to be optimally bound to a memory type
     * based on their use case and the architecture of the system.
     *
     * It is important to understand the distinction between device and host.
     *
     * On a traditional high-performance desktop computer,
     * the "device" would be the GPU, and the "host" would be the CPU.
     * Memory being copied between these two must cross the PCI bus.
     * On these systems we have to be concerned about bandwidth limitations
     * and causing memory stalls, so we have taken a great deal of care
     * to structure this API to guide the client towards optimal usage.
     *
     * Other kinds of devices do not necessarily have this distinction.
     * On an iPhone or Nintendo Switch, all memory is accessible both to the
     * GPU and the CPU at all times. These kinds of systems are known as
     * UMA, or Unified Memory Architecture. A desktop computer using the
     * CPU's integrated graphics can also be thought of as UMA.
     *
     * Vulkan memory types have several memory properties.
     * The relevant memory properties are as follows:
     *
     * DEVICE_LOCAL:
     *   This memory is on-device and most efficient for device access.
     *   On UMA systems all memory is device-local.
     *   If memory is not device-local, then it is host-local.
     *
     * HOST_VISIBLE:
     *   This memory can be mapped for host access, meaning we can obtain
     *   a pointer to directly access the memory.
     *
     * HOST_COHERENT:
     *   Host-coherent memory does not require cache management operations
     *   when mapped, so we always set this alongside HOST_VISIBLE
     *   to avoid extra record keeping.
     *
     * HOST_CACHED:
     *   Host-cached memory is faster to access than uncached memory
     *   but memory of this type might not always be available.
     *
     * GPU buffers, like vertex buffers, indirect buffers, etc
     * are optimally stored in device-local memory.
     * However, if device-local memory is low, these buffers
     * can be accessed from host-local memory with a performance penalty.
     *
     * Uniform buffers must be host-visible and coherent because
     * the client uses them to quickly push small amounts of data.
     * We prefer uniform buffers to also be device-local because
     * they are accessed by shaders, but the amount of memory
     * that is both device-local and host-visible
     * is often constrained, particularly on low-end devices.
     *
     * Transfer buffers must be host-visible and coherent because
     * the client uses them to stage data to be transferred
     * to device-local memory, or to read back data transferred
     * from the device. We prefer the cache bit for performance
     * but it isn't strictly necessary. We tolerate device-local
     * memory in this situation because, as mentioned above,
     * on certain devices all memory is device-local, and even
     * though the transfer isn't strictly necessary it is still
     * useful for correctly timelining data.
     */
    if (type == VULKAN_BUFFER_TYPE_GPU) {
        preferredMemoryPropertyFlags |=
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    } else if (type == VULKAN_BUFFER_TYPE_UNIFORM) {
        requiredMemoryPropertyFlags |=
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

        preferredMemoryPropertyFlags |=
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    } else if (type == VULKAN_BUFFER_TYPE_TRANSFER) {
        requiredMemoryPropertyFlags |=
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

        preferredMemoryPropertyFlags |=
            VK_MEMORY_PROPERTY_HOST_CACHED_BIT;

        tolerableMemoryPropertyFlags |=
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    } else {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Unrecognized buffer type!");
        return 0;
    }

    memoryTypesToTry = VULKAN_INTERNAL_FindBestBufferMemoryTypes(
        renderer,
        buffer,
        requiredMemoryPropertyFlags,
        preferredMemoryPropertyFlags,
        tolerableMemoryPropertyFlags,
        &memoryRequirements,
        &memoryTypeCount);

    for (i = 0; i < memoryTypeCount; i += 1) {
        bindResult = VULKAN_INTERNAL_BindResourceMemory(
            renderer,
            memoryTypesToTry[i],
            &memoryRequirements,
            size,
            dedicated,
            buffer,
            VK_NULL_HANDLE,
            usedRegion);

        if (bindResult == 1) {
            selectedMemoryTypeIndex = memoryTypesToTry[i];
            break;
        }
    }

    SDL_free(memoryTypesToTry);

    // Check for warnings on success
    if (bindResult == 1) {
        if (type == VULKAN_BUFFER_TYPE_GPU) {
            if (!renderer->outOfDeviceLocalMemoryWarning) {
                if ((renderer->memoryProperties.memoryTypes[selectedMemoryTypeIndex].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) == 0) {
                    SDL_LogWarn(SDL_LOG_CATEGORY_GPU, "Out of device-local memory, allocating buffers on host-local memory, expect degraded performance!");
                    renderer->outOfDeviceLocalMemoryWarning = 1;
                }
            }
        } else if (type == VULKAN_BUFFER_TYPE_UNIFORM) {
            if (!renderer->outofBARMemoryWarning) {
                if ((renderer->memoryProperties.memoryTypes[selectedMemoryTypeIndex].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) == 0) {
                    SDL_LogWarn(SDL_LOG_CATEGORY_GPU, "Out of BAR memory, allocating uniform buffers on host-local memory, expect degraded performance!");
                    renderer->outofBARMemoryWarning = 1;
                }
            }
        } else if (type == VULKAN_BUFFER_TYPE_TRANSFER) {
            if (!renderer->integratedMemoryNotification) {
                if ((renderer->memoryProperties.memoryTypes[selectedMemoryTypeIndex].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) == VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
                    SDL_LogInfo(SDL_LOG_CATEGORY_GPU, "Integrated memory detected, allocating TransferBuffers on device-local memory!");
                    renderer->integratedMemoryNotification = 1;
                }
            }
        }
    }

    return bindResult;
}

// Resource tracking

#define TRACK_RESOURCE(resource, type, array, count, capacity)  \
    for (Sint32 i = commandBuffer->count - 1; i >= 0; i -= 1) { \
        if (commandBuffer->array[i] == resource) {              \
            return;                                             \
        }                                                       \
    }                                                           \
                                                                \
    if (commandBuffer->count == commandBuffer->capacity) {      \
        commandBuffer->capacity += 1;                           \
        commandBuffer->array = SDL_realloc(                     \
            commandBuffer->array,                               \
            commandBuffer->capacity * sizeof(type));            \
    }                                                           \
    commandBuffer->array[commandBuffer->count] = resource;      \
    commandBuffer->count += 1;                                  \
    SDL_AtomicIncRef(&resource->referenceCount);

static void VULKAN_INTERNAL_TrackBuffer(
    VulkanCommandBuffer *commandBuffer,
    VulkanBuffer *buffer)
{
    TRACK_RESOURCE(
        buffer,
        VulkanBuffer *,
        usedBuffers,
        usedBufferCount,
        usedBufferCapacity)
}

static void VULKAN_INTERNAL_TrackTexture(
    VulkanCommandBuffer *commandBuffer,
    VulkanTexture *texture)
{
    TRACK_RESOURCE(
        texture,
        VulkanTexture *,
        usedTextures,
        usedTextureCount,
        usedTextureCapacity)
}

static void VULKAN_INTERNAL_TrackSampler(
    VulkanCommandBuffer *commandBuffer,
    VulkanSampler *sampler)
{
    TRACK_RESOURCE(
        sampler,
        VulkanSampler *,
        usedSamplers,
        usedSamplerCount,
        usedSamplerCapacity)
}

static void VULKAN_INTERNAL_TrackGraphicsPipeline(
    VulkanCommandBuffer *commandBuffer,
    VulkanGraphicsPipeline *graphicsPipeline)
{
    TRACK_RESOURCE(
        graphicsPipeline,
        VulkanGraphicsPipeline *,
        usedGraphicsPipelines,
        usedGraphicsPipelineCount,
        usedGraphicsPipelineCapacity)
}

static void VULKAN_INTERNAL_TrackComputePipeline(
    VulkanCommandBuffer *commandBuffer,
    VulkanComputePipeline *computePipeline)
{
    TRACK_RESOURCE(
        computePipeline,
        VulkanComputePipeline *,
        usedComputePipelines,
        usedComputePipelineCount,
        usedComputePipelineCapacity)
}

static void VULKAN_INTERNAL_TrackFramebuffer(
    VulkanCommandBuffer *commandBuffer,
    VulkanFramebuffer *framebuffer)
{
    TRACK_RESOURCE(
        framebuffer,
        VulkanFramebuffer *,
        usedFramebuffers,
        usedFramebufferCount,
        usedFramebufferCapacity);
}

static void VULKAN_INTERNAL_TrackUniformBuffer(
    VulkanCommandBuffer *commandBuffer,
    VulkanUniformBuffer *uniformBuffer)
{
    for (Sint32 i = commandBuffer->usedUniformBufferCount - 1; i >= 0; i -= 1) {
        if (commandBuffer->usedUniformBuffers[i] == uniformBuffer) {
            return;
        }
    }

    if (commandBuffer->usedUniformBufferCount == commandBuffer->usedUniformBufferCapacity) {
        commandBuffer->usedUniformBufferCapacity += 1;
        commandBuffer->usedUniformBuffers = SDL_realloc(
            commandBuffer->usedUniformBuffers,
            commandBuffer->usedUniformBufferCapacity * sizeof(VulkanUniformBuffer *));
    }
    commandBuffer->usedUniformBuffers[commandBuffer->usedUniformBufferCount] = uniformBuffer;
    commandBuffer->usedUniformBufferCount += 1;

    VULKAN_INTERNAL_TrackBuffer(
        commandBuffer,
        uniformBuffer->buffer);
}

#undef TRACK_RESOURCE

// Memory Barriers

/*
 * In Vulkan, we must manually synchronize operations that write to resources on the GPU
 * so that read-after-write, write-after-read, and write-after-write hazards do not occur.
 * Additionally, textures are required to be in specific layouts for specific use cases.
 * Both of these tasks are accomplished with vkCmdPipelineBarrier.
 *
 * To insert the correct barriers, we keep track of "usage modes" for buffers and textures.
 * These indicate the current usage of that resource on the command buffer.
 * The transition from one usage mode to another indicates how the barrier should be constructed.
 *
 * Pipeline barriers cannot be inserted during a render pass, but they can be inserted
 * during a compute or copy pass.
 *
 * This means that the "default" usage mode of any given resource should be that it should be
 * ready for a graphics-read operation, because we cannot barrier during a render pass.
 * In the case where a resource is only used in compute, its default usage mode can be compute-read.
 * This strategy allows us to avoid expensive record keeping of command buffer/resource usage mode pairs,
 * and it fully covers synchronization between all combinations of stages.
 *
 * In Upload and Copy functions, we transition the resource immediately before and after the copy command.
 *
 * When binding a resource for compute, we transition when the Bind functions are called.
 * If a bind slot containing a resource is overwritten, we transition the resource in that slot back to its default.
 * When EndComputePass is called we transition all bound resources back to their default state.
 *
 * When binding a texture as a render pass attachment, we transition the resource on BeginRenderPass
 * and transition it back to its default on EndRenderPass.
 *
 * This strategy imposes certain limitations on resource usage flags.
 * For example, a texture cannot have both the SAMPLER and GRAPHICS_STORAGE usage flags,
 * because then it is impossible for the backend to infer which default usage mode the texture should use.
 *
 * Sync hazards can be detected by setting VK_KHRONOS_VALIDATION_VALIDATE_SYNC=1 when using validation layers.
 */

static void VULKAN_INTERNAL_BufferMemoryBarrier(
    VulkanRenderer *renderer,
    VulkanCommandBuffer *commandBuffer,
    VulkanBufferUsageMode sourceUsageMode,
    VulkanBufferUsageMode destinationUsageMode,
    VulkanBuffer *buffer)
{
    VkPipelineStageFlags srcStages = 0;
    VkPipelineStageFlags dstStages = 0;
    VkBufferMemoryBarrier memoryBarrier;

    memoryBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    memoryBarrier.pNext = NULL;
    memoryBarrier.srcAccessMask = 0;
    memoryBarrier.dstAccessMask = 0;
    memoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    memoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    memoryBarrier.buffer = buffer->buffer;
    memoryBarrier.offset = 0;
    memoryBarrier.size = buffer->size;

    if (sourceUsageMode == VULKAN_BUFFER_USAGE_MODE_COPY_SOURCE) {
        srcStages = VK_PIPELINE_STAGE_TRANSFER_BIT;
        memoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    } else if (sourceUsageMode == VULKAN_BUFFER_USAGE_MODE_COPY_DESTINATION) {
        srcStages = VK_PIPELINE_STAGE_TRANSFER_BIT;
        memoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    } else if (sourceUsageMode == VULKAN_BUFFER_USAGE_MODE_VERTEX_READ) {
        srcStages = VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
        memoryBarrier.srcAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
    } else if (sourceUsageMode == VULKAN_BUFFER_USAGE_MODE_INDEX_READ) {
        srcStages = VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
        memoryBarrier.srcAccessMask = VK_ACCESS_INDEX_READ_BIT;
    } else if (sourceUsageMode == VULKAN_BUFFER_USAGE_MODE_INDIRECT) {
        srcStages = VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;
        memoryBarrier.srcAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    } else if (sourceUsageMode == VULKAN_BUFFER_USAGE_MODE_GRAPHICS_STORAGE_READ) {
        srcStages = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    } else if (sourceUsageMode == VULKAN_BUFFER_USAGE_MODE_COMPUTE_STORAGE_READ) {
        srcStages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    } else if (sourceUsageMode == VULKAN_BUFFER_USAGE_MODE_COMPUTE_STORAGE_READ_WRITE) {
        srcStages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    } else {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Unrecognized buffer source barrier type!");
        return;
    }

    if (destinationUsageMode == VULKAN_BUFFER_USAGE_MODE_COPY_SOURCE) {
        dstStages = VK_PIPELINE_STAGE_TRANSFER_BIT;
        memoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    } else if (destinationUsageMode == VULKAN_BUFFER_USAGE_MODE_COPY_DESTINATION) {
        dstStages = VK_PIPELINE_STAGE_TRANSFER_BIT;
        memoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    } else if (destinationUsageMode == VULKAN_BUFFER_USAGE_MODE_VERTEX_READ) {
        dstStages = VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
        memoryBarrier.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
    } else if (destinationUsageMode == VULKAN_BUFFER_USAGE_MODE_INDEX_READ) {
        dstStages = VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
        memoryBarrier.dstAccessMask = VK_ACCESS_INDEX_READ_BIT;
    } else if (destinationUsageMode == VULKAN_BUFFER_USAGE_MODE_INDIRECT) {
        dstStages = VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;
        memoryBarrier.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    } else if (destinationUsageMode == VULKAN_BUFFER_USAGE_MODE_GRAPHICS_STORAGE_READ) {
        dstStages = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    } else if (destinationUsageMode == VULKAN_BUFFER_USAGE_MODE_COMPUTE_STORAGE_READ) {
        dstStages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    } else if (destinationUsageMode == VULKAN_BUFFER_USAGE_MODE_COMPUTE_STORAGE_READ_WRITE) {
        dstStages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    } else {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Unrecognized buffer destination barrier type!");
        return;
    }

    renderer->vkCmdPipelineBarrier(
        commandBuffer->commandBuffer,
        srcStages,
        dstStages,
        0,
        0,
        NULL,
        1,
        &memoryBarrier,
        0,
        NULL);

    buffer->transitioned = true;
}

static void VULKAN_INTERNAL_TextureSubresourceMemoryBarrier(
    VulkanRenderer *renderer,
    VulkanCommandBuffer *commandBuffer,
    VulkanTextureUsageMode sourceUsageMode,
    VulkanTextureUsageMode destinationUsageMode,
    VulkanTextureSubresource *textureSubresource)
{
    VkPipelineStageFlags srcStages = 0;
    VkPipelineStageFlags dstStages = 0;
    VkImageMemoryBarrier memoryBarrier;

    memoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    memoryBarrier.pNext = NULL;
    memoryBarrier.srcAccessMask = 0;
    memoryBarrier.dstAccessMask = 0;
    memoryBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    memoryBarrier.newLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    memoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    memoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    memoryBarrier.image = textureSubresource->parent->image;
    memoryBarrier.subresourceRange.aspectMask = textureSubresource->parent->aspectFlags;
    memoryBarrier.subresourceRange.baseArrayLayer = textureSubresource->layer;
    memoryBarrier.subresourceRange.layerCount = 1;
    memoryBarrier.subresourceRange.baseMipLevel = textureSubresource->level;
    memoryBarrier.subresourceRange.levelCount = 1;

    if (sourceUsageMode == VULKAN_TEXTURE_USAGE_MODE_UNINITIALIZED) {
        srcStages = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        memoryBarrier.srcAccessMask = 0;
        memoryBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    } else if (sourceUsageMode == VULKAN_TEXTURE_USAGE_MODE_COPY_SOURCE) {
        srcStages = VK_PIPELINE_STAGE_TRANSFER_BIT;
        memoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        memoryBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    } else if (sourceUsageMode == VULKAN_TEXTURE_USAGE_MODE_COPY_DESTINATION) {
        srcStages = VK_PIPELINE_STAGE_TRANSFER_BIT;
        memoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        memoryBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    } else if (sourceUsageMode == VULKAN_TEXTURE_USAGE_MODE_SAMPLER) {
        srcStages = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        memoryBarrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    } else if (sourceUsageMode == VULKAN_TEXTURE_USAGE_MODE_GRAPHICS_STORAGE_READ) {
        srcStages = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        memoryBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    } else if (sourceUsageMode == VULKAN_TEXTURE_USAGE_MODE_COMPUTE_STORAGE_READ) {
        srcStages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        memoryBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    } else if (sourceUsageMode == VULKAN_TEXTURE_USAGE_MODE_COMPUTE_STORAGE_READ_WRITE) {
        srcStages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        memoryBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    } else if (sourceUsageMode == VULKAN_TEXTURE_USAGE_MODE_COLOR_ATTACHMENT) {
        srcStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        memoryBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        memoryBarrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    } else if (sourceUsageMode == VULKAN_TEXTURE_USAGE_MODE_DEPTH_STENCIL_ATTACHMENT) {
        srcStages = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        memoryBarrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        memoryBarrier.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    } else {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Unrecognized texture source barrier type!");
        return;
    }

    if (destinationUsageMode == VULKAN_TEXTURE_USAGE_MODE_COPY_SOURCE) {
        dstStages = VK_PIPELINE_STAGE_TRANSFER_BIT;
        memoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        memoryBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    } else if (destinationUsageMode == VULKAN_TEXTURE_USAGE_MODE_COPY_DESTINATION) {
        dstStages = VK_PIPELINE_STAGE_TRANSFER_BIT;
        memoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        memoryBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    } else if (destinationUsageMode == VULKAN_TEXTURE_USAGE_MODE_SAMPLER) {
        dstStages = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        memoryBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    } else if (destinationUsageMode == VULKAN_TEXTURE_USAGE_MODE_GRAPHICS_STORAGE_READ) {
        dstStages = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        memoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    } else if (destinationUsageMode == VULKAN_TEXTURE_USAGE_MODE_COMPUTE_STORAGE_READ) {
        dstStages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        memoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    } else if (destinationUsageMode == VULKAN_TEXTURE_USAGE_MODE_COMPUTE_STORAGE_READ_WRITE) {
        dstStages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        memoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    } else if (destinationUsageMode == VULKAN_TEXTURE_USAGE_MODE_COLOR_ATTACHMENT) {
        dstStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        memoryBarrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        memoryBarrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    } else if (destinationUsageMode == VULKAN_TEXTURE_USAGE_MODE_DEPTH_STENCIL_ATTACHMENT) {
        dstStages = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        memoryBarrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        memoryBarrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    } else if (destinationUsageMode == VULKAN_TEXTURE_USAGE_MODE_PRESENT) {
        dstStages = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        memoryBarrier.dstAccessMask = 0;
        memoryBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    } else {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Unrecognized texture destination barrier type!");
        return;
    }

    renderer->vkCmdPipelineBarrier(
        commandBuffer->commandBuffer,
        srcStages,
        dstStages,
        0,
        0,
        NULL,
        0,
        NULL,
        1,
        &memoryBarrier);
}

static VulkanBufferUsageMode VULKAN_INTERNAL_DefaultBufferUsageMode(
    VulkanBuffer *buffer)
{
    // NOTE: order matters here!

    if (buffer->usage & SDL_GPU_BUFFERUSAGE_VERTEX) {
        return VULKAN_BUFFER_USAGE_MODE_VERTEX_READ;
    } else if (buffer->usage & SDL_GPU_BUFFERUSAGE_INDEX) {
        return VULKAN_BUFFER_USAGE_MODE_INDEX_READ;
    } else if (buffer->usage & SDL_GPU_BUFFERUSAGE_INDIRECT) {
        return VULKAN_BUFFER_USAGE_MODE_INDIRECT;
    } else if (buffer->usage & SDL_GPU_BUFFERUSAGE_GRAPHICS_STORAGE_READ) {
        return VULKAN_BUFFER_USAGE_MODE_GRAPHICS_STORAGE_READ;
    } else if (buffer->usage & SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ) {
        return VULKAN_BUFFER_USAGE_MODE_COMPUTE_STORAGE_READ;
    } else if (buffer->usage & SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE) {
        return VULKAN_BUFFER_USAGE_MODE_COMPUTE_STORAGE_READ_WRITE;
    } else {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Buffer has no default usage mode!");
        return VULKAN_BUFFER_USAGE_MODE_VERTEX_READ;
    }
}

static VulkanTextureUsageMode VULKAN_INTERNAL_DefaultTextureUsageMode(
    VulkanTexture *texture)
{
    // NOTE: order matters here!
    // NOTE: graphics storage bits and sampler bit are mutually exclusive!

    if (texture->usage & SDL_GPU_TEXTUREUSAGE_SAMPLER) {
        return VULKAN_TEXTURE_USAGE_MODE_SAMPLER;
    } else if (texture->usage & SDL_GPU_TEXTUREUSAGE_GRAPHICS_STORAGE_READ) {
        return VULKAN_TEXTURE_USAGE_MODE_GRAPHICS_STORAGE_READ;
    } else if (texture->usage & SDL_GPU_TEXTUREUSAGE_COLOR_TARGET) {
        return VULKAN_TEXTURE_USAGE_MODE_COLOR_ATTACHMENT;
    } else if (texture->usage & SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET) {
        return VULKAN_TEXTURE_USAGE_MODE_DEPTH_STENCIL_ATTACHMENT;
    } else if (texture->usage & SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_READ) {
        return VULKAN_TEXTURE_USAGE_MODE_COMPUTE_STORAGE_READ;
    } else if (texture->usage & SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE) {
        return VULKAN_TEXTURE_USAGE_MODE_COMPUTE_STORAGE_READ_WRITE;
    } else if (texture->usage & SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_SIMULTANEOUS_READ_WRITE) {
        return VULKAN_TEXTURE_USAGE_MODE_COMPUTE_STORAGE_READ_WRITE;
    } else {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Texture has no default usage mode!");
        return VULKAN_TEXTURE_USAGE_MODE_SAMPLER;
    }
}

static void VULKAN_INTERNAL_BufferTransitionFromDefaultUsage(
    VulkanRenderer *renderer,
    VulkanCommandBuffer *commandBuffer,
    VulkanBufferUsageMode destinationUsageMode,
    VulkanBuffer *buffer)
{
    VULKAN_INTERNAL_BufferMemoryBarrier(
        renderer,
        commandBuffer,
        VULKAN_INTERNAL_DefaultBufferUsageMode(buffer),
        destinationUsageMode,
        buffer);
}

static void VULKAN_INTERNAL_BufferTransitionToDefaultUsage(
    VulkanRenderer *renderer,
    VulkanCommandBuffer *commandBuffer,
    VulkanBufferUsageMode sourceUsageMode,
    VulkanBuffer *buffer)
{
    VULKAN_INTERNAL_BufferMemoryBarrier(
        renderer,
        commandBuffer,
        sourceUsageMode,
        VULKAN_INTERNAL_DefaultBufferUsageMode(buffer),
        buffer);
}

static void VULKAN_INTERNAL_TextureSubresourceTransitionFromDefaultUsage(
    VulkanRenderer *renderer,
    VulkanCommandBuffer *commandBuffer,
    VulkanTextureUsageMode destinationUsageMode,
    VulkanTextureSubresource *textureSubresource)
{
    VULKAN_INTERNAL_TextureSubresourceMemoryBarrier(
        renderer,
        commandBuffer,
        VULKAN_INTERNAL_DefaultTextureUsageMode(textureSubresource->parent),
        destinationUsageMode,
        textureSubresource);
}

static void VULKAN_INTERNAL_TextureTransitionFromDefaultUsage(
    VulkanRenderer *renderer,
    VulkanCommandBuffer *commandBuffer,
    VulkanTextureUsageMode destinationUsageMode,
    VulkanTexture *texture)
{
    for (Uint32 i = 0; i < texture->subresourceCount; i += 1) {
        VULKAN_INTERNAL_TextureSubresourceTransitionFromDefaultUsage(
            renderer,
            commandBuffer,
            destinationUsageMode,
            &texture->subresources[i]);
    }
}

static void VULKAN_INTERNAL_TextureSubresourceTransitionToDefaultUsage(
    VulkanRenderer *renderer,
    VulkanCommandBuffer *commandBuffer,
    VulkanTextureUsageMode sourceUsageMode,
    VulkanTextureSubresource *textureSubresource)
{
    VULKAN_INTERNAL_TextureSubresourceMemoryBarrier(
        renderer,
        commandBuffer,
        sourceUsageMode,
        VULKAN_INTERNAL_DefaultTextureUsageMode(textureSubresource->parent),
        textureSubresource);
}

static void VULKAN_INTERNAL_TextureTransitionToDefaultUsage(
    VulkanRenderer *renderer,
    VulkanCommandBuffer *commandBuffer,
    VulkanTextureUsageMode sourceUsageMode,
    VulkanTexture *texture)
{
    // FIXME: could optimize this barrier
    for (Uint32 i = 0; i < texture->subresourceCount; i += 1) {
        VULKAN_INTERNAL_TextureSubresourceTransitionToDefaultUsage(
            renderer,
            commandBuffer,
            sourceUsageMode,
            &texture->subresources[i]);
    }
}

// Resource Disposal

static void VULKAN_INTERNAL_ReleaseFramebuffer(
    VulkanRenderer *renderer,
    VulkanFramebuffer *framebuffer)
{
    SDL_LockMutex(renderer->disposeLock);

    EXPAND_ARRAY_IF_NEEDED(
        renderer->framebuffersToDestroy,
        VulkanFramebuffer *,
        renderer->framebuffersToDestroyCount + 1,
        renderer->framebuffersToDestroyCapacity,
        renderer->framebuffersToDestroyCapacity * 2);

    renderer->framebuffersToDestroy[renderer->framebuffersToDestroyCount] = framebuffer;
    renderer->framebuffersToDestroyCount += 1;

    SDL_UnlockMutex(renderer->disposeLock);
}

static void VULKAN_INTERNAL_DestroyFramebuffer(
    VulkanRenderer *renderer,
    VulkanFramebuffer *framebuffer)
{
    renderer->vkDestroyFramebuffer(
        renderer->logicalDevice,
        framebuffer->framebuffer,
        NULL);

    SDL_free(framebuffer);
}

typedef struct CheckOneFramebufferForRemovalData
{
    Uint32 keysToRemoveCapacity;
    Uint32 keysToRemoveCount;
    FramebufferHashTableKey **keysToRemove;
    VkImageView view;
} CheckOneFramebufferForRemovalData;

static bool SDLCALL CheckOneFramebufferForRemoval(void *userdata, const SDL_HashTable *table, const void *vkey, const void *vvalue)
{
    CheckOneFramebufferForRemovalData *data = (CheckOneFramebufferForRemovalData *) userdata;
    FramebufferHashTableKey *key = (FramebufferHashTableKey *) vkey;
    VkImageView view = data->view;
    bool remove = false;

    for (Uint32 i = 0; i < key->numColorTargets; i += 1) {
        if (key->colorAttachmentViews[i] == view) {
            remove = true;
        }
    }
    for (Uint32 i = 0; i < key->numResolveAttachments; i += 1) {
        if (key->resolveAttachmentViews[i] == view) {
            remove = true;
        }
    }
    if (key->depthStencilAttachmentView == view) {
        remove = true;
    }

    if (remove) {
        if (data->keysToRemoveCount == data->keysToRemoveCapacity) {
            data->keysToRemoveCapacity *= 2;
            void *ptr = SDL_realloc(data->keysToRemove, data->keysToRemoveCapacity * sizeof(FramebufferHashTableKey *));
            if (!ptr) {
                return false;  // ugh, stop iterating. We're in trouble.
            }
            data->keysToRemove = (FramebufferHashTableKey **) ptr;
        }
        data->keysToRemove[data->keysToRemoveCount] = key;
        data->keysToRemoveCount++;
    }

    return true;  // keep iterating.
}

static void VULKAN_INTERNAL_RemoveFramebuffersContainingView(
    VulkanRenderer *renderer,
    VkImageView view)
{
    // Can't remove while iterating!

    CheckOneFramebufferForRemovalData data = { 8, 0, NULL, view };
    data.keysToRemove = (FramebufferHashTableKey **) SDL_malloc(data.keysToRemoveCapacity * sizeof(FramebufferHashTableKey *));
    if (!data.keysToRemove) {
        return;  // uhoh.
    }

    SDL_LockMutex(renderer->framebufferFetchLock);

    SDL_IterateHashTable(renderer->framebufferHashTable, CheckOneFramebufferForRemoval, &data);

    for (Uint32 i = 0; i < data.keysToRemoveCount; i += 1) {
        SDL_RemoveFromHashTable(renderer->framebufferHashTable, (void *)data.keysToRemove[i]);
    }

    SDL_UnlockMutex(renderer->framebufferFetchLock);

    SDL_free(data.keysToRemove);
}

static void VULKAN_INTERNAL_DestroyTexture(
    VulkanRenderer *renderer,
    VulkanTexture *texture)
{
    // Clean up subresources
    for (Uint32 subresourceIndex = 0; subresourceIndex < texture->subresourceCount; subresourceIndex += 1) {
        if (texture->subresources[subresourceIndex].renderTargetViews != NULL) {
            for (Uint32 depthIndex = 0; depthIndex < texture->depth; depthIndex += 1) {
                VULKAN_INTERNAL_RemoveFramebuffersContainingView(
                    renderer,
                    texture->subresources[subresourceIndex].renderTargetViews[depthIndex]);
            }

            for (Uint32 depthIndex = 0; depthIndex < texture->depth; depthIndex += 1) {
                renderer->vkDestroyImageView(
                    renderer->logicalDevice,
                    texture->subresources[subresourceIndex].renderTargetViews[depthIndex],
                    NULL);
            }
            SDL_free(texture->subresources[subresourceIndex].renderTargetViews);
        }

        if (texture->subresources[subresourceIndex].computeWriteView != VK_NULL_HANDLE) {
            renderer->vkDestroyImageView(
                renderer->logicalDevice,
                texture->subresources[subresourceIndex].computeWriteView,
                NULL);
        }

        if (texture->subresources[subresourceIndex].depthStencilView != VK_NULL_HANDLE) {
            VULKAN_INTERNAL_RemoveFramebuffersContainingView(
                renderer,
                texture->subresources[subresourceIndex].depthStencilView);
            renderer->vkDestroyImageView(
                renderer->logicalDevice,
                texture->subresources[subresourceIndex].depthStencilView,
                NULL);
        }
    }

    SDL_free(texture->subresources);

    if (texture->fullView) {
        renderer->vkDestroyImageView(
            renderer->logicalDevice,
            texture->fullView,
            NULL);
    }

    if (texture->image) {
        renderer->vkDestroyImage(
            renderer->logicalDevice,
            texture->image,
            NULL);
    }

    if (texture->usedRegion) {
        VULKAN_INTERNAL_RemoveMemoryUsedRegion(
            renderer,
            texture->usedRegion);
    }

    SDL_free(texture);
}

static void VULKAN_INTERNAL_DestroyBuffer(
    VulkanRenderer *renderer,
    VulkanBuffer *buffer)
{
    renderer->vkDestroyBuffer(
        renderer->logicalDevice,
        buffer->buffer,
        NULL);

    VULKAN_INTERNAL_RemoveMemoryUsedRegion(
        renderer,
        buffer->usedRegion);

    SDL_free(buffer);
}

static void VULKAN_INTERNAL_DestroyCommandPool(
    VulkanRenderer *renderer,
    VulkanCommandPool *commandPool)
{
    Uint32 i;
    VulkanCommandBuffer *commandBuffer;

    renderer->vkDestroyCommandPool(
        renderer->logicalDevice,
        commandPool->commandPool,
        NULL);

    for (i = 0; i < commandPool->inactiveCommandBufferCount; i += 1) {
        commandBuffer = commandPool->inactiveCommandBuffers[i];

        SDL_free(commandBuffer->presentDatas);
        SDL_free(commandBuffer->waitSemaphores);
        SDL_free(commandBuffer->signalSemaphores);
        SDL_free(commandBuffer->usedBuffers);
        SDL_free(commandBuffer->usedTextures);
        SDL_free(commandBuffer->usedSamplers);
        SDL_free(commandBuffer->usedGraphicsPipelines);
        SDL_free(commandBuffer->usedComputePipelines);
        SDL_free(commandBuffer->usedFramebuffers);
        SDL_free(commandBuffer->usedUniformBuffers);

        SDL_free(commandBuffer);
    }

    SDL_free(commandPool->inactiveCommandBuffers);
    SDL_free(commandPool);
}

static void VULKAN_INTERNAL_DestroyDescriptorSetLayout(
    VulkanRenderer *renderer,
    DescriptorSetLayout *layout)
{
    if (layout == NULL) {
        return;
    }

    if (layout->descriptorSetLayout != VK_NULL_HANDLE) {
        renderer->vkDestroyDescriptorSetLayout(
            renderer->logicalDevice,
            layout->descriptorSetLayout,
            NULL);
    }

    SDL_free(layout);
}

static void VULKAN_INTERNAL_DestroyGraphicsPipeline(
    VulkanRenderer *renderer,
    VulkanGraphicsPipeline *graphicsPipeline)
{
    renderer->vkDestroyPipeline(
        renderer->logicalDevice,
        graphicsPipeline->pipeline,
        NULL);

    (void)SDL_AtomicDecRef(&graphicsPipeline->vertexShader->referenceCount);
    (void)SDL_AtomicDecRef(&graphicsPipeline->fragmentShader->referenceCount);

    SDL_free(graphicsPipeline);
}

static void VULKAN_INTERNAL_DestroyComputePipeline(
    VulkanRenderer *renderer,
    VulkanComputePipeline *computePipeline)
{
    if (computePipeline->pipeline != VK_NULL_HANDLE) {
        renderer->vkDestroyPipeline(
            renderer->logicalDevice,
            computePipeline->pipeline,
            NULL);
    }

    if (computePipeline->shaderModule != VK_NULL_HANDLE) {
        renderer->vkDestroyShaderModule(
            renderer->logicalDevice,
            computePipeline->shaderModule,
            NULL);
    }

    SDL_free(computePipeline);
}

static void VULKAN_INTERNAL_DestroyShader(
    VulkanRenderer *renderer,
    VulkanShader *vulkanShader)
{
    renderer->vkDestroyShaderModule(
        renderer->logicalDevice,
        vulkanShader->shaderModule,
        NULL);

    SDL_free(vulkanShader->entrypointName);
    SDL_free(vulkanShader);
}

static void VULKAN_INTERNAL_DestroySampler(
    VulkanRenderer *renderer,
    VulkanSampler *vulkanSampler)
{
    renderer->vkDestroySampler(
        renderer->logicalDevice,
        vulkanSampler->sampler,
        NULL);

    SDL_free(vulkanSampler);
}

static void VULKAN_INTERNAL_DestroySwapchain(
    VulkanRenderer *renderer,
    WindowData *windowData)
{
    Uint32 i;

    if (windowData == NULL) {
        return;
    }

    for (i = 0; i < windowData->imageCount; i += 1) {
        VULKAN_INTERNAL_RemoveFramebuffersContainingView(
            renderer,
            windowData->textureContainers[i].activeTexture->subresources[0].renderTargetViews[0]);
        renderer->vkDestroyImageView(
            renderer->logicalDevice,
            windowData->textureContainers[i].activeTexture->subresources[0].renderTargetViews[0],
            NULL);
        SDL_free(windowData->textureContainers[i].activeTexture->subresources[0].renderTargetViews);
        SDL_free(windowData->textureContainers[i].activeTexture->subresources);
        SDL_free(windowData->textureContainers[i].activeTexture);
    }
    windowData->imageCount = 0;

    SDL_free(windowData->textureContainers);
    windowData->textureContainers = NULL;

    if (windowData->swapchain) {
        renderer->vkDestroySwapchainKHR(
            renderer->logicalDevice,
            windowData->swapchain,
            NULL);
        windowData->swapchain = VK_NULL_HANDLE;
    }

    if (windowData->surface) {
        renderer->vkDestroySurfaceKHR(
            renderer->instance,
            windowData->surface,
            NULL);
        windowData->surface = VK_NULL_HANDLE;
    }

    for (i = 0; i < MAX_FRAMES_IN_FLIGHT; i += 1) {
        if (windowData->imageAvailableSemaphore[i]) {
            renderer->vkDestroySemaphore(
                renderer->logicalDevice,
                windowData->imageAvailableSemaphore[i],
                NULL);
            windowData->imageAvailableSemaphore[i] = VK_NULL_HANDLE;
        }

        if (windowData->renderFinishedSemaphore[i]) {
            renderer->vkDestroySemaphore(
                renderer->logicalDevice,
                windowData->renderFinishedSemaphore[i],
                NULL);
            windowData->renderFinishedSemaphore[i] = VK_NULL_HANDLE;
        }
    }
}

static void VULKAN_INTERNAL_DestroyGraphicsPipelineResourceLayout(
    VulkanRenderer *renderer,
    VulkanGraphicsPipelineResourceLayout *resourceLayout)
{
    if (resourceLayout->pipelineLayout != VK_NULL_HANDLE) {
        renderer->vkDestroyPipelineLayout(
            renderer->logicalDevice,
            resourceLayout->pipelineLayout,
            NULL);
    }

    SDL_free(resourceLayout);
}

static void VULKAN_INTERNAL_DestroyComputePipelineResourceLayout(
    VulkanRenderer *renderer,
    VulkanComputePipelineResourceLayout *resourceLayout)
{
    if (resourceLayout->pipelineLayout != VK_NULL_HANDLE) {
        renderer->vkDestroyPipelineLayout(
            renderer->logicalDevice,
            resourceLayout->pipelineLayout,
            NULL);
    }

    SDL_free(resourceLayout);
}

static void VULKAN_INTERNAL_DestroyDescriptorSetCache(
    VulkanRenderer *renderer,
    DescriptorSetCache *descriptorSetCache)
{
    for (Uint32 i = 0; i < descriptorSetCache->poolCount; i += 1) {
        for (Uint32 j = 0; j < descriptorSetCache->pools[i].poolCount; j += 1) {
            renderer->vkDestroyDescriptorPool(
                renderer->logicalDevice,
                descriptorSetCache->pools[i].descriptorPools[j],
                NULL);
        }
        SDL_free(descriptorSetCache->pools[i].descriptorSets);
        SDL_free(descriptorSetCache->pools[i].descriptorPools);
    }
    SDL_free(descriptorSetCache->pools);
    SDL_free(descriptorSetCache);
}

// Hashtable functions

static Uint32 SDLCALL VULKAN_INTERNAL_GraphicsPipelineResourceLayoutHashFunction(void *userdata, const void *key)
{
    GraphicsPipelineResourceLayoutHashTableKey *hashTableKey = (GraphicsPipelineResourceLayoutHashTableKey *)key;
    /* The algorithm for this hashing function
     * is taken from Josh Bloch's "Effective Java".
     * (https://stackoverflow.com/a/113600/12492383)
     */
    const Uint32 hashFactor = 31;
    Uint32 result = 1;
    result = result * hashFactor + hashTableKey->vertexSamplerCount;
    result = result * hashFactor + hashTableKey->vertexStorageBufferCount;
    result = result * hashFactor + hashTableKey->vertexStorageTextureCount;
    result = result * hashFactor + hashTableKey->vertexUniformBufferCount;
    result = result * hashFactor + hashTableKey->fragmentSamplerCount;
    result = result * hashFactor + hashTableKey->fragmentStorageBufferCount;
    result = result * hashFactor + hashTableKey->fragmentStorageTextureCount;
    result = result * hashFactor + hashTableKey->fragmentUniformBufferCount;
    return result;
}
static bool SDLCALL VULKAN_INTERNAL_GraphicsPipelineResourceLayoutHashKeyMatch(void *userdata, const void *aKey, const void *bKey)
{
    return SDL_memcmp(aKey, bKey, sizeof(GraphicsPipelineResourceLayoutHashTableKey)) == 0;
}
static void SDLCALL VULKAN_INTERNAL_GraphicsPipelineResourceLayoutHashDestroy(void *userdata, const void *key, const void *value)
{
    VulkanRenderer *renderer = (VulkanRenderer *)userdata;
    VulkanGraphicsPipelineResourceLayout *resourceLayout = (VulkanGraphicsPipelineResourceLayout *)value;
    VULKAN_INTERNAL_DestroyGraphicsPipelineResourceLayout(renderer, resourceLayout);
    SDL_free((void*)key);
}

static Uint32 SDLCALL VULKAN_INTERNAL_ComputePipelineResourceLayoutHashFunction(void *userdata, const void *key)
{
    ComputePipelineResourceLayoutHashTableKey *hashTableKey = (ComputePipelineResourceLayoutHashTableKey *)key;
    /* The algorithm for this hashing function
     * is taken from Josh Bloch's "Effective Java".
     * (https://stackoverflow.com/a/113600/12492383)
     */
    const Uint32 hashFactor = 31;
    Uint32 result = 1;
    result = result * hashFactor + hashTableKey->samplerCount;
    result = result * hashFactor + hashTableKey->readonlyStorageTextureCount;
    result = result * hashFactor + hashTableKey->readonlyStorageBufferCount;
    result = result * hashFactor + hashTableKey->readWriteStorageTextureCount;
    result = result * hashFactor + hashTableKey->readWriteStorageBufferCount;
    result = result * hashFactor + hashTableKey->uniformBufferCount;
    return result;
}

static bool SDLCALL VULKAN_INTERNAL_ComputePipelineResourceLayoutHashKeyMatch(void *userdata, const void *aKey, const void *bKey)
{
    return SDL_memcmp(aKey, bKey, sizeof(ComputePipelineResourceLayoutHashTableKey)) == 0;
}

static void SDLCALL VULKAN_INTERNAL_ComputePipelineResourceLayoutHashDestroy(void *userdata, const void *key, const void *value)
{
    VulkanRenderer *renderer = (VulkanRenderer *)userdata;
    VulkanComputePipelineResourceLayout *resourceLayout = (VulkanComputePipelineResourceLayout *)value;
    VULKAN_INTERNAL_DestroyComputePipelineResourceLayout(renderer, resourceLayout);
    SDL_free((void*)key);
}

static Uint32 SDLCALL VULKAN_INTERNAL_DescriptorSetLayoutHashFunction(void *userdata, const void *key)
{
    DescriptorSetLayoutHashTableKey *hashTableKey = (DescriptorSetLayoutHashTableKey *)key;

    /* The algorithm for this hashing function
     * is taken from Josh Bloch's "Effective Java".
     * (https://stackoverflow.com/a/113600/12492383)
     */
    const Uint32 hashFactor = 31;
    Uint32 result = 1;
    result = result * hashFactor + hashTableKey->shaderStage;
    result = result * hashFactor + hashTableKey->samplerCount;
    result = result * hashFactor + hashTableKey->storageTextureCount;
    result = result * hashFactor + hashTableKey->storageBufferCount;
    result = result * hashFactor + hashTableKey->writeStorageTextureCount;
    result = result * hashFactor + hashTableKey->writeStorageBufferCount;
    result = result * hashFactor + hashTableKey->uniformBufferCount;
    return result;
}

static bool SDLCALL VULKAN_INTERNAL_DescriptorSetLayoutHashKeyMatch(void *userdata, const void *aKey, const void *bKey)
{
    return SDL_memcmp(aKey, bKey, sizeof(DescriptorSetLayoutHashTableKey)) == 0;
}

static void SDLCALL VULKAN_INTERNAL_DescriptorSetLayoutHashDestroy(void *userdata, const void *key, const void *value)
{
    VulkanRenderer *renderer = (VulkanRenderer *)userdata;
    DescriptorSetLayout *layout = (DescriptorSetLayout *)value;
    VULKAN_INTERNAL_DestroyDescriptorSetLayout(renderer, layout);
    SDL_free((void*)key);
}

static Uint32 SDLCALL VULKAN_INTERNAL_CommandPoolHashFunction(void *userdata, const void *key)
{
    return (Uint32)((CommandPoolHashTableKey *)key)->threadID;
}

static bool SDLCALL VULKAN_INTERNAL_CommandPoolHashKeyMatch(void *userdata, const void *aKey, const void *bKey)
{
    CommandPoolHashTableKey *a = (CommandPoolHashTableKey *)aKey;
    CommandPoolHashTableKey *b = (CommandPoolHashTableKey *)bKey;
    return a->threadID == b->threadID;
}

static void SDLCALL VULKAN_INTERNAL_CommandPoolHashDestroy(void *userdata, const void *key, const void *value)
{
    VulkanRenderer *renderer = (VulkanRenderer *)userdata;
    VulkanCommandPool *pool = (VulkanCommandPool *)value;
    VULKAN_INTERNAL_DestroyCommandPool(renderer, pool);
    SDL_free((void *)key);
}

static Uint32 SDLCALL VULKAN_INTERNAL_RenderPassHashFunction(void *userdata, const void *key)
{
    RenderPassHashTableKey *hashTableKey = (RenderPassHashTableKey *)key;

    /* The algorithm for this hashing function
     * is taken from Josh Bloch's "Effective Java".
     * (https://stackoverflow.com/a/113600/12492383)
     */
    const Uint32 hashFactor = 31;
    Uint32 result = 1;

    for (Uint32 i = 0; i < hashTableKey->numColorTargets; i += 1) {
        result = result * hashFactor + hashTableKey->colorTargetDescriptions[i].loadOp;
        result = result * hashFactor + hashTableKey->colorTargetDescriptions[i].storeOp;
        result = result * hashFactor + hashTableKey->colorTargetDescriptions[i].format;
    }

    for (Uint32 i = 0; i < hashTableKey->numResolveTargets; i += 1) {
        result = result * hashFactor + hashTableKey->resolveTargetFormats[i];
    }

    result = result * hashFactor + hashTableKey->depthStencilTargetDescription.loadOp;
    result = result * hashFactor + hashTableKey->depthStencilTargetDescription.storeOp;
    result = result * hashFactor + hashTableKey->depthStencilTargetDescription.stencilLoadOp;
    result = result * hashFactor + hashTableKey->depthStencilTargetDescription.stencilStoreOp;
    result = result * hashFactor + hashTableKey->depthStencilTargetDescription.format;

    result = result * hashFactor + hashTableKey->sampleCount;

    return result;
}

static bool SDLCALL VULKAN_INTERNAL_RenderPassHashKeyMatch(void *userdata, const void *aKey, const void *bKey)
{
    RenderPassHashTableKey *a = (RenderPassHashTableKey *)aKey;
    RenderPassHashTableKey *b = (RenderPassHashTableKey *)bKey;

    if (a->numColorTargets != b->numColorTargets) {
        return 0;
    }

    if (a->numResolveTargets != b->numResolveTargets) {
        return 0;
    }

    if (a->sampleCount != b->sampleCount) {
        return 0;
    }

    for (Uint32 i = 0; i < a->numColorTargets; i += 1) {
        if (a->colorTargetDescriptions[i].format != b->colorTargetDescriptions[i].format) {
            return 0;
        }

        if (a->colorTargetDescriptions[i].loadOp != b->colorTargetDescriptions[i].loadOp) {
            return 0;
        }

        if (a->colorTargetDescriptions[i].storeOp != b->colorTargetDescriptions[i].storeOp) {
            return 0;
        }
    }

    for (Uint32 i = 0; i < a->numResolveTargets; i += 1) {
        if (a->resolveTargetFormats[i] != b->resolveTargetFormats[i]) {
            return 0;
        }
    }

    if (a->depthStencilTargetDescription.format != b->depthStencilTargetDescription.format) {
        return 0;
    }

    if (a->depthStencilTargetDescription.loadOp != b->depthStencilTargetDescription.loadOp) {
        return 0;
    }

    if (a->depthStencilTargetDescription.storeOp != b->depthStencilTargetDescription.storeOp) {
        return 0;
    }

    if (a->depthStencilTargetDescription.stencilLoadOp != b->depthStencilTargetDescription.stencilLoadOp) {
        return 0;
    }

    if (a->depthStencilTargetDescription.stencilStoreOp != b->depthStencilTargetDescription.stencilStoreOp) {
        return 0;
    }

    return 1;
}

static void SDLCALL VULKAN_INTERNAL_RenderPassHashDestroy(void *userdata, const void *key, const void *value)
{
    VulkanRenderer *renderer = (VulkanRenderer *)userdata;
    VulkanRenderPassHashTableValue *renderPassWrapper = (VulkanRenderPassHashTableValue *)value;
    renderer->vkDestroyRenderPass(
        renderer->logicalDevice,
        renderPassWrapper->handle,
        NULL);
    SDL_free(renderPassWrapper);
    SDL_free((void *)key);
}

static Uint32 SDLCALL VULKAN_INTERNAL_FramebufferHashFunction(void *userdata, const void *key)
{
    FramebufferHashTableKey *hashTableKey = (FramebufferHashTableKey *)key;

    /* The algorithm for this hashing function
     * is taken from Josh Bloch's "Effective Java".
     * (https://stackoverflow.com/a/113600/12492383)
     */
    const Uint32 hashFactor = 31;
    Uint32 result = 1;

    for (Uint32 i = 0; i < hashTableKey->numColorTargets; i += 1) {
        result = result * hashFactor + (Uint32)(uintptr_t)hashTableKey->colorAttachmentViews[i];
    }
    for (Uint32 i = 0; i < hashTableKey->numResolveAttachments; i += 1) {
        result = result * hashFactor + (Uint32)(uintptr_t)hashTableKey->resolveAttachmentViews[i];
    }

    result = result * hashFactor + (Uint32)(uintptr_t)hashTableKey->depthStencilAttachmentView;
    result = result * hashFactor + hashTableKey->width;
    result = result * hashFactor + hashTableKey->height;

    return result;
}

static bool SDLCALL VULKAN_INTERNAL_FramebufferHashKeyMatch(void *userdata, const void *aKey, const void *bKey)
{
    FramebufferHashTableKey *a = (FramebufferHashTableKey *)aKey;
    FramebufferHashTableKey *b = (FramebufferHashTableKey *)bKey;

    if (a->numColorTargets != b->numColorTargets) {
        return 0;
    }

    if (a->numResolveAttachments != b->numResolveAttachments) {
        return 0;
    }

    for (Uint32 i = 0; i < a->numColorTargets; i += 1) {
        if (a->colorAttachmentViews[i] != b->colorAttachmentViews[i]) {
            return 0;
        }
    }

    for (Uint32 i = 0; i < a->numResolveAttachments; i += 1) {
        if (a->resolveAttachmentViews[i] != b->resolveAttachmentViews[i]) {
            return 0;
        }
    }

    if (a->depthStencilAttachmentView != b->depthStencilAttachmentView) {
        return 0;
    }

    if (a->width != b->width) {
        return 0;
    }

    if (a->height != b->height) {
        return 0;
    }

    return 1;
}

static void SDLCALL VULKAN_INTERNAL_FramebufferHashDestroy(void *userdata, const void *key, const void *value)
{
    VulkanRenderer *renderer = (VulkanRenderer *)userdata;
    VulkanFramebuffer *framebuffer = (VulkanFramebuffer *)value;
    VULKAN_INTERNAL_ReleaseFramebuffer(renderer, framebuffer);
    SDL_free((void *)key);
}

// Descriptor pools

static bool VULKAN_INTERNAL_AllocateDescriptorSets(
    VulkanRenderer *renderer,
    VkDescriptorPool descriptorPool,
    VkDescriptorSetLayout descriptorSetLayout,
    Uint32 descriptorSetCount,
    VkDescriptorSet *descriptorSetArray)
{
    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo;
    VkDescriptorSetLayout *descriptorSetLayouts = SDL_stack_alloc(VkDescriptorSetLayout, descriptorSetCount);
    VkResult vulkanResult;
    Uint32 i;

    for (i = 0; i < descriptorSetCount; i += 1) {
        descriptorSetLayouts[i] = descriptorSetLayout;
    }

    descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptorSetAllocateInfo.pNext = NULL;
    descriptorSetAllocateInfo.descriptorPool = descriptorPool;
    descriptorSetAllocateInfo.descriptorSetCount = descriptorSetCount;
    descriptorSetAllocateInfo.pSetLayouts = descriptorSetLayouts;

    vulkanResult = renderer->vkAllocateDescriptorSets(
        renderer->logicalDevice,
        &descriptorSetAllocateInfo,
        descriptorSetArray);

    SDL_stack_free(descriptorSetLayouts);

    CHECK_VULKAN_ERROR_AND_RETURN(vulkanResult, vkAllocateDescriptorSets, false);

    return true;
}

static bool VULKAN_INTERNAL_AllocateDescriptorsFromPool(
    VulkanRenderer *renderer,
    DescriptorSetLayout *descriptorSetLayout,
    DescriptorSetPool *descriptorSetPool)
{
    VkDescriptorPoolSize descriptorPoolSizes[
        MAX_TEXTURE_SAMPLERS_PER_STAGE +
        MAX_STORAGE_TEXTURES_PER_STAGE +
        MAX_STORAGE_BUFFERS_PER_STAGE +
        MAX_COMPUTE_WRITE_TEXTURES +
        MAX_COMPUTE_WRITE_BUFFERS +
        MAX_UNIFORM_BUFFERS_PER_STAGE];
    VkDescriptorPoolCreateInfo descriptorPoolInfo;
    VkDescriptorPool pool;
    VkResult vulkanResult;

    // Category 1
    for (Uint32 i = 0; i < descriptorSetLayout->samplerCount; i += 1) {
        descriptorPoolSizes[i].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorPoolSizes[i].descriptorCount = DESCRIPTOR_POOL_SIZE;
    }

    for (Uint32 i = descriptorSetLayout->samplerCount; i < descriptorSetLayout->samplerCount + descriptorSetLayout->storageTextureCount; i += 1) {
        descriptorPoolSizes[i].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; // Yes, we are declaring the storage image as a sampled image, because shaders are stupid.
        descriptorPoolSizes[i].descriptorCount = DESCRIPTOR_POOL_SIZE;
    }

    for (Uint32 i = descriptorSetLayout->samplerCount + descriptorSetLayout->storageTextureCount; i < descriptorSetLayout->samplerCount + descriptorSetLayout->storageTextureCount + descriptorSetLayout->storageBufferCount; i += 1) {
        descriptorPoolSizes[i].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorPoolSizes[i].descriptorCount = DESCRIPTOR_POOL_SIZE;
    }

    // Category 2
    for (Uint32 i = 0; i < descriptorSetLayout->writeStorageTextureCount; i += 1) {
        descriptorPoolSizes[i].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptorPoolSizes[i].descriptorCount = DESCRIPTOR_POOL_SIZE;
    }

    for (Uint32 i = descriptorSetLayout->writeStorageTextureCount; i < descriptorSetLayout->writeStorageTextureCount + descriptorSetLayout->writeStorageBufferCount; i += 1) {
        descriptorPoolSizes[i].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorPoolSizes[i].descriptorCount = DESCRIPTOR_POOL_SIZE;
    }

    // Category 3
    for (Uint32 i = 0; i < descriptorSetLayout->uniformBufferCount; i += 1) {
        descriptorPoolSizes[i].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        descriptorPoolSizes[i].descriptorCount = DESCRIPTOR_POOL_SIZE;
    }

    descriptorPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptorPoolInfo.pNext = NULL;
    descriptorPoolInfo.flags = 0;
    descriptorPoolInfo.maxSets = DESCRIPTOR_POOL_SIZE;
    descriptorPoolInfo.poolSizeCount =
        descriptorSetLayout->samplerCount +
        descriptorSetLayout->storageTextureCount +
        descriptorSetLayout->storageBufferCount +
        descriptorSetLayout->writeStorageTextureCount +
        descriptorSetLayout->writeStorageBufferCount +
        descriptorSetLayout->uniformBufferCount;
    descriptorPoolInfo.pPoolSizes = descriptorPoolSizes;

    vulkanResult = renderer->vkCreateDescriptorPool(
        renderer->logicalDevice,
        &descriptorPoolInfo,
        NULL,
        &pool);

    CHECK_VULKAN_ERROR_AND_RETURN(vulkanResult, vkCreateDescriptorPool, false);

    descriptorSetPool->poolCount += 1;
    descriptorSetPool->descriptorPools = SDL_realloc(
        descriptorSetPool->descriptorPools,
        sizeof(VkDescriptorPool) * descriptorSetPool->poolCount);

    descriptorSetPool->descriptorPools[descriptorSetPool->poolCount - 1] = pool;

    descriptorSetPool->descriptorSets = SDL_realloc(
        descriptorSetPool->descriptorSets,
        sizeof(VkDescriptorSet) * descriptorSetPool->poolCount * DESCRIPTOR_POOL_SIZE);

    if (!VULKAN_INTERNAL_AllocateDescriptorSets(
        renderer,
        pool,
        descriptorSetLayout->descriptorSetLayout,
        DESCRIPTOR_POOL_SIZE,
        &descriptorSetPool->descriptorSets[descriptorSetPool->descriptorSetCount])) {
        return false;
    }

    descriptorSetPool->descriptorSetCount += DESCRIPTOR_POOL_SIZE;

    return true;
}

// NOTE: these categories should be mutually exclusive
static DescriptorSetLayout *VULKAN_INTERNAL_FetchDescriptorSetLayout(
    VulkanRenderer *renderer,
    VkShaderStageFlagBits shaderStage,
    // Category 1: read resources
    Uint32 samplerCount,
    Uint32 storageTextureCount,
    Uint32 storageBufferCount,
    // Category 2: write resources
    Uint32 writeStorageTextureCount,
    Uint32 writeStorageBufferCount,
    // Category 3: uniform buffers
    Uint32 uniformBufferCount)
{
    DescriptorSetLayoutHashTableKey key;
    SDL_zero(key);
    DescriptorSetLayout *layout = NULL;

    key.shaderStage = shaderStage;
    key.samplerCount = samplerCount;
    key.storageTextureCount = storageTextureCount;
    key.storageBufferCount = storageBufferCount;
    key.writeStorageTextureCount = writeStorageTextureCount;
    key.writeStorageBufferCount = writeStorageBufferCount;
    key.uniformBufferCount = uniformBufferCount;

    SDL_LockMutex(renderer->descriptorSetLayoutFetchLock);

    if (SDL_FindInHashTable(
        renderer->descriptorSetLayoutHashTable,
        (const void *)&key,
        (const void **)&layout)) {
        SDL_UnlockMutex(renderer->descriptorSetLayoutFetchLock);
        return layout;
    }

    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[
        MAX_TEXTURE_SAMPLERS_PER_STAGE +
        MAX_STORAGE_TEXTURES_PER_STAGE +
        MAX_STORAGE_BUFFERS_PER_STAGE +
        MAX_COMPUTE_WRITE_TEXTURES +
        MAX_COMPUTE_WRITE_BUFFERS];

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo;
    descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorSetLayoutCreateInfo.pNext = NULL;
    descriptorSetLayoutCreateInfo.flags = 0;

    // Category 1
    for (Uint32 i = 0; i < samplerCount; i += 1) {
        descriptorSetLayoutBindings[i].binding = i;
        descriptorSetLayoutBindings[i].descriptorCount = 1;
        descriptorSetLayoutBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorSetLayoutBindings[i].stageFlags = shaderStage;
        descriptorSetLayoutBindings[i].pImmutableSamplers = NULL;
    }

    for (Uint32 i = samplerCount; i < samplerCount + storageTextureCount; i += 1) {
        descriptorSetLayoutBindings[i].binding = i;
        descriptorSetLayoutBindings[i].descriptorCount = 1;
        descriptorSetLayoutBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; // Yes, we are declaring the storage image as a sampled image, because shaders are stupid.
        descriptorSetLayoutBindings[i].stageFlags = shaderStage;
        descriptorSetLayoutBindings[i].pImmutableSamplers = NULL;
    }

    for (Uint32 i = samplerCount + storageTextureCount; i < samplerCount + storageTextureCount + storageBufferCount; i += 1) {
        descriptorSetLayoutBindings[i].binding = i;
        descriptorSetLayoutBindings[i].descriptorCount = 1;
        descriptorSetLayoutBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorSetLayoutBindings[i].stageFlags = shaderStage;
        descriptorSetLayoutBindings[i].pImmutableSamplers = NULL;
    }

    // Category 2
    for (Uint32 i = 0; i < writeStorageTextureCount; i += 1) {
        descriptorSetLayoutBindings[i].binding = i;
        descriptorSetLayoutBindings[i].descriptorCount = 1;
        descriptorSetLayoutBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptorSetLayoutBindings[i].stageFlags = shaderStage;
        descriptorSetLayoutBindings[i].pImmutableSamplers = NULL;
    }

    for (Uint32 i = writeStorageTextureCount; i < writeStorageTextureCount + writeStorageBufferCount; i += 1) {
        descriptorSetLayoutBindings[i].binding = i;
        descriptorSetLayoutBindings[i].descriptorCount = 1;
        descriptorSetLayoutBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorSetLayoutBindings[i].stageFlags = shaderStage;
        descriptorSetLayoutBindings[i].pImmutableSamplers = NULL;
    }

    // Category 3
    for (Uint32 i = 0; i < uniformBufferCount; i += 1) {
        descriptorSetLayoutBindings[i].binding = i;
        descriptorSetLayoutBindings[i].descriptorCount = 1;
        descriptorSetLayoutBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        descriptorSetLayoutBindings[i].stageFlags = shaderStage;
        descriptorSetLayoutBindings[i].pImmutableSamplers = NULL;
    }

    descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;
    descriptorSetLayoutCreateInfo.bindingCount =
        samplerCount +
        storageTextureCount +
        storageBufferCount +
        writeStorageTextureCount +
        writeStorageBufferCount +
        uniformBufferCount;

    VkResult vulkanResult = renderer->vkCreateDescriptorSetLayout(
        renderer->logicalDevice,
        &descriptorSetLayoutCreateInfo,
        NULL,
        &descriptorSetLayout);

    if (vulkanResult != VK_SUCCESS) {
        SDL_UnlockMutex(renderer->descriptorSetLayoutFetchLock);
        CHECK_VULKAN_ERROR_AND_RETURN(vulkanResult, vkCreateDescriptorSetLayout, NULL);
    }

    layout = SDL_malloc(sizeof(DescriptorSetLayout));
    layout->descriptorSetLayout = descriptorSetLayout;

    layout->samplerCount = samplerCount;
    layout->storageBufferCount = storageBufferCount;
    layout->storageTextureCount = storageTextureCount;
    layout->writeStorageBufferCount = writeStorageBufferCount;
    layout->writeStorageTextureCount = writeStorageTextureCount;
    layout->uniformBufferCount = uniformBufferCount;

    layout->ID = SDL_AtomicIncRef(&renderer->layoutResourceID);

    DescriptorSetLayoutHashTableKey *allocedKey = SDL_malloc(sizeof(DescriptorSetLayoutHashTableKey));
    SDL_memcpy(allocedKey, &key, sizeof(DescriptorSetLayoutHashTableKey));

    SDL_InsertIntoHashTable(
        renderer->descriptorSetLayoutHashTable,
        (const void *)allocedKey,
        (const void *)layout, true);

    SDL_UnlockMutex(renderer->descriptorSetLayoutFetchLock);
    return layout;
}

static VulkanGraphicsPipelineResourceLayout *VULKAN_INTERNAL_FetchGraphicsPipelineResourceLayout(
    VulkanRenderer *renderer,
    VulkanShader *vertexShader,
    VulkanShader *fragmentShader)
{
    GraphicsPipelineResourceLayoutHashTableKey key;
    SDL_zero(key);
    VulkanGraphicsPipelineResourceLayout *pipelineResourceLayout = NULL;

    key.vertexSamplerCount = vertexShader->numSamplers;
    key.vertexStorageTextureCount = vertexShader->numStorageTextures;
    key.vertexStorageBufferCount = vertexShader->numStorageBuffers;
    key.vertexUniformBufferCount = vertexShader->numUniformBuffers;
    key.fragmentSamplerCount = fragmentShader->numSamplers;
    key.fragmentStorageTextureCount = fragmentShader->numStorageTextures;
    key.fragmentStorageBufferCount = fragmentShader->numStorageBuffers;
    key.fragmentUniformBufferCount = fragmentShader->numUniformBuffers;

    SDL_LockMutex(renderer->graphicsPipelineLayoutFetchLock);

    if (SDL_FindInHashTable(
        renderer->graphicsPipelineResourceLayoutHashTable,
        (const void *)&key,
        (const void **)&pipelineResourceLayout)) {
        SDL_UnlockMutex(renderer->graphicsPipelineLayoutFetchLock);
        return pipelineResourceLayout;
    }

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo;
    VkDescriptorSetLayout descriptorSetLayouts[4];
    VkResult vulkanResult;

    pipelineResourceLayout = SDL_calloc(1, sizeof(VulkanGraphicsPipelineResourceLayout));

    pipelineResourceLayout->descriptorSetLayouts[0] = VULKAN_INTERNAL_FetchDescriptorSetLayout(
        renderer,
        VK_SHADER_STAGE_VERTEX_BIT,
        vertexShader->numSamplers,
        vertexShader->numStorageTextures,
        vertexShader->numStorageBuffers,
        0,
        0,
        0);

    pipelineResourceLayout->descriptorSetLayouts[1] = VULKAN_INTERNAL_FetchDescriptorSetLayout(
        renderer,
        VK_SHADER_STAGE_VERTEX_BIT,
        0,
        0,
        0,
        0,
        0,
        vertexShader->numUniformBuffers);

    pipelineResourceLayout->descriptorSetLayouts[2] = VULKAN_INTERNAL_FetchDescriptorSetLayout(
        renderer,
        VK_SHADER_STAGE_FRAGMENT_BIT,
        fragmentShader->numSamplers,
        fragmentShader->numStorageTextures,
        fragmentShader->numStorageBuffers,
        0,
        0,
        0);

    pipelineResourceLayout->descriptorSetLayouts[3] = VULKAN_INTERNAL_FetchDescriptorSetLayout(
        renderer,
        VK_SHADER_STAGE_FRAGMENT_BIT,
        0,
        0,
        0,
        0,
        0,
        fragmentShader->numUniformBuffers);

    descriptorSetLayouts[0] = pipelineResourceLayout->descriptorSetLayouts[0]->descriptorSetLayout;
    descriptorSetLayouts[1] = pipelineResourceLayout->descriptorSetLayouts[1]->descriptorSetLayout;
    descriptorSetLayouts[2] = pipelineResourceLayout->descriptorSetLayouts[2]->descriptorSetLayout;
    descriptorSetLayouts[3] = pipelineResourceLayout->descriptorSetLayouts[3]->descriptorSetLayout;

    pipelineResourceLayout->vertexSamplerCount = vertexShader->numSamplers;
    pipelineResourceLayout->vertexStorageTextureCount = vertexShader->numStorageTextures;
    pipelineResourceLayout->vertexStorageBufferCount = vertexShader->numStorageBuffers;
    pipelineResourceLayout->vertexUniformBufferCount = vertexShader->numUniformBuffers;

    pipelineResourceLayout->fragmentSamplerCount = fragmentShader->numSamplers;
    pipelineResourceLayout->fragmentStorageTextureCount = fragmentShader->numStorageTextures;
    pipelineResourceLayout->fragmentStorageBufferCount = fragmentShader->numStorageBuffers;
    pipelineResourceLayout->fragmentUniformBufferCount = fragmentShader->numUniformBuffers;

    // Create the pipeline layout

    pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.pNext = NULL;
    pipelineLayoutCreateInfo.flags = 0;
    pipelineLayoutCreateInfo.setLayoutCount = 4;
    pipelineLayoutCreateInfo.pSetLayouts = descriptorSetLayouts;
    pipelineLayoutCreateInfo.pushConstantRangeCount = 0;
    pipelineLayoutCreateInfo.pPushConstantRanges = NULL;

    vulkanResult = renderer->vkCreatePipelineLayout(
        renderer->logicalDevice,
        &pipelineLayoutCreateInfo,
        NULL,
        &pipelineResourceLayout->pipelineLayout);

    if (vulkanResult != VK_SUCCESS) {
        VULKAN_INTERNAL_DestroyGraphicsPipelineResourceLayout(renderer, pipelineResourceLayout);
        SDL_UnlockMutex(renderer->graphicsPipelineLayoutFetchLock);
        CHECK_VULKAN_ERROR_AND_RETURN(vulkanResult, vkCreatePipelineLayout, NULL);
    }

    GraphicsPipelineResourceLayoutHashTableKey *allocedKey = SDL_malloc(sizeof(GraphicsPipelineResourceLayoutHashTableKey));
    SDL_memcpy(allocedKey, &key, sizeof(GraphicsPipelineResourceLayoutHashTableKey));

    SDL_InsertIntoHashTable(
        renderer->graphicsPipelineResourceLayoutHashTable,
        (const void *)allocedKey,
        (const void *)pipelineResourceLayout, true);

    SDL_UnlockMutex(renderer->graphicsPipelineLayoutFetchLock);
    return pipelineResourceLayout;
}

static VulkanComputePipelineResourceLayout *VULKAN_INTERNAL_FetchComputePipelineResourceLayout(
    VulkanRenderer *renderer,
    const SDL_GPUComputePipelineCreateInfo *createinfo)
{
    ComputePipelineResourceLayoutHashTableKey key;
    SDL_zero(key);
    VulkanComputePipelineResourceLayout *pipelineResourceLayout = NULL;

    key.samplerCount = createinfo->num_samplers;
    key.readonlyStorageTextureCount = createinfo->num_readonly_storage_textures;
    key.readonlyStorageBufferCount = createinfo->num_readonly_storage_buffers;
    key.readWriteStorageTextureCount = createinfo->num_readwrite_storage_textures;
    key.readWriteStorageBufferCount = createinfo->num_readwrite_storage_buffers;
    key.uniformBufferCount = createinfo->num_uniform_buffers;

    SDL_LockMutex(renderer->computePipelineLayoutFetchLock);

    if (SDL_FindInHashTable(
        renderer->computePipelineResourceLayoutHashTable,
        (const void *)&key,
        (const void **)&pipelineResourceLayout)) {
        SDL_UnlockMutex(renderer->computePipelineLayoutFetchLock);
        return pipelineResourceLayout;
    }

    VkDescriptorSetLayout descriptorSetLayouts[3];
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo;
    VkResult vulkanResult;

    pipelineResourceLayout = SDL_calloc(1, sizeof(VulkanComputePipelineResourceLayout));

    pipelineResourceLayout->descriptorSetLayouts[0] = VULKAN_INTERNAL_FetchDescriptorSetLayout(
        renderer,
        VK_SHADER_STAGE_COMPUTE_BIT,
        createinfo->num_samplers,
        createinfo->num_readonly_storage_textures,
        createinfo->num_readonly_storage_buffers,
        0,
        0,
        0);

    pipelineResourceLayout->descriptorSetLayouts[1] = VULKAN_INTERNAL_FetchDescriptorSetLayout(
        renderer,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        0,
        0,
        createinfo->num_readwrite_storage_textures,
        createinfo->num_readwrite_storage_buffers,
        0);

    pipelineResourceLayout->descriptorSetLayouts[2] = VULKAN_INTERNAL_FetchDescriptorSetLayout(
        renderer,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        0,
        0,
        0,
        0,
        createinfo->num_uniform_buffers);

    descriptorSetLayouts[0] = pipelineResourceLayout->descriptorSetLayouts[0]->descriptorSetLayout;
    descriptorSetLayouts[1] = pipelineResourceLayout->descriptorSetLayouts[1]->descriptorSetLayout;
    descriptorSetLayouts[2] = pipelineResourceLayout->descriptorSetLayouts[2]->descriptorSetLayout;

    pipelineResourceLayout->numSamplers = createinfo->num_samplers;
    pipelineResourceLayout->numReadonlyStorageTextures = createinfo->num_readonly_storage_textures;
    pipelineResourceLayout->numReadonlyStorageBuffers = createinfo->num_readonly_storage_buffers;
    pipelineResourceLayout->numReadWriteStorageTextures = createinfo->num_readwrite_storage_textures;
    pipelineResourceLayout->numReadWriteStorageBuffers = createinfo->num_readwrite_storage_buffers;
    pipelineResourceLayout->numUniformBuffers = createinfo->num_uniform_buffers;

    // Create the pipeline layout

    pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.pNext = NULL;
    pipelineLayoutCreateInfo.flags = 0;
    pipelineLayoutCreateInfo.setLayoutCount = 3;
    pipelineLayoutCreateInfo.pSetLayouts = descriptorSetLayouts;
    pipelineLayoutCreateInfo.pushConstantRangeCount = 0;
    pipelineLayoutCreateInfo.pPushConstantRanges = NULL;

    vulkanResult = renderer->vkCreatePipelineLayout(
        renderer->logicalDevice,
        &pipelineLayoutCreateInfo,
        NULL,
        &pipelineResourceLayout->pipelineLayout);

    if (vulkanResult != VK_SUCCESS) {
        VULKAN_INTERNAL_DestroyComputePipelineResourceLayout(renderer, pipelineResourceLayout);
        SDL_UnlockMutex(renderer->computePipelineLayoutFetchLock);
        CHECK_VULKAN_ERROR_AND_RETURN(vulkanResult, vkCreatePipelineLayout, NULL);
    }

    ComputePipelineResourceLayoutHashTableKey *allocedKey = SDL_malloc(sizeof(ComputePipelineResourceLayoutHashTableKey));
    SDL_memcpy(allocedKey, &key, sizeof(ComputePipelineResourceLayoutHashTableKey));

    SDL_InsertIntoHashTable(
        renderer->computePipelineResourceLayoutHashTable,
        (const void *)allocedKey,
        (const void *)pipelineResourceLayout, true);

    SDL_UnlockMutex(renderer->computePipelineLayoutFetchLock);
    return pipelineResourceLayout;
}

// Data Buffer

static VulkanBuffer *VULKAN_INTERNAL_CreateBuffer(
    VulkanRenderer *renderer,
    VkDeviceSize size,
    SDL_GPUBufferUsageFlags usageFlags,
    VulkanBufferType type,
    bool dedicated,
    const char *debugName)
{
    VulkanBuffer *buffer;
    VkResult vulkanResult;
    VkBufferCreateInfo createinfo;
    VkBufferUsageFlags vulkanUsageFlags = 0;
    Uint8 bindResult;

    if (usageFlags & SDL_GPU_BUFFERUSAGE_VERTEX) {
        vulkanUsageFlags |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    }

    if (usageFlags & SDL_GPU_BUFFERUSAGE_INDEX) {
        vulkanUsageFlags |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    }

    if (usageFlags & (SDL_GPU_BUFFERUSAGE_GRAPHICS_STORAGE_READ |
                      SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ |
                      SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE)) {
        vulkanUsageFlags |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    }

    if (usageFlags & SDL_GPU_BUFFERUSAGE_INDIRECT) {
        vulkanUsageFlags |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
    }

    if (type == VULKAN_BUFFER_TYPE_UNIFORM) {
        vulkanUsageFlags |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    } else {
        // GPU buffers need transfer bits for defrag, transfer buffers need them for transfers
        vulkanUsageFlags |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    }

    buffer = SDL_calloc(1, sizeof(VulkanBuffer));

    buffer->size = size;
    buffer->usage = usageFlags;
    buffer->type = type;
    buffer->markedForDestroy = false;
    buffer->transitioned = false;

    createinfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    createinfo.pNext = NULL;
    createinfo.flags = 0;
    createinfo.size = size;
    createinfo.usage = vulkanUsageFlags;
    createinfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    createinfo.queueFamilyIndexCount = 1;
    createinfo.pQueueFamilyIndices = &renderer->queueFamilyIndex;

    // Set transfer bits so we can defrag
    createinfo.usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    vulkanResult = renderer->vkCreateBuffer(
        renderer->logicalDevice,
        &createinfo,
        NULL,
        &buffer->buffer);

    if (vulkanResult != VK_SUCCESS) {
        SDL_free(buffer);
        CHECK_VULKAN_ERROR_AND_RETURN(vulkanResult, vkCreateBuffer, NULL);
    }

    bindResult = VULKAN_INTERNAL_BindMemoryForBuffer(
        renderer,
        buffer->buffer,
        buffer->size,
        buffer->type,
        dedicated,
        &buffer->usedRegion);

    if (bindResult != 1) {
        renderer->vkDestroyBuffer(
            renderer->logicalDevice,
            buffer->buffer,
            NULL);

        SDL_free(buffer);
        return NULL;
    }

    buffer->usedRegion->vulkanBuffer = buffer; // lol

    SDL_SetAtomicInt(&buffer->referenceCount, 0);

    if (renderer->debugMode && renderer->supportsDebugUtils && debugName != NULL) {
        VkDebugUtilsObjectNameInfoEXT nameInfo;
        nameInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
        nameInfo.pNext = NULL;
        nameInfo.pObjectName = debugName;
        nameInfo.objectType = VK_OBJECT_TYPE_BUFFER;
        nameInfo.objectHandle = (uint64_t)buffer->buffer;

        renderer->vkSetDebugUtilsObjectNameEXT(
            renderer->logicalDevice,
            &nameInfo);
    }

    return buffer;
}

static VulkanBufferContainer *VULKAN_INTERNAL_CreateBufferContainer(
    VulkanRenderer *renderer,
    VkDeviceSize size,
    SDL_GPUBufferUsageFlags usageFlags,
    VulkanBufferType type,
    bool dedicated,
    const char *debugName)
{
    VulkanBufferContainer *bufferContainer;
    VulkanBuffer *buffer;

    buffer = VULKAN_INTERNAL_CreateBuffer(
        renderer,
        size,
        usageFlags,
        type,
        dedicated,
        debugName);

    if (buffer == NULL) {
        return NULL;
    }

    bufferContainer = SDL_calloc(1, sizeof(VulkanBufferContainer));

    bufferContainer->activeBuffer = buffer;
    buffer->container = bufferContainer;
    buffer->containerIndex = 0;

    bufferContainer->bufferCapacity = 1;
    bufferContainer->bufferCount = 1;
    bufferContainer->buffers = SDL_calloc(bufferContainer->bufferCapacity, sizeof(VulkanBuffer *));
    bufferContainer->buffers[0] = bufferContainer->activeBuffer;
    bufferContainer->dedicated = dedicated;
    bufferContainer->debugName = NULL;

    if (debugName != NULL) {
        bufferContainer->debugName = SDL_strdup(debugName);
    }

    return bufferContainer;
}

// Texture Subresource Utilities

static Uint32 VULKAN_INTERNAL_GetTextureSubresourceIndex(
    Uint32 mipLevel,
    Uint32 layer,
    Uint32 numLevels)
{
    return mipLevel + (layer * numLevels);
}

static VulkanTextureSubresource *VULKAN_INTERNAL_FetchTextureSubresource(
    VulkanTextureContainer *textureContainer,
    Uint32 layer,
    Uint32 level)
{
    Uint32 index = VULKAN_INTERNAL_GetTextureSubresourceIndex(
        level,
        layer,
        textureContainer->header.info.num_levels);

    return &textureContainer->activeTexture->subresources[index];
}

static bool VULKAN_INTERNAL_CreateRenderTargetView(
    VulkanRenderer *renderer,
    VulkanTexture *texture,
    Uint32 layerOrDepth,
    Uint32 level,
    VkFormat format,
    VkComponentMapping swizzle,
    VkImageView *pView)
{
    VkResult vulkanResult;
    VkImageViewCreateInfo imageViewCreateInfo;

    // create framebuffer compatible views for RenderTarget
    imageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    imageViewCreateInfo.pNext = NULL;
    imageViewCreateInfo.flags = 0;
    imageViewCreateInfo.image = texture->image;
    imageViewCreateInfo.format = format;
    imageViewCreateInfo.components = swizzle;
    imageViewCreateInfo.subresourceRange.aspectMask = texture->aspectFlags;
    imageViewCreateInfo.subresourceRange.baseMipLevel = level;
    imageViewCreateInfo.subresourceRange.levelCount = 1;
    imageViewCreateInfo.subresourceRange.baseArrayLayer = layerOrDepth;
    imageViewCreateInfo.subresourceRange.layerCount = 1;
    imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;

    vulkanResult = renderer->vkCreateImageView(
        renderer->logicalDevice,
        &imageViewCreateInfo,
        NULL,
        pView);

    if (vulkanResult != VK_SUCCESS) {
        *pView = (VkImageView)VK_NULL_HANDLE;
        CHECK_VULKAN_ERROR_AND_RETURN(vulkanResult, vkCreateImageView, false);
    }

    return true;
}

static bool VULKAN_INTERNAL_CreateSubresourceView(
    VulkanRenderer *renderer,
    const SDL_GPUTextureCreateInfo *createinfo,
    VulkanTexture *texture,
    Uint32 layer,
    Uint32 level,
    VkComponentMapping swizzle,
    VkImageView *pView)
{
    VkResult vulkanResult;
    VkImageViewCreateInfo imageViewCreateInfo;

    // create framebuffer compatible views for RenderTarget
    imageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    imageViewCreateInfo.pNext = NULL;
    imageViewCreateInfo.flags = 0;
    imageViewCreateInfo.image = texture->image;
    imageViewCreateInfo.format = SDLToVK_TextureFormat[createinfo->format];
    imageViewCreateInfo.components = swizzle;
    imageViewCreateInfo.subresourceRange.aspectMask = texture->aspectFlags;
    imageViewCreateInfo.subresourceRange.baseMipLevel = level;
    imageViewCreateInfo.subresourceRange.levelCount = 1;
    imageViewCreateInfo.subresourceRange.baseArrayLayer = layer;
    imageViewCreateInfo.subresourceRange.layerCount = 1;
    imageViewCreateInfo.viewType = (createinfo->type == SDL_GPU_TEXTURETYPE_3D) ? VK_IMAGE_VIEW_TYPE_3D : VK_IMAGE_VIEW_TYPE_2D;

    vulkanResult = renderer->vkCreateImageView(
        renderer->logicalDevice,
        &imageViewCreateInfo,
        NULL,
        pView);

    if (vulkanResult != VK_SUCCESS) {
        *pView = (VkImageView)VK_NULL_HANDLE;
        CHECK_VULKAN_ERROR_AND_RETURN(vulkanResult, vkCreateImageView, false);
    }

    return true;
}

// Swapchain

static bool VULKAN_INTERNAL_QuerySwapchainSupport(
    VulkanRenderer *renderer,
    VkPhysicalDevice physicalDevice,
    VkSurfaceKHR surface,
    SwapchainSupportDetails *outputDetails)
{
    VkResult result;
    VkBool32 supportsPresent;

    renderer->vkGetPhysicalDeviceSurfaceSupportKHR(
        physicalDevice,
        renderer->queueFamilyIndex,
        surface,
        &supportsPresent);

    // Initialize these in case anything fails
    outputDetails->formatsLength = 0;
    outputDetails->presentModesLength = 0;

    if (!supportsPresent) {
        SET_STRING_ERROR_AND_RETURN("This surface does not support presenting!", false);
    }

    // Run the device surface queries
    result = renderer->vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
        physicalDevice,
        surface,
        &outputDetails->capabilities);
    CHECK_VULKAN_ERROR_AND_RETURN(result, vkGetPhysicalDeviceSurfaceCapabilitiesKHR, false);

    if (!(outputDetails->capabilities.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR)) {
        SDL_LogWarn(SDL_LOG_CATEGORY_GPU, "Opaque presentation unsupported! Expect weird transparency bugs!");
    }

    result = renderer->vkGetPhysicalDeviceSurfaceFormatsKHR(
        physicalDevice,
        surface,
        &outputDetails->formatsLength,
        NULL);
    CHECK_VULKAN_ERROR_AND_RETURN(result, vkGetPhysicalDeviceSurfaceFormatsKHR, false);
    result = renderer->vkGetPhysicalDeviceSurfacePresentModesKHR(
        physicalDevice,
        surface,
        &outputDetails->presentModesLength,
        NULL);
    CHECK_VULKAN_ERROR_AND_RETURN(result, vkGetPhysicalDeviceSurfacePresentModesKHR, false);

    // Generate the arrays, if applicable

    outputDetails->formats = NULL;
    if (outputDetails->formatsLength != 0) {
        outputDetails->formats = (VkSurfaceFormatKHR *)SDL_malloc(
            sizeof(VkSurfaceFormatKHR) * outputDetails->formatsLength);

        if (!outputDetails->formats) { // OOM
            return false;
        }

        result = renderer->vkGetPhysicalDeviceSurfaceFormatsKHR(
            physicalDevice,
            surface,
            &outputDetails->formatsLength,
            outputDetails->formats);
        if (result != VK_SUCCESS) {
            SDL_free(outputDetails->formats);
            CHECK_VULKAN_ERROR_AND_RETURN(result, vkGetPhysicalDeviceSurfaceFormatsKHR, false);
        }
    }

    outputDetails->presentModes = NULL;
    if (outputDetails->presentModesLength != 0) {
        outputDetails->presentModes = (VkPresentModeKHR *)SDL_malloc(
            sizeof(VkPresentModeKHR) * outputDetails->presentModesLength);

        if (!outputDetails->presentModes) { // OOM
            SDL_free(outputDetails->formats);
            return false;
        }

        result = renderer->vkGetPhysicalDeviceSurfacePresentModesKHR(
            physicalDevice,
            surface,
            &outputDetails->presentModesLength,
            outputDetails->presentModes);
        if (result != VK_SUCCESS) {
            SDL_free(outputDetails->formats);
            SDL_free(outputDetails->presentModes);
            CHECK_VULKAN_ERROR_AND_RETURN(result, vkGetPhysicalDeviceSurfacePresentModesKHR, false);
        }
    }

    /* If we made it here, all the queries were successful. This does NOT
     * necessarily mean there are any supported formats or present modes!
     */
    return true;
}

static bool VULKAN_INTERNAL_VerifySwapSurfaceFormat(
    VkFormat desiredFormat,
    VkColorSpaceKHR desiredColorSpace,
    VkSurfaceFormatKHR *availableFormats,
    Uint32 availableFormatsLength)
{
    Uint32 i;
    for (i = 0; i < availableFormatsLength; i += 1) {
        if (availableFormats[i].format == desiredFormat &&
            availableFormats[i].colorSpace == desiredColorSpace) {
            return true;
        }
    }
    return false;
}

static bool VULKAN_INTERNAL_VerifySwapPresentMode(
    VkPresentModeKHR presentMode,
    const VkPresentModeKHR *availablePresentModes,
    Uint32 availablePresentModesLength)
{
    Uint32 i;
    for (i = 0; i < availablePresentModesLength; i += 1) {
        if (availablePresentModes[i] == presentMode) {
            return true;
        }
    }
    return false;
}

/* It would be nice if VULKAN_INTERNAL_CreateSwapchain could return a bool.
 * Unfortunately, some Win32 NVIDIA drivers are stupid
 * and will return surface extents of (0, 0)
 * in certain edge cases, and the swapchain extents are not allowed to be 0.
 * In this case, the client probably still wants to claim the window
 * or recreate the swapchain, so we should return 2 to indicate retry.
 * -cosmonaut
 */
#define VULKAN_INTERNAL_TRY_AGAIN 2

static Uint32 VULKAN_INTERNAL_CreateSwapchain(
    VulkanRenderer *renderer,
    WindowData *windowData)
{
    VkResult vulkanResult;
    VkSwapchainCreateInfoKHR swapchainCreateInfo;
    VkImage *swapchainImages;
    VkSemaphoreCreateInfo semaphoreCreateInfo;
    SwapchainSupportDetails swapchainSupportDetails;
    bool hasValidSwapchainComposition, hasValidPresentMode;
    VkCompositeAlphaFlagsKHR compositeAlphaFlag = 0;
    Uint32 i;

    windowData->frameCounter = 0;

    SDL_VideoDevice *_this = SDL_GetVideoDevice();
    SDL_assert(_this && _this->Vulkan_CreateSurface);

    // Each swapchain must have its own surface.
    if (!_this->Vulkan_CreateSurface(
            _this,
            windowData->window,
            renderer->instance,
            NULL, // FIXME: VAllocationCallbacks
            &windowData->surface)) {
        return false;
    }
    SDL_assert(windowData->surface);

    if (!VULKAN_INTERNAL_QuerySwapchainSupport(
            renderer,
            renderer->physicalDevice,
            windowData->surface,
            &swapchainSupportDetails)) {
        renderer->vkDestroySurfaceKHR(
            renderer->instance,
            windowData->surface,
            NULL);
        windowData->surface = VK_NULL_HANDLE;
        if (swapchainSupportDetails.formatsLength > 0) {
            SDL_free(swapchainSupportDetails.formats);
        }
        if (swapchainSupportDetails.presentModesLength > 0) {
            SDL_free(swapchainSupportDetails.presentModes);
        }
        return false;
    }

    // Verify that we can use the requested composition and present mode
    windowData->format = SwapchainCompositionToFormat[windowData->swapchainComposition];
    windowData->colorSpace = SwapchainCompositionToColorSpace[windowData->swapchainComposition];
    windowData->swapchainSwizzle = SwapchainCompositionSwizzle[windowData->swapchainComposition];
    windowData->usingFallbackFormat = false;

    hasValidSwapchainComposition = VULKAN_INTERNAL_VerifySwapSurfaceFormat(
        windowData->format,
        windowData->colorSpace,
        swapchainSupportDetails.formats,
        swapchainSupportDetails.formatsLength);

    if (!hasValidSwapchainComposition) {
        // Let's try again with the fallback format...
        windowData->format = SwapchainCompositionToFallbackFormat[windowData->swapchainComposition];
        windowData->usingFallbackFormat = true;
        hasValidSwapchainComposition = VULKAN_INTERNAL_VerifySwapSurfaceFormat(
            windowData->format,
            windowData->colorSpace,
            swapchainSupportDetails.formats,
            swapchainSupportDetails.formatsLength);
    }

    hasValidPresentMode = VULKAN_INTERNAL_VerifySwapPresentMode(
        SDLToVK_PresentMode[windowData->presentMode],
        swapchainSupportDetails.presentModes,
        swapchainSupportDetails.presentModesLength);

    if (!hasValidSwapchainComposition || !hasValidPresentMode) {
        renderer->vkDestroySurfaceKHR(
            renderer->instance,
            windowData->surface,
            NULL);
        windowData->surface = VK_NULL_HANDLE;

        if (swapchainSupportDetails.formatsLength > 0) {
            SDL_free(swapchainSupportDetails.formats);
        }

        if (swapchainSupportDetails.presentModesLength > 0) {
            SDL_free(swapchainSupportDetails.presentModes);
        }

        if (!hasValidSwapchainComposition) {
            SET_STRING_ERROR_AND_RETURN("Device does not support requested swapchain composition!", false);
        }
        if (!hasValidPresentMode) {
            SET_STRING_ERROR_AND_RETURN("Device does not support requested present_mode!", false);
        }
        return false;
    }

    // NVIDIA + Win32 can return 0 extent when the window is minimized. Try again!
    if (swapchainSupportDetails.capabilities.currentExtent.width == 0 ||
        swapchainSupportDetails.capabilities.currentExtent.height == 0) {
        renderer->vkDestroySurfaceKHR(
            renderer->instance,
            windowData->surface,
            NULL);
        windowData->surface = VK_NULL_HANDLE;
        if (swapchainSupportDetails.formatsLength > 0) {
            SDL_free(swapchainSupportDetails.formats);
        }
        if (swapchainSupportDetails.presentModesLength > 0) {
            SDL_free(swapchainSupportDetails.presentModes);
        }
        return VULKAN_INTERNAL_TRY_AGAIN;
    }

    Uint32 requestedImageCount = renderer->allowedFramesInFlight;

#ifdef SDL_PLATFORM_APPLE
    windowData->width = swapchainSupportDetails.capabilities.currentExtent.width;
    windowData->height = swapchainSupportDetails.capabilities.currentExtent.height;
#else
    windowData->width = SDL_clamp(
        windowData->swapchainCreateWidth,
        swapchainSupportDetails.capabilities.minImageExtent.width,
        swapchainSupportDetails.capabilities.maxImageExtent.width);
    windowData->height = SDL_clamp(windowData->swapchainCreateHeight,
        swapchainSupportDetails.capabilities.minImageExtent.height,
        swapchainSupportDetails.capabilities.maxImageExtent.height);
#endif

    if (swapchainSupportDetails.capabilities.maxImageCount > 0 &&
        requestedImageCount > swapchainSupportDetails.capabilities.maxImageCount) {
        requestedImageCount = swapchainSupportDetails.capabilities.maxImageCount;
    }

    if (requestedImageCount < swapchainSupportDetails.capabilities.minImageCount) {
        requestedImageCount = swapchainSupportDetails.capabilities.minImageCount;
    }

    if (windowData->presentMode == SDL_GPU_PRESENTMODE_MAILBOX) {
        /* Required for proper triple-buffering.
         * Note that this is below the above maxImageCount check!
         * If the driver advertises MAILBOX but does not support 3 swap
         * images, it's not real mailbox support, so let it fail hard.
         * -flibit
         */
        requestedImageCount = SDL_max(requestedImageCount, 3);
    }

    // Default to opaque, if available, followed by inherit, and overwrite with a value that supports transparency, if necessary.
    if (swapchainSupportDetails.capabilities.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR) {
        compositeAlphaFlag = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    } else if (swapchainSupportDetails.capabilities.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR) {
        compositeAlphaFlag = VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
    }

    if ((windowData->window->flags & SDL_WINDOW_TRANSPARENT) || !compositeAlphaFlag) {
        if (swapchainSupportDetails.capabilities.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR) {
            compositeAlphaFlag = VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR;
        } else if (swapchainSupportDetails.capabilities.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR) {
            compositeAlphaFlag = VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR;
        } else if (swapchainSupportDetails.capabilities.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR) {
            compositeAlphaFlag = VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
        } else {
            SDL_LogWarn(SDL_LOG_CATEGORY_GPU, "SDL_WINDOW_TRANSPARENT flag set, but no suitable swapchain composite alpha value supported!");
        }
    }

    swapchainCreateInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swapchainCreateInfo.pNext = NULL;
    swapchainCreateInfo.flags = 0;
    swapchainCreateInfo.surface = windowData->surface;
    swapchainCreateInfo.minImageCount = requestedImageCount;
    swapchainCreateInfo.imageFormat = windowData->format;
    swapchainCreateInfo.imageColorSpace = windowData->colorSpace;
    swapchainCreateInfo.imageExtent.width = windowData->width;
    swapchainCreateInfo.imageExtent.height = windowData->height;
    swapchainCreateInfo.imageArrayLayers = 1;
    swapchainCreateInfo.imageUsage =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
        VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    swapchainCreateInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    swapchainCreateInfo.queueFamilyIndexCount = 0;
    swapchainCreateInfo.pQueueFamilyIndices = NULL;
#ifdef SDL_PLATFORM_ANDROID
    swapchainCreateInfo.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
#else
    swapchainCreateInfo.preTransform = swapchainSupportDetails.capabilities.currentTransform;
#endif
    swapchainCreateInfo.compositeAlpha = compositeAlphaFlag;
    swapchainCreateInfo.presentMode = SDLToVK_PresentMode[windowData->presentMode];
    swapchainCreateInfo.clipped = VK_TRUE;
    swapchainCreateInfo.oldSwapchain = VK_NULL_HANDLE;

    vulkanResult = renderer->vkCreateSwapchainKHR(
        renderer->logicalDevice,
        &swapchainCreateInfo,
        NULL,
        &windowData->swapchain);

    if (swapchainSupportDetails.formatsLength > 0) {
        SDL_free(swapchainSupportDetails.formats);
    }
    if (swapchainSupportDetails.presentModesLength > 0) {
        SDL_free(swapchainSupportDetails.presentModes);
    }

    if (vulkanResult != VK_SUCCESS) {
        renderer->vkDestroySurfaceKHR(
            renderer->instance,
            windowData->surface,
            NULL);
        windowData->surface = VK_NULL_HANDLE;
        CHECK_VULKAN_ERROR_AND_RETURN(vulkanResult, vkCreateSwapchainKHR, false);
    }

    vulkanResult = renderer->vkGetSwapchainImagesKHR(
        renderer->logicalDevice,
        windowData->swapchain,
        &windowData->imageCount,
        NULL);
    CHECK_VULKAN_ERROR_AND_RETURN(vulkanResult, vkGetSwapchainImagesKHR, false);

    windowData->textureContainers = SDL_malloc(
        sizeof(VulkanTextureContainer) * windowData->imageCount);

    if (!windowData->textureContainers) { // OOM
        renderer->vkDestroySurfaceKHR(
            renderer->instance,
            windowData->surface,
            NULL);
        renderer->vkDestroySwapchainKHR(
            renderer->logicalDevice,
            windowData->swapchain,
            NULL);
        windowData->surface = VK_NULL_HANDLE;
        windowData->swapchain = VK_NULL_HANDLE;
        return false;
    }

    swapchainImages = SDL_stack_alloc(VkImage, windowData->imageCount);

    vulkanResult = renderer->vkGetSwapchainImagesKHR(
        renderer->logicalDevice,
        windowData->swapchain,
        &windowData->imageCount,
        swapchainImages);
    CHECK_VULKAN_ERROR_AND_RETURN(vulkanResult, vkGetSwapchainImagesKHR, false);

    for (i = 0; i < windowData->imageCount; i += 1) {

        // Initialize dummy container
        SDL_zero(windowData->textureContainers[i]);
        windowData->textureContainers[i].canBeCycled = false;
        windowData->textureContainers[i].header.info.width = windowData->width;
        windowData->textureContainers[i].header.info.height = windowData->height;
        windowData->textureContainers[i].header.info.layer_count_or_depth = 1;
        windowData->textureContainers[i].header.info.format = SwapchainCompositionToSDLFormat(
            windowData->swapchainComposition,
            windowData->usingFallbackFormat);
        windowData->textureContainers[i].header.info.type = SDL_GPU_TEXTURETYPE_2D;
        windowData->textureContainers[i].header.info.num_levels = 1;
        windowData->textureContainers[i].header.info.sample_count = SDL_GPU_SAMPLECOUNT_1;
        windowData->textureContainers[i].header.info.usage = SDL_GPU_TEXTUREUSAGE_COLOR_TARGET;

        windowData->textureContainers[i].activeTexture = SDL_malloc(sizeof(VulkanTexture));
        windowData->textureContainers[i].activeTexture->image = swapchainImages[i];

        // Swapchain memory is managed by the driver
        windowData->textureContainers[i].activeTexture->usedRegion = NULL;

        windowData->textureContainers[i].activeTexture->swizzle = windowData->swapchainSwizzle;
        windowData->textureContainers[i].activeTexture->aspectFlags = VK_IMAGE_ASPECT_COLOR_BIT;
        windowData->textureContainers[i].activeTexture->depth = 1;
        windowData->textureContainers[i].activeTexture->usage = SDL_GPU_TEXTUREUSAGE_COLOR_TARGET;
        windowData->textureContainers[i].activeTexture->container = &windowData->textureContainers[i];
        SDL_SetAtomicInt(&windowData->textureContainers[i].activeTexture->referenceCount, 0);

        // Create slice
        windowData->textureContainers[i].activeTexture->subresourceCount = 1;
        windowData->textureContainers[i].activeTexture->subresources = SDL_malloc(sizeof(VulkanTextureSubresource));
        windowData->textureContainers[i].activeTexture->subresources[0].parent = windowData->textureContainers[i].activeTexture;
        windowData->textureContainers[i].activeTexture->subresources[0].layer = 0;
        windowData->textureContainers[i].activeTexture->subresources[0].level = 0;
        windowData->textureContainers[i].activeTexture->subresources[0].renderTargetViews = SDL_malloc(sizeof(VkImageView));
        if (!VULKAN_INTERNAL_CreateRenderTargetView(
            renderer,
            windowData->textureContainers[i].activeTexture,
            0,
            0,
            windowData->format,
            windowData->swapchainSwizzle,
            &windowData->textureContainers[i].activeTexture->subresources[0].renderTargetViews[0])) {
            renderer->vkDestroySurfaceKHR(
                renderer->instance,
                windowData->surface,
                NULL);
            renderer->vkDestroySwapchainKHR(
                renderer->logicalDevice,
                windowData->swapchain,
                NULL);
            windowData->surface = VK_NULL_HANDLE;
            windowData->swapchain = VK_NULL_HANDLE;
            return false;
        }
    }

    SDL_stack_free(swapchainImages);

    semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semaphoreCreateInfo.pNext = NULL;
    semaphoreCreateInfo.flags = 0;

    for (i = 0; i < MAX_FRAMES_IN_FLIGHT; i += 1) {
        vulkanResult = renderer->vkCreateSemaphore(
            renderer->logicalDevice,
            &semaphoreCreateInfo,
            NULL,
            &windowData->imageAvailableSemaphore[i]);

        if (vulkanResult != VK_SUCCESS) {
            renderer->vkDestroySurfaceKHR(
                renderer->instance,
                windowData->surface,
                NULL);
            renderer->vkDestroySwapchainKHR(
                renderer->logicalDevice,
                windowData->swapchain,
                NULL);
            windowData->surface = VK_NULL_HANDLE;
            windowData->swapchain = VK_NULL_HANDLE;
            CHECK_VULKAN_ERROR_AND_RETURN(vulkanResult, vkCreateSemaphore, false);
        }

        vulkanResult = renderer->vkCreateSemaphore(
            renderer->logicalDevice,
            &semaphoreCreateInfo,
            NULL,
            &windowData->renderFinishedSemaphore[i]);

        if (vulkanResult != VK_SUCCESS) {
            renderer->vkDestroySurfaceKHR(
                renderer->instance,
                windowData->surface,
                NULL);
            renderer->vkDestroySwapchainKHR(
                renderer->logicalDevice,
                windowData->swapchain,
                NULL);
            windowData->surface = VK_NULL_HANDLE;
            windowData->swapchain = VK_NULL_HANDLE;
            CHECK_VULKAN_ERROR_AND_RETURN(vulkanResult, vkCreateSemaphore, false);
        }

        windowData->inFlightFences[i] = NULL;
    }

    windowData->needsSwapchainRecreate = false;
    return true;
}

// Command Buffers

static bool VULKAN_INTERNAL_BeginCommandBuffer(
    VulkanRenderer *renderer,
    VulkanCommandBuffer *commandBuffer)
{
    VkCommandBufferBeginInfo beginInfo;
    VkResult result;

    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.pNext = NULL;
    beginInfo.flags = 0;
    beginInfo.pInheritanceInfo = NULL;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    result = renderer->vkBeginCommandBuffer(
        commandBuffer->commandBuffer,
        &beginInfo);

    CHECK_VULKAN_ERROR_AND_RETURN(result, vkBeginCommandBuffer, false);

    return true;
}

static bool VULKAN_INTERNAL_EndCommandBuffer(
    VulkanRenderer *renderer,
    VulkanCommandBuffer *commandBuffer)
{
    VkResult result = renderer->vkEndCommandBuffer(
        commandBuffer->commandBuffer);

    CHECK_VULKAN_ERROR_AND_RETURN(result, vkEndCommandBuffer, false);

    return true;
}

static void VULKAN_DestroyDevice(
    SDL_GPUDevice *device)
{
    VulkanRenderer *renderer = (VulkanRenderer *)device->driverData;
    VulkanMemorySubAllocator *allocator;

    VULKAN_Wait(device->driverData);

    for (Sint32 i = renderer->claimedWindowCount - 1; i >= 0; i -= 1) {
        VULKAN_ReleaseWindow(device->driverData, renderer->claimedWindows[i]->window);
    }

    SDL_free(renderer->claimedWindows);

    VULKAN_Wait(device->driverData);

    SDL_free(renderer->submittedCommandBuffers);

    for (Uint32 i = 0; i < renderer->uniformBufferPoolCount; i += 1) {
        VULKAN_INTERNAL_DestroyBuffer(
            renderer,
            renderer->uniformBufferPool[i]->buffer);
        SDL_free(renderer->uniformBufferPool[i]);
    }
    SDL_free(renderer->uniformBufferPool);

    for (Uint32 i = 0; i < renderer->descriptorSetCachePoolCount; i += 1) {
        VULKAN_INTERNAL_DestroyDescriptorSetCache(
            renderer,
            renderer->descriptorSetCachePool[i]);
    }
    SDL_free(renderer->descriptorSetCachePool);

    for (Uint32 i = 0; i < renderer->fencePool.availableFenceCount; i += 1) {
        renderer->vkDestroyFence(
            renderer->logicalDevice,
            renderer->fencePool.availableFences[i]->fence,
            NULL);

        SDL_free(renderer->fencePool.availableFences[i]);
    }

    SDL_free(renderer->fencePool.availableFences);
    SDL_DestroyMutex(renderer->fencePool.lock);

    SDL_DestroyHashTable(renderer->commandPoolHashTable);
    SDL_DestroyHashTable(renderer->renderPassHashTable);
    SDL_DestroyHashTable(renderer->framebufferHashTable);
    SDL_DestroyHashTable(renderer->graphicsPipelineResourceLayoutHashTable);
    SDL_DestroyHashTable(renderer->computePipelineResourceLayoutHashTable);
    SDL_DestroyHashTable(renderer->descriptorSetLayoutHashTable);

    for (Uint32 i = 0; i < VK_MAX_MEMORY_TYPES; i += 1) {
        allocator = &renderer->memoryAllocator->subAllocators[i];

        for (Sint32 j = allocator->allocationCount - 1; j >= 0; j -= 1) {
            for (Sint32 k = allocator->allocations[j]->usedRegionCount - 1; k >= 0; k -= 1) {
                VULKAN_INTERNAL_RemoveMemoryUsedRegion(
                    renderer,
                    allocator->allocations[j]->usedRegions[k]);
            }

            VULKAN_INTERNAL_DeallocateMemory(
                renderer,
                allocator,
                j);
        }

        if (renderer->memoryAllocator->subAllocators[i].allocations != NULL) {
            SDL_free(renderer->memoryAllocator->subAllocators[i].allocations);
        }

        SDL_free(renderer->memoryAllocator->subAllocators[i].sortedFreeRegions);
    }

    SDL_free(renderer->memoryAllocator);

    SDL_free(renderer->texturesToDestroy);
    SDL_free(renderer->buffersToDestroy);
    SDL_free(renderer->graphicsPipelinesToDestroy);
    SDL_free(renderer->computePipelinesToDestroy);
    SDL_free(renderer->shadersToDestroy);
    SDL_free(renderer->samplersToDestroy);
    SDL_free(renderer->framebuffersToDestroy);
    SDL_free(renderer->allocationsToDefrag);

    SDL_DestroyMutex(renderer->allocatorLock);
    SDL_DestroyMutex(renderer->disposeLock);
    SDL_DestroyMutex(renderer->submitLock);
    SDL_DestroyMutex(renderer->acquireCommandBufferLock);
    SDL_DestroyMutex(renderer->acquireUniformBufferLock);
    SDL_DestroyMutex(renderer->renderPassFetchLock);
    SDL_DestroyMutex(renderer->framebufferFetchLock);
    SDL_DestroyMutex(renderer->graphicsPipelineLayoutFetchLock);
    SDL_DestroyMutex(renderer->computePipelineLayoutFetchLock);
    SDL_DestroyMutex(renderer->descriptorSetLayoutFetchLock);
    SDL_DestroyMutex(renderer->windowLock);

    renderer->vkDestroyDevice(renderer->logicalDevice, NULL);
    renderer->vkDestroyInstance(renderer->instance, NULL);

    SDL_DestroyProperties(renderer->props);

    SDL_free(renderer);
    SDL_free(device);
    SDL_Vulkan_UnloadLibrary();
}

static SDL_PropertiesID VULKAN_GetDeviceProperties(
    SDL_GPUDevice *device)
{
    VulkanRenderer *renderer = (VulkanRenderer *)device->driverData;
    return renderer->props;
}

static DescriptorSetCache *VULKAN_INTERNAL_AcquireDescriptorSetCache(
    VulkanRenderer *renderer)
{
    DescriptorSetCache *cache;

    if (renderer->descriptorSetCachePoolCount == 0) {
        cache = SDL_malloc(sizeof(DescriptorSetCache));
        cache->poolCount = 0;
        cache->pools = NULL;
    } else {
        cache = renderer->descriptorSetCachePool[renderer->descriptorSetCachePoolCount - 1];
        renderer->descriptorSetCachePoolCount -= 1;
    }

    return cache;
}

static void VULKAN_INTERNAL_ReturnDescriptorSetCacheToPool(
    VulkanRenderer *renderer,
    DescriptorSetCache *descriptorSetCache)
{
    EXPAND_ARRAY_IF_NEEDED(
        renderer->descriptorSetCachePool,
        DescriptorSetCache *,
        renderer->descriptorSetCachePoolCount + 1,
        renderer->descriptorSetCachePoolCapacity,
        renderer->descriptorSetCachePoolCapacity * 2);

    renderer->descriptorSetCachePool[renderer->descriptorSetCachePoolCount] = descriptorSetCache;
    renderer->descriptorSetCachePoolCount += 1;

    for (Uint32 i = 0; i < descriptorSetCache->poolCount; i += 1) {
        descriptorSetCache->pools[i].descriptorSetIndex = 0;
    }
}

static VkDescriptorSet VULKAN_INTERNAL_FetchDescriptorSet(
    VulkanRenderer *renderer,
    VulkanCommandBuffer *vulkanCommandBuffer,
    DescriptorSetLayout *descriptorSetLayout)
{
    // Grow the pool to meet the descriptor set layout ID
    if (descriptorSetLayout->ID >= vulkanCommandBuffer->descriptorSetCache->poolCount) {
        vulkanCommandBuffer->descriptorSetCache->pools = SDL_realloc(
            vulkanCommandBuffer->descriptorSetCache->pools,
            sizeof(DescriptorSetPool) * (descriptorSetLayout->ID + 1));

        for (Uint32 i = vulkanCommandBuffer->descriptorSetCache->poolCount; i < descriptorSetLayout->ID + 1; i += 1) {
            SDL_zero(vulkanCommandBuffer->descriptorSetCache->pools[i]);
        }

        vulkanCommandBuffer->descriptorSetCache->poolCount = descriptorSetLayout->ID + 1;
    }

    DescriptorSetPool *pool =
        &vulkanCommandBuffer->descriptorSetCache->pools[descriptorSetLayout->ID];

    if (pool->descriptorSetIndex == pool->descriptorSetCount) {
        if (!VULKAN_INTERNAL_AllocateDescriptorsFromPool(
            renderer,
            descriptorSetLayout,
            pool)) {
            return VK_NULL_HANDLE;
        }
    }

    VkDescriptorSet descriptorSet = pool->descriptorSets[pool->descriptorSetIndex];
    pool->descriptorSetIndex += 1;

    return descriptorSet;
}

static void VULKAN_INTERNAL_BindGraphicsDescriptorSets(
    VulkanRenderer *renderer,
    VulkanCommandBuffer *commandBuffer)
{
    VulkanGraphicsPipelineResourceLayout *resourceLayout;
    DescriptorSetLayout *descriptorSetLayout;
    VkWriteDescriptorSet writeDescriptorSets[
        (MAX_TEXTURE_SAMPLERS_PER_STAGE +
        MAX_STORAGE_TEXTURES_PER_STAGE +
        MAX_STORAGE_BUFFERS_PER_STAGE +
        MAX_UNIFORM_BUFFERS_PER_STAGE) * 2];
    VkDescriptorBufferInfo bufferInfos[MAX_STORAGE_BUFFERS_PER_STAGE * 2];
    VkDescriptorImageInfo imageInfos[(MAX_TEXTURE_SAMPLERS_PER_STAGE + MAX_STORAGE_TEXTURES_PER_STAGE) * 2];
    Uint32 dynamicOffsets[MAX_UNIFORM_BUFFERS_PER_STAGE * 2];
    Uint32 writeCount = 0;
    Uint32 bufferInfoCount = 0;
    Uint32 imageInfoCount = 0;
    Uint32 dynamicOffsetCount = 0;

    if (
        !commandBuffer->needVertexBufferBind &&
        !commandBuffer->needNewVertexResourceDescriptorSet &&
        !commandBuffer->needNewVertexUniformDescriptorSet &&
        !commandBuffer->needNewVertexUniformOffsets &&
        !commandBuffer->needNewFragmentResourceDescriptorSet &&
        !commandBuffer->needNewFragmentUniformDescriptorSet &&
        !commandBuffer->needNewFragmentUniformOffsets
    ) {
        return;
    }

    if (commandBuffer->needVertexBufferBind && commandBuffer->vertexBufferCount > 0) {
        renderer->vkCmdBindVertexBuffers(
            commandBuffer->commandBuffer,
            0,
            commandBuffer->vertexBufferCount,
            commandBuffer->vertexBuffers,
            commandBuffer->vertexBufferOffsets);

        commandBuffer->needVertexBufferBind = false;
    }

    resourceLayout = commandBuffer->currentGraphicsPipeline->resourceLayout;

    if (commandBuffer->needNewVertexResourceDescriptorSet) {
        descriptorSetLayout = resourceLayout->descriptorSetLayouts[0];

        commandBuffer->vertexResourceDescriptorSet = VULKAN_INTERNAL_FetchDescriptorSet(
            renderer,
            commandBuffer,
            descriptorSetLayout);

        for (Uint32 i = 0; i < resourceLayout->vertexSamplerCount; i += 1) {
            VkWriteDescriptorSet *currentWriteDescriptorSet = &writeDescriptorSets[writeCount];

            currentWriteDescriptorSet->sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            currentWriteDescriptorSet->pNext = NULL;
            currentWriteDescriptorSet->descriptorCount = 1;
            currentWriteDescriptorSet->descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            currentWriteDescriptorSet->dstArrayElement = 0;
            currentWriteDescriptorSet->dstBinding = i;
            currentWriteDescriptorSet->dstSet = commandBuffer->vertexResourceDescriptorSet;
            currentWriteDescriptorSet->pTexelBufferView = NULL;
            currentWriteDescriptorSet->pBufferInfo = NULL;

            imageInfos[imageInfoCount].sampler = commandBuffer->vertexSamplers[i]->sampler;
            imageInfos[imageInfoCount].imageView = commandBuffer->vertexSamplerTextures[i]->fullView;
            imageInfos[imageInfoCount].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            currentWriteDescriptorSet->pImageInfo = &imageInfos[imageInfoCount];

            writeCount += 1;
            imageInfoCount += 1;
        }

        for (Uint32 i = 0; i < resourceLayout->vertexStorageTextureCount; i += 1) {
            VkWriteDescriptorSet *currentWriteDescriptorSet = &writeDescriptorSets[writeCount];

            currentWriteDescriptorSet->sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            currentWriteDescriptorSet->pNext = NULL;
            currentWriteDescriptorSet->descriptorCount = 1;
            currentWriteDescriptorSet->descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; // Yes, we are declaring a storage image as a sampled image, because shaders are stupid.
            currentWriteDescriptorSet->dstArrayElement = 0;
            currentWriteDescriptorSet->dstBinding = resourceLayout->vertexSamplerCount + i;
            currentWriteDescriptorSet->dstSet = commandBuffer->vertexResourceDescriptorSet;
            currentWriteDescriptorSet->pTexelBufferView = NULL;
            currentWriteDescriptorSet->pBufferInfo = NULL;

            imageInfos[imageInfoCount].sampler = VK_NULL_HANDLE;
            imageInfos[imageInfoCount].imageView = commandBuffer->vertexStorageTextures[i]->fullView;
            imageInfos[imageInfoCount].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            currentWriteDescriptorSet->pImageInfo = &imageInfos[imageInfoCount];

            writeCount += 1;
            imageInfoCount += 1;
        }

        for (Uint32 i = 0; i < resourceLayout->vertexStorageBufferCount; i += 1) {
            VkWriteDescriptorSet *currentWriteDescriptorSet = &writeDescriptorSets[writeCount];

            currentWriteDescriptorSet->sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            currentWriteDescriptorSet->pNext = NULL;
            currentWriteDescriptorSet->descriptorCount = 1;
            currentWriteDescriptorSet->descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            currentWriteDescriptorSet->dstArrayElement = 0;
            currentWriteDescriptorSet->dstBinding = resourceLayout->vertexSamplerCount + resourceLayout->vertexStorageTextureCount + i;
            currentWriteDescriptorSet->dstSet = commandBuffer->vertexResourceDescriptorSet;
            currentWriteDescriptorSet->pTexelBufferView = NULL;
            currentWriteDescriptorSet->pImageInfo = NULL;

            bufferInfos[bufferInfoCount].buffer = commandBuffer->vertexStorageBuffers[i]->buffer;
            bufferInfos[bufferInfoCount].offset = 0;
            bufferInfos[bufferInfoCount].range = VK_WHOLE_SIZE;

            currentWriteDescriptorSet->pBufferInfo = &bufferInfos[bufferInfoCount];

            writeCount += 1;
            bufferInfoCount += 1;
        }

        commandBuffer->needNewVertexResourceDescriptorSet = false;
    }

    if (commandBuffer->needNewVertexUniformDescriptorSet) {
        descriptorSetLayout = resourceLayout->descriptorSetLayouts[1];

        commandBuffer->vertexUniformDescriptorSet = VULKAN_INTERNAL_FetchDescriptorSet(
            renderer,
            commandBuffer,
            descriptorSetLayout);

        for (Uint32 i = 0; i < resourceLayout->vertexUniformBufferCount; i += 1) {
            VkWriteDescriptorSet *currentWriteDescriptorSet = &writeDescriptorSets[writeCount];

            currentWriteDescriptorSet->sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            currentWriteDescriptorSet->pNext = NULL;
            currentWriteDescriptorSet->descriptorCount = 1;
            currentWriteDescriptorSet->descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
            currentWriteDescriptorSet->dstArrayElement = 0;
            currentWriteDescriptorSet->dstBinding = i;
            currentWriteDescriptorSet->dstSet = commandBuffer->vertexUniformDescriptorSet;
            currentWriteDescriptorSet->pTexelBufferView = NULL;
            currentWriteDescriptorSet->pImageInfo = NULL;

            bufferInfos[bufferInfoCount].buffer = commandBuffer->vertexUniformBuffers[i]->buffer->buffer;
            bufferInfos[bufferInfoCount].offset = 0;
            bufferInfos[bufferInfoCount].range = MAX_UBO_SECTION_SIZE;

            currentWriteDescriptorSet->pBufferInfo = &bufferInfos[bufferInfoCount];

            writeCount += 1;
            bufferInfoCount += 1;
        }

        commandBuffer->needNewVertexUniformDescriptorSet = false;
    }

    for (Uint32 i = 0; i < resourceLayout->vertexUniformBufferCount; i += 1) {
        dynamicOffsets[dynamicOffsetCount] = commandBuffer->vertexUniformBuffers[i]->drawOffset;
        dynamicOffsetCount += 1;
    }

    if (commandBuffer->needNewFragmentResourceDescriptorSet) {
        descriptorSetLayout = resourceLayout->descriptorSetLayouts[2];

        commandBuffer->fragmentResourceDescriptorSet = VULKAN_INTERNAL_FetchDescriptorSet(
            renderer,
            commandBuffer,
            descriptorSetLayout);

        for (Uint32 i = 0; i < resourceLayout->fragmentSamplerCount; i += 1) {
            VkWriteDescriptorSet *currentWriteDescriptorSet = &writeDescriptorSets[writeCount];

            currentWriteDescriptorSet->sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            currentWriteDescriptorSet->pNext = NULL;
            currentWriteDescriptorSet->descriptorCount = 1;
            currentWriteDescriptorSet->descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            currentWriteDescriptorSet->dstArrayElement = 0;
            currentWriteDescriptorSet->dstBinding = i;
            currentWriteDescriptorSet->dstSet = commandBuffer->fragmentResourceDescriptorSet;
            currentWriteDescriptorSet->pTexelBufferView = NULL;
            currentWriteDescriptorSet->pBufferInfo = NULL;

            imageInfos[imageInfoCount].sampler = commandBuffer->fragmentSamplers[i]->sampler;
            imageInfos[imageInfoCount].imageView = commandBuffer->fragmentSamplerTextures[i]->fullView;
            imageInfos[imageInfoCount].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            currentWriteDescriptorSet->pImageInfo = &imageInfos[imageInfoCount];

            writeCount += 1;
            imageInfoCount += 1;
        }

        for (Uint32 i = 0; i < resourceLayout->fragmentStorageTextureCount; i += 1) {
            VkWriteDescriptorSet *currentWriteDescriptorSet = &writeDescriptorSets[writeCount];

            currentWriteDescriptorSet->sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            currentWriteDescriptorSet->pNext = NULL;
            currentWriteDescriptorSet->descriptorCount = 1;
            currentWriteDescriptorSet->descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; // Yes, we are declaring a storage image as a sampled image, because shaders are stupid.
            currentWriteDescriptorSet->dstArrayElement = 0;
            currentWriteDescriptorSet->dstBinding = resourceLayout->fragmentSamplerCount + i;
            currentWriteDescriptorSet->dstSet = commandBuffer->fragmentResourceDescriptorSet;
            currentWriteDescriptorSet->pTexelBufferView = NULL;
            currentWriteDescriptorSet->pBufferInfo = NULL;

            imageInfos[imageInfoCount].sampler = VK_NULL_HANDLE;
            imageInfos[imageInfoCount].imageView = commandBuffer->fragmentStorageTextures[i]->fullView;
            imageInfos[imageInfoCount].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            currentWriteDescriptorSet->pImageInfo = &imageInfos[imageInfoCount];

            writeCount += 1;
            imageInfoCount += 1;
        }

        for (Uint32 i = 0; i < resourceLayout->fragmentStorageBufferCount; i += 1) {
            VkWriteDescriptorSet *currentWriteDescriptorSet = &writeDescriptorSets[writeCount];

            currentWriteDescriptorSet->sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            currentWriteDescriptorSet->pNext = NULL;
            currentWriteDescriptorSet->descriptorCount = 1;
            currentWriteDescriptorSet->descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            currentWriteDescriptorSet->dstArrayElement = 0;
            currentWriteDescriptorSet->dstBinding = resourceLayout->fragmentSamplerCount + resourceLayout->fragmentStorageTextureCount + i;
            currentWriteDescriptorSet->dstSet = commandBuffer->fragmentResourceDescriptorSet;
            currentWriteDescriptorSet->pTexelBufferView = NULL;
            currentWriteDescriptorSet->pImageInfo = NULL;

            bufferInfos[bufferInfoCount].buffer = commandBuffer->fragmentStorageBuffers[i]->buffer;
            bufferInfos[bufferInfoCount].offset = 0;
            bufferInfos[bufferInfoCount].range = VK_WHOLE_SIZE;

            currentWriteDescriptorSet->pBufferInfo = &bufferInfos[bufferInfoCount];

            writeCount += 1;
            bufferInfoCount += 1;
        }

        commandBuffer->needNewFragmentResourceDescriptorSet = false;
    }

    if (commandBuffer->needNewFragmentUniformDescriptorSet) {
        descriptorSetLayout = resourceLayout->descriptorSetLayouts[3];

        commandBuffer->fragmentUniformDescriptorSet = VULKAN_INTERNAL_FetchDescriptorSet(
            renderer,
            commandBuffer,
            descriptorSetLayout);

        for (Uint32 i = 0; i < resourceLayout->fragmentUniformBufferCount; i += 1) {
            VkWriteDescriptorSet *currentWriteDescriptorSet = &writeDescriptorSets[writeCount];

            currentWriteDescriptorSet->sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            currentWriteDescriptorSet->pNext = NULL;
            currentWriteDescriptorSet->descriptorCount = 1;
            currentWriteDescriptorSet->descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
            currentWriteDescriptorSet->dstArrayElement = 0;
            currentWriteDescriptorSet->dstBinding = i;
            currentWriteDescriptorSet->dstSet = commandBuffer->fragmentUniformDescriptorSet;
            currentWriteDescriptorSet->pTexelBufferView = NULL;
            currentWriteDescriptorSet->pImageInfo = NULL;

            bufferInfos[bufferInfoCount].buffer = commandBuffer->fragmentUniformBuffers[i]->buffer->buffer;
            bufferInfos[bufferInfoCount].offset = 0;
            bufferInfos[bufferInfoCount].range = MAX_UBO_SECTION_SIZE;

            currentWriteDescriptorSet->pBufferInfo = &bufferInfos[bufferInfoCount];

            writeCount += 1;
            bufferInfoCount += 1;
        }

        commandBuffer->needNewFragmentUniformDescriptorSet = false;
    }

    for (Uint32 i = 0; i < resourceLayout->fragmentUniformBufferCount; i += 1) {
        dynamicOffsets[dynamicOffsetCount] = commandBuffer->fragmentUniformBuffers[i]->drawOffset;
        dynamicOffsetCount += 1;
    }

    renderer->vkUpdateDescriptorSets(
        renderer->logicalDevice,
        writeCount,
        writeDescriptorSets,
        0,
        NULL);

    VkDescriptorSet sets[4];
    sets[0] = commandBuffer->vertexResourceDescriptorSet;
    sets[1] = commandBuffer->vertexUniformDescriptorSet;
    sets[2] = commandBuffer->fragmentResourceDescriptorSet;
    sets[3] = commandBuffer->fragmentUniformDescriptorSet;

    renderer->vkCmdBindDescriptorSets(
        commandBuffer->commandBuffer,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        resourceLayout->pipelineLayout,
        0,
        4,
        sets,
        dynamicOffsetCount,
        dynamicOffsets);

    commandBuffer->needNewVertexUniformOffsets = false;
    commandBuffer->needNewFragmentUniformOffsets = false;
}

static void VULKAN_DrawIndexedPrimitives(
    SDL_GPUCommandBuffer *commandBuffer,
    Uint32 numIndices,
    Uint32 numInstances,
    Uint32 firstIndex,
    Sint32 vertexOffset,
    Uint32 firstInstance)
{
    VulkanCommandBuffer *vulkanCommandBuffer = (VulkanCommandBuffer *)commandBuffer;
    VulkanRenderer *renderer = vulkanCommandBuffer->renderer;

    VULKAN_INTERNAL_BindGraphicsDescriptorSets(renderer, vulkanCommandBuffer);

    renderer->vkCmdDrawIndexed(
        vulkanCommandBuffer->commandBuffer,
        numIndices,
        numInstances,
        firstIndex,
        vertexOffset,
        firstInstance);
}

static void VULKAN_DrawPrimitives(
    SDL_GPUCommandBuffer *commandBuffer,
    Uint32 numVertices,
    Uint32 numInstances,
    Uint32 firstVertex,
    Uint32 firstInstance)
{
    VulkanCommandBuffer *vulkanCommandBuffer = (VulkanCommandBuffer *)commandBuffer;
    VulkanRenderer *renderer = vulkanCommandBuffer->renderer;

    VULKAN_INTERNAL_BindGraphicsDescriptorSets(renderer, vulkanCommandBuffer);

    renderer->vkCmdDraw(
        vulkanCommandBuffer->commandBuffer,
        numVertices,
        numInstances,
        firstVertex,
        firstInstance);
}

static void VULKAN_DrawPrimitivesIndirect(
    SDL_GPUCommandBuffer *commandBuffer,
    SDL_GPUBuffer *buffer,
    Uint32 offset,
    Uint32 drawCount)
{
    VulkanCommandBuffer *vulkanCommandBuffer = (VulkanCommandBuffer *)commandBuffer;
    VulkanRenderer *renderer = vulkanCommandBuffer->renderer;
    VulkanBuffer *vulkanBuffer = ((VulkanBufferContainer *)buffer)->activeBuffer;
    Uint32 pitch = sizeof(SDL_GPUIndirectDrawCommand);
    Uint32 i;

    VULKAN_INTERNAL_BindGraphicsDescriptorSets(renderer, vulkanCommandBuffer);

    if (renderer->supportsMultiDrawIndirect) {
        // Real multi-draw!
        renderer->vkCmdDrawIndirect(
            vulkanCommandBuffer->commandBuffer,
            vulkanBuffer->buffer,
            offset,
            drawCount,
            pitch);
    } else {
        // Fake multi-draw...
        for (i = 0; i < drawCount; i += 1) {
            renderer->vkCmdDrawIndirect(
                vulkanCommandBuffer->commandBuffer,
                vulkanBuffer->buffer,
                offset + (pitch * i),
                1,
                pitch);
        }
    }

    VULKAN_INTERNAL_TrackBuffer(vulkanCommandBuffer, vulkanBuffer);
}

static void VULKAN_DrawIndexedPrimitivesIndirect(
    SDL_GPUCommandBuffer *commandBuffer,
    SDL_GPUBuffer *buffer,
    Uint32 offset,
    Uint32 drawCount)
{
    VulkanCommandBuffer *vulkanCommandBuffer = (VulkanCommandBuffer *)commandBuffer;
    VulkanRenderer *renderer = vulkanCommandBuffer->renderer;
    VulkanBuffer *vulkanBuffer = ((VulkanBufferContainer *)buffer)->activeBuffer;
    Uint32 pitch = sizeof(SDL_GPUIndexedIndirectDrawCommand);
    Uint32 i;

    VULKAN_INTERNAL_BindGraphicsDescriptorSets(renderer, vulkanCommandBuffer);

    if (renderer->supportsMultiDrawIndirect) {
        // Real multi-draw!
        renderer->vkCmdDrawIndexedIndirect(
            vulkanCommandBuffer->commandBuffer,
            vulkanBuffer->buffer,
            offset,
            drawCount,
            pitch);
    } else {
        // Fake multi-draw...
        for (i = 0; i < drawCount; i += 1) {
            renderer->vkCmdDrawIndexedIndirect(
                vulkanCommandBuffer->commandBuffer,
                vulkanBuffer->buffer,
                offset + (pitch * i),
                1,
                pitch);
        }
    }

    VULKAN_INTERNAL_TrackBuffer(vulkanCommandBuffer, vulkanBuffer);
}

// Debug Naming

static void VULKAN_INTERNAL_SetBufferName(
    VulkanRenderer *renderer,
    VulkanBuffer *buffer,
    const char *text)
{
    VkDebugUtilsObjectNameInfoEXT nameInfo;

    if (renderer->debugMode && renderer->supportsDebugUtils) {
        nameInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
        nameInfo.pNext = NULL;
        nameInfo.pObjectName = text;
        nameInfo.objectType = VK_OBJECT_TYPE_BUFFER;
        nameInfo.objectHandle = (uint64_t)buffer->buffer;

        renderer->vkSetDebugUtilsObjectNameEXT(
            renderer->logicalDevice,
            &nameInfo);
    }
}

static void VULKAN_SetBufferName(
    SDL_GPURenderer *driverData,
    SDL_GPUBuffer *buffer,
    const char *text)
{
    VulkanRenderer *renderer = (VulkanRenderer *)driverData;
    VulkanBufferContainer *container = (VulkanBufferContainer *)buffer;
    size_t textLength = SDL_strlen(text) + 1;

    if (renderer->debugMode && renderer->supportsDebugUtils) {
        container->debugName = SDL_realloc(
            container->debugName,
            textLength);

        SDL_utf8strlcpy(
            container->debugName,
            text,
            textLength);

        for (Uint32 i = 0; i < container->bufferCount; i += 1) {
            VULKAN_INTERNAL_SetBufferName(
                renderer,
                container->buffers[i],
                text);
        }
    }
}

static void VULKAN_INTERNAL_SetTextureName(
    VulkanRenderer *renderer,
    VulkanTexture *texture,
    const char *text)
{
    VkDebugUtilsObjectNameInfoEXT nameInfo;

    if (renderer->debugMode && renderer->supportsDebugUtils) {
        nameInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
        nameInfo.pNext = NULL;
        nameInfo.pObjectName = text;
        nameInfo.objectType = VK_OBJECT_TYPE_IMAGE;
        nameInfo.objectHandle = (uint64_t)texture->image;

        renderer->vkSetDebugUtilsObjectNameEXT(
            renderer->logicalDevice,
            &nameInfo);
    }
}

static void VULKAN_SetTextureName(
    SDL_GPURenderer *driverData,
    SDL_GPUTexture *texture,
    const char *text)
{
    VulkanRenderer *renderer = (VulkanRenderer *)driverData;
    VulkanTextureContainer *container = (VulkanTextureContainer *)texture;
    size_t textLength = SDL_strlen(text) + 1;

    if (renderer->debugMode && renderer->supportsDebugUtils) {
        container->debugName = SDL_realloc(
            container->debugName,
            textLength);

        SDL_utf8strlcpy(
            container->debugName,
            text,
            textLength);

        for (Uint32 i = 0; i < container->textureCount; i += 1) {
            VULKAN_INTERNAL_SetTextureName(
                renderer,
                container->textures[i],
                text);
        }
    }
}

static void VULKAN_InsertDebugLabel(
    SDL_GPUCommandBuffer *commandBuffer,
    const char *text)
{
    VulkanCommandBuffer *vulkanCommandBuffer = (VulkanCommandBuffer *)commandBuffer;
    VulkanRenderer *renderer = vulkanCommandBuffer->renderer;
    VkDebugUtilsLabelEXT labelInfo;

    if (renderer->supportsDebugUtils) {
        SDL_zero(labelInfo);
        labelInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
        labelInfo.pLabelName = text;

        renderer->vkCmdInsertDebugUtilsLabelEXT(
            vulkanCommandBuffer->commandBuffer,
            &labelInfo);
    }
}

static void VULKAN_PushDebugGroup(
    SDL_GPUCommandBuffer *commandBuffer,
    const char *name)
{
    VulkanCommandBuffer *vulkanCommandBuffer = (VulkanCommandBuffer *)commandBuffer;
    VulkanRenderer *renderer = vulkanCommandBuffer->renderer;
    VkDebugUtilsLabelEXT labelInfo;

    if (renderer->supportsDebugUtils) {
        SDL_zero(labelInfo);
        labelInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
        labelInfo.pLabelName = name;

        renderer->vkCmdBeginDebugUtilsLabelEXT(
            vulkanCommandBuffer->commandBuffer,
            &labelInfo);
    }
}

static void VULKAN_PopDebugGroup(
    SDL_GPUCommandBuffer *commandBuffer)
{
    VulkanCommandBuffer *vulkanCommandBuffer = (VulkanCommandBuffer *)commandBuffer;
    VulkanRenderer *renderer = vulkanCommandBuffer->renderer;

    if (renderer->supportsDebugUtils) {
        renderer->vkCmdEndDebugUtilsLabelEXT(vulkanCommandBuffer->commandBuffer);
    }
}

static VulkanTexture *VULKAN_INTERNAL_CreateTexture(
    VulkanRenderer *renderer,
    bool transitionToDefaultLayout,
    const SDL_GPUTextureCreateInfo *createinfo)
{
    VkResult vulkanResult;
    VkImageCreateInfo imageCreateInfo;
    VkImageCreateFlags imageCreateFlags = 0;
    VkImageViewCreateInfo imageViewCreateInfo;
    Uint8 bindResult;
    VkImageUsageFlags vkUsageFlags = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    Uint32 layerCount = (createinfo->type == SDL_GPU_TEXTURETYPE_3D) ? 1 : createinfo->layer_count_or_depth;
    Uint32 depth = (createinfo->type == SDL_GPU_TEXTURETYPE_3D) ? createinfo->layer_count_or_depth : 1;

    VulkanTexture *texture = SDL_calloc(1, sizeof(VulkanTexture));
    texture->swizzle = SwizzleForSDLFormat(createinfo->format);
    texture->depth = depth;
    texture->usage = createinfo->usage;
    SDL_SetAtomicInt(&texture->referenceCount, 0);

    if (IsDepthFormat(createinfo->format)) {
        texture->aspectFlags = VK_IMAGE_ASPECT_DEPTH_BIT;

        if (IsStencilFormat(createinfo->format)) {
            texture->aspectFlags |= VK_IMAGE_ASPECT_STENCIL_BIT;
        }
    } else {
        texture->aspectFlags = VK_IMAGE_ASPECT_COLOR_BIT;
    }

    if (createinfo->type == SDL_GPU_TEXTURETYPE_CUBE || createinfo->type == SDL_GPU_TEXTURETYPE_CUBE_ARRAY) {
        imageCreateFlags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    } else if (createinfo->type == SDL_GPU_TEXTURETYPE_3D) {
        imageCreateFlags |= VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT;
    }

    if (createinfo->usage & (SDL_GPU_TEXTUREUSAGE_SAMPLER |
                             SDL_GPU_TEXTUREUSAGE_GRAPHICS_STORAGE_READ |
                             SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_READ)) {
        vkUsageFlags |= VK_IMAGE_USAGE_SAMPLED_BIT;
    }
    if (createinfo->usage & SDL_GPU_TEXTUREUSAGE_COLOR_TARGET) {
        vkUsageFlags |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    }
    if (createinfo->usage & SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET) {
        vkUsageFlags |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    }
    if (createinfo->usage & (SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE |
                             SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_SIMULTANEOUS_READ_WRITE)) {
        vkUsageFlags |= VK_IMAGE_USAGE_STORAGE_BIT;
    }

    imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCreateInfo.pNext = NULL;
    imageCreateInfo.flags = imageCreateFlags;
    imageCreateInfo.imageType = createinfo->type == SDL_GPU_TEXTURETYPE_3D ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
    imageCreateInfo.format = SDLToVK_TextureFormat[createinfo->format];
    imageCreateInfo.extent.width = createinfo->width;
    imageCreateInfo.extent.height = createinfo->height;
    imageCreateInfo.extent.depth = depth;
    imageCreateInfo.mipLevels = createinfo->num_levels;
    imageCreateInfo.arrayLayers = layerCount;
    imageCreateInfo.samples = SDLToVK_SampleCount[createinfo->sample_count];
    imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCreateInfo.usage = vkUsageFlags;
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCreateInfo.queueFamilyIndexCount = 0;
    imageCreateInfo.pQueueFamilyIndices = NULL;
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    vulkanResult = renderer->vkCreateImage(
        renderer->logicalDevice,
        &imageCreateInfo,
        NULL,
        &texture->image);

    if (vulkanResult != VK_SUCCESS) {
        VULKAN_INTERNAL_DestroyTexture(renderer, texture);
        CHECK_VULKAN_ERROR_AND_RETURN(vulkanResult, vkCreateImage, NULL);
    }

    bindResult = VULKAN_INTERNAL_BindMemoryForImage(
        renderer,
        texture->image,
        &texture->usedRegion);

    if (bindResult != 1) {
        renderer->vkDestroyImage(
            renderer->logicalDevice,
            texture->image,
            NULL);

        VULKAN_INTERNAL_DestroyTexture(renderer, texture);
        SET_STRING_ERROR_AND_RETURN("Unable to bind memory for texture!", NULL);
    }

    texture->usedRegion->vulkanTexture = texture; // lol

    if (createinfo->usage & (SDL_GPU_TEXTUREUSAGE_SAMPLER | SDL_GPU_TEXTUREUSAGE_GRAPHICS_STORAGE_READ | SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_READ)) {

        imageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        imageViewCreateInfo.pNext = NULL;
        imageViewCreateInfo.flags = 0;
        imageViewCreateInfo.image = texture->image;
        imageViewCreateInfo.format = SDLToVK_TextureFormat[createinfo->format];
        imageViewCreateInfo.components = texture->swizzle;
        imageViewCreateInfo.subresourceRange.aspectMask = texture->aspectFlags & ~VK_IMAGE_ASPECT_STENCIL_BIT; // Can't sample stencil values
        imageViewCreateInfo.subresourceRange.baseMipLevel = 0;
        imageViewCreateInfo.subresourceRange.levelCount = createinfo->num_levels;
        imageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
        imageViewCreateInfo.subresourceRange.layerCount = layerCount;

        if (createinfo->type == SDL_GPU_TEXTURETYPE_CUBE) {
            imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
        } else if (createinfo->type == SDL_GPU_TEXTURETYPE_CUBE_ARRAY) {
            imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
        } else if (createinfo->type == SDL_GPU_TEXTURETYPE_3D) {
            imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_3D;
        } else if (createinfo->type == SDL_GPU_TEXTURETYPE_2D_ARRAY) {
            imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        } else {
            imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        }

        vulkanResult = renderer->vkCreateImageView(
            renderer->logicalDevice,
            &imageViewCreateInfo,
            NULL,
            &texture->fullView);

        if (vulkanResult != VK_SUCCESS) {
            VULKAN_INTERNAL_DestroyTexture(renderer, texture);
            CHECK_VULKAN_ERROR_AND_RETURN(vulkanResult, "vkCreateImageView", NULL);
        }
    }

    // Define slices
    texture->subresourceCount = layerCount * createinfo->num_levels;
    texture->subresources = SDL_calloc(
        texture->subresourceCount,
        sizeof(VulkanTextureSubresource));

    for (Uint32 i = 0; i < layerCount; i += 1) {
        for (Uint32 j = 0; j < createinfo->num_levels; j += 1) {
            Uint32 subresourceIndex = VULKAN_INTERNAL_GetTextureSubresourceIndex(
                j,
                i,
                createinfo->num_levels);

            if (createinfo->usage & SDL_GPU_TEXTUREUSAGE_COLOR_TARGET) {
                texture->subresources[subresourceIndex].renderTargetViews = SDL_malloc(
                    depth * sizeof(VkImageView));

                if (depth > 1) {
                    for (Uint32 k = 0; k < depth; k += 1) {
                        if (!VULKAN_INTERNAL_CreateRenderTargetView(
                            renderer,
                            texture,
                            k,
                            j,
                            SDLToVK_TextureFormat[createinfo->format],
                            texture->swizzle,
                            &texture->subresources[subresourceIndex].renderTargetViews[k])) {
                            VULKAN_INTERNAL_DestroyTexture(renderer, texture);
                            return NULL;
                        }
                    }
                } else {
                    if (!VULKAN_INTERNAL_CreateRenderTargetView(
                        renderer,
                        texture,
                        i,
                        j,
                        SDLToVK_TextureFormat[createinfo->format],
                        texture->swizzle,
                        &texture->subresources[subresourceIndex].renderTargetViews[0])) {
                        VULKAN_INTERNAL_DestroyTexture(renderer, texture);
                        return NULL;
                    }
                }
            }

            if ((createinfo->usage & SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE) || (createinfo->usage & SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_SIMULTANEOUS_READ_WRITE)) {
                if (!VULKAN_INTERNAL_CreateSubresourceView(
                    renderer,
                    createinfo,
                    texture,
                    i,
                    j,
                    texture->swizzle,
                    &texture->subresources[subresourceIndex].computeWriteView)) {
                    VULKAN_INTERNAL_DestroyTexture(renderer, texture);
                    return NULL;
                }
            }

            if (createinfo->usage & SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET) {
                if (!VULKAN_INTERNAL_CreateSubresourceView(
                    renderer,
                    createinfo,
                    texture,
                    i,
                    j,
                    texture->swizzle,
                    &texture->subresources[subresourceIndex].depthStencilView)) {
                    VULKAN_INTERNAL_DestroyTexture(renderer, texture);
                    return NULL;
                }
            }

            texture->subresources[subresourceIndex].parent = texture;
            texture->subresources[subresourceIndex].layer = i;
            texture->subresources[subresourceIndex].level = j;
        }
    }

    // Set debug name if applicable
    if (renderer->debugMode && renderer->supportsDebugUtils && SDL_HasProperty(createinfo->props, SDL_PROP_GPU_TEXTURE_CREATE_NAME_STRING)) {
        VkDebugUtilsObjectNameInfoEXT nameInfo;
        nameInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
        nameInfo.pNext = NULL;
        nameInfo.pObjectName = SDL_GetStringProperty(createinfo->props, SDL_PROP_GPU_TEXTURE_CREATE_NAME_STRING, NULL);
        nameInfo.objectType = VK_OBJECT_TYPE_IMAGE;
        nameInfo.objectHandle = (uint64_t)texture->image;

        renderer->vkSetDebugUtilsObjectNameEXT(
            renderer->logicalDevice,
            &nameInfo);
    }

    if (transitionToDefaultLayout) {
        // Let's transition to the default barrier state, because for some reason Vulkan doesn't let us do that with initialLayout.
        VulkanCommandBuffer *barrierCommandBuffer = (VulkanCommandBuffer *)VULKAN_AcquireCommandBuffer((SDL_GPURenderer *)renderer);
        VULKAN_INTERNAL_TextureTransitionToDefaultUsage(
            renderer,
            barrierCommandBuffer,
            VULKAN_TEXTURE_USAGE_MODE_UNINITIALIZED,
            texture);
        VULKAN_INTERNAL_TrackTexture(barrierCommandBuffer, texture);
        VULKAN_Submit((SDL_GPUCommandBuffer *)barrierCommandBuffer);
    }

    return texture;
}

static void VULKAN_INTERNAL_CycleActiveBuffer(
    VulkanRenderer *renderer,
    VulkanBufferContainer *container)
{
    VulkanBuffer *buffer;

    // If a previously-cycled buffer is available, we can use that.
    for (Uint32 i = 0; i < container->bufferCount; i += 1) {
        buffer = container->buffers[i];
        if (SDL_GetAtomicInt(&buffer->referenceCount) == 0) {
            container->activeBuffer = buffer;
            return;
        }
    }

    // No buffer handle is available, create a new one.
    buffer = VULKAN_INTERNAL_CreateBuffer(
        renderer,
        container->activeBuffer->size,
        container->activeBuffer->usage,
        container->activeBuffer->type,
        container->dedicated,
        container->debugName);

    if (!buffer) {
        return;
    }

    EXPAND_ARRAY_IF_NEEDED(
        container->buffers,
        VulkanBuffer *,
        container->bufferCount + 1,
        container->bufferCapacity,
        container->bufferCapacity * 2);

    container->buffers[container->bufferCount] = buffer;
    buffer->container = container;
    buffer->containerIndex = container->bufferCount;
    container->bufferCount += 1;

    container->activeBuffer = buffer;
}

static void VULKAN_INTERNAL_CycleActiveTexture(
    VulkanRenderer *renderer,
    VulkanCommandBuffer *commandBuffer,
    VulkanTextureContainer *container)
{
    VulkanTexture *texture;

    // If a previously-cycled texture is available, we can use that.
    for (Uint32 i = 0; i < container->textureCount; i += 1) {
        texture = container->textures[i];

        if (SDL_GetAtomicInt(&texture->referenceCount) == 0) {
            container->activeTexture = texture;
            return;
        }
    }

    // No texture is available, generate a new one.
    texture = VULKAN_INTERNAL_CreateTexture(
        renderer,
        false,
        &container->header.info);

    VULKAN_INTERNAL_TextureTransitionToDefaultUsage(
        renderer,
        commandBuffer,
        VULKAN_TEXTURE_USAGE_MODE_UNINITIALIZED,
        texture);

    if (!texture) {
        return;
    }

    EXPAND_ARRAY_IF_NEEDED(
        container->textures,
        VulkanTexture *,
        container->textureCount + 1,
        container->textureCapacity,
        container->textureCapacity * 2);

    container->textures[container->textureCount] = texture;
    texture->container = container;
    texture->containerIndex = container->textureCount;
    container->textureCount += 1;

    container->activeTexture = texture;
}

static VulkanBuffer *VULKAN_INTERNAL_PrepareBufferForWrite(
    VulkanRenderer *renderer,
    VulkanCommandBuffer *commandBuffer,
    VulkanBufferContainer *bufferContainer,
    bool cycle,
    VulkanBufferUsageMode destinationUsageMode)
{
    if (
        cycle &&
        SDL_GetAtomicInt(&bufferContainer->activeBuffer->referenceCount) > 0) {
        VULKAN_INTERNAL_CycleActiveBuffer(
            renderer,
            bufferContainer);
    }

    VULKAN_INTERNAL_BufferTransitionFromDefaultUsage(
        renderer,
        commandBuffer,
        destinationUsageMode,
        bufferContainer->activeBuffer);

    return bufferContainer->activeBuffer;
}

static VulkanTextureSubresource *VULKAN_INTERNAL_PrepareTextureSubresourceForWrite(
    VulkanRenderer *renderer,
    VulkanCommandBuffer *commandBuffer,
    VulkanTextureContainer *textureContainer,
    Uint32 layer,
    Uint32 level,
    bool cycle,
    VulkanTextureUsageMode destinationUsageMode)
{
    VulkanTextureSubresource *textureSubresource = VULKAN_INTERNAL_FetchTextureSubresource(
        textureContainer,
        layer,
        level);

    if (
        cycle &&
        textureContainer->canBeCycled &&
        SDL_GetAtomicInt(&textureContainer->activeTexture->referenceCount) > 0) {
        VULKAN_INTERNAL_CycleActiveTexture(
            renderer,
            commandBuffer,
            textureContainer);

        textureSubresource = VULKAN_INTERNAL_FetchTextureSubresource(
            textureContainer,
            layer,
            level);
    }

    // always do barrier because of layout transitions
    VULKAN_INTERNAL_TextureSubresourceTransitionFromDefaultUsage(
        renderer,
        commandBuffer,
        destinationUsageMode,
        textureSubresource);

    return textureSubresource;
}

static VkRenderPass VULKAN_INTERNAL_CreateRenderPass(
    VulkanRenderer *renderer,
    const SDL_GPUColorTargetInfo *colorTargetInfos,
    Uint32 numColorTargets,
    const SDL_GPUDepthStencilTargetInfo *depthStencilTargetInfo)
{
    VkResult vulkanResult;
    VkAttachmentDescription attachmentDescriptions[2 * MAX_COLOR_TARGET_BINDINGS + 1 /* depth */];
    VkAttachmentReference colorAttachmentReferences[MAX_COLOR_TARGET_BINDINGS];
    VkAttachmentReference resolveReferences[MAX_COLOR_TARGET_BINDINGS];
    VkAttachmentReference depthStencilAttachmentReference;
    VkRenderPassCreateInfo renderPassCreateInfo;
    VkSubpassDescription subpass;
    VkRenderPass renderPass;
    Uint32 i;

    Uint32 attachmentDescriptionCount = 0;
    Uint32 colorAttachmentReferenceCount = 0;
    Uint32 resolveReferenceCount = 0;

    for (i = 0; i < numColorTargets; i += 1) {
        VulkanTextureContainer *container = (VulkanTextureContainer *)colorTargetInfos[i].texture;
        attachmentDescriptions[attachmentDescriptionCount].flags = 0;
        attachmentDescriptions[attachmentDescriptionCount].format = SDLToVK_TextureFormat[container->header.info.format];
        attachmentDescriptions[attachmentDescriptionCount].samples = SDLToVK_SampleCount[container->header.info.sample_count];
        attachmentDescriptions[attachmentDescriptionCount].loadOp = SDLToVK_LoadOp[colorTargetInfos[i].load_op];
        attachmentDescriptions[attachmentDescriptionCount].storeOp = SDLToVK_StoreOp[colorTargetInfos[i].store_op];
        attachmentDescriptions[attachmentDescriptionCount].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachmentDescriptions[attachmentDescriptionCount].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachmentDescriptions[attachmentDescriptionCount].initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        attachmentDescriptions[attachmentDescriptionCount].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        colorAttachmentReferences[colorAttachmentReferenceCount].attachment = attachmentDescriptionCount;
        colorAttachmentReferences[colorAttachmentReferenceCount].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        attachmentDescriptionCount += 1;
        colorAttachmentReferenceCount += 1;

        if (colorTargetInfos[i].store_op == SDL_GPU_STOREOP_RESOLVE || colorTargetInfos[i].store_op == SDL_GPU_STOREOP_RESOLVE_AND_STORE) {
            VulkanTextureContainer *resolveContainer = (VulkanTextureContainer *)colorTargetInfos[i].resolve_texture;

            attachmentDescriptions[attachmentDescriptionCount].flags = 0;
            attachmentDescriptions[attachmentDescriptionCount].format = SDLToVK_TextureFormat[resolveContainer->header.info.format];
            attachmentDescriptions[attachmentDescriptionCount].samples = SDLToVK_SampleCount[resolveContainer->header.info.sample_count];
            attachmentDescriptions[attachmentDescriptionCount].loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; // The texture will be overwritten anyway
            attachmentDescriptions[attachmentDescriptionCount].storeOp = VK_ATTACHMENT_STORE_OP_STORE; // Always store the resolve texture
            attachmentDescriptions[attachmentDescriptionCount].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            attachmentDescriptions[attachmentDescriptionCount].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            attachmentDescriptions[attachmentDescriptionCount].initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            attachmentDescriptions[attachmentDescriptionCount].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

            resolveReferences[resolveReferenceCount].attachment = attachmentDescriptionCount;
            resolveReferences[resolveReferenceCount].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

            attachmentDescriptionCount += 1;
            resolveReferenceCount += 1;
        }
    }

    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.flags = 0;
    subpass.inputAttachmentCount = 0;
    subpass.pInputAttachments = NULL;
    subpass.colorAttachmentCount = numColorTargets;
    subpass.pColorAttachments = colorAttachmentReferences;
    subpass.preserveAttachmentCount = 0;
    subpass.pPreserveAttachments = NULL;

    if (depthStencilTargetInfo == NULL) {
        subpass.pDepthStencilAttachment = NULL;
    } else {
        VulkanTextureContainer *container = (VulkanTextureContainer *)depthStencilTargetInfo->texture;

        attachmentDescriptions[attachmentDescriptionCount].flags = 0;
        attachmentDescriptions[attachmentDescriptionCount].format = SDLToVK_TextureFormat[container->header.info.format];
        attachmentDescriptions[attachmentDescriptionCount].samples = SDLToVK_SampleCount[container->header.info.sample_count];
        attachmentDescriptions[attachmentDescriptionCount].loadOp = SDLToVK_LoadOp[depthStencilTargetInfo->load_op];
        attachmentDescriptions[attachmentDescriptionCount].storeOp = SDLToVK_StoreOp[depthStencilTargetInfo->store_op];
        attachmentDescriptions[attachmentDescriptionCount].stencilLoadOp = SDLToVK_LoadOp[depthStencilTargetInfo->stencil_load_op];
        attachmentDescriptions[attachmentDescriptionCount].stencilStoreOp = SDLToVK_StoreOp[depthStencilTargetInfo->stencil_store_op];
        attachmentDescriptions[attachmentDescriptionCount].initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        attachmentDescriptions[attachmentDescriptionCount].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        depthStencilAttachmentReference.attachment = attachmentDescriptionCount;
        depthStencilAttachmentReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        subpass.pDepthStencilAttachment = &depthStencilAttachmentReference;

        attachmentDescriptionCount += 1;
    }

    if (resolveReferenceCount > 0) {
        subpass.pResolveAttachments = resolveReferences;
    } else {
        subpass.pResolveAttachments = NULL;
    }

    renderPassCreateInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassCreateInfo.pNext = NULL;
    renderPassCreateInfo.flags = 0;
    renderPassCreateInfo.pAttachments = attachmentDescriptions;
    renderPassCreateInfo.attachmentCount = attachmentDescriptionCount;
    renderPassCreateInfo.subpassCount = 1;
    renderPassCreateInfo.pSubpasses = &subpass;
    renderPassCreateInfo.dependencyCount = 0;
    renderPassCreateInfo.pDependencies = NULL;

    vulkanResult = renderer->vkCreateRenderPass(
        renderer->logicalDevice,
        &renderPassCreateInfo,
        NULL,
        &renderPass);

    CHECK_VULKAN_ERROR_AND_RETURN(vulkanResult, vkCreateRenderPass, VK_NULL_HANDLE);

    return renderPass;
}

static VkRenderPass VULKAN_INTERNAL_CreateTransientRenderPass(
    VulkanRenderer *renderer,
    SDL_GPUGraphicsPipelineTargetInfo targetInfo,
    VkSampleCountFlagBits sampleCount)
{
    VkAttachmentDescription attachmentDescriptions[MAX_COLOR_TARGET_BINDINGS + 1 /* depth */];
    VkAttachmentReference colorAttachmentReferences[MAX_COLOR_TARGET_BINDINGS];
    VkAttachmentReference depthStencilAttachmentReference;
    SDL_GPUColorTargetDescription attachmentDescription;
    VkSubpassDescription subpass;
    VkRenderPassCreateInfo renderPassCreateInfo;
    VkRenderPass renderPass;
    VkResult result;

    Uint32 attachmentDescriptionCount = 0;
    Uint32 colorAttachmentReferenceCount = 0;
    Uint32 i;

    for (i = 0; i < targetInfo.num_color_targets; i += 1) {
        attachmentDescription = targetInfo.color_target_descriptions[i];

        attachmentDescriptions[attachmentDescriptionCount].flags = 0;
        attachmentDescriptions[attachmentDescriptionCount].format = SDLToVK_TextureFormat[attachmentDescription.format];
        attachmentDescriptions[attachmentDescriptionCount].samples = sampleCount;
        attachmentDescriptions[attachmentDescriptionCount].loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachmentDescriptions[attachmentDescriptionCount].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachmentDescriptions[attachmentDescriptionCount].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachmentDescriptions[attachmentDescriptionCount].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachmentDescriptions[attachmentDescriptionCount].initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        attachmentDescriptions[attachmentDescriptionCount].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        colorAttachmentReferences[colorAttachmentReferenceCount].attachment = attachmentDescriptionCount;
        colorAttachmentReferences[colorAttachmentReferenceCount].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        attachmentDescriptionCount += 1;
        colorAttachmentReferenceCount += 1;
    }

    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.flags = 0;
    subpass.inputAttachmentCount = 0;
    subpass.pInputAttachments = NULL;
    subpass.colorAttachmentCount = targetInfo.num_color_targets;
    subpass.pColorAttachments = colorAttachmentReferences;
    subpass.preserveAttachmentCount = 0;
    subpass.pPreserveAttachments = NULL;

    if (targetInfo.has_depth_stencil_target) {
        attachmentDescriptions[attachmentDescriptionCount].flags = 0;
        attachmentDescriptions[attachmentDescriptionCount].format = SDLToVK_TextureFormat[targetInfo.depth_stencil_format];
        attachmentDescriptions[attachmentDescriptionCount].samples = sampleCount;
        attachmentDescriptions[attachmentDescriptionCount].loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachmentDescriptions[attachmentDescriptionCount].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachmentDescriptions[attachmentDescriptionCount].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachmentDescriptions[attachmentDescriptionCount].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachmentDescriptions[attachmentDescriptionCount].initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        attachmentDescriptions[attachmentDescriptionCount].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        depthStencilAttachmentReference.attachment = attachmentDescriptionCount;
        depthStencilAttachmentReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        subpass.pDepthStencilAttachment = &depthStencilAttachmentReference;

        attachmentDescriptionCount += 1;
    } else {
        subpass.pDepthStencilAttachment = NULL;
    }

    // Resolve attachments aren't needed for transient passes
    subpass.pResolveAttachments = NULL;

    renderPassCreateInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassCreateInfo.pNext = NULL;
    renderPassCreateInfo.flags = 0;
    renderPassCreateInfo.pAttachments = attachmentDescriptions;
    renderPassCreateInfo.attachmentCount = attachmentDescriptionCount;
    renderPassCreateInfo.subpassCount = 1;
    renderPassCreateInfo.pSubpasses = &subpass;
    renderPassCreateInfo.dependencyCount = 0;
    renderPassCreateInfo.pDependencies = NULL;

    result = renderer->vkCreateRenderPass(
        renderer->logicalDevice,
        &renderPassCreateInfo,
        NULL,
        &renderPass);

    CHECK_VULKAN_ERROR_AND_RETURN(result, vkCreateRenderPass, VK_NULL_HANDLE);

    return renderPass;
}

static SDL_GPUGraphicsPipeline *VULKAN_CreateGraphicsPipeline(
    SDL_GPURenderer *driverData,
    const SDL_GPUGraphicsPipelineCreateInfo *createinfo)
{
    VkResult vulkanResult;
    Uint32 i;

    VulkanGraphicsPipeline *graphicsPipeline = (VulkanGraphicsPipeline *)SDL_malloc(sizeof(VulkanGraphicsPipeline));
    VkGraphicsPipelineCreateInfo vkPipelineCreateInfo;

    VkPipelineShaderStageCreateInfo shaderStageCreateInfos[2];

    VkPipelineVertexInputStateCreateInfo vertexInputStateCreateInfo;
    VkVertexInputBindingDescription *vertexInputBindingDescriptions = SDL_stack_alloc(VkVertexInputBindingDescription, createinfo->vertex_input_state.num_vertex_buffers);
    VkVertexInputAttributeDescription *vertexInputAttributeDescriptions = SDL_stack_alloc(VkVertexInputAttributeDescription, createinfo->vertex_input_state.num_vertex_attributes);

    VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCreateInfo;

    VkPipelineViewportStateCreateInfo viewportStateCreateInfo;

    VkPipelineRasterizationStateCreateInfo rasterizationStateCreateInfo;

    VkPipelineMultisampleStateCreateInfo multisampleStateCreateInfo;

    VkPipelineDepthStencilStateCreateInfo depthStencilStateCreateInfo;
    VkStencilOpState frontStencilState;
    VkStencilOpState backStencilState;

    VkPipelineColorBlendStateCreateInfo colorBlendStateCreateInfo;
    VkPipelineColorBlendAttachmentState *colorBlendAttachmentStates = SDL_stack_alloc(
        VkPipelineColorBlendAttachmentState,
        createinfo->target_info.num_color_targets);

    static const VkDynamicState dynamicStates[] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
        VK_DYNAMIC_STATE_BLEND_CONSTANTS,
        VK_DYNAMIC_STATE_STENCIL_REFERENCE
    };
    VkPipelineDynamicStateCreateInfo dynamicStateCreateInfo;

    VulkanRenderer *renderer = (VulkanRenderer *)driverData;

    // Create a "compatible" render pass

    VkRenderPass transientRenderPass = VULKAN_INTERNAL_CreateTransientRenderPass(
        renderer,
        createinfo->target_info,
        SDLToVK_SampleCount[createinfo->multisample_state.sample_count]);

    // Dynamic state

    dynamicStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicStateCreateInfo.pNext = NULL;
    dynamicStateCreateInfo.flags = 0;
    dynamicStateCreateInfo.dynamicStateCount = SDL_arraysize(dynamicStates);
    dynamicStateCreateInfo.pDynamicStates = dynamicStates;

    // Shader stages

    graphicsPipeline->vertexShader = (VulkanShader *)createinfo->vertex_shader;
    SDL_AtomicIncRef(&graphicsPipeline->vertexShader->referenceCount);

    shaderStageCreateInfos[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageCreateInfos[0].pNext = NULL;
    shaderStageCreateInfos[0].flags = 0;
    shaderStageCreateInfos[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    shaderStageCreateInfos[0].module = graphicsPipeline->vertexShader->shaderModule;
    shaderStageCreateInfos[0].pName = graphicsPipeline->vertexShader->entrypointName;
    shaderStageCreateInfos[0].pSpecializationInfo = NULL;

    graphicsPipeline->fragmentShader = (VulkanShader *)createinfo->fragment_shader;
    SDL_AtomicIncRef(&graphicsPipeline->fragmentShader->referenceCount);

    shaderStageCreateInfos[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageCreateInfos[1].pNext = NULL;
    shaderStageCreateInfos[1].flags = 0;
    shaderStageCreateInfos[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    shaderStageCreateInfos[1].module = graphicsPipeline->fragmentShader->shaderModule;
    shaderStageCreateInfos[1].pName = graphicsPipeline->fragmentShader->entrypointName;
    shaderStageCreateInfos[1].pSpecializationInfo = NULL;

    if (renderer->debugMode) {
        if (graphicsPipeline->vertexShader->stage != SDL_GPU_SHADERSTAGE_VERTEX) {
            SDL_assert_release(!"CreateGraphicsPipeline was passed a fragment shader for the vertex stage");
        }
        if (graphicsPipeline->fragmentShader->stage != SDL_GPU_SHADERSTAGE_FRAGMENT) {
            SDL_assert_release(!"CreateGraphicsPipeline was passed a vertex shader for the fragment stage");
        }
    }

    // Vertex input

    for (i = 0; i < createinfo->vertex_input_state.num_vertex_buffers; i += 1) {
        vertexInputBindingDescriptions[i].binding = createinfo->vertex_input_state.vertex_buffer_descriptions[i].slot;
        vertexInputBindingDescriptions[i].inputRate = SDLToVK_VertexInputRate[createinfo->vertex_input_state.vertex_buffer_descriptions[i].input_rate];
        vertexInputBindingDescriptions[i].stride = createinfo->vertex_input_state.vertex_buffer_descriptions[i].pitch;
    }

    for (i = 0; i < createinfo->vertex_input_state.num_vertex_attributes; i += 1) {
        vertexInputAttributeDescriptions[i].binding = createinfo->vertex_input_state.vertex_attributes[i].buffer_slot;
        vertexInputAttributeDescriptions[i].format = SDLToVK_VertexFormat[createinfo->vertex_input_state.vertex_attributes[i].format];
        vertexInputAttributeDescriptions[i].location = createinfo->vertex_input_state.vertex_attributes[i].location;
        vertexInputAttributeDescriptions[i].offset = createinfo->vertex_input_state.vertex_attributes[i].offset;
    }

    vertexInputStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputStateCreateInfo.pNext = NULL;
    vertexInputStateCreateInfo.flags = 0;
    vertexInputStateCreateInfo.vertexBindingDescriptionCount = createinfo->vertex_input_state.num_vertex_buffers;
    vertexInputStateCreateInfo.pVertexBindingDescriptions = vertexInputBindingDescriptions;
    vertexInputStateCreateInfo.vertexAttributeDescriptionCount = createinfo->vertex_input_state.num_vertex_attributes;
    vertexInputStateCreateInfo.pVertexAttributeDescriptions = vertexInputAttributeDescriptions;

    // Topology

    inputAssemblyStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssemblyStateCreateInfo.pNext = NULL;
    inputAssemblyStateCreateInfo.flags = 0;
    inputAssemblyStateCreateInfo.primitiveRestartEnable = VK_FALSE;
    inputAssemblyStateCreateInfo.topology = SDLToVK_PrimitiveType[createinfo->primitive_type];

    graphicsPipeline->primitiveType = createinfo->primitive_type;

    // Viewport

    // NOTE: viewport and scissor are dynamic, and must be set using the command buffer

    viewportStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportStateCreateInfo.pNext = NULL;
    viewportStateCreateInfo.flags = 0;
    viewportStateCreateInfo.viewportCount = 1;
    viewportStateCreateInfo.pViewports = NULL;
    viewportStateCreateInfo.scissorCount = 1;
    viewportStateCreateInfo.pScissors = NULL;

    // Rasterization

    rasterizationStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizationStateCreateInfo.pNext = NULL;
    rasterizationStateCreateInfo.flags = 0;
    rasterizationStateCreateInfo.depthClampEnable = !createinfo->rasterizer_state.enable_depth_clip;
    rasterizationStateCreateInfo.rasterizerDiscardEnable = VK_FALSE;
    rasterizationStateCreateInfo.polygonMode = SDLToVK_PolygonMode(
        renderer,
        createinfo->rasterizer_state.fill_mode);
    rasterizationStateCreateInfo.cullMode = SDLToVK_CullMode[createinfo->rasterizer_state.cull_mode];
    rasterizationStateCreateInfo.frontFace = SDLToVK_FrontFace[createinfo->rasterizer_state.front_face];
    rasterizationStateCreateInfo.depthBiasEnable =
        createinfo->rasterizer_state.enable_depth_bias;
    rasterizationStateCreateInfo.depthBiasConstantFactor =
        createinfo->rasterizer_state.depth_bias_constant_factor;
    rasterizationStateCreateInfo.depthBiasClamp =
        createinfo->rasterizer_state.depth_bias_clamp;
    rasterizationStateCreateInfo.depthBiasSlopeFactor =
        createinfo->rasterizer_state.depth_bias_slope_factor;
    rasterizationStateCreateInfo.lineWidth = 1.0f;

    // Multisample

    Uint32 sampleMask = 0xFFFFFFFF;

    multisampleStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampleStateCreateInfo.pNext = NULL;
    multisampleStateCreateInfo.flags = 0;
    multisampleStateCreateInfo.rasterizationSamples = SDLToVK_SampleCount[createinfo->multisample_state.sample_count];
    multisampleStateCreateInfo.sampleShadingEnable = VK_FALSE;
    multisampleStateCreateInfo.minSampleShading = 1.0f;
    multisampleStateCreateInfo.pSampleMask = &sampleMask;
    multisampleStateCreateInfo.alphaToCoverageEnable = createinfo->multisample_state.enable_alpha_to_coverage;
    multisampleStateCreateInfo.alphaToOneEnable = VK_FALSE;

    // Depth Stencil State

    frontStencilState.failOp = SDLToVK_StencilOp[createinfo->depth_stencil_state.front_stencil_state.fail_op];
    frontStencilState.passOp = SDLToVK_StencilOp[createinfo->depth_stencil_state.front_stencil_state.pass_op];
    frontStencilState.depthFailOp = SDLToVK_StencilOp[createinfo->depth_stencil_state.front_stencil_state.depth_fail_op];
    frontStencilState.compareOp = SDLToVK_CompareOp[createinfo->depth_stencil_state.front_stencil_state.compare_op];
    frontStencilState.compareMask =
        createinfo->depth_stencil_state.compare_mask;
    frontStencilState.writeMask =
        createinfo->depth_stencil_state.write_mask;
    frontStencilState.reference = 0;

    backStencilState.failOp = SDLToVK_StencilOp[createinfo->depth_stencil_state.back_stencil_state.fail_op];
    backStencilState.passOp = SDLToVK_StencilOp[createinfo->depth_stencil_state.back_stencil_state.pass_op];
    backStencilState.depthFailOp = SDLToVK_StencilOp[createinfo->depth_stencil_state.back_stencil_state.depth_fail_op];
    backStencilState.compareOp = SDLToVK_CompareOp[createinfo->depth_stencil_state.back_stencil_state.compare_op];
    backStencilState.compareMask =
        createinfo->depth_stencil_state.compare_mask;
    backStencilState.writeMask =
        createinfo->depth_stencil_state.write_mask;
    backStencilState.reference = 0;

    depthStencilStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencilStateCreateInfo.pNext = NULL;
    depthStencilStateCreateInfo.flags = 0;
    depthStencilStateCreateInfo.depthTestEnable =
        createinfo->depth_stencil_state.enable_depth_test;
    depthStencilStateCreateInfo.depthWriteEnable =
        createinfo->depth_stencil_state.enable_depth_write;
    depthStencilStateCreateInfo.depthCompareOp = SDLToVK_CompareOp[createinfo->depth_stencil_state.compare_op];
    depthStencilStateCreateInfo.depthBoundsTestEnable = VK_FALSE;
    depthStencilStateCreateInfo.stencilTestEnable =
        createinfo->depth_stencil_state.enable_stencil_test;
    depthStencilStateCreateInfo.front = frontStencilState;
    depthStencilStateCreateInfo.back = backStencilState;
    depthStencilStateCreateInfo.minDepthBounds = 0; // unused
    depthStencilStateCreateInfo.maxDepthBounds = 0; // unused

    // Color Blend

    for (i = 0; i < createinfo->target_info.num_color_targets; i += 1) {
        SDL_GPUColorTargetBlendState blendState = createinfo->target_info.color_target_descriptions[i].blend_state;
        SDL_GPUColorComponentFlags colorWriteMask = blendState.enable_color_write_mask ?
            blendState.color_write_mask :
            0xF;

        colorBlendAttachmentStates[i].blendEnable =
            blendState.enable_blend;
        colorBlendAttachmentStates[i].srcColorBlendFactor = SDLToVK_BlendFactor[blendState.src_color_blendfactor];
        colorBlendAttachmentStates[i].dstColorBlendFactor = SDLToVK_BlendFactor[blendState.dst_color_blendfactor];
        colorBlendAttachmentStates[i].colorBlendOp = SDLToVK_BlendOp[blendState.color_blend_op];
        colorBlendAttachmentStates[i].srcAlphaBlendFactor = SDLToVK_BlendFactor[blendState.src_alpha_blendfactor];
        colorBlendAttachmentStates[i].dstAlphaBlendFactor = SDLToVK_BlendFactor[blendState.dst_alpha_blendfactor];
        colorBlendAttachmentStates[i].alphaBlendOp = SDLToVK_BlendOp[blendState.alpha_blend_op];
        colorBlendAttachmentStates[i].colorWriteMask =
            colorWriteMask;
    }

    colorBlendStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlendStateCreateInfo.pNext = NULL;
    colorBlendStateCreateInfo.flags = 0;
    colorBlendStateCreateInfo.attachmentCount =
        createinfo->target_info.num_color_targets;
    colorBlendStateCreateInfo.pAttachments =
        colorBlendAttachmentStates;
    colorBlendStateCreateInfo.blendConstants[0] = 1.0f;
    colorBlendStateCreateInfo.blendConstants[1] = 1.0f;
    colorBlendStateCreateInfo.blendConstants[2] = 1.0f;
    colorBlendStateCreateInfo.blendConstants[3] = 1.0f;

    // We don't support LogicOp, so this is easy.
    colorBlendStateCreateInfo.logicOpEnable = VK_FALSE;
    colorBlendStateCreateInfo.logicOp = 0;

    // Pipeline Layout

    graphicsPipeline->resourceLayout =
        VULKAN_INTERNAL_FetchGraphicsPipelineResourceLayout(
            renderer,
            graphicsPipeline->vertexShader,
            graphicsPipeline->fragmentShader);

    if (graphicsPipeline->resourceLayout == NULL) {
        SDL_stack_free(vertexInputBindingDescriptions);
        SDL_stack_free(vertexInputAttributeDescriptions);
        SDL_stack_free(colorBlendAttachmentStates);
        SDL_free(graphicsPipeline);
        SET_STRING_ERROR_AND_RETURN("Failed to initialize pipeline resource layout!", NULL);
    }

    // Pipeline

    vkPipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    vkPipelineCreateInfo.pNext = NULL;
    vkPipelineCreateInfo.flags = 0;
    vkPipelineCreateInfo.stageCount = 2;
    vkPipelineCreateInfo.pStages = shaderStageCreateInfos;
    vkPipelineCreateInfo.pVertexInputState = &vertexInputStateCreateInfo;
    vkPipelineCreateInfo.pInputAssemblyState = &inputAssemblyStateCreateInfo;
    vkPipelineCreateInfo.pTessellationState = VK_NULL_HANDLE;
    vkPipelineCreateInfo.pViewportState = &viewportStateCreateInfo;
    vkPipelineCreateInfo.pRasterizationState = &rasterizationStateCreateInfo;
    vkPipelineCreateInfo.pMultisampleState = &multisampleStateCreateInfo;
    vkPipelineCreateInfo.pDepthStencilState = &depthStencilStateCreateInfo;
    vkPipelineCreateInfo.pColorBlendState = &colorBlendStateCreateInfo;
    vkPipelineCreateInfo.pDynamicState = &dynamicStateCreateInfo;
    vkPipelineCreateInfo.layout = graphicsPipeline->resourceLayout->pipelineLayout;
    vkPipelineCreateInfo.renderPass = transientRenderPass;
    vkPipelineCreateInfo.subpass = 0;
    vkPipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
    vkPipelineCreateInfo.basePipelineIndex = 0;

    // TODO: enable pipeline caching
    vulkanResult = renderer->vkCreateGraphicsPipelines(
        renderer->logicalDevice,
        VK_NULL_HANDLE,
        1,
        &vkPipelineCreateInfo,
        NULL,
        &graphicsPipeline->pipeline);

    SDL_stack_free(vertexInputBindingDescriptions);
    SDL_stack_free(vertexInputAttributeDescriptions);
    SDL_stack_free(colorBlendAttachmentStates);

    renderer->vkDestroyRenderPass(
        renderer->logicalDevice,
        transientRenderPass,
        NULL);

    if (vulkanResult != VK_SUCCESS) {
        SDL_free(graphicsPipeline);
        CHECK_VULKAN_ERROR_AND_RETURN(vulkanResult, vkCreateGraphicsPipelines, NULL);
    }

    SDL_SetAtomicInt(&graphicsPipeline->referenceCount, 0);

    if (renderer->debugMode && renderer->supportsDebugUtils && SDL_HasProperty(createinfo->props, SDL_PROP_GPU_GRAPHICSPIPELINE_CREATE_NAME_STRING)) {
        VkDebugUtilsObjectNameInfoEXT nameInfo;
        nameInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
        nameInfo.pNext = NULL;
        nameInfo.pObjectName = SDL_GetStringProperty(createinfo->props, SDL_PROP_GPU_GRAPHICSPIPELINE_CREATE_NAME_STRING, NULL);
        nameInfo.objectType = VK_OBJECT_TYPE_PIPELINE;
        nameInfo.objectHandle = (uint64_t)graphicsPipeline->pipeline;

        renderer->vkSetDebugUtilsObjectNameEXT(
            renderer->logicalDevice,
            &nameInfo);
    }

    return (SDL_GPUGraphicsPipeline *)graphicsPipeline;
}

static bool VULKAN_INTERNAL_IsValidShaderBytecode(
    const Uint8 *code,
    size_t codeSize)
{
    // SPIR-V bytecode has a 4 byte header containing 0x07230203. SPIR-V is
    // defined as a stream of words and not a stream of bytes so both byte
    // orders need to be considered.
    //
    // FIXME: It is uncertain if drivers are able to load both byte orders. If
    // needed we may need to do an optional swizzle internally so apps can
    // continue to treat shader code as an opaque blob.
    if (codeSize < 4 || code == NULL) {
        return false;
    }
    const Uint32 magic = 0x07230203;
    const Uint32 magicInv = 0x03022307;
    return SDL_memcmp(code, &magic, 4) == 0 || SDL_memcmp(code, &magicInv, 4) == 0;
}

static SDL_GPUComputePipeline *VULKAN_CreateComputePipeline(
    SDL_GPURenderer *driverData,
    const SDL_GPUComputePipelineCreateInfo *createinfo)
{
    VkShaderModuleCreateInfo shaderModuleCreateInfo;
    VkComputePipelineCreateInfo vkShaderCreateInfo;
    VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo;
    VkResult vulkanResult;
    VulkanRenderer *renderer = (VulkanRenderer *)driverData;
    VulkanComputePipeline *vulkanComputePipeline;

    if (createinfo->format != SDL_GPU_SHADERFORMAT_SPIRV) {
        SET_STRING_ERROR_AND_RETURN("Incompatible shader format for Vulkan!", NULL);
    }

    if (!VULKAN_INTERNAL_IsValidShaderBytecode(createinfo->code, createinfo->code_size)) {
        SET_STRING_ERROR_AND_RETURN("The provided shader code is not valid SPIR-V!", NULL);
    }

    vulkanComputePipeline = SDL_malloc(sizeof(VulkanComputePipeline));
    shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderModuleCreateInfo.pNext = NULL;
    shaderModuleCreateInfo.flags = 0;
    shaderModuleCreateInfo.codeSize = createinfo->code_size;
    shaderModuleCreateInfo.pCode = (Uint32 *)createinfo->code;

    vulkanResult = renderer->vkCreateShaderModule(
        renderer->logicalDevice,
        &shaderModuleCreateInfo,
        NULL,
        &vulkanComputePipeline->shaderModule);

    if (vulkanResult != VK_SUCCESS) {
        SDL_free(vulkanComputePipeline);
        CHECK_VULKAN_ERROR_AND_RETURN(vulkanResult, vkCreateShaderModule, NULL);
    }

    pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineShaderStageCreateInfo.pNext = NULL;
    pipelineShaderStageCreateInfo.flags = 0;
    pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineShaderStageCreateInfo.module = vulkanComputePipeline->shaderModule;
    pipelineShaderStageCreateInfo.pName = createinfo->entrypoint;
    pipelineShaderStageCreateInfo.pSpecializationInfo = NULL;

    vulkanComputePipeline->resourceLayout = VULKAN_INTERNAL_FetchComputePipelineResourceLayout(
        renderer,
        createinfo);

    if (vulkanComputePipeline->resourceLayout == NULL) {
        renderer->vkDestroyShaderModule(
            renderer->logicalDevice,
            vulkanComputePipeline->shaderModule,
            NULL);
        SDL_free(vulkanComputePipeline);
        return NULL;
    }

    vkShaderCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    vkShaderCreateInfo.pNext = NULL;
    vkShaderCreateInfo.flags = 0;
    vkShaderCreateInfo.stage = pipelineShaderStageCreateInfo;
    vkShaderCreateInfo.layout = vulkanComputePipeline->resourceLayout->pipelineLayout;
    vkShaderCreateInfo.basePipelineHandle = (VkPipeline)VK_NULL_HANDLE;
    vkShaderCreateInfo.basePipelineIndex = 0;

    vulkanResult = renderer->vkCreateComputePipelines(
        renderer->logicalDevice,
        (VkPipelineCache)VK_NULL_HANDLE,
        1,
        &vkShaderCreateInfo,
        NULL,
        &vulkanComputePipeline->pipeline);

    if (vulkanResult != VK_SUCCESS) {
        VULKAN_INTERNAL_DestroyComputePipeline(renderer, vulkanComputePipeline);
        CHECK_VULKAN_ERROR_AND_RETURN(vulkanResult, vkCreateComputePipeline, NULL);
        return NULL;
    }

    SDL_SetAtomicInt(&vulkanComputePipeline->referenceCount, 0);

    if (renderer->debugMode && renderer->supportsDebugUtils && SDL_HasProperty(createinfo->props, SDL_PROP_GPU_COMPUTEPIPELINE_CREATE_NAME_STRING)) {
        VkDebugUtilsObjectNameInfoEXT nameInfo;
        nameInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
        nameInfo.pNext = NULL;
        nameInfo.pObjectName = SDL_GetStringProperty(createinfo->props, SDL_PROP_GPU_COMPUTEPIPELINE_CREATE_NAME_STRING, NULL);
        nameInfo.objectType = VK_OBJECT_TYPE_PIPELINE;
        nameInfo.objectHandle = (uint64_t)vulkanComputePipeline->pipeline;

        renderer->vkSetDebugUtilsObjectNameEXT(
            renderer->logicalDevice,
            &nameInfo);
    }

    return (SDL_GPUComputePipeline *)vulkanComputePipeline;
}

static SDL_GPUSampler *VULKAN_CreateSampler(
    SDL_GPURenderer *driverData,
    const SDL_GPUSamplerCreateInfo *createinfo)
{
    VulkanRenderer *renderer = (VulkanRenderer *)driverData;
    VulkanSampler *vulkanSampler = SDL_malloc(sizeof(VulkanSampler));
    VkResult vulkanResult;

    VkSamplerCreateInfo vkSamplerCreateInfo;
    vkSamplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    vkSamplerCreateInfo.pNext = NULL;
    vkSamplerCreateInfo.flags = 0;
    vkSamplerCreateInfo.magFilter = SDLToVK_Filter[createinfo->mag_filter];
    vkSamplerCreateInfo.minFilter = SDLToVK_Filter[createinfo->min_filter];
    vkSamplerCreateInfo.mipmapMode = SDLToVK_SamplerMipmapMode[createinfo->mipmap_mode];
    vkSamplerCreateInfo.addressModeU = SDLToVK_SamplerAddressMode[createinfo->address_mode_u];
    vkSamplerCreateInfo.addressModeV = SDLToVK_SamplerAddressMode[createinfo->address_mode_v];
    vkSamplerCreateInfo.addressModeW = SDLToVK_SamplerAddressMode[createinfo->address_mode_w];
    vkSamplerCreateInfo.mipLodBias = createinfo->mip_lod_bias;
    vkSamplerCreateInfo.anisotropyEnable = createinfo->enable_anisotropy;
    vkSamplerCreateInfo.maxAnisotropy = createinfo->max_anisotropy;
    vkSamplerCreateInfo.compareEnable = createinfo->enable_compare;
    vkSamplerCreateInfo.compareOp = SDLToVK_CompareOp[createinfo->compare_op];
    vkSamplerCreateInfo.minLod = createinfo->min_lod;
    vkSamplerCreateInfo.maxLod = createinfo->max_lod;
    vkSamplerCreateInfo.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK; // arbitrary, unused
    vkSamplerCreateInfo.unnormalizedCoordinates = VK_FALSE;

    vulkanResult = renderer->vkCreateSampler(
        renderer->logicalDevice,
        &vkSamplerCreateInfo,
        NULL,
        &vulkanSampler->sampler);

    if (vulkanResult != VK_SUCCESS) {
        SDL_free(vulkanSampler);
        CHECK_VULKAN_ERROR_AND_RETURN(vulkanResult, vkCreateSampler, NULL);
    }

    SDL_SetAtomicInt(&vulkanSampler->referenceCount, 0);

    if (renderer->debugMode && renderer->supportsDebugUtils && SDL_HasProperty(createinfo->props, SDL_PROP_GPU_SAMPLER_CREATE_NAME_STRING)) {
        VkDebugUtilsObjectNameInfoEXT nameInfo;
        nameInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
        nameInfo.pNext = NULL;
        nameInfo.pObjectName = SDL_GetStringProperty(createinfo->props, SDL_PROP_GPU_SAMPLER_CREATE_NAME_STRING, NULL);
        nameInfo.objectType = VK_OBJECT_TYPE_SAMPLER;
        nameInfo.objectHandle = (uint64_t)vulkanSampler->sampler;

        renderer->vkSetDebugUtilsObjectNameEXT(
            renderer->logicalDevice,
            &nameInfo);
    }

    return (SDL_GPUSampler *)vulkanSampler;
}

static SDL_GPUShader *VULKAN_CreateShader(
    SDL_GPURenderer *driverData,
    const SDL_GPUShaderCreateInfo *createinfo)
{
    VulkanShader *vulkanShader;
    VkResult vulkanResult;
    VkShaderModuleCreateInfo vkShaderModuleCreateInfo;
    VulkanRenderer *renderer = (VulkanRenderer *)driverData;

    if (!VULKAN_INTERNAL_IsValidShaderBytecode(createinfo->code, createinfo->code_size)) {
        SET_STRING_ERROR_AND_RETURN("The provided shader code is not valid SPIR-V!", NULL);
    }

    vulkanShader = SDL_malloc(sizeof(VulkanShader));
    vkShaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    vkShaderModuleCreateInfo.pNext = NULL;
    vkShaderModuleCreateInfo.flags = 0;
    vkShaderModuleCreateInfo.codeSize = createinfo->code_size;
    vkShaderModuleCreateInfo.pCode = (Uint32 *)createinfo->code;

    vulkanResult = renderer->vkCreateShaderModule(
        renderer->logicalDevice,
        &vkShaderModuleCreateInfo,
        NULL,
        &vulkanShader->shaderModule);

    if (vulkanResult != VK_SUCCESS) {
        SDL_free(vulkanShader);
        CHECK_VULKAN_ERROR_AND_RETURN(vulkanResult, vkCreateShaderModule, NULL);
    }

    const char *entrypoint = createinfo->entrypoint;
    if (!entrypoint) {
        entrypoint = "main";
    }
    vulkanShader->entrypointName = SDL_strdup(entrypoint);
    vulkanShader->stage = createinfo->stage;
    vulkanShader->numSamplers = createinfo->num_samplers;
    vulkanShader->numStorageTextures = createinfo->num_storage_textures;
    vulkanShader->numStorageBuffers = createinfo->num_storage_buffers;
    vulkanShader->numUniformBuffers = createinfo->num_uniform_buffers;

    SDL_SetAtomicInt(&vulkanShader->referenceCount, 0);

    if (renderer->debugMode && SDL_HasProperty(createinfo->props, SDL_PROP_GPU_SHADER_CREATE_NAME_STRING)) {
        VkDebugUtilsObjectNameInfoEXT nameInfo;
        nameInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
        nameInfo.pNext = NULL;
        nameInfo.pObjectName = SDL_GetStringProperty(createinfo->props, SDL_PROP_GPU_SHADER_CREATE_NAME_STRING, NULL);
        nameInfo.objectType = VK_OBJECT_TYPE_SHADER_MODULE;
        nameInfo.objectHandle = (uint64_t)vulkanShader->shaderModule;

        renderer->vkSetDebugUtilsObjectNameEXT(
            renderer->logicalDevice,
            &nameInfo);
    }

    return (SDL_GPUShader *)vulkanShader;
}

static bool VULKAN_SupportsSampleCount(
    SDL_GPURenderer *driverData,
    SDL_GPUTextureFormat format,
    SDL_GPUSampleCount sampleCount)
{
    VulkanRenderer *renderer = (VulkanRenderer *)driverData;
    VkSampleCountFlags bits = IsDepthFormat(format) ? renderer->physicalDeviceProperties.properties.limits.framebufferDepthSampleCounts : renderer->physicalDeviceProperties.properties.limits.framebufferColorSampleCounts;
    VkSampleCountFlagBits vkSampleCount = SDLToVK_SampleCount[sampleCount];
    return !!(bits & vkSampleCount);
}

static SDL_GPUTexture *VULKAN_CreateTexture(
    SDL_GPURenderer *driverData,
    const SDL_GPUTextureCreateInfo *createinfo)
{
    VulkanRenderer *renderer = (VulkanRenderer *)driverData;
    VulkanTexture *texture;
    VulkanTextureContainer *container;

    texture = VULKAN_INTERNAL_CreateTexture(
        renderer,
        true,
        createinfo);

    if (texture == NULL) {
        return NULL;
    }

    container = SDL_malloc(sizeof(VulkanTextureContainer));

    // Copy properties so we don't lose information when the client destroys them
    container->header.info = *createinfo;
    container->header.info.props = SDL_CreateProperties();
    if (createinfo->props) {
        SDL_CopyProperties(createinfo->props, container->header.info.props);
    }

    container->canBeCycled = true;
    container->activeTexture = texture;
    container->textureCapacity = 1;
    container->textureCount = 1;
    container->textures = SDL_malloc(
        container->textureCapacity * sizeof(VulkanTexture *));
    container->textures[0] = container->activeTexture;
    container->debugName = NULL;

    if (SDL_HasProperty(createinfo->props, SDL_PROP_GPU_TEXTURE_CREATE_NAME_STRING)) {
        container->debugName = SDL_strdup(SDL_GetStringProperty(createinfo->props, SDL_PROP_GPU_TEXTURE_CREATE_NAME_STRING, NULL));
    }

    texture->container = container;
    texture->containerIndex = 0;

    return (SDL_GPUTexture *)container;
}

static SDL_GPUBuffer *VULKAN_CreateBuffer(
    SDL_GPURenderer *driverData,
    SDL_GPUBufferUsageFlags usageFlags,
    Uint32 size,
    const char *debugName)
{
    return (SDL_GPUBuffer *)VULKAN_INTERNAL_CreateBufferContainer(
        (VulkanRenderer *)driverData,
        (VkDeviceSize)size,
        usageFlags,
        VULKAN_BUFFER_TYPE_GPU,
        false,
        debugName);
}

static VulkanUniformBuffer *VULKAN_INTERNAL_CreateUniformBuffer(
    VulkanRenderer *renderer,
    Uint32 size)
{
    VulkanUniformBuffer *uniformBuffer = SDL_calloc(1, sizeof(VulkanUniformBuffer));

    uniformBuffer->buffer = VULKAN_INTERNAL_CreateBuffer(
        renderer,
        (VkDeviceSize)size,
        0,
        VULKAN_BUFFER_TYPE_UNIFORM,
        false,
        NULL);

    uniformBuffer->drawOffset = 0;
    uniformBuffer->writeOffset = 0;
    uniformBuffer->buffer->uniformBufferForDefrag = uniformBuffer;

    return uniformBuffer;
}

static SDL_GPUTransferBuffer *VULKAN_CreateTransferBuffer(
    SDL_GPURenderer *driverData,
    SDL_GPUTransferBufferUsage usage,
    Uint32 size,
    const char *debugName)
{
    return (SDL_GPUTransferBuffer *)VULKAN_INTERNAL_CreateBufferContainer(
        (VulkanRenderer *)driverData,
        (VkDeviceSize)size,
        0,
        VULKAN_BUFFER_TYPE_TRANSFER,
        true, // Dedicated allocations preserve the data even if a defrag is triggered.
        debugName);
}

static void VULKAN_INTERNAL_ReleaseTexture(
    VulkanRenderer *renderer,
    VulkanTexture *vulkanTexture)
{
    if (vulkanTexture->markedForDestroy) {
        return;
    }

    SDL_LockMutex(renderer->disposeLock);

    EXPAND_ARRAY_IF_NEEDED(
        renderer->texturesToDestroy,
        VulkanTexture *,
        renderer->texturesToDestroyCount + 1,
        renderer->texturesToDestroyCapacity,
        renderer->texturesToDestroyCapacity * 2);

    renderer->texturesToDestroy[renderer->texturesToDestroyCount] = vulkanTexture;
    renderer->texturesToDestroyCount += 1;

    vulkanTexture->markedForDestroy = true;

    SDL_UnlockMutex(renderer->disposeLock);
}

static void VULKAN_ReleaseTexture(
    SDL_GPURenderer *driverData,
    SDL_GPUTexture *texture)
{
    VulkanRenderer *renderer = (VulkanRenderer *)driverData;
    VulkanTextureContainer *vulkanTextureContainer = (VulkanTextureContainer *)texture;
    Uint32 i;

    SDL_LockMutex(renderer->disposeLock);

    for (i = 0; i < vulkanTextureContainer->textureCount; i += 1) {
        VULKAN_INTERNAL_ReleaseTexture(renderer, vulkanTextureContainer->textures[i]);
    }

    // Containers are just client handles, so we can destroy immediately
    if (vulkanTextureContainer->debugName != NULL) {
        SDL_free(vulkanTextureContainer->debugName);
    }
    SDL_free(vulkanTextureContainer->textures);
    SDL_free(vulkanTextureContainer);

    SDL_UnlockMutex(renderer->disposeLock);
}

static void VULKAN_ReleaseSampler(
    SDL_GPURenderer *driverData,
    SDL_GPUSampler *sampler)
{
    VulkanRenderer *renderer = (VulkanRenderer *)driverData;
    VulkanSampler *vulkanSampler = (VulkanSampler *)sampler;

    SDL_LockMutex(renderer->disposeLock);

    EXPAND_ARRAY_IF_NEEDED(
        renderer->samplersToDestroy,
        VulkanSampler *,
        renderer->samplersToDestroyCount + 1,
        renderer->samplersToDestroyCapacity,
        renderer->samplersToDestroyCapacity * 2);

    renderer->samplersToDestroy[renderer->samplersToDestroyCount] = vulkanSampler;
    renderer->samplersToDestroyCount += 1;

    SDL_UnlockMutex(renderer->disposeLock);
}

static void VULKAN_INTERNAL_ReleaseBuffer(
    VulkanRenderer *renderer,
    VulkanBuffer *vulkanBuffer)
{
    if (vulkanBuffer->markedForDestroy) {
        return;
    }

    SDL_LockMutex(renderer->disposeLock);

    EXPAND_ARRAY_IF_NEEDED(
        renderer->buffersToDestroy,
        VulkanBuffer *,
        renderer->buffersToDestroyCount + 1,
        renderer->buffersToDestroyCapacity,
        renderer->buffersToDestroyCapacity * 2);

    renderer->buffersToDestroy[renderer->buffersToDestroyCount] = vulkanBuffer;
    renderer->buffersToDestroyCount += 1;

    vulkanBuffer->markedForDestroy = true;
    vulkanBuffer->container = NULL;

    SDL_UnlockMutex(renderer->disposeLock);
}

static void VULKAN_INTERNAL_ReleaseBufferContainer(
    VulkanRenderer *renderer,
    VulkanBufferContainer *bufferContainer)
{
    Uint32 i;

    SDL_LockMutex(renderer->disposeLock);

    for (i = 0; i < bufferContainer->bufferCount; i += 1) {
        VULKAN_INTERNAL_ReleaseBuffer(renderer, bufferContainer->buffers[i]);
    }

    // Containers are just client handles, so we can free immediately
    if (bufferContainer->debugName != NULL) {
        SDL_free(bufferContainer->debugName);
        bufferContainer->debugName = NULL;
    }
    SDL_free(bufferContainer->buffers);
    SDL_free(bufferContainer);

    SDL_UnlockMutex(renderer->disposeLock);
}

static void VULKAN_ReleaseBuffer(
    SDL_GPURenderer *driverData,
    SDL_GPUBuffer *buffer)
{
    VulkanRenderer *renderer = (VulkanRenderer *)driverData;
    VulkanBufferContainer *vulkanBufferContainer = (VulkanBufferContainer *)buffer;

    VULKAN_INTERNAL_ReleaseBufferContainer(
        renderer,
        vulkanBufferContainer);
}

static void VULKAN_ReleaseTransferBuffer(
    SDL_GPURenderer *driverData,
    SDL_GPUTransferBuffer *transferBuffer)
{
    VulkanRenderer *renderer = (VulkanRenderer *)driverData;
    VulkanBufferContainer *transferBufferContainer = (VulkanBufferContainer *)transferBuffer;

    VULKAN_INTERNAL_ReleaseBufferContainer(
        renderer,
        transferBufferContainer);
}

static void VULKAN_ReleaseShader(
    SDL_GPURenderer *driverData,
    SDL_GPUShader *shader)
{
    VulkanRenderer *renderer = (VulkanRenderer *)driverData;
    VulkanShader *vulkanShader = (VulkanShader *)shader;

    SDL_LockMutex(renderer->disposeLock);

    EXPAND_ARRAY_IF_NEEDED(
        renderer->shadersToDestroy,
        VulkanShader *,
        renderer->shadersToDestroyCount + 1,
        renderer->shadersToDestroyCapacity,
        renderer->shadersToDestroyCapacity * 2);

    renderer->shadersToDestroy[renderer->shadersToDestroyCount] = vulkanShader;
    renderer->shadersToDestroyCount += 1;

    SDL_UnlockMutex(renderer->disposeLock);
}

static void VULKAN_ReleaseComputePipeline(
    SDL_GPURenderer *driverData,
    SDL_GPUComputePipeline *computePipeline)
{
    VulkanRenderer *renderer = (VulkanRenderer *)driverData;
    VulkanComputePipeline *vulkanComputePipeline = (VulkanComputePipeline *)computePipeline;

    SDL_LockMutex(renderer->disposeLock);

    EXPAND_ARRAY_IF_NEEDED(
        renderer->computePipelinesToDestroy,
        VulkanComputePipeline *,
        renderer->computePipelinesToDestroyCount + 1,
        renderer->computePipelinesToDestroyCapacity,
        renderer->computePipelinesToDestroyCapacity * 2);

    renderer->computePipelinesToDestroy[renderer->computePipelinesToDestroyCount] = vulkanComputePipeline;
    renderer->computePipelinesToDestroyCount += 1;

    SDL_UnlockMutex(renderer->disposeLock);
}

static void VULKAN_ReleaseGraphicsPipeline(
    SDL_GPURenderer *driverData,
    SDL_GPUGraphicsPipeline *graphicsPipeline)
{
    VulkanRenderer *renderer = (VulkanRenderer *)driverData;
    VulkanGraphicsPipeline *vulkanGraphicsPipeline = (VulkanGraphicsPipeline *)graphicsPipeline;

    SDL_LockMutex(renderer->disposeLock);

    EXPAND_ARRAY_IF_NEEDED(
        renderer->graphicsPipelinesToDestroy,
        VulkanGraphicsPipeline *,
        renderer->graphicsPipelinesToDestroyCount + 1,
        renderer->graphicsPipelinesToDestroyCapacity,
        renderer->graphicsPipelinesToDestroyCapacity * 2);

    renderer->graphicsPipelinesToDestroy[renderer->graphicsPipelinesToDestroyCount] = vulkanGraphicsPipeline;
    renderer->graphicsPipelinesToDestroyCount += 1;

    SDL_UnlockMutex(renderer->disposeLock);
}

// Command Buffer render state

static VkRenderPass VULKAN_INTERNAL_FetchRenderPass(
    VulkanRenderer *renderer,
    const SDL_GPUColorTargetInfo *colorTargetInfos,
    Uint32 numColorTargets,
    const SDL_GPUDepthStencilTargetInfo *depthStencilTargetInfo)
{
    VulkanRenderPassHashTableValue *renderPassWrapper = NULL;
    VkRenderPass renderPassHandle;
    RenderPassHashTableKey key;
    Uint32 i;

    SDL_zero(key);

    for (i = 0; i < numColorTargets; i += 1) {
        key.colorTargetDescriptions[i].format = SDLToVK_TextureFormat[((VulkanTextureContainer *)colorTargetInfos[i].texture)->header.info.format];
        key.colorTargetDescriptions[i].loadOp = colorTargetInfos[i].load_op;
        key.colorTargetDescriptions[i].storeOp = colorTargetInfos[i].store_op;

        if (colorTargetInfos[i].resolve_texture != NULL) {
            key.resolveTargetFormats[key.numResolveTargets] = SDLToVK_TextureFormat[((VulkanTextureContainer *)colorTargetInfos[i].resolve_texture)->header.info.format];
            key.numResolveTargets += 1;
        }
    }

    key.sampleCount = VK_SAMPLE_COUNT_1_BIT;
    if (numColorTargets > 0) {
        key.sampleCount = SDLToVK_SampleCount[((VulkanTextureContainer *)colorTargetInfos[0].texture)->header.info.sample_count];
    }

    key.numColorTargets = numColorTargets;

    if (depthStencilTargetInfo == NULL) {
        key.depthStencilTargetDescription.format = 0;
        key.depthStencilTargetDescription.loadOp = SDL_GPU_LOADOP_DONT_CARE;
        key.depthStencilTargetDescription.storeOp = SDL_GPU_STOREOP_DONT_CARE;
        key.depthStencilTargetDescription.stencilLoadOp = SDL_GPU_LOADOP_DONT_CARE;
        key.depthStencilTargetDescription.stencilStoreOp = SDL_GPU_STOREOP_DONT_CARE;
    } else {
        key.depthStencilTargetDescription.format = SDLToVK_TextureFormat[((VulkanTextureContainer *)depthStencilTargetInfo->texture)->header.info.format];
        key.depthStencilTargetDescription.loadOp = depthStencilTargetInfo->load_op;
        key.depthStencilTargetDescription.storeOp = depthStencilTargetInfo->store_op;
        key.depthStencilTargetDescription.stencilLoadOp = depthStencilTargetInfo->stencil_load_op;
        key.depthStencilTargetDescription.stencilStoreOp = depthStencilTargetInfo->stencil_store_op;
    }

    SDL_LockMutex(renderer->renderPassFetchLock);

    bool result = SDL_FindInHashTable(
        renderer->renderPassHashTable,
        (const void *)&key,
        (const void **)&renderPassWrapper);

    if (result) {
        SDL_UnlockMutex(renderer->renderPassFetchLock);
        return renderPassWrapper->handle;
    }

    renderPassHandle = VULKAN_INTERNAL_CreateRenderPass(
        renderer,
        colorTargetInfos,
        numColorTargets,
        depthStencilTargetInfo);

    if (renderPassHandle == VK_NULL_HANDLE) {
        SDL_UnlockMutex(renderer->renderPassFetchLock);
        return VK_NULL_HANDLE;
    }

    // Have to malloc the key to store it in the hashtable
    RenderPassHashTableKey *allocedKey = SDL_malloc(sizeof(RenderPassHashTableKey));
    SDL_memcpy(allocedKey, &key, sizeof(RenderPassHashTableKey));

    renderPassWrapper = SDL_malloc(sizeof(VulkanRenderPassHashTableValue));
    renderPassWrapper->handle = renderPassHandle;

    SDL_InsertIntoHashTable(
        renderer->renderPassHashTable,
        (const void *)allocedKey,
        (const void *)renderPassWrapper, true);

    SDL_UnlockMutex(renderer->renderPassFetchLock);

    return renderPassHandle;
}

static VulkanFramebuffer *VULKAN_INTERNAL_FetchFramebuffer(
    VulkanRenderer *renderer,
    VkRenderPass renderPass,
    const SDL_GPUColorTargetInfo *colorTargetInfos,
    Uint32 numColorTargets,
    const SDL_GPUDepthStencilTargetInfo *depthStencilTargetInfo,
    Uint32 width,
    Uint32 height)
{
    VulkanFramebuffer *vulkanFramebuffer = NULL;
    VkFramebufferCreateInfo framebufferInfo;
    VkResult result;
    VkImageView imageViewAttachments[2 * MAX_COLOR_TARGET_BINDINGS + 1 /* depth */];
    FramebufferHashTableKey key;
    Uint32 attachmentCount = 0;
    Uint32 i;

    SDL_zero(imageViewAttachments);
    SDL_zero(key);

    key.numColorTargets = numColorTargets;

    for (i = 0; i < numColorTargets; i += 1) {
        VulkanTextureContainer *container = (VulkanTextureContainer *)colorTargetInfos[i].texture;
        VulkanTextureSubresource *subresource = VULKAN_INTERNAL_FetchTextureSubresource(
            container,
            container->header.info.type == SDL_GPU_TEXTURETYPE_3D ? 0 : colorTargetInfos[i].layer_or_depth_plane,
            colorTargetInfos[i].mip_level);

        Uint32 rtvIndex =
            container->header.info.type == SDL_GPU_TEXTURETYPE_3D ? colorTargetInfos[i].layer_or_depth_plane : 0;
        key.colorAttachmentViews[i] = subresource->renderTargetViews[rtvIndex];

        if (colorTargetInfos[i].resolve_texture != NULL) {
            VulkanTextureContainer *resolveTextureContainer = (VulkanTextureContainer *)colorTargetInfos[i].resolve_texture;
            VulkanTextureSubresource *resolveSubresource = VULKAN_INTERNAL_FetchTextureSubresource(
                resolveTextureContainer,
                colorTargetInfos[i].layer_or_depth_plane,
                colorTargetInfos[i].mip_level);

            key.resolveAttachmentViews[key.numResolveAttachments] = resolveSubresource->renderTargetViews[0];
            key.numResolveAttachments += 1;
        }
    }

    if (depthStencilTargetInfo == NULL) {
        key.depthStencilAttachmentView = VK_NULL_HANDLE;
    } else {
        VulkanTextureSubresource *subresource = VULKAN_INTERNAL_FetchTextureSubresource(
            (VulkanTextureContainer *)depthStencilTargetInfo->texture,
            0,
            0);
        key.depthStencilAttachmentView = subresource->depthStencilView;
    }

    key.width = width;
    key.height = height;

    SDL_LockMutex(renderer->framebufferFetchLock);

    bool findResult = SDL_FindInHashTable(
        renderer->framebufferHashTable,
        (const void *)&key,
        (const void **)&vulkanFramebuffer);

    if (findResult) {
        SDL_UnlockMutex(renderer->framebufferFetchLock);
        return vulkanFramebuffer;
    }

    vulkanFramebuffer = SDL_malloc(sizeof(VulkanFramebuffer));

    SDL_SetAtomicInt(&vulkanFramebuffer->referenceCount, 0);

    // Create a new framebuffer

    for (i = 0; i < numColorTargets; i += 1) {
        VulkanTextureContainer *container = (VulkanTextureContainer *)colorTargetInfos[i].texture;
        VulkanTextureSubresource *subresource = VULKAN_INTERNAL_FetchTextureSubresource(
            container,
            container->header.info.type == SDL_GPU_TEXTURETYPE_3D ? 0 : colorTargetInfos[i].layer_or_depth_plane,
            colorTargetInfos[i].mip_level);

        Uint32 rtvIndex =
            container->header.info.type == SDL_GPU_TEXTURETYPE_3D ? colorTargetInfos[i].layer_or_depth_plane : 0;

        imageViewAttachments[attachmentCount] = subresource->renderTargetViews[rtvIndex];

        attachmentCount += 1;

        if (colorTargetInfos[i].store_op == SDL_GPU_STOREOP_RESOLVE || colorTargetInfos[i].store_op == SDL_GPU_STOREOP_RESOLVE_AND_STORE) {
            VulkanTextureContainer *resolveContainer = (VulkanTextureContainer *)colorTargetInfos[i].resolve_texture;
            VulkanTextureSubresource *resolveSubresource = VULKAN_INTERNAL_FetchTextureSubresource(
                resolveContainer,
                colorTargetInfos[i].resolve_layer,
                colorTargetInfos[i].resolve_mip_level);

            imageViewAttachments[attachmentCount] = resolveSubresource->renderTargetViews[0];

            attachmentCount += 1;
        }
    }

    if (depthStencilTargetInfo != NULL) {
        VulkanTextureSubresource *subresource = VULKAN_INTERNAL_FetchTextureSubresource(
            (VulkanTextureContainer *)depthStencilTargetInfo->texture,
            0,
            0);
        imageViewAttachments[attachmentCount] = subresource->depthStencilView;

        attachmentCount += 1;
    }

    framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferInfo.pNext = NULL;
    framebufferInfo.flags = 0;
    framebufferInfo.renderPass = renderPass;
    framebufferInfo.attachmentCount = attachmentCount;
    framebufferInfo.pAttachments = imageViewAttachments;
    framebufferInfo.width = key.width;
    framebufferInfo.height = key.height;
    framebufferInfo.layers = 1;

    result = renderer->vkCreateFramebuffer(
        renderer->logicalDevice,
        &framebufferInfo,
        NULL,
        &vulkanFramebuffer->framebuffer);

    if (result == VK_SUCCESS) {
        // Have to malloc the key to store it in the hashtable
        FramebufferHashTableKey *allocedKey = SDL_malloc(sizeof(FramebufferHashTableKey));
        SDL_memcpy(allocedKey, &key, sizeof(FramebufferHashTableKey));

        SDL_InsertIntoHashTable(
            renderer->framebufferHashTable,
            (const void *)allocedKey,
            (const void *)vulkanFramebuffer, true);

    } else {
        SDL_free(vulkanFramebuffer);
        SDL_UnlockMutex(renderer->framebufferFetchLock);
        CHECK_VULKAN_ERROR_AND_RETURN(result, vkCreateFramebuffer, NULL);
    }

    SDL_UnlockMutex(renderer->framebufferFetchLock);
    return vulkanFramebuffer;
}

static void VULKAN_INTERNAL_SetCurrentViewport(
    VulkanCommandBuffer *commandBuffer,
    const SDL_GPUViewport *viewport)
{
    VulkanCommandBuffer *vulkanCommandBuffer = commandBuffer;
    VulkanRenderer *renderer = vulkanCommandBuffer->renderer;

    vulkanCommandBuffer->currentViewport.x = viewport->x;
    vulkanCommandBuffer->currentViewport.width = viewport->w;
    vulkanCommandBuffer->currentViewport.minDepth = viewport->min_depth;
    vulkanCommandBuffer->currentViewport.maxDepth = viewport->max_depth;

    // Viewport flip for consistency with other backends
    vulkanCommandBuffer->currentViewport.y = viewport->y + viewport->h;
    vulkanCommandBuffer->currentViewport.height = -viewport->h;

    renderer->vkCmdSetViewport(
        vulkanCommandBuffer->commandBuffer,
        0,
        1,
        &vulkanCommandBuffer->currentViewport);
}

static void VULKAN_SetViewport(
    SDL_GPUCommandBuffer *commandBuffer,
    const SDL_GPUViewport *viewport)
{
    VulkanCommandBuffer *vulkanCommandBuffer = (VulkanCommandBuffer *)commandBuffer;

    VULKAN_INTERNAL_SetCurrentViewport(
        vulkanCommandBuffer,
        viewport);
}

static void VULKAN_INTERNAL_SetCurrentScissor(
    VulkanCommandBuffer *vulkanCommandBuffer,
    const SDL_Rect *scissor)
{
    VulkanRenderer *renderer = vulkanCommandBuffer->renderer;

    vulkanCommandBuffer->currentScissor.offset.x = scissor->x;
    vulkanCommandBuffer->currentScissor.offset.y = scissor->y;
    vulkanCommandBuffer->currentScissor.extent.width = scissor->w;
    vulkanCommandBuffer->currentScissor.extent.height = scissor->h;

    renderer->vkCmdSetScissor(
        vulkanCommandBuffer->commandBuffer,
        0,
        1,
        &vulkanCommandBuffer->currentScissor);
}

static void VULKAN_SetScissor(
    SDL_GPUCommandBuffer *commandBuffer,
    const SDL_Rect *scissor)
{
    VulkanCommandBuffer *vulkanCommandBuffer = (VulkanCommandBuffer *)commandBuffer;

    VULKAN_INTERNAL_SetCurrentScissor(
        vulkanCommandBuffer,
        scissor);
}

static void VULKAN_INTERNAL_SetCurrentBlendConstants(
    VulkanCommandBuffer *vulkanCommandBuffer,
    SDL_FColor blendConstants)
{
    VulkanRenderer *renderer = vulkanCommandBuffer->renderer;

    vulkanCommandBuffer->blendConstants[0] = blendConstants.r;
    vulkanCommandBuffer->blendConstants[1] = blendConstants.g;
    vulkanCommandBuffer->blendConstants[2] = blendConstants.b;
    vulkanCommandBuffer->blendConstants[3] = blendConstants.a;

    renderer->vkCmdSetBlendConstants(
        vulkanCommandBuffer->commandBuffer,
        vulkanCommandBuffer->blendConstants);
}

static void VULKAN_SetBlendConstants(
    SDL_GPUCommandBuffer *commandBuffer,
    SDL_FColor blendConstants)
{
    VulkanCommandBuffer *vulkanCommandBuffer = (VulkanCommandBuffer *)commandBuffer;

    VULKAN_INTERNAL_SetCurrentBlendConstants(
        vulkanCommandBuffer,
        blendConstants);
}

static void VULKAN_INTERNAL_SetCurrentStencilReference(
    VulkanCommandBuffer *vulkanCommandBuffer,
    Uint8 reference)
{
    VulkanRenderer *renderer = vulkanCommandBuffer->renderer;

    vulkanCommandBuffer->stencilRef = reference;

    renderer->vkCmdSetStencilReference(
        vulkanCommandBuffer->commandBuffer,
        VK_STENCIL_FACE_FRONT_AND_BACK,
        vulkanCommandBuffer->stencilRef);
}

static void VULKAN_SetStencilReference(
    SDL_GPUCommandBuffer *commandBuffer,
    Uint8 reference)
{
    VulkanCommandBuffer *vulkanCommandBuffer = (VulkanCommandBuffer *)commandBuffer;

    VULKAN_INTERNAL_SetCurrentStencilReference(
        vulkanCommandBuffer,
        reference);
}

static void VULKAN_BindVertexSamplers(
    SDL_GPUCommandBuffer *commandBuffer,
    Uint32 firstSlot,
    const SDL_GPUTextureSamplerBinding *textureSamplerBindings,
    Uint32 numBindings)
{
    VulkanCommandBuffer *vulkanCommandBuffer = (VulkanCommandBuffer *)commandBuffer;

    for (Uint32 i = 0; i < numBindings; i += 1) {
        VulkanTextureContainer *textureContainer = (VulkanTextureContainer *)textureSamplerBindings[i].texture;
        VulkanSampler *sampler = (VulkanSampler *)textureSamplerBindings[i].sampler;

        if (vulkanCommandBuffer->vertexSamplers[firstSlot + i] != sampler) {
            VULKAN_INTERNAL_TrackSampler(
                vulkanCommandBuffer,
                (VulkanSampler *)textureSamplerBindings[i].sampler);

            vulkanCommandBuffer->vertexSamplers[firstSlot + i] = (VulkanSampler *)textureSamplerBindings[i].sampler;
            vulkanCommandBuffer->needNewVertexResourceDescriptorSet = true;
        }

        if (vulkanCommandBuffer->vertexSamplerTextures[firstSlot + i] != textureContainer->activeTexture) {
            VULKAN_INTERNAL_TrackTexture(
                vulkanCommandBuffer,
                textureContainer->activeTexture);

            vulkanCommandBuffer->vertexSamplerTextures[firstSlot + i] = textureContainer->activeTexture;
            vulkanCommandBuffer->needNewVertexResourceDescriptorSet = true;
        }
    }
}

static void VULKAN_BindVertexStorageTextures(
    SDL_GPUCommandBuffer *commandBuffer,
    Uint32 firstSlot,
    SDL_GPUTexture *const *storageTextures,
    Uint32 numBindings)
{
    VulkanCommandBuffer *vulkanCommandBuffer = (VulkanCommandBuffer *)commandBuffer;

    for (Uint32 i = 0; i < numBindings; i += 1) {
        VulkanTextureContainer *textureContainer = (VulkanTextureContainer *)storageTextures[i];

        if (vulkanCommandBuffer->vertexStorageTextures[firstSlot + i] != textureContainer->activeTexture) {
            VULKAN_INTERNAL_TrackTexture(
                vulkanCommandBuffer,
                textureContainer->activeTexture);

            vulkanCommandBuffer->vertexStorageTextures[firstSlot + i] = textureContainer->activeTexture;
            vulkanCommandBuffer->needNewVertexResourceDescriptorSet = true;
        }
    }
}

static void VULKAN_BindVertexStorageBuffers(
    SDL_GPUCommandBuffer *commandBuffer,
    Uint32 firstSlot,
    SDL_GPUBuffer *const *storageBuffers,
    Uint32 numBindings)
{
    VulkanCommandBuffer *vulkanCommandBuffer = (VulkanCommandBuffer *)commandBuffer;

    for (Uint32 i = 0; i < numBindings; i += 1) {
        VulkanBufferContainer *bufferContainer = (VulkanBufferContainer *)storageBuffers[i];

        if (vulkanCommandBuffer->vertexStorageBuffers[firstSlot + i] != bufferContainer->activeBuffer) {
            VULKAN_INTERNAL_TrackBuffer(
                vulkanCommandBuffer,
                bufferContainer->activeBuffer);

            vulkanCommandBuffer->vertexStorageBuffers[firstSlot + i] = bufferContainer->activeBuffer;
            vulkanCommandBuffer->needNewVertexResourceDescriptorSet = true;
        }
    }
}

static void VULKAN_BindFragmentSamplers(
    SDL_GPUCommandBuffer *commandBuffer,
    Uint32 firstSlot,
    const SDL_GPUTextureSamplerBinding *textureSamplerBindings,
    Uint32 numBindings)
{
    VulkanCommandBuffer *vulkanCommandBuffer = (VulkanCommandBuffer *)commandBuffer;

    for (Uint32 i = 0; i < numBindings; i += 1) {
        VulkanTextureContainer *textureContainer = (VulkanTextureContainer *)textureSamplerBindings[i].texture;
        VulkanSampler *sampler = (VulkanSampler *)textureSamplerBindings[i].sampler;

        if (vulkanCommandBuffer->fragmentSamplers[firstSlot + i] != sampler) {
            VULKAN_INTERNAL_TrackSampler(
                vulkanCommandBuffer,
                (VulkanSampler *)textureSamplerBindings[i].sampler);

            vulkanCommandBuffer->fragmentSamplers[firstSlot + i] = (VulkanSampler *)textureSamplerBindings[i].sampler;
            vulkanCommandBuffer->needNewFragmentResourceDescriptorSet = true;
        }

        if (vulkanCommandBuffer->fragmentSamplerTextures[firstSlot + i] != textureContainer->activeTexture) {
            VULKAN_INTERNAL_TrackTexture(
                vulkanCommandBuffer,
                textureContainer->activeTexture);

            vulkanCommandBuffer->fragmentSamplerTextures[firstSlot + i] = textureContainer->activeTexture;
            vulkanCommandBuffer->needNewFragmentResourceDescriptorSet = true;
        }
    }
}

static void VULKAN_BindFragmentStorageTextures(
    SDL_GPUCommandBuffer *commandBuffer,
    Uint32 firstSlot,
    SDL_GPUTexture *const *storageTextures,
    Uint32 numBindings)
{
    VulkanCommandBuffer *vulkanCommandBuffer = (VulkanCommandBuffer *)commandBuffer;

    for (Uint32 i = 0; i < numBindings; i += 1) {
        VulkanTextureContainer *textureContainer = (VulkanTextureContainer *)storageTextures[i];

        if (vulkanCommandBuffer->fragmentStorageTextures[firstSlot + i] != textureContainer->activeTexture) {
            VULKAN_INTERNAL_TrackTexture(
                vulkanCommandBuffer,
                textureContainer->activeTexture);

            vulkanCommandBuffer->fragmentStorageTextures[firstSlot + i] = textureContainer->activeTexture;
            vulkanCommandBuffer->needNewFragmentResourceDescriptorSet = true;
        }
    }
}

static void VULKAN_BindFragmentStorageBuffers(
    SDL_GPUCommandBuffer *commandBuffer,
    Uint32 firstSlot,
    SDL_GPUBuffer *const *storageBuffers,
    Uint32 numBindings)
{
    VulkanCommandBuffer *vulkanCommandBuffer = (VulkanCommandBuffer *)commandBuffer;
    VulkanBufferContainer *bufferContainer;
    Uint32 i;

    for (i = 0; i < numBindings; i += 1) {
        bufferContainer = (VulkanBufferContainer *)storageBuffers[i];

        if (vulkanCommandBuffer->fragmentStorageBuffers[firstSlot + i] != bufferContainer->activeBuffer) {
            VULKAN_INTERNAL_TrackBuffer(
                vulkanCommandBuffer,
                bufferContainer->activeBuffer);

            vulkanCommandBuffer->fragmentStorageBuffers[firstSlot + i] = bufferContainer->activeBuffer;
            vulkanCommandBuffer->needNewFragmentResourceDescriptorSet = true;
        }
    }
}

static VulkanUniformBuffer *VULKAN_INTERNAL_AcquireUniformBufferFromPool(
    VulkanCommandBuffer *commandBuffer)
{
    VulkanRenderer *renderer = commandBuffer->renderer;
    VulkanUniformBuffer *uniformBuffer;

    SDL_LockMutex(renderer->acquireUniformBufferLock);

    if (renderer->uniformBufferPoolCount > 0) {
        uniformBuffer = renderer->uniformBufferPool[renderer->uniformBufferPoolCount - 1];
        renderer->uniformBufferPoolCount -= 1;
    } else {
        uniformBuffer = VULKAN_INTERNAL_CreateUniformBuffer(
            renderer,
            UNIFORM_BUFFER_SIZE);
    }

    SDL_UnlockMutex(renderer->acquireUniformBufferLock);

    VULKAN_INTERNAL_TrackUniformBuffer(commandBuffer, uniformBuffer);

    return uniformBuffer;
}

static void VULKAN_INTERNAL_ReturnUniformBufferToPool(
    VulkanRenderer *renderer,
    VulkanUniformBuffer *uniformBuffer)
{
    if (renderer->uniformBufferPoolCount >= renderer->uniformBufferPoolCapacity) {
        renderer->uniformBufferPoolCapacity *= 2;
        renderer->uniformBufferPool = SDL_realloc(
            renderer->uniformBufferPool,
            renderer->uniformBufferPoolCapacity * sizeof(VulkanUniformBuffer *));
    }

    renderer->uniformBufferPool[renderer->uniformBufferPoolCount] = uniformBuffer;
    renderer->uniformBufferPoolCount += 1;

    uniformBuffer->writeOffset = 0;
    uniformBuffer->drawOffset = 0;
}

static void VULKAN_INTERNAL_PushUniformData(
    VulkanCommandBuffer *commandBuffer,
    VulkanUniformBufferStage uniformBufferStage,
    Uint32 slotIndex,
    const void *data,
    Uint32 length)
{
    Uint32 blockSize =
        VULKAN_INTERNAL_NextHighestAlignment32(
            length,
            commandBuffer->renderer->minUBOAlignment);

    VulkanUniformBuffer *uniformBuffer;

    if (uniformBufferStage == VULKAN_UNIFORM_BUFFER_STAGE_VERTEX) {
        if (commandBuffer->vertexUniformBuffers[slotIndex] == NULL) {
            commandBuffer->vertexUniformBuffers[slotIndex] = VULKAN_INTERNAL_AcquireUniformBufferFromPool(
                commandBuffer);
        }
        uniformBuffer = commandBuffer->vertexUniformBuffers[slotIndex];
    } else if (uniformBufferStage == VULKAN_UNIFORM_BUFFER_STAGE_FRAGMENT) {
        if (commandBuffer->fragmentUniformBuffers[slotIndex] == NULL) {
            commandBuffer->fragmentUniformBuffers[slotIndex] = VULKAN_INTERNAL_AcquireUniformBufferFromPool(
                commandBuffer);
        }
        uniformBuffer = commandBuffer->fragmentUniformBuffers[slotIndex];
    } else if (uniformBufferStage == VULKAN_UNIFORM_BUFFER_STAGE_COMPUTE) {
        if (commandBuffer->computeUniformBuffers[slotIndex] == NULL) {
            commandBuffer->computeUniformBuffers[slotIndex] = VULKAN_INTERNAL_AcquireUniformBufferFromPool(
                commandBuffer);
        }
        uniformBuffer = commandBuffer->computeUniformBuffers[slotIndex];
    } else {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Unrecognized shader stage!");
        return;
    }

    // If there is no more room, acquire a new uniform buffer
    if (uniformBuffer->writeOffset + blockSize + MAX_UBO_SECTION_SIZE >= uniformBuffer->buffer->size) {
        uniformBuffer = VULKAN_INTERNAL_AcquireUniformBufferFromPool(commandBuffer);

        uniformBuffer->drawOffset = 0;
        uniformBuffer->writeOffset = 0;

        if (uniformBufferStage == VULKAN_UNIFORM_BUFFER_STAGE_VERTEX) {
            commandBuffer->vertexUniformBuffers[slotIndex] = uniformBuffer;
            commandBuffer->needNewVertexUniformDescriptorSet = true;
        } else if (uniformBufferStage == VULKAN_UNIFORM_BUFFER_STAGE_FRAGMENT) {
            commandBuffer->fragmentUniformBuffers[slotIndex] = uniformBuffer;
            commandBuffer->needNewFragmentUniformDescriptorSet = true;
        } else if (uniformBufferStage == VULKAN_UNIFORM_BUFFER_STAGE_COMPUTE) {
            commandBuffer->computeUniformBuffers[slotIndex] = uniformBuffer;
            commandBuffer->needNewComputeUniformDescriptorSet = true;
        } else {
            SDL_LogError(SDL_LOG_CATEGORY_GPU, "Unrecognized shader stage!");
            return;
        }
    }

    uniformBuffer->drawOffset = uniformBuffer->writeOffset;

    Uint8 *dst =
        uniformBuffer->buffer->usedRegion->allocation->mapPointer +
        uniformBuffer->buffer->usedRegion->resourceOffset +
        uniformBuffer->writeOffset;

    SDL_memcpy(
        dst,
        data,
        length);

    uniformBuffer->writeOffset += blockSize;

    if (uniformBufferStage == VULKAN_UNIFORM_BUFFER_STAGE_VERTEX) {
        commandBuffer->needNewVertexUniformOffsets = true;
    } else if (uniformBufferStage == VULKAN_UNIFORM_BUFFER_STAGE_FRAGMENT) {
        commandBuffer->needNewFragmentUniformOffsets = true;
    } else if (uniformBufferStage == VULKAN_UNIFORM_BUFFER_STAGE_COMPUTE) {
        commandBuffer->needNewComputeUniformOffsets = true;
    } else {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Unrecognized shader stage!");
        return;
    }
}

static void VULKAN_BeginRenderPass(
    SDL_GPUCommandBuffer *commandBuffer,
    const SDL_GPUColorTargetInfo *colorTargetInfos,
    Uint32 numColorTargets,
    const SDL_GPUDepthStencilTargetInfo *depthStencilTargetInfo)
{
    VulkanCommandBuffer *vulkanCommandBuffer = (VulkanCommandBuffer *)commandBuffer;
    VulkanRenderer *renderer = vulkanCommandBuffer->renderer;
    VkRenderPass renderPass;
    VulkanFramebuffer *framebuffer;

    Uint32 w, h;
    VkClearValue *clearValues;
    Uint32 clearCount = 0;
    Uint32 totalColorAttachmentCount = 0;
    Uint32 i;
    SDL_GPUViewport defaultViewport;
    SDL_Rect defaultScissor;
    SDL_FColor defaultBlendConstants;
    Uint32 framebufferWidth = SDL_MAX_UINT32;
    Uint32 framebufferHeight = SDL_MAX_UINT32;

    for (i = 0; i < numColorTargets; i += 1) {
        VulkanTextureContainer *textureContainer = (VulkanTextureContainer *)colorTargetInfos[i].texture;

        w = textureContainer->header.info.width >> colorTargetInfos[i].mip_level;
        h = textureContainer->header.info.height >> colorTargetInfos[i].mip_level;

        // The framebuffer cannot be larger than the smallest attachment.

        if (w < framebufferWidth) {
            framebufferWidth = w;
        }

        if (h < framebufferHeight) {
            framebufferHeight = h;
        }
    }

    if (depthStencilTargetInfo != NULL) {
        VulkanTextureContainer *textureContainer = (VulkanTextureContainer *)depthStencilTargetInfo->texture;

        w = textureContainer->header.info.width;
        h = textureContainer->header.info.height;

        // The framebuffer cannot be larger than the smallest attachment.

        if (w < framebufferWidth) {
            framebufferWidth = w;
        }

        if (h < framebufferHeight) {
            framebufferHeight = h;
        }
    }

    for (i = 0; i < numColorTargets; i += 1) {
        VulkanTextureContainer *textureContainer = (VulkanTextureContainer *)colorTargetInfos[i].texture;
        VulkanTextureSubresource *subresource = VULKAN_INTERNAL_PrepareTextureSubresourceForWrite(
            renderer,
            vulkanCommandBuffer,
            textureContainer,
            textureContainer->header.info.type == SDL_GPU_TEXTURETYPE_3D ? 0 : colorTargetInfos[i].layer_or_depth_plane,
            colorTargetInfos[i].mip_level,
            colorTargetInfos[i].cycle,
            VULKAN_TEXTURE_USAGE_MODE_COLOR_ATTACHMENT);

        vulkanCommandBuffer->colorAttachmentSubresources[vulkanCommandBuffer->colorAttachmentSubresourceCount] = subresource;
        vulkanCommandBuffer->colorAttachmentSubresourceCount += 1;
        VULKAN_INTERNAL_TrackTexture(vulkanCommandBuffer, subresource->parent);
        totalColorAttachmentCount += 1;
        clearCount += 1;

        if (colorTargetInfos[i].store_op == SDL_GPU_STOREOP_RESOLVE || colorTargetInfos[i].store_op == SDL_GPU_STOREOP_RESOLVE_AND_STORE) {
            VulkanTextureContainer *resolveContainer = (VulkanTextureContainer *)colorTargetInfos[i].resolve_texture;
            VulkanTextureSubresource *resolveSubresource = VULKAN_INTERNAL_PrepareTextureSubresourceForWrite(
                renderer,
                vulkanCommandBuffer,
                resolveContainer,
                colorTargetInfos[i].resolve_layer,
                colorTargetInfos[i].resolve_mip_level,
                colorTargetInfos[i].cycle_resolve_texture,
                VULKAN_TEXTURE_USAGE_MODE_COLOR_ATTACHMENT);

            vulkanCommandBuffer->resolveAttachmentSubresources[vulkanCommandBuffer->resolveAttachmentSubresourceCount] = resolveSubresource;
            vulkanCommandBuffer->resolveAttachmentSubresourceCount += 1;
            VULKAN_INTERNAL_TrackTexture(vulkanCommandBuffer, resolveSubresource->parent);
            totalColorAttachmentCount += 1;
            clearCount += 1;
        }
    }

    if (depthStencilTargetInfo != NULL) {
        VulkanTextureContainer *textureContainer = (VulkanTextureContainer *)depthStencilTargetInfo->texture;
        VulkanTextureSubresource *subresource = VULKAN_INTERNAL_PrepareTextureSubresourceForWrite(
            renderer,
            vulkanCommandBuffer,
            textureContainer,
            0,
            0,
            depthStencilTargetInfo->cycle,
            VULKAN_TEXTURE_USAGE_MODE_DEPTH_STENCIL_ATTACHMENT);

        vulkanCommandBuffer->depthStencilAttachmentSubresource = subresource;
        VULKAN_INTERNAL_TrackTexture(vulkanCommandBuffer, subresource->parent);
        clearCount += 1;
    }

    // Fetch required render objects

    renderPass = VULKAN_INTERNAL_FetchRenderPass(
        renderer,
        colorTargetInfos,
        numColorTargets,
        depthStencilTargetInfo);

    if (renderPass == VK_NULL_HANDLE) {
        return;
    }

    framebuffer = VULKAN_INTERNAL_FetchFramebuffer(
        renderer,
        renderPass,
        colorTargetInfos,
        numColorTargets,
        depthStencilTargetInfo,
        framebufferWidth,
        framebufferHeight);

    if (framebuffer == NULL) {
        return;
    }

    VULKAN_INTERNAL_TrackFramebuffer(vulkanCommandBuffer, framebuffer);

    // Set clear values

    clearValues = SDL_stack_alloc(VkClearValue, clearCount);

    for (i = 0; i < totalColorAttachmentCount; i += 1) {
        clearValues[i].color.float32[0] = colorTargetInfos[i].clear_color.r;
        clearValues[i].color.float32[1] = colorTargetInfos[i].clear_color.g;
        clearValues[i].color.float32[2] = colorTargetInfos[i].clear_color.b;
        clearValues[i].color.float32[3] = colorTargetInfos[i].clear_color.a;

        if (colorTargetInfos[i].store_op == SDL_GPU_STOREOP_RESOLVE || colorTargetInfos[i].store_op == SDL_GPU_STOREOP_RESOLVE_AND_STORE) {
            // Skip over the resolve texture, we're not clearing it
            i += 1;
        }
    }

    if (depthStencilTargetInfo != NULL) {
        clearValues[totalColorAttachmentCount].depthStencil.depth =
            depthStencilTargetInfo->clear_depth;
        clearValues[totalColorAttachmentCount].depthStencil.stencil =
            depthStencilTargetInfo->clear_stencil;
    }

    VkRenderPassBeginInfo renderPassBeginInfo;
    renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassBeginInfo.pNext = NULL;
    renderPassBeginInfo.renderPass = renderPass;
    renderPassBeginInfo.framebuffer = framebuffer->framebuffer;
    renderPassBeginInfo.pClearValues = clearValues;
    renderPassBeginInfo.clearValueCount = clearCount;
    renderPassBeginInfo.renderArea.extent.width = framebufferWidth;
    renderPassBeginInfo.renderArea.extent.height = framebufferHeight;
    renderPassBeginInfo.renderArea.offset.x = 0;
    renderPassBeginInfo.renderArea.offset.y = 0;

    renderer->vkCmdBeginRenderPass(
        vulkanCommandBuffer->commandBuffer,
        &renderPassBeginInfo,
        VK_SUBPASS_CONTENTS_INLINE);

    SDL_stack_free(clearValues);

    // Set sensible default states

    defaultViewport.x = 0;
    defaultViewport.y = 0;
    defaultViewport.w = (float)framebufferWidth;
    defaultViewport.h = (float)framebufferHeight;
    defaultViewport.min_depth = 0;
    defaultViewport.max_depth = 1;

    VULKAN_INTERNAL_SetCurrentViewport(
        vulkanCommandBuffer,
        &defaultViewport);

    defaultScissor.x = 0;
    defaultScissor.y = 0;
    defaultScissor.w = (Sint32)framebufferWidth;
    defaultScissor.h = (Sint32)framebufferHeight;

    VULKAN_INTERNAL_SetCurrentScissor(
        vulkanCommandBuffer,
        &defaultScissor);

    defaultBlendConstants.r = 1.0f;
    defaultBlendConstants.g = 1.0f;
    defaultBlendConstants.b = 1.0f;
    defaultBlendConstants.a = 1.0f;

    VULKAN_INTERNAL_SetCurrentBlendConstants(
        vulkanCommandBuffer,
        defaultBlendConstants);

    VULKAN_INTERNAL_SetCurrentStencilReference(
        vulkanCommandBuffer,
        0);
}

static void VULKAN_BindGraphicsPipeline(
    SDL_GPUCommandBuffer *commandBuffer,
    SDL_GPUGraphicsPipeline *graphicsPipeline)
{
    VulkanCommandBuffer *vulkanCommandBuffer = (VulkanCommandBuffer *)commandBuffer;
    VulkanRenderer *renderer = vulkanCommandBuffer->renderer;
    VulkanGraphicsPipeline *pipeline = (VulkanGraphicsPipeline *)graphicsPipeline;

    renderer->vkCmdBindPipeline(
        vulkanCommandBuffer->commandBuffer,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        pipeline->pipeline);

    vulkanCommandBuffer->currentGraphicsPipeline = pipeline;

    VULKAN_INTERNAL_TrackGraphicsPipeline(vulkanCommandBuffer, pipeline);

    // Acquire uniform buffers if necessary
    for (Uint32 i = 0; i < pipeline->resourceLayout->vertexUniformBufferCount; i += 1) {
        if (vulkanCommandBuffer->vertexUniformBuffers[i] == NULL) {
            vulkanCommandBuffer->vertexUniformBuffers[i] = VULKAN_INTERNAL_AcquireUniformBufferFromPool(
                vulkanCommandBuffer);
        }
    }

    for (Uint32 i = 0; i < pipeline->resourceLayout->fragmentUniformBufferCount; i += 1) {
        if (vulkanCommandBuffer->fragmentUniformBuffers[i] == NULL) {
            vulkanCommandBuffer->fragmentUniformBuffers[i] = VULKAN_INTERNAL_AcquireUniformBufferFromPool(
                vulkanCommandBuffer);
        }
    }

    // Mark bindings as needed
    vulkanCommandBuffer->needNewVertexResourceDescriptorSet = true;
    vulkanCommandBuffer->needNewFragmentResourceDescriptorSet = true;
    vulkanCommandBuffer->needNewVertexUniformDescriptorSet = true;
    vulkanCommandBuffer->needNewFragmentUniformDescriptorSet = true;
    vulkanCommandBuffer->needNewVertexUniformOffsets = true;
    vulkanCommandBuffer->needNewFragmentUniformOffsets = true;
}

static void VULKAN_BindVertexBuffers(
    SDL_GPUCommandBuffer *commandBuffer,
    Uint32 firstSlot,
    const SDL_GPUBufferBinding *bindings,
    Uint32 numBindings)
{
    VulkanCommandBuffer *vulkanCommandBuffer = (VulkanCommandBuffer *)commandBuffer;

    for (Uint32 i = 0; i < numBindings; i += 1) {
        VulkanBuffer *buffer = ((VulkanBufferContainer *)bindings[i].buffer)->activeBuffer;
        if (vulkanCommandBuffer->vertexBuffers[firstSlot + i] != buffer->buffer || vulkanCommandBuffer->vertexBufferOffsets[firstSlot + i] != bindings[i].offset) {
            VULKAN_INTERNAL_TrackBuffer(vulkanCommandBuffer, buffer);

            vulkanCommandBuffer->vertexBuffers[firstSlot + i] = buffer->buffer;
            vulkanCommandBuffer->vertexBufferOffsets[firstSlot + i] = bindings[i].offset;
            vulkanCommandBuffer->needVertexBufferBind = true;
        }
    }

    vulkanCommandBuffer->vertexBufferCount =
        SDL_max(vulkanCommandBuffer->vertexBufferCount, firstSlot + numBindings);
}

static void VULKAN_BindIndexBuffer(
    SDL_GPUCommandBuffer *commandBuffer,
    const SDL_GPUBufferBinding *binding,
    SDL_GPUIndexElementSize indexElementSize)
{
    VulkanCommandBuffer *vulkanCommandBuffer = (VulkanCommandBuffer *)commandBuffer;
    VulkanRenderer *renderer = vulkanCommandBuffer->renderer;
    VulkanBuffer *vulkanBuffer = ((VulkanBufferContainer *)binding->buffer)->activeBuffer;

    VULKAN_INTERNAL_TrackBuffer(vulkanCommandBuffer, vulkanBuffer);

    renderer->vkCmdBindIndexBuffer(
        vulkanCommandBuffer->commandBuffer,
        vulkanBuffer->buffer,
        (VkDeviceSize)binding->offset,
        SDLToVK_IndexType[indexElementSize]);
}

static void VULKAN_PushVertexUniformData(
    SDL_GPUCommandBuffer *commandBuffer,
    Uint32 slotIndex,
    const void *data,
    Uint32 length)
{
    VulkanCommandBuffer *vulkanCommandBuffer = (VulkanCommandBuffer *)commandBuffer;

    VULKAN_INTERNAL_PushUniformData(
        vulkanCommandBuffer,
        VULKAN_UNIFORM_BUFFER_STAGE_VERTEX,
        slotIndex,
        data,
        length);
}

static void VULKAN_PushFragmentUniformData(
    SDL_GPUCommandBuffer *commandBuffer,
    Uint32 slotIndex,
    const void *data,
    Uint32 length)
{
    VulkanCommandBuffer *vulkanCommandBuffer = (VulkanCommandBuffer *)commandBuffer;

    VULKAN_INTERNAL_PushUniformData(
        vulkanCommandBuffer,
        VULKAN_UNIFORM_BUFFER_STAGE_FRAGMENT,
        slotIndex,
        data,
        length);
}

static void VULKAN_EndRenderPass(
    SDL_GPUCommandBuffer *commandBuffer)
{
    VulkanCommandBuffer *vulkanCommandBuffer = (VulkanCommandBuffer *)commandBuffer;
    VulkanRenderer *renderer = vulkanCommandBuffer->renderer;
    Uint32 i;

    renderer->vkCmdEndRenderPass(
        vulkanCommandBuffer->commandBuffer);

    for (i = 0; i < vulkanCommandBuffer->colorAttachmentSubresourceCount; i += 1) {
        VULKAN_INTERNAL_TextureSubresourceTransitionToDefaultUsage(
            renderer,
            vulkanCommandBuffer,
            VULKAN_TEXTURE_USAGE_MODE_COLOR_ATTACHMENT,
            vulkanCommandBuffer->colorAttachmentSubresources[i]);
    }
    vulkanCommandBuffer->colorAttachmentSubresourceCount = 0;

    for (i = 0; i < vulkanCommandBuffer->resolveAttachmentSubresourceCount; i += 1) {
        VULKAN_INTERNAL_TextureSubresourceTransitionToDefaultUsage(
            renderer,
            vulkanCommandBuffer,
            VULKAN_TEXTURE_USAGE_MODE_COLOR_ATTACHMENT,
            vulkanCommandBuffer->resolveAttachmentSubresources[i]);
    }
    vulkanCommandBuffer->resolveAttachmentSubresourceCount = 0;

    if (vulkanCommandBuffer->depthStencilAttachmentSubresource != NULL) {
        VULKAN_INTERNAL_TextureSubresourceTransitionToDefaultUsage(
            renderer,
            vulkanCommandBuffer,
            VULKAN_TEXTURE_USAGE_MODE_DEPTH_STENCIL_ATTACHMENT,
            vulkanCommandBuffer->depthStencilAttachmentSubresource);
        vulkanCommandBuffer->depthStencilAttachmentSubresource = NULL;
    }

    vulkanCommandBuffer->currentGraphicsPipeline = NULL;

    vulkanCommandBuffer->vertexResourceDescriptorSet = VK_NULL_HANDLE;
    vulkanCommandBuffer->vertexUniformDescriptorSet = VK_NULL_HANDLE;
    vulkanCommandBuffer->fragmentResourceDescriptorSet = VK_NULL_HANDLE;
    vulkanCommandBuffer->fragmentUniformDescriptorSet = VK_NULL_HANDLE;

    // Reset bind state
    SDL_zeroa(vulkanCommandBuffer->colorAttachmentSubresources);
    SDL_zeroa(vulkanCommandBuffer->resolveAttachmentSubresources);
    vulkanCommandBuffer->depthStencilAttachmentSubresource = NULL;

    SDL_zeroa(vulkanCommandBuffer->vertexBuffers);
    SDL_zeroa(vulkanCommandBuffer->vertexBufferOffsets);
    vulkanCommandBuffer->vertexBufferCount = 0;

    SDL_zeroa(vulkanCommandBuffer->vertexSamplers);
    SDL_zeroa(vulkanCommandBuffer->vertexSamplerTextures);
    SDL_zeroa(vulkanCommandBuffer->vertexStorageTextures);
    SDL_zeroa(vulkanCommandBuffer->vertexStorageBuffers);

    SDL_zeroa(vulkanCommandBuffer->fragmentSamplers);
    SDL_zeroa(vulkanCommandBuffer->fragmentSamplerTextures);
    SDL_zeroa(vulkanCommandBuffer->fragmentStorageTextures);
    SDL_zeroa(vulkanCommandBuffer->fragmentStorageBuffers);
}

static void VULKAN_BeginComputePass(
    SDL_GPUCommandBuffer *commandBuffer,
    const SDL_GPUStorageTextureReadWriteBinding *storageTextureBindings,
    Uint32 numStorageTextureBindings,
    const SDL_GPUStorageBufferReadWriteBinding *storageBufferBindings,
    Uint32 numStorageBufferBindings)
{
    VulkanCommandBuffer *vulkanCommandBuffer = (VulkanCommandBuffer *)commandBuffer;
    VulkanRenderer *renderer = vulkanCommandBuffer->renderer;
    VulkanBufferContainer *bufferContainer;
    VulkanBuffer *buffer;
    Uint32 i;

    vulkanCommandBuffer->readWriteComputeStorageTextureSubresourceCount = numStorageTextureBindings;

    for (i = 0; i < numStorageTextureBindings; i += 1) {
        VulkanTextureContainer *textureContainer = (VulkanTextureContainer *)storageTextureBindings[i].texture;
        VulkanTextureSubresource *subresource = VULKAN_INTERNAL_PrepareTextureSubresourceForWrite(
            renderer,
            vulkanCommandBuffer,
            textureContainer,
            storageTextureBindings[i].layer,
            storageTextureBindings[i].mip_level,
            storageTextureBindings[i].cycle,
            VULKAN_TEXTURE_USAGE_MODE_COMPUTE_STORAGE_READ_WRITE);

        vulkanCommandBuffer->readWriteComputeStorageTextureSubresources[i] = subresource;

        VULKAN_INTERNAL_TrackTexture(
            vulkanCommandBuffer,
            subresource->parent);
    }

    for (i = 0; i < numStorageBufferBindings; i += 1) {
        bufferContainer = (VulkanBufferContainer *)storageBufferBindings[i].buffer;
        buffer = VULKAN_INTERNAL_PrepareBufferForWrite(
            renderer,
            vulkanCommandBuffer,
            bufferContainer,
            storageBufferBindings[i].cycle,
            VULKAN_BUFFER_USAGE_MODE_COMPUTE_STORAGE_READ);

        vulkanCommandBuffer->readWriteComputeStorageBuffers[i] = buffer;

        VULKAN_INTERNAL_TrackBuffer(
            vulkanCommandBuffer,
            buffer);
    }
}

static void VULKAN_BindComputePipeline(
    SDL_GPUCommandBuffer *commandBuffer,
    SDL_GPUComputePipeline *computePipeline)
{
    VulkanCommandBuffer *vulkanCommandBuffer = (VulkanCommandBuffer *)commandBuffer;
    VulkanRenderer *renderer = vulkanCommandBuffer->renderer;
    VulkanComputePipeline *vulkanComputePipeline = (VulkanComputePipeline *)computePipeline;

    renderer->vkCmdBindPipeline(
        vulkanCommandBuffer->commandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        vulkanComputePipeline->pipeline);

    vulkanCommandBuffer->currentComputePipeline = vulkanComputePipeline;

    VULKAN_INTERNAL_TrackComputePipeline(vulkanCommandBuffer, vulkanComputePipeline);

    // Acquire uniform buffers if necessary
    for (Uint32 i = 0; i < vulkanComputePipeline->resourceLayout->numUniformBuffers; i += 1) {
        if (vulkanCommandBuffer->computeUniformBuffers[i] == NULL) {
            vulkanCommandBuffer->computeUniformBuffers[i] = VULKAN_INTERNAL_AcquireUniformBufferFromPool(
                vulkanCommandBuffer);
        }
    }

    // Mark binding as needed
    vulkanCommandBuffer->needNewComputeReadWriteDescriptorSet = true;
    vulkanCommandBuffer->needNewComputeReadOnlyDescriptorSet = true;
    vulkanCommandBuffer->needNewComputeUniformDescriptorSet = true;
    vulkanCommandBuffer->needNewComputeUniformOffsets = true;
}

static void VULKAN_BindComputeSamplers(
    SDL_GPUCommandBuffer *commandBuffer,
    Uint32 firstSlot,
    const SDL_GPUTextureSamplerBinding *textureSamplerBindings,
    Uint32 numBindings)
{
    VulkanCommandBuffer *vulkanCommandBuffer = (VulkanCommandBuffer *)commandBuffer;

    for (Uint32 i = 0; i < numBindings; i += 1) {
        VulkanTextureContainer *textureContainer = (VulkanTextureContainer *)textureSamplerBindings[i].texture;
        VulkanSampler *sampler = (VulkanSampler *)textureSamplerBindings[i].sampler;

        if (vulkanCommandBuffer->computeSamplers[firstSlot + i] != sampler) {
            VULKAN_INTERNAL_TrackSampler(
                vulkanCommandBuffer,
                sampler);

            vulkanCommandBuffer->computeSamplers[firstSlot + i] = sampler;
            vulkanCommandBuffer->needNewComputeReadOnlyDescriptorSet = true;
        }

        if (vulkanCommandBuffer->computeSamplerTextures[firstSlot + i] != textureContainer->activeTexture) {
            VULKAN_INTERNAL_TrackTexture(
                vulkanCommandBuffer,
                textureContainer->activeTexture);

            vulkanCommandBuffer->computeSamplerTextures[firstSlot + i] = textureContainer->activeTexture;
            vulkanCommandBuffer->needNewComputeReadOnlyDescriptorSet = true;
        }
    }
}

static void VULKAN_BindComputeStorageTextures(
    SDL_GPUCommandBuffer *commandBuffer,
    Uint32 firstSlot,
    SDL_GPUTexture *const *storageTextures,
    Uint32 numBindings)
{
    VulkanCommandBuffer *vulkanCommandBuffer = (VulkanCommandBuffer *)commandBuffer;
    VulkanRenderer *renderer = vulkanCommandBuffer->renderer;

    for (Uint32 i = 0; i < numBindings; i += 1) {
        VulkanTextureContainer *textureContainer = (VulkanTextureContainer *)storageTextures[i];

        if (vulkanCommandBuffer->readOnlyComputeStorageTextures[firstSlot + i] != textureContainer->activeTexture) {
            /* If a different texture as in this slot, transition it back to its default usage */
            if (vulkanCommandBuffer->readOnlyComputeStorageTextures[firstSlot + i] != NULL) {
                VULKAN_INTERNAL_TextureTransitionToDefaultUsage(
                    renderer,
                    vulkanCommandBuffer,
                    VULKAN_TEXTURE_USAGE_MODE_COMPUTE_STORAGE_READ,
                    vulkanCommandBuffer->readOnlyComputeStorageTextures[firstSlot + i]);
            }

            /* Then transition the new texture and prepare it for binding */
            VULKAN_INTERNAL_TextureTransitionFromDefaultUsage(
                renderer,
                vulkanCommandBuffer,
                VULKAN_TEXTURE_USAGE_MODE_COMPUTE_STORAGE_READ,
                textureContainer->activeTexture);


            VULKAN_INTERNAL_TrackTexture(
                vulkanCommandBuffer,
                textureContainer->activeTexture);

            vulkanCommandBuffer->readOnlyComputeStorageTextures[firstSlot + i] = textureContainer->activeTexture;
            vulkanCommandBuffer->needNewComputeReadOnlyDescriptorSet = true;
        }
    }
}

static void VULKAN_BindComputeStorageBuffers(
    SDL_GPUCommandBuffer *commandBuffer,
    Uint32 firstSlot,
    SDL_GPUBuffer *const *storageBuffers,
    Uint32 numBindings)
{
    VulkanCommandBuffer *vulkanCommandBuffer = (VulkanCommandBuffer *)commandBuffer;
    VulkanRenderer *renderer = vulkanCommandBuffer->renderer;

    for (Uint32 i = 0; i < numBindings; i += 1) {
        VulkanBufferContainer *bufferContainer = (VulkanBufferContainer *)storageBuffers[i];

        if (vulkanCommandBuffer->readOnlyComputeStorageBuffers[firstSlot + i] != bufferContainer->activeBuffer) {
            /* If a different buffer was in this slot, transition it back to its default usage */
            if (vulkanCommandBuffer->readOnlyComputeStorageBuffers[firstSlot + i] != NULL) {
                VULKAN_INTERNAL_BufferTransitionToDefaultUsage(
                    renderer,
                    vulkanCommandBuffer,
                    VULKAN_BUFFER_USAGE_MODE_COMPUTE_STORAGE_READ,
                    vulkanCommandBuffer->readOnlyComputeStorageBuffers[firstSlot + i]);
            }

            /* Then transition the new buffer and prepare it for binding */
            VULKAN_INTERNAL_BufferTransitionFromDefaultUsage(
                renderer,
                vulkanCommandBuffer,
                VULKAN_BUFFER_USAGE_MODE_COMPUTE_STORAGE_READ,
                bufferContainer->activeBuffer);

            VULKAN_INTERNAL_TrackBuffer(
                vulkanCommandBuffer,
                bufferContainer->activeBuffer);

            vulkanCommandBuffer->readOnlyComputeStorageBuffers[firstSlot + i] = bufferContainer->activeBuffer;
            vulkanCommandBuffer->needNewComputeReadOnlyDescriptorSet = true;
        }
    }
}

static void VULKAN_PushComputeUniformData(
    SDL_GPUCommandBuffer *commandBuffer,
    Uint32 slotIndex,
    const void *data,
    Uint32 length)
{
    VulkanCommandBuffer *vulkanCommandBuffer = (VulkanCommandBuffer *)commandBuffer;

    VULKAN_INTERNAL_PushUniformData(
        vulkanCommandBuffer,
        VULKAN_UNIFORM_BUFFER_STAGE_COMPUTE,
        slotIndex,
        data,
        length);
}

static void VULKAN_INTERNAL_BindComputeDescriptorSets(
    VulkanRenderer *renderer,
    VulkanCommandBuffer *commandBuffer)
{
    VulkanComputePipelineResourceLayout *resourceLayout;
    DescriptorSetLayout *descriptorSetLayout;
    VkWriteDescriptorSet writeDescriptorSets[
        MAX_TEXTURE_SAMPLERS_PER_STAGE +
        MAX_STORAGE_TEXTURES_PER_STAGE +
        MAX_STORAGE_BUFFERS_PER_STAGE +
        MAX_COMPUTE_WRITE_TEXTURES +
        MAX_COMPUTE_WRITE_BUFFERS +
        MAX_UNIFORM_BUFFERS_PER_STAGE];
    VkDescriptorBufferInfo bufferInfos[MAX_STORAGE_BUFFERS_PER_STAGE + MAX_COMPUTE_WRITE_BUFFERS + MAX_UNIFORM_BUFFERS_PER_STAGE];
    VkDescriptorImageInfo imageInfos[MAX_TEXTURE_SAMPLERS_PER_STAGE + MAX_STORAGE_TEXTURES_PER_STAGE + MAX_COMPUTE_WRITE_TEXTURES];
    Uint32 dynamicOffsets[MAX_UNIFORM_BUFFERS_PER_STAGE];
    Uint32 writeCount = 0;
    Uint32 bufferInfoCount = 0;
    Uint32 imageInfoCount = 0;
    Uint32 dynamicOffsetCount = 0;

    if (
        !commandBuffer->needNewComputeReadOnlyDescriptorSet &&
        !commandBuffer->needNewComputeReadWriteDescriptorSet &&
        !commandBuffer->needNewComputeUniformDescriptorSet &&
        !commandBuffer->needNewComputeUniformOffsets
    ) {
        return;
    }

    resourceLayout = commandBuffer->currentComputePipeline->resourceLayout;

    if (commandBuffer->needNewComputeReadOnlyDescriptorSet) {
        descriptorSetLayout = resourceLayout->descriptorSetLayouts[0];

        commandBuffer->computeReadOnlyDescriptorSet = VULKAN_INTERNAL_FetchDescriptorSet(
            renderer,
            commandBuffer,
            descriptorSetLayout);

        for (Uint32 i = 0; i < resourceLayout->numSamplers; i += 1) {
            VkWriteDescriptorSet *currentWriteDescriptorSet = &writeDescriptorSets[writeCount];

            currentWriteDescriptorSet->sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            currentWriteDescriptorSet->pNext = NULL;
            currentWriteDescriptorSet->descriptorCount = 1;
            currentWriteDescriptorSet->descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            currentWriteDescriptorSet->dstArrayElement = 0;
            currentWriteDescriptorSet->dstBinding = i;
            currentWriteDescriptorSet->dstSet = commandBuffer->computeReadOnlyDescriptorSet;
            currentWriteDescriptorSet->pTexelBufferView = NULL;
            currentWriteDescriptorSet->pBufferInfo = NULL;

            imageInfos[imageInfoCount].sampler = commandBuffer->computeSamplers[i]->sampler;
            imageInfos[imageInfoCount].imageView = commandBuffer->computeSamplerTextures[i]->fullView;
            imageInfos[imageInfoCount].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            currentWriteDescriptorSet->pImageInfo = &imageInfos[imageInfoCount];

            writeCount += 1;
            imageInfoCount += 1;
        }

        for (Uint32 i = 0; i < resourceLayout->numReadonlyStorageTextures; i += 1) {
            VkWriteDescriptorSet *currentWriteDescriptorSet = &writeDescriptorSets[writeCount];

            currentWriteDescriptorSet->sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            currentWriteDescriptorSet->pNext = NULL;
            currentWriteDescriptorSet->descriptorCount = 1;
            currentWriteDescriptorSet->descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; // Yes, we are declaring the readonly storage texture as a sampled image, because shaders are stupid.
            currentWriteDescriptorSet->dstArrayElement = 0;
            currentWriteDescriptorSet->dstBinding = resourceLayout->numSamplers + i;
            currentWriteDescriptorSet->dstSet = commandBuffer->computeReadOnlyDescriptorSet;
            currentWriteDescriptorSet->pTexelBufferView = NULL;
            currentWriteDescriptorSet->pBufferInfo = NULL;

            imageInfos[imageInfoCount].sampler = VK_NULL_HANDLE;
            imageInfos[imageInfoCount].imageView = commandBuffer->readOnlyComputeStorageTextures[i]->fullView;
            imageInfos[imageInfoCount].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            currentWriteDescriptorSet->pImageInfo = &imageInfos[imageInfoCount];

            writeCount += 1;
            imageInfoCount += 1;
        }

        for (Uint32 i = 0; i < resourceLayout->numReadonlyStorageBuffers; i += 1) {
            VkWriteDescriptorSet *currentWriteDescriptorSet = &writeDescriptorSets[writeCount];

            currentWriteDescriptorSet->sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            currentWriteDescriptorSet->pNext = NULL;
            currentWriteDescriptorSet->descriptorCount = 1;
            currentWriteDescriptorSet->descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            currentWriteDescriptorSet->dstArrayElement = 0;
            currentWriteDescriptorSet->dstBinding = resourceLayout->numSamplers + resourceLayout->numReadonlyStorageTextures + i;
            currentWriteDescriptorSet->dstSet = commandBuffer->computeReadOnlyDescriptorSet;
            currentWriteDescriptorSet->pTexelBufferView = NULL;
            currentWriteDescriptorSet->pImageInfo = NULL;

            bufferInfos[bufferInfoCount].buffer = commandBuffer->readOnlyComputeStorageBuffers[i]->buffer;
            bufferInfos[bufferInfoCount].offset = 0;
            bufferInfos[bufferInfoCount].range = VK_WHOLE_SIZE;

            currentWriteDescriptorSet->pBufferInfo = &bufferInfos[bufferInfoCount];

            writeCount += 1;
            bufferInfoCount += 1;
        }

        commandBuffer->needNewComputeReadOnlyDescriptorSet = false;
    }

    if (commandBuffer->needNewComputeReadWriteDescriptorSet) {
        descriptorSetLayout = resourceLayout->descriptorSetLayouts[1];

        commandBuffer->computeReadWriteDescriptorSet = VULKAN_INTERNAL_FetchDescriptorSet(
            renderer,
            commandBuffer,
            descriptorSetLayout);

        for (Uint32 i = 0; i < resourceLayout->numReadWriteStorageTextures; i += 1) {
            VkWriteDescriptorSet *currentWriteDescriptorSet = &writeDescriptorSets[writeCount];

            currentWriteDescriptorSet->sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            currentWriteDescriptorSet->pNext = NULL;
            currentWriteDescriptorSet->descriptorCount = 1;
            currentWriteDescriptorSet->descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            currentWriteDescriptorSet->dstArrayElement = 0;
            currentWriteDescriptorSet->dstBinding = i;
            currentWriteDescriptorSet->dstSet = commandBuffer->computeReadWriteDescriptorSet;
            currentWriteDescriptorSet->pTexelBufferView = NULL;
            currentWriteDescriptorSet->pBufferInfo = NULL;

            imageInfos[imageInfoCount].sampler = VK_NULL_HANDLE;
            imageInfos[imageInfoCount].imageView = commandBuffer->readWriteComputeStorageTextureSubresources[i]->computeWriteView;
            imageInfos[imageInfoCount].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            currentWriteDescriptorSet->pImageInfo = &imageInfos[imageInfoCount];

            writeCount += 1;
            imageInfoCount += 1;
        }

        for (Uint32 i = 0; i < resourceLayout->numReadWriteStorageBuffers; i += 1) {
            VkWriteDescriptorSet *currentWriteDescriptorSet = &writeDescriptorSets[writeCount];

            currentWriteDescriptorSet->sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            currentWriteDescriptorSet->pNext = NULL;
            currentWriteDescriptorSet->descriptorCount = 1;
            currentWriteDescriptorSet->descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            currentWriteDescriptorSet->dstArrayElement = 0;
            currentWriteDescriptorSet->dstBinding = resourceLayout->numReadWriteStorageTextures + i;
            currentWriteDescriptorSet->dstSet = commandBuffer->computeReadWriteDescriptorSet;
            currentWriteDescriptorSet->pTexelBufferView = NULL;
            currentWriteDescriptorSet->pImageInfo = NULL;

            bufferInfos[bufferInfoCount].buffer = commandBuffer->readWriteComputeStorageBuffers[i]->buffer;
            bufferInfos[bufferInfoCount].offset = 0;
            bufferInfos[bufferInfoCount].range = VK_WHOLE_SIZE;

            currentWriteDescriptorSet->pBufferInfo = &bufferInfos[bufferInfoCount];

            writeCount += 1;
            bufferInfoCount += 1;
        }

        commandBuffer->needNewComputeReadWriteDescriptorSet = false;
    }

    if (commandBuffer->needNewComputeUniformDescriptorSet) {
        descriptorSetLayout = resourceLayout->descriptorSetLayouts[2];

        commandBuffer->computeUniformDescriptorSet = VULKAN_INTERNAL_FetchDescriptorSet(
            renderer,
            commandBuffer,
            descriptorSetLayout);


        for (Uint32 i = 0; i < resourceLayout->numUniformBuffers; i += 1) {
            VkWriteDescriptorSet *currentWriteDescriptorSet = &writeDescriptorSets[writeCount];

            currentWriteDescriptorSet->sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            currentWriteDescriptorSet->pNext = NULL;
            currentWriteDescriptorSet->descriptorCount = 1;
            currentWriteDescriptorSet->descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
            currentWriteDescriptorSet->dstArrayElement = 0;
            currentWriteDescriptorSet->dstBinding = i;
            currentWriteDescriptorSet->dstSet = commandBuffer->computeUniformDescriptorSet;
            currentWriteDescriptorSet->pTexelBufferView = NULL;
            currentWriteDescriptorSet->pImageInfo = NULL;

            bufferInfos[bufferInfoCount].buffer = commandBuffer->computeUniformBuffers[i]->buffer->buffer;
            bufferInfos[bufferInfoCount].offset = 0;
            bufferInfos[bufferInfoCount].range = MAX_UBO_SECTION_SIZE;

            currentWriteDescriptorSet->pBufferInfo = &bufferInfos[bufferInfoCount];

            writeCount += 1;
            bufferInfoCount += 1;
        }

        commandBuffer->needNewComputeUniformDescriptorSet = false;
    }

    for (Uint32 i = 0; i < resourceLayout->numUniformBuffers; i += 1) {
        dynamicOffsets[i] = commandBuffer->computeUniformBuffers[i]->drawOffset;
        dynamicOffsetCount += 1;
    }

    renderer->vkUpdateDescriptorSets(
        renderer->logicalDevice,
        writeCount,
        writeDescriptorSets,
        0,
        NULL);

    VkDescriptorSet sets[3];
    sets[0] = commandBuffer->computeReadOnlyDescriptorSet;
    sets[1] = commandBuffer->computeReadWriteDescriptorSet;
    sets[2] = commandBuffer->computeUniformDescriptorSet;

    renderer->vkCmdBindDescriptorSets(
        commandBuffer->commandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        resourceLayout->pipelineLayout,
        0,
        3,
        sets,
        dynamicOffsetCount,
        dynamicOffsets);

    commandBuffer->needNewVertexUniformOffsets = false;
}

static void VULKAN_DispatchCompute(
    SDL_GPUCommandBuffer *commandBuffer,
    Uint32 groupcountX,
    Uint32 groupcountY,
    Uint32 groupcountZ)
{
    VulkanCommandBuffer *vulkanCommandBuffer = (VulkanCommandBuffer *)commandBuffer;
    VulkanRenderer *renderer = vulkanCommandBuffer->renderer;

    VULKAN_INTERNAL_BindComputeDescriptorSets(renderer, vulkanCommandBuffer);

    renderer->vkCmdDispatch(
        vulkanCommandBuffer->commandBuffer,
        groupcountX,
        groupcountY,
        groupcountZ);
}

static void VULKAN_DispatchComputeIndirect(
    SDL_GPUCommandBuffer *commandBuffer,
    SDL_GPUBuffer *buffer,
    Uint32 offset)
{
    VulkanCommandBuffer *vulkanCommandBuffer = (VulkanCommandBuffer *)commandBuffer;
    VulkanRenderer *renderer = vulkanCommandBuffer->renderer;
    VulkanBuffer *vulkanBuffer = ((VulkanBufferContainer *)buffer)->activeBuffer;

    VULKAN_INTERNAL_BindComputeDescriptorSets(renderer, vulkanCommandBuffer);

    renderer->vkCmdDispatchIndirect(
        vulkanCommandBuffer->commandBuffer,
        vulkanBuffer->buffer,
        offset);

    VULKAN_INTERNAL_TrackBuffer(vulkanCommandBuffer, vulkanBuffer);
}

static void VULKAN_EndComputePass(
    SDL_GPUCommandBuffer *commandBuffer)
{
    VulkanCommandBuffer *vulkanCommandBuffer = (VulkanCommandBuffer *)commandBuffer;
    Uint32 i;

    for (i = 0; i < vulkanCommandBuffer->readWriteComputeStorageTextureSubresourceCount; i += 1) {
        VULKAN_INTERNAL_TextureSubresourceTransitionToDefaultUsage(
            vulkanCommandBuffer->renderer,
            vulkanCommandBuffer,
            VULKAN_TEXTURE_USAGE_MODE_COMPUTE_STORAGE_READ_WRITE,
            vulkanCommandBuffer->readWriteComputeStorageTextureSubresources[i]);
        vulkanCommandBuffer->readWriteComputeStorageTextureSubresources[i] = NULL;
    }
    vulkanCommandBuffer->readWriteComputeStorageTextureSubresourceCount = 0;

    for (i = 0; i < MAX_COMPUTE_WRITE_BUFFERS; i += 1) {
        if (vulkanCommandBuffer->readWriteComputeStorageBuffers[i] != NULL) {
            VULKAN_INTERNAL_BufferTransitionToDefaultUsage(
                vulkanCommandBuffer->renderer,
                vulkanCommandBuffer,
                VULKAN_BUFFER_USAGE_MODE_COMPUTE_STORAGE_READ_WRITE,
                vulkanCommandBuffer->readWriteComputeStorageBuffers[i]);

            vulkanCommandBuffer->readWriteComputeStorageBuffers[i] = NULL;
        }
    }

    for (i = 0; i < MAX_STORAGE_TEXTURES_PER_STAGE; i += 1) {
        if (vulkanCommandBuffer->readOnlyComputeStorageTextures[i] != NULL) {
            VULKAN_INTERNAL_TextureTransitionToDefaultUsage(
                vulkanCommandBuffer->renderer,
                vulkanCommandBuffer,
                VULKAN_TEXTURE_USAGE_MODE_COMPUTE_STORAGE_READ,
                vulkanCommandBuffer->readOnlyComputeStorageTextures[i]);

            vulkanCommandBuffer->readOnlyComputeStorageTextures[i] = NULL;
        }
    }

    for (i = 0; i < MAX_STORAGE_BUFFERS_PER_STAGE; i += 1) {
        if (vulkanCommandBuffer->readOnlyComputeStorageBuffers[i] != NULL) {
            VULKAN_INTERNAL_BufferTransitionToDefaultUsage(
                vulkanCommandBuffer->renderer,
                vulkanCommandBuffer,
                VULKAN_BUFFER_USAGE_MODE_COMPUTE_STORAGE_READ,
                vulkanCommandBuffer->readOnlyComputeStorageBuffers[i]);

            vulkanCommandBuffer->readOnlyComputeStorageBuffers[i] = NULL;
        }
    }

    // we don't need a barrier because sampler state is always the default if sampler bit is set
    SDL_zeroa(vulkanCommandBuffer->computeSamplerTextures);
    SDL_zeroa(vulkanCommandBuffer->computeSamplers);

    vulkanCommandBuffer->currentComputePipeline = NULL;

    vulkanCommandBuffer->computeReadOnlyDescriptorSet = VK_NULL_HANDLE;
    vulkanCommandBuffer->computeReadWriteDescriptorSet = VK_NULL_HANDLE;
    vulkanCommandBuffer->computeUniformDescriptorSet = VK_NULL_HANDLE;
}

static void *VULKAN_MapTransferBuffer(
    SDL_GPURenderer *driverData,
    SDL_GPUTransferBuffer *transferBuffer,
    bool cycle)
{
    VulkanRenderer *renderer = (VulkanRenderer *)driverData;
    VulkanBufferContainer *transferBufferContainer = (VulkanBufferContainer *)transferBuffer;

    if (
        cycle &&
        SDL_GetAtomicInt(&transferBufferContainer->activeBuffer->referenceCount) > 0) {
        VULKAN_INTERNAL_CycleActiveBuffer(
            renderer,
            transferBufferContainer);
    }

    Uint8 *bufferPointer =
        transferBufferContainer->activeBuffer->usedRegion->allocation->mapPointer +
        transferBufferContainer->activeBuffer->usedRegion->resourceOffset;

    return bufferPointer;
}

static void VULKAN_UnmapTransferBuffer(
    SDL_GPURenderer *driverData,
    SDL_GPUTransferBuffer *transferBuffer)
{
    // no-op because transfer buffers are persistently mapped
    (void)driverData;
    (void)transferBuffer;
}

static void VULKAN_BeginCopyPass(
    SDL_GPUCommandBuffer *commandBuffer)
{
    // no-op
    (void)commandBuffer;
}

static void VULKAN_UploadToTexture(
    SDL_GPUCommandBuffer *commandBuffer,
    const SDL_GPUTextureTransferInfo *source,
    const SDL_GPUTextureRegion *destination,
    bool cycle)
{
    VulkanCommandBuffer *vulkanCommandBuffer = (VulkanCommandBuffer *)commandBuffer;
    VulkanRenderer *renderer = vulkanCommandBuffer->renderer;
    VulkanBufferContainer *transferBufferContainer = (VulkanBufferContainer *)source->transfer_buffer;
    VulkanTextureContainer *vulkanTextureContainer = (VulkanTextureContainer *)destination->texture;
    VulkanTextureSubresource *vulkanTextureSubresource;
    VkBufferImageCopy imageCopy;

    // Note that the transfer buffer does not need a barrier, as it is synced by the client

    vulkanTextureSubresource = VULKAN_INTERNAL_PrepareTextureSubresourceForWrite(
        renderer,
        vulkanCommandBuffer,
        vulkanTextureContainer,
        destination->layer,
        destination->mip_level,
        cycle,
        VULKAN_TEXTURE_USAGE_MODE_COPY_DESTINATION);

    imageCopy.imageExtent.width = destination->w;
    imageCopy.imageExtent.height = destination->h;
    imageCopy.imageExtent.depth = destination->d;
    imageCopy.imageOffset.x = destination->x;
    imageCopy.imageOffset.y = destination->y;
    imageCopy.imageOffset.z = destination->z;
    imageCopy.imageSubresource.aspectMask = vulkanTextureSubresource->parent->aspectFlags;
    imageCopy.imageSubresource.baseArrayLayer = destination->layer;
    imageCopy.imageSubresource.layerCount = 1;
    imageCopy.imageSubresource.mipLevel = destination->mip_level;
    imageCopy.bufferOffset = source->offset;
    imageCopy.bufferRowLength = source->pixels_per_row;
    imageCopy.bufferImageHeight = source->rows_per_layer;

    renderer->vkCmdCopyBufferToImage(
        vulkanCommandBuffer->commandBuffer,
        transferBufferContainer->activeBuffer->buffer,
        vulkanTextureSubresource->parent->image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &imageCopy);

    VULKAN_INTERNAL_TextureSubresourceTransitionToDefaultUsage(
        renderer,
        vulkanCommandBuffer,
        VULKAN_TEXTURE_USAGE_MODE_COPY_DESTINATION,
        vulkanTextureSubresource);

    VULKAN_INTERNAL_TrackBuffer(vulkanCommandBuffer, transferBufferContainer->activeBuffer);
    VULKAN_INTERNAL_TrackTexture(vulkanCommandBuffer, vulkanTextureSubresource->parent);
}

static void VULKAN_UploadToBuffer(
    SDL_GPUCommandBuffer *commandBuffer,
    const SDL_GPUTransferBufferLocation *source,
    const SDL_GPUBufferRegion *destination,
    bool cycle)
{
    VulkanCommandBuffer *vulkanCommandBuffer = (VulkanCommandBuffer *)commandBuffer;
    VulkanRenderer *renderer = vulkanCommandBuffer->renderer;
    VulkanBufferContainer *transferBufferContainer = (VulkanBufferContainer *)source->transfer_buffer;
    VulkanBufferContainer *bufferContainer = (VulkanBufferContainer *)destination->buffer;
    VkBufferCopy bufferCopy;

    // Note that the transfer buffer does not need a barrier, as it is synced by the client

    VulkanBuffer *vulkanBuffer = VULKAN_INTERNAL_PrepareBufferForWrite(
        renderer,
        vulkanCommandBuffer,
        bufferContainer,
        cycle,
        VULKAN_BUFFER_USAGE_MODE_COPY_DESTINATION);

    bufferCopy.srcOffset = source->offset;
    bufferCopy.dstOffset = destination->offset;
    bufferCopy.size = destination->size;

    renderer->vkCmdCopyBuffer(
        vulkanCommandBuffer->commandBuffer,
        transferBufferContainer->activeBuffer->buffer,
        vulkanBuffer->buffer,
        1,
        &bufferCopy);

    VULKAN_INTERNAL_BufferTransitionToDefaultUsage(
        renderer,
        vulkanCommandBuffer,
        VULKAN_BUFFER_USAGE_MODE_COPY_DESTINATION,
        vulkanBuffer);

    VULKAN_INTERNAL_TrackBuffer(vulkanCommandBuffer, transferBufferContainer->activeBuffer);
    VULKAN_INTERNAL_TrackBuffer(vulkanCommandBuffer, vulkanBuffer);
}

// Readback

static void VULKAN_DownloadFromTexture(
    SDL_GPUCommandBuffer *commandBuffer,
    const SDL_GPUTextureRegion *source,
    const SDL_GPUTextureTransferInfo *destination)
{
    VulkanCommandBuffer *vulkanCommandBuffer = (VulkanCommandBuffer *)commandBuffer;
    VulkanRenderer *renderer = vulkanCommandBuffer->renderer;
    VulkanTextureContainer *textureContainer = (VulkanTextureContainer *)source->texture;
    VulkanTextureSubresource *vulkanTextureSubresource;
    VulkanBufferContainer *transferBufferContainer = (VulkanBufferContainer *)destination->transfer_buffer;
    VkBufferImageCopy imageCopy;
    vulkanTextureSubresource = VULKAN_INTERNAL_FetchTextureSubresource(
        textureContainer,
        source->layer,
        source->mip_level);

    // Note that the transfer buffer does not need a barrier, as it is synced by the client

    VULKAN_INTERNAL_TextureSubresourceTransitionFromDefaultUsage(
        renderer,
        vulkanCommandBuffer,
        VULKAN_TEXTURE_USAGE_MODE_COPY_SOURCE,
        vulkanTextureSubresource);

    imageCopy.imageExtent.width = source->w;
    imageCopy.imageExtent.height = source->h;
    imageCopy.imageExtent.depth = source->d;
    imageCopy.imageOffset.x = source->x;
    imageCopy.imageOffset.y = source->y;
    imageCopy.imageOffset.z = source->z;
    imageCopy.imageSubresource.aspectMask = vulkanTextureSubresource->parent->aspectFlags;
    imageCopy.imageSubresource.baseArrayLayer = source->layer;
    imageCopy.imageSubresource.layerCount = 1;
    imageCopy.imageSubresource.mipLevel = source->mip_level;
    imageCopy.bufferOffset = destination->offset;
    imageCopy.bufferRowLength = destination->pixels_per_row;
    imageCopy.bufferImageHeight = destination->rows_per_layer;

    renderer->vkCmdCopyImageToBuffer(
        vulkanCommandBuffer->commandBuffer,
        vulkanTextureSubresource->parent->image,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        transferBufferContainer->activeBuffer->buffer,
        1,
        &imageCopy);

    VULKAN_INTERNAL_TextureSubresourceTransitionToDefaultUsage(
        renderer,
        vulkanCommandBuffer,
        VULKAN_TEXTURE_USAGE_MODE_COPY_SOURCE,
        vulkanTextureSubresource);

    VULKAN_INTERNAL_TrackBuffer(vulkanCommandBuffer, transferBufferContainer->activeBuffer);
    VULKAN_INTERNAL_TrackTexture(vulkanCommandBuffer, vulkanTextureSubresource->parent);
}

static void VULKAN_DownloadFromBuffer(
    SDL_GPUCommandBuffer *commandBuffer,
    const SDL_GPUBufferRegion *source,
    const SDL_GPUTransferBufferLocation *destination)
{
    VulkanCommandBuffer *vulkanCommandBuffer = (VulkanCommandBuffer *)commandBuffer;
    VulkanRenderer *renderer = vulkanCommandBuffer->renderer;
    VulkanBufferContainer *bufferContainer = (VulkanBufferContainer *)source->buffer;
    VulkanBufferContainer *transferBufferContainer = (VulkanBufferContainer *)destination->transfer_buffer;
    VkBufferCopy bufferCopy;

    // Note that transfer buffer does not need a barrier, as it is synced by the client

    VULKAN_INTERNAL_BufferTransitionFromDefaultUsage(
        renderer,
        vulkanCommandBuffer,
        VULKAN_BUFFER_USAGE_MODE_COPY_SOURCE,
        bufferContainer->activeBuffer);

    bufferCopy.srcOffset = source->offset;
    bufferCopy.dstOffset = destination->offset;
    bufferCopy.size = source->size;

    renderer->vkCmdCopyBuffer(
        vulkanCommandBuffer->commandBuffer,
        bufferContainer->activeBuffer->buffer,
        transferBufferContainer->activeBuffer->buffer,
        1,
        &bufferCopy);

    VULKAN_INTERNAL_BufferTransitionToDefaultUsage(
        renderer,
        vulkanCommandBuffer,
        VULKAN_BUFFER_USAGE_MODE_COPY_SOURCE,
        bufferContainer->activeBuffer);

    VULKAN_INTERNAL_TrackBuffer(vulkanCommandBuffer, transferBufferContainer->activeBuffer);
    VULKAN_INTERNAL_TrackBuffer(vulkanCommandBuffer, bufferContainer->activeBuffer);
}

static void VULKAN_CopyTextureToTexture(
    SDL_GPUCommandBuffer *commandBuffer,
    const SDL_GPUTextureLocation *source,
    const SDL_GPUTextureLocation *destination,
    Uint32 w,
    Uint32 h,
    Uint32 d,
    bool cycle)
{
    VulkanCommandBuffer *vulkanCommandBuffer = (VulkanCommandBuffer *)commandBuffer;
    VulkanRenderer *renderer = vulkanCommandBuffer->renderer;
    VulkanTextureSubresource *srcSubresource;
    VulkanTextureSubresource *dstSubresource;
    VkImageCopy imageCopy;

    srcSubresource = VULKAN_INTERNAL_FetchTextureSubresource(
        (VulkanTextureContainer *)source->texture,
        source->layer,
        source->mip_level);

    dstSubresource = VULKAN_INTERNAL_PrepareTextureSubresourceForWrite(
        renderer,
        vulkanCommandBuffer,
        (VulkanTextureContainer *)destination->texture,
        destination->layer,
        destination->mip_level,
        cycle,
        VULKAN_TEXTURE_USAGE_MODE_COPY_DESTINATION);

    VULKAN_INTERNAL_TextureSubresourceTransitionFromDefaultUsage(
        renderer,
        vulkanCommandBuffer,
        VULKAN_TEXTURE_USAGE_MODE_COPY_SOURCE,
        srcSubresource);

    imageCopy.srcOffset.x = source->x;
    imageCopy.srcOffset.y = source->y;
    imageCopy.srcOffset.z = source->z;
    imageCopy.srcSubresource.aspectMask = srcSubresource->parent->aspectFlags;
    imageCopy.srcSubresource.baseArrayLayer = source->layer;
    imageCopy.srcSubresource.layerCount = 1;
    imageCopy.srcSubresource.mipLevel = source->mip_level;
    imageCopy.dstOffset.x = destination->x;
    imageCopy.dstOffset.y = destination->y;
    imageCopy.dstOffset.z = destination->z;
    imageCopy.dstSubresource.aspectMask = dstSubresource->parent->aspectFlags;
    imageCopy.dstSubresource.baseArrayLayer = destination->layer;
    imageCopy.dstSubresource.layerCount = 1;
    imageCopy.dstSubresource.mipLevel = destination->mip_level;
    imageCopy.extent.width = w;
    imageCopy.extent.height = h;
    imageCopy.extent.depth = d;

    renderer->vkCmdCopyImage(
        vulkanCommandBuffer->commandBuffer,
        srcSubresource->parent->image,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        dstSubresource->parent->image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &imageCopy);

    VULKAN_INTERNAL_TextureSubresourceTransitionToDefaultUsage(
        renderer,
        vulkanCommandBuffer,
        VULKAN_TEXTURE_USAGE_MODE_COPY_SOURCE,
        srcSubresource);

    VULKAN_INTERNAL_TextureSubresourceTransitionToDefaultUsage(
        renderer,
        vulkanCommandBuffer,
        VULKAN_TEXTURE_USAGE_MODE_COPY_DESTINATION,
        dstSubresource);

    VULKAN_INTERNAL_TrackTexture(vulkanCommandBuffer, srcSubresource->parent);
    VULKAN_INTERNAL_TrackTexture(vulkanCommandBuffer, dstSubresource->parent);
}

static void VULKAN_CopyBufferToBuffer(
    SDL_GPUCommandBuffer *commandBuffer,
    const SDL_GPUBufferLocation *source,
    const SDL_GPUBufferLocation *destination,
    Uint32 size,
    bool cycle)
{
    VulkanCommandBuffer *vulkanCommandBuffer = (VulkanCommandBuffer *)commandBuffer;
    VulkanRenderer *renderer = vulkanCommandBuffer->renderer;
    VulkanBufferContainer *srcContainer = (VulkanBufferContainer *)source->buffer;
    VulkanBufferContainer *dstContainer = (VulkanBufferContainer *)destination->buffer;
    VkBufferCopy bufferCopy;

    VulkanBuffer *dstBuffer = VULKAN_INTERNAL_PrepareBufferForWrite(
        renderer,
        vulkanCommandBuffer,
        dstContainer,
        cycle,
        VULKAN_BUFFER_USAGE_MODE_COPY_DESTINATION);

    VULKAN_INTERNAL_BufferTransitionFromDefaultUsage(
        renderer,
        vulkanCommandBuffer,
        VULKAN_BUFFER_USAGE_MODE_COPY_SOURCE,
        srcContainer->activeBuffer);

    bufferCopy.srcOffset = source->offset;
    bufferCopy.dstOffset = destination->offset;
    bufferCopy.size = size;

    renderer->vkCmdCopyBuffer(
        vulkanCommandBuffer->commandBuffer,
        srcContainer->activeBuffer->buffer,
        dstBuffer->buffer,
        1,
        &bufferCopy);

    VULKAN_INTERNAL_BufferTransitionToDefaultUsage(
        renderer,
        vulkanCommandBuffer,
        VULKAN_BUFFER_USAGE_MODE_COPY_SOURCE,
        srcContainer->activeBuffer);

    VULKAN_INTERNAL_BufferTransitionToDefaultUsage(
        renderer,
        vulkanCommandBuffer,
        VULKAN_BUFFER_USAGE_MODE_COPY_DESTINATION,
        dstBuffer);

    VULKAN_INTERNAL_TrackBuffer(vulkanCommandBuffer, srcContainer->activeBuffer);
    VULKAN_INTERNAL_TrackBuffer(vulkanCommandBuffer, dstBuffer);
}

static void VULKAN_GenerateMipmaps(
    SDL_GPUCommandBuffer *commandBuffer,
    SDL_GPUTexture *texture)
{
    VulkanCommandBuffer *vulkanCommandBuffer = (VulkanCommandBuffer *)commandBuffer;
    VulkanRenderer *renderer = vulkanCommandBuffer->renderer;
    VulkanTextureContainer *container = (VulkanTextureContainer *)texture;
    VulkanTextureSubresource *srcTextureSubresource;
    VulkanTextureSubresource *dstTextureSubresource;
    VkImageBlit blit;

    // Blit each slice sequentially. Barriers, barriers everywhere!
    for (Uint32 layerOrDepthIndex = 0; layerOrDepthIndex < container->header.info.layer_count_or_depth; layerOrDepthIndex += 1)
        for (Uint32 level = 1; level < container->header.info.num_levels; level += 1) {
            Uint32 layer = container->header.info.type == SDL_GPU_TEXTURETYPE_3D ? 0 : layerOrDepthIndex;
            Uint32 depth = container->header.info.type == SDL_GPU_TEXTURETYPE_3D ? layerOrDepthIndex : 0;

            Uint32 srcSubresourceIndex = VULKAN_INTERNAL_GetTextureSubresourceIndex(
                level - 1,
                layer,
                container->header.info.num_levels);
            Uint32 dstSubresourceIndex = VULKAN_INTERNAL_GetTextureSubresourceIndex(
                level,
                layer,
                container->header.info.num_levels);

            srcTextureSubresource = &container->activeTexture->subresources[srcSubresourceIndex];
            dstTextureSubresource = &container->activeTexture->subresources[dstSubresourceIndex];

            VULKAN_INTERNAL_TextureSubresourceTransitionFromDefaultUsage(
                renderer,
                vulkanCommandBuffer,
                VULKAN_TEXTURE_USAGE_MODE_COPY_SOURCE,
                srcTextureSubresource);

            VULKAN_INTERNAL_TextureSubresourceTransitionFromDefaultUsage(
                renderer,
                vulkanCommandBuffer,
                VULKAN_TEXTURE_USAGE_MODE_COPY_DESTINATION,
                dstTextureSubresource);

            blit.srcOffsets[0].x = 0;
            blit.srcOffsets[0].y = 0;
            blit.srcOffsets[0].z = depth;

            blit.srcOffsets[1].x = container->header.info.width >> (level - 1);
            blit.srcOffsets[1].y = container->header.info.height >> (level - 1);
            blit.srcOffsets[1].z = depth + 1;

            blit.dstOffsets[0].x = 0;
            blit.dstOffsets[0].y = 0;
            blit.dstOffsets[0].z = depth;

            blit.dstOffsets[1].x = container->header.info.width >> level;
            blit.dstOffsets[1].y = container->header.info.height >> level;
            blit.dstOffsets[1].z = depth + 1;

            blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            blit.srcSubresource.baseArrayLayer = layer;
            blit.srcSubresource.layerCount = 1;
            blit.srcSubresource.mipLevel = level - 1;

            blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            blit.dstSubresource.baseArrayLayer = layer;
            blit.dstSubresource.layerCount = 1;
            blit.dstSubresource.mipLevel = level;

            renderer->vkCmdBlitImage(
                vulkanCommandBuffer->commandBuffer,
                container->activeTexture->image,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                container->activeTexture->image,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1,
                &blit,
                VK_FILTER_LINEAR);

            VULKAN_INTERNAL_TextureSubresourceTransitionToDefaultUsage(
                renderer,
                vulkanCommandBuffer,
                VULKAN_TEXTURE_USAGE_MODE_COPY_SOURCE,
                srcTextureSubresource);

            VULKAN_INTERNAL_TextureSubresourceTransitionToDefaultUsage(
                renderer,
                vulkanCommandBuffer,
                VULKAN_TEXTURE_USAGE_MODE_COPY_DESTINATION,
                dstTextureSubresource);

            VULKAN_INTERNAL_TrackTexture(vulkanCommandBuffer, srcTextureSubresource->parent);
            VULKAN_INTERNAL_TrackTexture(vulkanCommandBuffer, dstTextureSubresource->parent);
        }
}

static void VULKAN_EndCopyPass(
    SDL_GPUCommandBuffer *commandBuffer)
{
    // no-op
    (void)commandBuffer;
}

static void VULKAN_Blit(
    SDL_GPUCommandBuffer *commandBuffer,
    const SDL_GPUBlitInfo *info)
{
    VulkanCommandBuffer *vulkanCommandBuffer = (VulkanCommandBuffer *)commandBuffer;
    VulkanRenderer *renderer = vulkanCommandBuffer->renderer;
    TextureCommonHeader *srcHeader = (TextureCommonHeader *)info->source.texture;
    TextureCommonHeader *dstHeader = (TextureCommonHeader *)info->destination.texture;
    VkImageBlit region;
    Uint32 srcLayer = srcHeader->info.type == SDL_GPU_TEXTURETYPE_3D ? 0 : info->source.layer_or_depth_plane;
    Uint32 srcDepth = srcHeader->info.type == SDL_GPU_TEXTURETYPE_3D ? info->source.layer_or_depth_plane : 0;
    Uint32 dstLayer = dstHeader->info.type == SDL_GPU_TEXTURETYPE_3D ? 0 : info->destination.layer_or_depth_plane;
    Uint32 dstDepth = dstHeader->info.type == SDL_GPU_TEXTURETYPE_3D ? info->destination.layer_or_depth_plane : 0;
    int32_t swap;

    // Using BeginRenderPass to clear because vkCmdClearColorImage requires barriers anyway
    if (info->load_op == SDL_GPU_LOADOP_CLEAR) {
        SDL_GPUColorTargetInfo targetInfo;
        SDL_zero(targetInfo);
        targetInfo.texture = info->destination.texture;
        targetInfo.mip_level = info->destination.mip_level;
        targetInfo.layer_or_depth_plane = info->destination.layer_or_depth_plane;
        targetInfo.load_op = SDL_GPU_LOADOP_CLEAR;
        targetInfo.store_op = SDL_GPU_STOREOP_STORE;
        targetInfo.clear_color = info->clear_color;
        targetInfo.cycle = info->cycle;
        VULKAN_BeginRenderPass(
            commandBuffer,
            &targetInfo,
            1,
            NULL);
        VULKAN_EndRenderPass(commandBuffer);
    }

    VulkanTextureSubresource *srcSubresource = VULKAN_INTERNAL_FetchTextureSubresource(
        (VulkanTextureContainer *)info->source.texture,
        srcLayer,
        info->source.mip_level);

    VulkanTextureSubresource *dstSubresource = VULKAN_INTERNAL_PrepareTextureSubresourceForWrite(
        renderer,
        vulkanCommandBuffer,
        (VulkanTextureContainer *)info->destination.texture,
        dstLayer,
        info->destination.mip_level,
        info->cycle,
        VULKAN_TEXTURE_USAGE_MODE_COPY_DESTINATION);

    VULKAN_INTERNAL_TextureSubresourceTransitionFromDefaultUsage(
        renderer,
        vulkanCommandBuffer,
        VULKAN_TEXTURE_USAGE_MODE_COPY_SOURCE,
        srcSubresource);

    region.srcSubresource.aspectMask = srcSubresource->parent->aspectFlags;
    region.srcSubresource.baseArrayLayer = srcSubresource->layer;
    region.srcSubresource.layerCount = 1;
    region.srcSubresource.mipLevel = srcSubresource->level;
    region.srcOffsets[0].x = info->source.x;
    region.srcOffsets[0].y = info->source.y;
    region.srcOffsets[0].z = srcDepth;
    region.srcOffsets[1].x = info->source.x + info->source.w;
    region.srcOffsets[1].y = info->source.y + info->source.h;
    region.srcOffsets[1].z = srcDepth + 1;

    if (info->flip_mode & SDL_FLIP_HORIZONTAL) {
        // flip the x positions
        swap = region.srcOffsets[0].x;
        region.srcOffsets[0].x = region.srcOffsets[1].x;
        region.srcOffsets[1].x = swap;
    }

    if (info->flip_mode & SDL_FLIP_VERTICAL) {
        // flip the y positions
        swap = region.srcOffsets[0].y;
        region.srcOffsets[0].y = region.srcOffsets[1].y;
        region.srcOffsets[1].y = swap;
    }

    region.dstSubresource.aspectMask = dstSubresource->parent->aspectFlags;
    region.dstSubresource.baseArrayLayer = dstSubresource->layer;
    region.dstSubresource.layerCount = 1;
    region.dstSubresource.mipLevel = dstSubresource->level;
    region.dstOffsets[0].x = info->destination.x;
    region.dstOffsets[0].y = info->destination.y;
    region.dstOffsets[0].z = dstDepth;
    region.dstOffsets[1].x = info->destination.x + info->destination.w;
    region.dstOffsets[1].y = info->destination.y + info->destination.h;
    region.dstOffsets[1].z = dstDepth + 1;

    renderer->vkCmdBlitImage(
        vulkanCommandBuffer->commandBuffer,
        srcSubresource->parent->image,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        dstSubresource->parent->image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &region,
        SDLToVK_Filter[info->filter]);

    VULKAN_INTERNAL_TextureSubresourceTransitionToDefaultUsage(
        renderer,
        vulkanCommandBuffer,
        VULKAN_TEXTURE_USAGE_MODE_COPY_SOURCE,
        srcSubresource);

    VULKAN_INTERNAL_TextureSubresourceTransitionToDefaultUsage(
        renderer,
        vulkanCommandBuffer,
        VULKAN_TEXTURE_USAGE_MODE_COPY_DESTINATION,
        dstSubresource);

    VULKAN_INTERNAL_TrackTexture(vulkanCommandBuffer, srcSubresource->parent);
    VULKAN_INTERNAL_TrackTexture(vulkanCommandBuffer, dstSubresource->parent);
}

static bool VULKAN_INTERNAL_AllocateCommandBuffer(
    VulkanRenderer *renderer,
    VulkanCommandPool *vulkanCommandPool)
{
    VkCommandBufferAllocateInfo allocateInfo;
    VkResult vulkanResult;
    VkCommandBuffer commandBufferHandle;
    VulkanCommandBuffer *commandBuffer;

    vulkanCommandPool->inactiveCommandBufferCapacity += 1;

    vulkanCommandPool->inactiveCommandBuffers = SDL_realloc(
        vulkanCommandPool->inactiveCommandBuffers,
        sizeof(VulkanCommandBuffer *) *
            vulkanCommandPool->inactiveCommandBufferCapacity);

    allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocateInfo.pNext = NULL;
    allocateInfo.commandPool = vulkanCommandPool->commandPool;
    allocateInfo.commandBufferCount = 1;
    allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

    vulkanResult = renderer->vkAllocateCommandBuffers(
        renderer->logicalDevice,
        &allocateInfo,
        &commandBufferHandle);

    CHECK_VULKAN_ERROR_AND_RETURN(vulkanResult, vkAllocateCommandBuffers, false);

    commandBuffer = SDL_malloc(sizeof(VulkanCommandBuffer));
    commandBuffer->renderer = renderer;
    commandBuffer->commandPool = vulkanCommandPool;
    commandBuffer->commandBuffer = commandBufferHandle;

    commandBuffer->inFlightFence = VK_NULL_HANDLE;

    // Presentation tracking

    commandBuffer->presentDataCapacity = 1;
    commandBuffer->presentDataCount = 0;
    commandBuffer->presentDatas = SDL_malloc(
        commandBuffer->presentDataCapacity * sizeof(VulkanPresentData));

    commandBuffer->waitSemaphoreCapacity = 1;
    commandBuffer->waitSemaphoreCount = 0;
    commandBuffer->waitSemaphores = SDL_malloc(
        commandBuffer->waitSemaphoreCapacity * sizeof(VkSemaphore));

    commandBuffer->signalSemaphoreCapacity = 1;
    commandBuffer->signalSemaphoreCount = 0;
    commandBuffer->signalSemaphores = SDL_malloc(
        commandBuffer->signalSemaphoreCapacity * sizeof(VkSemaphore));

    // Resource bind tracking

    commandBuffer->needVertexBufferBind = false;
    commandBuffer->needNewVertexResourceDescriptorSet = true;
    commandBuffer->needNewVertexUniformDescriptorSet = true;
    commandBuffer->needNewVertexUniformOffsets = true;
    commandBuffer->needNewFragmentResourceDescriptorSet = true;
    commandBuffer->needNewFragmentUniformDescriptorSet = true;
    commandBuffer->needNewFragmentUniformOffsets = true;

    commandBuffer->needNewComputeReadWriteDescriptorSet = true;
    commandBuffer->needNewComputeReadOnlyDescriptorSet = true;
    commandBuffer->needNewComputeUniformDescriptorSet = true;
    commandBuffer->needNewComputeUniformOffsets = true;

    commandBuffer->vertexResourceDescriptorSet = VK_NULL_HANDLE;
    commandBuffer->vertexUniformDescriptorSet = VK_NULL_HANDLE;
    commandBuffer->fragmentResourceDescriptorSet = VK_NULL_HANDLE;
    commandBuffer->fragmentUniformDescriptorSet = VK_NULL_HANDLE;

    commandBuffer->computeReadOnlyDescriptorSet = VK_NULL_HANDLE;
    commandBuffer->computeReadWriteDescriptorSet = VK_NULL_HANDLE;
    commandBuffer->computeUniformDescriptorSet = VK_NULL_HANDLE;

    // Resource tracking

    commandBuffer->usedBufferCapacity = 4;
    commandBuffer->usedBufferCount = 0;
    commandBuffer->usedBuffers = SDL_malloc(
        commandBuffer->usedBufferCapacity * sizeof(VulkanBuffer *));

    commandBuffer->usedTextureCapacity = 4;
    commandBuffer->usedTextureCount = 0;
    commandBuffer->usedTextures = SDL_malloc(
        commandBuffer->usedTextureCapacity * sizeof(VulkanTexture *));

    commandBuffer->usedSamplerCapacity = 4;
    commandBuffer->usedSamplerCount = 0;
    commandBuffer->usedSamplers = SDL_malloc(
        commandBuffer->usedSamplerCapacity * sizeof(VulkanSampler *));

    commandBuffer->usedGraphicsPipelineCapacity = 4;
    commandBuffer->usedGraphicsPipelineCount = 0;
    commandBuffer->usedGraphicsPipelines = SDL_malloc(
        commandBuffer->usedGraphicsPipelineCapacity * sizeof(VulkanGraphicsPipeline *));

    commandBuffer->usedComputePipelineCapacity = 4;
    commandBuffer->usedComputePipelineCount = 0;
    commandBuffer->usedComputePipelines = SDL_malloc(
        commandBuffer->usedComputePipelineCapacity * sizeof(VulkanComputePipeline *));

    commandBuffer->usedFramebufferCapacity = 4;
    commandBuffer->usedFramebufferCount = 0;
    commandBuffer->usedFramebuffers = SDL_malloc(
        commandBuffer->usedFramebufferCapacity * sizeof(VulkanFramebuffer *));

    commandBuffer->usedUniformBufferCapacity = 4;
    commandBuffer->usedUniformBufferCount = 0;
    commandBuffer->usedUniformBuffers = SDL_malloc(
        commandBuffer->usedUniformBufferCapacity * sizeof(VulkanUniformBuffer *));

    // Pool it!

    vulkanCommandPool->inactiveCommandBuffers[vulkanCommandPool->inactiveCommandBufferCount] = commandBuffer;
    vulkanCommandPool->inactiveCommandBufferCount += 1;

    return true;
}

static VulkanCommandPool *VULKAN_INTERNAL_FetchCommandPool(
    VulkanRenderer *renderer,
    SDL_ThreadID threadID)
{
    VulkanCommandPool *vulkanCommandPool = NULL;
    VkCommandPoolCreateInfo commandPoolCreateInfo;
    VkResult vulkanResult;
    CommandPoolHashTableKey key;
    key.threadID = threadID;

    bool result = SDL_FindInHashTable(
        renderer->commandPoolHashTable,
        (const void *)&key,
        (const void **)&vulkanCommandPool);

    if (result) {
        return vulkanCommandPool;
    }

    vulkanCommandPool = (VulkanCommandPool *)SDL_malloc(sizeof(VulkanCommandPool));

    commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolCreateInfo.pNext = NULL;
    commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    commandPoolCreateInfo.queueFamilyIndex = renderer->queueFamilyIndex;

    vulkanResult = renderer->vkCreateCommandPool(
        renderer->logicalDevice,
        &commandPoolCreateInfo,
        NULL,
        &vulkanCommandPool->commandPool);

    if (vulkanResult != VK_SUCCESS) {
        SDL_free(vulkanCommandPool);
        CHECK_VULKAN_ERROR_AND_RETURN(vulkanResult, vkCreateCommandPool, NULL);
        return NULL;
    }

    vulkanCommandPool->threadID = threadID;

    vulkanCommandPool->inactiveCommandBufferCapacity = 0;
    vulkanCommandPool->inactiveCommandBufferCount = 0;
    vulkanCommandPool->inactiveCommandBuffers = NULL;

    if (!VULKAN_INTERNAL_AllocateCommandBuffer(
        renderer,
        vulkanCommandPool)) {
        VULKAN_INTERNAL_DestroyCommandPool(renderer, vulkanCommandPool);
        return NULL;
    }

    CommandPoolHashTableKey *allocedKey = SDL_malloc(sizeof(CommandPoolHashTableKey));
    allocedKey->threadID = threadID;

    SDL_InsertIntoHashTable(
        renderer->commandPoolHashTable,
        (const void *)allocedKey,
        (const void *)vulkanCommandPool, true);

    return vulkanCommandPool;
}

static VulkanCommandBuffer *VULKAN_INTERNAL_GetInactiveCommandBufferFromPool(
    VulkanRenderer *renderer,
    SDL_ThreadID threadID)
{
    VulkanCommandPool *commandPool =
        VULKAN_INTERNAL_FetchCommandPool(renderer, threadID);
    VulkanCommandBuffer *commandBuffer;

    if (commandPool == NULL) {
        return NULL;
    }

    if (commandPool->inactiveCommandBufferCount == 0) {
        if (!VULKAN_INTERNAL_AllocateCommandBuffer(
            renderer,
            commandPool)) {
            return NULL;
        }
    }

    commandBuffer = commandPool->inactiveCommandBuffers[commandPool->inactiveCommandBufferCount - 1];
    commandPool->inactiveCommandBufferCount -= 1;

    return commandBuffer;
}

static SDL_GPUCommandBuffer *VULKAN_AcquireCommandBuffer(
    SDL_GPURenderer *driverData)
{
    VulkanRenderer *renderer = (VulkanRenderer *)driverData;
    VkResult result;
    Uint32 i;

    SDL_ThreadID threadID = SDL_GetCurrentThreadID();

    SDL_LockMutex(renderer->acquireCommandBufferLock);

    VulkanCommandBuffer *commandBuffer =
        VULKAN_INTERNAL_GetInactiveCommandBufferFromPool(renderer, threadID);

    commandBuffer->descriptorSetCache = VULKAN_INTERNAL_AcquireDescriptorSetCache(renderer);

    SDL_UnlockMutex(renderer->acquireCommandBufferLock);

    if (commandBuffer == NULL) {
        return NULL;
    }

    // Reset state

    commandBuffer->currentComputePipeline = NULL;
    commandBuffer->currentGraphicsPipeline = NULL;

    SDL_zeroa(commandBuffer->colorAttachmentSubresources);
    SDL_zeroa(commandBuffer->resolveAttachmentSubresources);
    commandBuffer->depthStencilAttachmentSubresource = NULL;
    commandBuffer->colorAttachmentSubresourceCount = 0;
    commandBuffer->resolveAttachmentSubresourceCount = 0;

    for (i = 0; i < MAX_UNIFORM_BUFFERS_PER_STAGE; i += 1) {
        commandBuffer->vertexUniformBuffers[i] = NULL;
        commandBuffer->fragmentUniformBuffers[i] = NULL;
        commandBuffer->computeUniformBuffers[i] = NULL;
    }

    commandBuffer->needVertexBufferBind = false;
    commandBuffer->needNewVertexResourceDescriptorSet = true;
    commandBuffer->needNewVertexUniformDescriptorSet = true;
    commandBuffer->needNewVertexUniformOffsets = true;
    commandBuffer->needNewFragmentResourceDescriptorSet = true;
    commandBuffer->needNewFragmentUniformDescriptorSet = true;
    commandBuffer->needNewFragmentUniformOffsets = true;

    commandBuffer->needNewComputeReadOnlyDescriptorSet = true;
    commandBuffer->needNewComputeUniformDescriptorSet = true;
    commandBuffer->needNewComputeUniformOffsets = true;

    commandBuffer->vertexResourceDescriptorSet = VK_NULL_HANDLE;
    commandBuffer->vertexUniformDescriptorSet = VK_NULL_HANDLE;
    commandBuffer->fragmentResourceDescriptorSet = VK_NULL_HANDLE;
    commandBuffer->fragmentUniformDescriptorSet = VK_NULL_HANDLE;

    commandBuffer->computeReadOnlyDescriptorSet = VK_NULL_HANDLE;
    commandBuffer->computeReadWriteDescriptorSet = VK_NULL_HANDLE;
    commandBuffer->computeUniformDescriptorSet = VK_NULL_HANDLE;

    SDL_zeroa(commandBuffer->vertexBuffers);
    SDL_zeroa(commandBuffer->vertexBufferOffsets);
    commandBuffer->vertexBufferCount = 0;

    SDL_zeroa(commandBuffer->vertexSamplerTextures);
    SDL_zeroa(commandBuffer->vertexSamplers);
    SDL_zeroa(commandBuffer->vertexStorageTextures);
    SDL_zeroa(commandBuffer->vertexStorageBuffers);

    SDL_zeroa(commandBuffer->fragmentSamplerTextures);
    SDL_zeroa(commandBuffer->fragmentSamplers);
    SDL_zeroa(commandBuffer->fragmentStorageTextures);
    SDL_zeroa(commandBuffer->fragmentStorageBuffers);

    SDL_zeroa(commandBuffer->readWriteComputeStorageTextureSubresources);
    commandBuffer->readWriteComputeStorageTextureSubresourceCount = 0;
    SDL_zeroa(commandBuffer->readWriteComputeStorageBuffers);
    SDL_zeroa(commandBuffer->computeSamplerTextures);
    SDL_zeroa(commandBuffer->computeSamplers);
    SDL_zeroa(commandBuffer->readOnlyComputeStorageTextures);
    SDL_zeroa(commandBuffer->readOnlyComputeStorageBuffers);

    commandBuffer->autoReleaseFence = true;

    commandBuffer->isDefrag = 0;

    /* Reset the command buffer here to avoid resets being called
     * from a separate thread than where the command buffer was acquired
     */
    result = renderer->vkResetCommandBuffer(
        commandBuffer->commandBuffer,
        VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT);

    CHECK_VULKAN_ERROR_AND_RETURN(result, vkResetCommandBuffer, NULL);

    if (!VULKAN_INTERNAL_BeginCommandBuffer(renderer, commandBuffer)) {
        return NULL;
    }

    return (SDL_GPUCommandBuffer *)commandBuffer;
}

static bool VULKAN_QueryFence(
    SDL_GPURenderer *driverData,
    SDL_GPUFence *fence)
{
    VulkanRenderer *renderer = (VulkanRenderer *)driverData;
    VkResult result;

    result = renderer->vkGetFenceStatus(
        renderer->logicalDevice,
        ((VulkanFenceHandle *)fence)->fence);

    if (result == VK_SUCCESS) {
        return true;
    } else if (result == VK_NOT_READY) {
        return false;
    } else {
        SET_ERROR_AND_RETURN("vkGetFenceStatus: %s", VkErrorMessages(result), false);
    }
}

static void VULKAN_INTERNAL_ReturnFenceToPool(
    VulkanRenderer *renderer,
    VulkanFenceHandle *fenceHandle)
{
    SDL_LockMutex(renderer->fencePool.lock);

    EXPAND_ARRAY_IF_NEEDED(
        renderer->fencePool.availableFences,
        VulkanFenceHandle *,
        renderer->fencePool.availableFenceCount + 1,
        renderer->fencePool.availableFenceCapacity,
        renderer->fencePool.availableFenceCapacity * 2);

    renderer->fencePool.availableFences[renderer->fencePool.availableFenceCount] = fenceHandle;
    renderer->fencePool.availableFenceCount += 1;

    SDL_UnlockMutex(renderer->fencePool.lock);
}

static void VULKAN_ReleaseFence(
    SDL_GPURenderer *driverData,
    SDL_GPUFence *fence)
{
    VulkanFenceHandle *handle = (VulkanFenceHandle *)fence;

    if (SDL_AtomicDecRef(&handle->referenceCount)) {
        VULKAN_INTERNAL_ReturnFenceToPool((VulkanRenderer *)driverData, handle);
    }
}

static WindowData *VULKAN_INTERNAL_FetchWindowData(
    SDL_Window *window)
{
    SDL_PropertiesID properties = SDL_GetWindowProperties(window);
    return (WindowData *)SDL_GetPointerProperty(properties, WINDOW_PROPERTY_DATA, NULL);
}

static bool VULKAN_INTERNAL_OnWindowResize(void *userdata, SDL_Event *e)
{
    SDL_Window *w = (SDL_Window *)userdata;
    WindowData *data;
    if (e->type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED && e->window.windowID == SDL_GetWindowID(w)) {
        data = VULKAN_INTERNAL_FetchWindowData(w);
        data->needsSwapchainRecreate = true;
        data->swapchainCreateWidth = e->window.data1;
        data->swapchainCreateHeight = e->window.data2;
    }

    return true;
}

static bool VULKAN_SupportsSwapchainComposition(
    SDL_GPURenderer *driverData,
    SDL_Window *window,
    SDL_GPUSwapchainComposition swapchainComposition)
{
    VulkanRenderer *renderer = (VulkanRenderer *)driverData;
    WindowData *windowData = VULKAN_INTERNAL_FetchWindowData(window);
    VkSurfaceKHR surface;
    SwapchainSupportDetails supportDetails;
    bool result = false;

    if (windowData == NULL) {
        SET_STRING_ERROR_AND_RETURN("Must claim window before querying swapchain composition support!", false);
    }

    surface = windowData->surface;
    if (!surface) {
        SET_STRING_ERROR_AND_RETURN("Window has no Vulkan surface", false);
    }

    if (VULKAN_INTERNAL_QuerySwapchainSupport(
            renderer,
            renderer->physicalDevice,
            surface,
            &supportDetails)) {

        result = VULKAN_INTERNAL_VerifySwapSurfaceFormat(
            SwapchainCompositionToFormat[swapchainComposition],
            SwapchainCompositionToColorSpace[swapchainComposition],
            supportDetails.formats,
            supportDetails.formatsLength);

        if (!result) {
            // Let's try again with the fallback format...
            result = VULKAN_INTERNAL_VerifySwapSurfaceFormat(
                SwapchainCompositionToFallbackFormat[swapchainComposition],
                SwapchainCompositionToColorSpace[swapchainComposition],
                supportDetails.formats,
                supportDetails.formatsLength);
        }

        SDL_free(supportDetails.formats);
        SDL_free(supportDetails.presentModes);
    }

    return result;
}

static bool VULKAN_SupportsPresentMode(
    SDL_GPURenderer *driverData,
    SDL_Window *window,
    SDL_GPUPresentMode presentMode)
{
    VulkanRenderer *renderer = (VulkanRenderer *)driverData;
    WindowData *windowData = VULKAN_INTERNAL_FetchWindowData(window);
    VkSurfaceKHR surface;
    SwapchainSupportDetails supportDetails;
    bool result = false;

    if (windowData == NULL) {
        SET_STRING_ERROR_AND_RETURN("Must claim window before querying present mode support!", false);
    }

    surface = windowData->surface;
    if (!surface) {
        SET_STRING_ERROR_AND_RETURN("Window has no Vulkan surface", false);
    }

    if (VULKAN_INTERNAL_QuerySwapchainSupport(
            renderer,
            renderer->physicalDevice,
            surface,
            &supportDetails)) {

        result = VULKAN_INTERNAL_VerifySwapPresentMode(
            SDLToVK_PresentMode[presentMode],
            supportDetails.presentModes,
            supportDetails.presentModesLength);

        SDL_free(supportDetails.formats);
        SDL_free(supportDetails.presentModes);
    }

    return result;
}

static bool VULKAN_ClaimWindow(
    SDL_GPURenderer *driverData,
    SDL_Window *window)
{
    VulkanRenderer *renderer = (VulkanRenderer *)driverData;
    WindowData *windowData = VULKAN_INTERNAL_FetchWindowData(window);

    if (windowData == NULL) {
        windowData = SDL_calloc(1, sizeof(WindowData));
        windowData->window = window;
        windowData->presentMode = SDL_GPU_PRESENTMODE_VSYNC;
        windowData->swapchainComposition = SDL_GPU_SWAPCHAINCOMPOSITION_SDR;

        // On non-Apple platforms the swapchain capability currentExtent can be different from the window,
        // so we have to query the window size.
#ifndef SDL_PLATFORM_APPLE
        int w, h;
        SDL_SyncWindow(window);
        SDL_GetWindowSizeInPixels(window, &w, &h);
        windowData->swapchainCreateWidth = w;
        windowData->swapchainCreateHeight = h;
#endif

        Uint32 createSwapchainResult = VULKAN_INTERNAL_CreateSwapchain(renderer, windowData);
        if (createSwapchainResult == 1) {
            SDL_SetPointerProperty(SDL_GetWindowProperties(window), WINDOW_PROPERTY_DATA, windowData);

            SDL_LockMutex(renderer->windowLock);
            if (renderer->claimedWindowCount >= renderer->claimedWindowCapacity) {
                renderer->claimedWindowCapacity *= 2;
                renderer->claimedWindows = SDL_realloc(
                    renderer->claimedWindows,
                    renderer->claimedWindowCapacity * sizeof(WindowData *));
            }

            renderer->claimedWindows[renderer->claimedWindowCount] = windowData;
            renderer->claimedWindowCount += 1;
            SDL_UnlockMutex(renderer->windowLock);

            SDL_AddEventWatch(VULKAN_INTERNAL_OnWindowResize, window);

            return true;
        } else if (createSwapchainResult == VULKAN_INTERNAL_TRY_AGAIN) {
            windowData->needsSwapchainRecreate = true;
            return true;
        } else {
            SDL_free(windowData);
            return false;
        }
    } else {
        SET_STRING_ERROR_AND_RETURN("Window already claimed!", false);
    }
}

static void VULKAN_ReleaseWindow(
    SDL_GPURenderer *driverData,
    SDL_Window *window)
{
    VulkanRenderer *renderer = (VulkanRenderer *)driverData;
    WindowData *windowData = VULKAN_INTERNAL_FetchWindowData(window);
    Uint32 i;

    if (windowData == NULL) {
        return;
    }

    VULKAN_Wait(driverData);

    for (i = 0; i < MAX_FRAMES_IN_FLIGHT; i += 1) {
        if (windowData->inFlightFences[i] != NULL) {
            VULKAN_ReleaseFence(
                driverData,
                windowData->inFlightFences[i]);
        }
    }

    VULKAN_INTERNAL_DestroySwapchain(
        (VulkanRenderer *)driverData,
        windowData);


    SDL_LockMutex(renderer->windowLock);
    for (i = 0; i < renderer->claimedWindowCount; i += 1) {
        if (renderer->claimedWindows[i]->window == window) {
            renderer->claimedWindows[i] = renderer->claimedWindows[renderer->claimedWindowCount - 1];
            renderer->claimedWindowCount -= 1;
            break;
        }
    }
    SDL_UnlockMutex(renderer->windowLock);

    SDL_free(windowData);

    SDL_ClearProperty(SDL_GetWindowProperties(window), WINDOW_PROPERTY_DATA);
    SDL_RemoveEventWatch(VULKAN_INTERNAL_OnWindowResize, window);
}

static Uint32 VULKAN_INTERNAL_RecreateSwapchain(
    VulkanRenderer *renderer,
    WindowData *windowData)
{
    Uint32 i;

    if (!VULKAN_Wait((SDL_GPURenderer *)renderer)) {
        return false;
    }

    for (i = 0; i < MAX_FRAMES_IN_FLIGHT; i += 1) {
        if (windowData->inFlightFences[i] != NULL) {
            VULKAN_ReleaseFence(
                (SDL_GPURenderer *)renderer,
                windowData->inFlightFences[i]);
            windowData->inFlightFences[i] = NULL;
        }
    }

    VULKAN_INTERNAL_DestroySwapchain(renderer, windowData);
    return VULKAN_INTERNAL_CreateSwapchain(renderer, windowData);
}

static bool VULKAN_WaitForSwapchain(
    SDL_GPURenderer *driverData,
    SDL_Window *window)
{
    VulkanRenderer *renderer = (VulkanRenderer *)driverData;
    WindowData *windowData = VULKAN_INTERNAL_FetchWindowData(window);

    if (windowData == NULL) {
        SET_STRING_ERROR_AND_RETURN("Cannot wait for a swapchain from an unclaimed window!", false);
    }

    if (windowData->inFlightFences[windowData->frameCounter] != NULL) {
        if (!VULKAN_WaitForFences(
            driverData,
            true,
            &windowData->inFlightFences[windowData->frameCounter],
            1)) {
            return false;
        }
    }

    return true;
}

static bool VULKAN_INTERNAL_AcquireSwapchainTexture(
    bool block,
    SDL_GPUCommandBuffer *commandBuffer,
    SDL_Window *window,
    SDL_GPUTexture **swapchainTexture,
    Uint32 *swapchainTextureWidth,
    Uint32 *swapchainTextureHeight)
{
    VulkanCommandBuffer *vulkanCommandBuffer = (VulkanCommandBuffer *)commandBuffer;
    VulkanRenderer *renderer = vulkanCommandBuffer->renderer;
    Uint32 swapchainImageIndex;
    WindowData *windowData;
    VkResult acquireResult = VK_SUCCESS;
    VulkanTextureContainer *swapchainTextureContainer = NULL;
    VulkanPresentData *presentData;

    *swapchainTexture = NULL;
    if (swapchainTextureWidth) {
        *swapchainTextureWidth = 0;
    }
    if (swapchainTextureHeight) {
        *swapchainTextureHeight = 0;
    }

    windowData = VULKAN_INTERNAL_FetchWindowData(window);
    if (windowData == NULL) {
        SET_STRING_ERROR_AND_RETURN("Cannot acquire a swapchain texture from an unclaimed window!", false);
    }

    // If window data marked as needing swapchain recreate, try to recreate
    if (windowData->needsSwapchainRecreate) {
        Uint32 recreateSwapchainResult = VULKAN_INTERNAL_RecreateSwapchain(renderer, windowData);
        if (!recreateSwapchainResult) {
            return false;
        } else if (recreateSwapchainResult == VULKAN_INTERNAL_TRY_AGAIN) {
            // Edge case, texture is filled in with NULL but not an error
            if (windowData->inFlightFences[windowData->frameCounter] != NULL) {
                VULKAN_ReleaseFence(
                    (SDL_GPURenderer *)renderer,
                    windowData->inFlightFences[windowData->frameCounter]);
                windowData->inFlightFences[windowData->frameCounter] = NULL;
            }
            return true;
        }
    }

    if (swapchainTextureWidth) {
        *swapchainTextureWidth = windowData->width;
    }
    if (swapchainTextureHeight) {
        *swapchainTextureHeight = windowData->height;
    }

    if (windowData->inFlightFences[windowData->frameCounter] != NULL) {
        if (block) {
            // If we are blocking, just wait for the fence!
            if (!VULKAN_WaitForFences(
                (SDL_GPURenderer *)renderer,
                true,
                &windowData->inFlightFences[windowData->frameCounter],
                1)) {
                return false;
            }
        } else {
            // If we are not blocking and the least recent fence is not signaled,
            // return true to indicate that there is no error but rendering should be skipped.
            if (!VULKAN_QueryFence(
                    (SDL_GPURenderer *)renderer,
                    windowData->inFlightFences[windowData->frameCounter])) {
                return true;
            }
        }

        VULKAN_ReleaseFence(
            (SDL_GPURenderer *)renderer,
            windowData->inFlightFences[windowData->frameCounter]);

        windowData->inFlightFences[windowData->frameCounter] = NULL;
    }

    // Finally, try to acquire!
    while (true) {
        acquireResult = renderer->vkAcquireNextImageKHR(
            renderer->logicalDevice,
            windowData->swapchain,
            SDL_MAX_UINT64,
            windowData->imageAvailableSemaphore[windowData->frameCounter],
            VK_NULL_HANDLE,
            &swapchainImageIndex);

        if (acquireResult == VK_SUCCESS || acquireResult == VK_SUBOPTIMAL_KHR) {
            break;  // we got the next image!
        }

        // If acquisition is invalid, let's try to recreate
        Uint32 recreateSwapchainResult = VULKAN_INTERNAL_RecreateSwapchain(renderer, windowData);
        if (!recreateSwapchainResult) {
            return false;
        } else if (recreateSwapchainResult == VULKAN_INTERNAL_TRY_AGAIN) {
            // Edge case, texture is filled in with NULL but not an error
            return true;
        }
    }

    swapchainTextureContainer = &windowData->textureContainers[swapchainImageIndex];

    // We need a special execution dependency with pWaitDstStageMask or image transition can start before acquire finishes

    VkImageMemoryBarrier imageBarrier;
    imageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    imageBarrier.pNext = NULL;
    imageBarrier.srcAccessMask = 0;
    imageBarrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    imageBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageBarrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    imageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageBarrier.image = swapchainTextureContainer->activeTexture->image;
    imageBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageBarrier.subresourceRange.baseMipLevel = 0;
    imageBarrier.subresourceRange.levelCount = 1;
    imageBarrier.subresourceRange.baseArrayLayer = 0;
    imageBarrier.subresourceRange.layerCount = 1;

    renderer->vkCmdPipelineBarrier(
        vulkanCommandBuffer->commandBuffer,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        0,
        0,
        NULL,
        0,
        NULL,
        1,
        &imageBarrier);

    // Set up present struct

    if (vulkanCommandBuffer->presentDataCount == vulkanCommandBuffer->presentDataCapacity) {
        vulkanCommandBuffer->presentDataCapacity += 1;
        vulkanCommandBuffer->presentDatas = SDL_realloc(
            vulkanCommandBuffer->presentDatas,
            vulkanCommandBuffer->presentDataCapacity * sizeof(VulkanPresentData));
    }

    presentData = &vulkanCommandBuffer->presentDatas[vulkanCommandBuffer->presentDataCount];
    vulkanCommandBuffer->presentDataCount += 1;

    presentData->windowData = windowData;
    presentData->swapchainImageIndex = swapchainImageIndex;

    // Set up present semaphores

    if (vulkanCommandBuffer->waitSemaphoreCount == vulkanCommandBuffer->waitSemaphoreCapacity) {
        vulkanCommandBuffer->waitSemaphoreCapacity += 1;
        vulkanCommandBuffer->waitSemaphores = SDL_realloc(
            vulkanCommandBuffer->waitSemaphores,
            vulkanCommandBuffer->waitSemaphoreCapacity * sizeof(VkSemaphore));
    }

    vulkanCommandBuffer->waitSemaphores[vulkanCommandBuffer->waitSemaphoreCount] =
        windowData->imageAvailableSemaphore[windowData->frameCounter];
    vulkanCommandBuffer->waitSemaphoreCount += 1;

    if (vulkanCommandBuffer->signalSemaphoreCount == vulkanCommandBuffer->signalSemaphoreCapacity) {
        vulkanCommandBuffer->signalSemaphoreCapacity += 1;
        vulkanCommandBuffer->signalSemaphores = SDL_realloc(
            vulkanCommandBuffer->signalSemaphores,
            vulkanCommandBuffer->signalSemaphoreCapacity * sizeof(VkSemaphore));
    }

    vulkanCommandBuffer->signalSemaphores[vulkanCommandBuffer->signalSemaphoreCount] =
        windowData->renderFinishedSemaphore[windowData->frameCounter];
    vulkanCommandBuffer->signalSemaphoreCount += 1;

    *swapchainTexture = (SDL_GPUTexture *)swapchainTextureContainer;
    return true;
}

static bool VULKAN_AcquireSwapchainTexture(
    SDL_GPUCommandBuffer *command_buffer,
    SDL_Window *window,
    SDL_GPUTexture **swapchain_texture,
    Uint32 *swapchain_texture_width,
    Uint32 *swapchain_texture_height
) {
    return VULKAN_INTERNAL_AcquireSwapchainTexture(
        false,
        command_buffer,
        window,
        swapchain_texture,
        swapchain_texture_width,
        swapchain_texture_height);
}

static bool VULKAN_WaitAndAcquireSwapchainTexture(
    SDL_GPUCommandBuffer *command_buffer,
    SDL_Window *window,
    SDL_GPUTexture **swapchain_texture,
    Uint32 *swapchain_texture_width,
    Uint32 *swapchain_texture_height
) {
    return VULKAN_INTERNAL_AcquireSwapchainTexture(
        true,
        command_buffer,
        window,
        swapchain_texture,
        swapchain_texture_width,
        swapchain_texture_height);
}

static SDL_GPUTextureFormat VULKAN_GetSwapchainTextureFormat(
    SDL_GPURenderer *driverData,
    SDL_Window *window)
{
    VulkanRenderer *renderer = (VulkanRenderer*)driverData;
    WindowData *windowData = VULKAN_INTERNAL_FetchWindowData(window);

    if (windowData == NULL) {
        SET_STRING_ERROR_AND_RETURN("Cannot get swapchain format, window has not been claimed!", SDL_GPU_TEXTUREFORMAT_INVALID);
    }

    return SwapchainCompositionToSDLFormat(
        windowData->swapchainComposition,
        windowData->usingFallbackFormat);
}

static bool VULKAN_SetSwapchainParameters(
    SDL_GPURenderer *driverData,
    SDL_Window *window,
    SDL_GPUSwapchainComposition swapchainComposition,
    SDL_GPUPresentMode presentMode)
{
    VulkanRenderer *renderer = (VulkanRenderer *)driverData;
    WindowData *windowData = VULKAN_INTERNAL_FetchWindowData(window);

    if (windowData == NULL) {
        SET_STRING_ERROR_AND_RETURN("Cannot set swapchain parameters on unclaimed window!", false);
    }

    if (!VULKAN_SupportsSwapchainComposition(driverData, window, swapchainComposition)) {
        SET_STRING_ERROR_AND_RETURN("Swapchain composition not supported!", false);
    }

    if (!VULKAN_SupportsPresentMode(driverData, window, presentMode)) {
        SET_STRING_ERROR_AND_RETURN("Present mode not supported!", false);
    }

    windowData->presentMode = presentMode;
    windowData->swapchainComposition = swapchainComposition;

    Uint32 recreateSwapchainResult = VULKAN_INTERNAL_RecreateSwapchain(renderer, windowData);
    if (!recreateSwapchainResult) {
        return false;
    } else if (recreateSwapchainResult == VULKAN_INTERNAL_TRY_AGAIN) {
        // Edge case, swapchain extent is (0, 0) but this is not an error
        windowData->needsSwapchainRecreate = true;
        return true;
    }

    return true;
}

static bool VULKAN_SetAllowedFramesInFlight(
    SDL_GPURenderer *driverData,
    Uint32 allowedFramesInFlight)
{
    VulkanRenderer *renderer = (VulkanRenderer *)driverData;

    renderer->allowedFramesInFlight = allowedFramesInFlight;

    for (Uint32 i = 0; i < renderer->claimedWindowCount; i += 1) {
        WindowData *windowData = renderer->claimedWindows[i];

        Uint32 recreateResult = VULKAN_INTERNAL_RecreateSwapchain(renderer, windowData);
        if (!recreateResult) {
            return false;
        } else if (recreateResult == VULKAN_INTERNAL_TRY_AGAIN) {
            // Edge case, swapchain extent is (0, 0) but this is not an error
            windowData->needsSwapchainRecreate = true;
        }
    }

    return true;
}

// Submission structure

static VulkanFenceHandle *VULKAN_INTERNAL_AcquireFenceFromPool(
    VulkanRenderer *renderer)
{
    VulkanFenceHandle *handle;
    VkFenceCreateInfo fenceCreateInfo;
    VkFence fence;
    VkResult vulkanResult;

    if (renderer->fencePool.availableFenceCount == 0) {
        // Create fence
        fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceCreateInfo.pNext = NULL;
        fenceCreateInfo.flags = 0;

        vulkanResult = renderer->vkCreateFence(
            renderer->logicalDevice,
            &fenceCreateInfo,
            NULL,
            &fence);

        CHECK_VULKAN_ERROR_AND_RETURN(vulkanResult, vkCreateFence, NULL);

        handle = SDL_malloc(sizeof(VulkanFenceHandle));
        handle->fence = fence;
        SDL_SetAtomicInt(&handle->referenceCount, 0);
        return handle;
    }

    SDL_LockMutex(renderer->fencePool.lock);

    handle = renderer->fencePool.availableFences[renderer->fencePool.availableFenceCount - 1];
    renderer->fencePool.availableFenceCount -= 1;

    vulkanResult = renderer->vkResetFences(
        renderer->logicalDevice,
        1,
        &handle->fence);

    SDL_UnlockMutex(renderer->fencePool.lock);

    CHECK_VULKAN_ERROR_AND_RETURN(vulkanResult, vkResetFences, NULL);

    return handle;
}

static void VULKAN_INTERNAL_PerformPendingDestroys(
    VulkanRenderer *renderer)
{
    SDL_LockMutex(renderer->disposeLock);

    for (Sint32 i = renderer->texturesToDestroyCount - 1; i >= 0; i -= 1) {
        if (SDL_GetAtomicInt(&renderer->texturesToDestroy[i]->referenceCount) == 0) {
            VULKAN_INTERNAL_DestroyTexture(
                renderer,
                renderer->texturesToDestroy[i]);

            renderer->texturesToDestroy[i] = renderer->texturesToDestroy[renderer->texturesToDestroyCount - 1];
            renderer->texturesToDestroyCount -= 1;
        }
    }

    for (Sint32 i = renderer->buffersToDestroyCount - 1; i >= 0; i -= 1) {
        if (SDL_GetAtomicInt(&renderer->buffersToDestroy[i]->referenceCount) == 0) {
            VULKAN_INTERNAL_DestroyBuffer(
                renderer,
                renderer->buffersToDestroy[i]);

            renderer->buffersToDestroy[i] = renderer->buffersToDestroy[renderer->buffersToDestroyCount - 1];
            renderer->buffersToDestroyCount -= 1;
        }
    }

    for (Sint32 i = renderer->graphicsPipelinesToDestroyCount - 1; i >= 0; i -= 1) {
        if (SDL_GetAtomicInt(&renderer->graphicsPipelinesToDestroy[i]->referenceCount) == 0) {
            VULKAN_INTERNAL_DestroyGraphicsPipeline(
                renderer,
                renderer->graphicsPipelinesToDestroy[i]);

            renderer->graphicsPipelinesToDestroy[i] = renderer->graphicsPipelinesToDestroy[renderer->graphicsPipelinesToDestroyCount - 1];
            renderer->graphicsPipelinesToDestroyCount -= 1;
        }
    }

    for (Sint32 i = renderer->computePipelinesToDestroyCount - 1; i >= 0; i -= 1) {
        if (SDL_GetAtomicInt(&renderer->computePipelinesToDestroy[i]->referenceCount) == 0) {
            VULKAN_INTERNAL_DestroyComputePipeline(
                renderer,
                renderer->computePipelinesToDestroy[i]);

            renderer->computePipelinesToDestroy[i] = renderer->computePipelinesToDestroy[renderer->computePipelinesToDestroyCount - 1];
            renderer->computePipelinesToDestroyCount -= 1;
        }
    }

    for (Sint32 i = renderer->shadersToDestroyCount - 1; i >= 0; i -= 1) {
        if (SDL_GetAtomicInt(&renderer->shadersToDestroy[i]->referenceCount) == 0) {
            VULKAN_INTERNAL_DestroyShader(
                renderer,
                renderer->shadersToDestroy[i]);

            renderer->shadersToDestroy[i] = renderer->shadersToDestroy[renderer->shadersToDestroyCount - 1];
            renderer->shadersToDestroyCount -= 1;
        }
    }

    for (Sint32 i = renderer->samplersToDestroyCount - 1; i >= 0; i -= 1) {
        if (SDL_GetAtomicInt(&renderer->samplersToDestroy[i]->referenceCount) == 0) {
            VULKAN_INTERNAL_DestroySampler(
                renderer,
                renderer->samplersToDestroy[i]);

            renderer->samplersToDestroy[i] = renderer->samplersToDestroy[renderer->samplersToDestroyCount - 1];
            renderer->samplersToDestroyCount -= 1;
        }
    }

    for (Sint32 i = renderer->framebuffersToDestroyCount - 1; i >= 0; i -= 1) {
        if (SDL_GetAtomicInt(&renderer->framebuffersToDestroy[i]->referenceCount) == 0) {
            VULKAN_INTERNAL_DestroyFramebuffer(
                renderer,
                renderer->framebuffersToDestroy[i]);

            renderer->framebuffersToDestroy[i] = renderer->framebuffersToDestroy[renderer->framebuffersToDestroyCount - 1];
            renderer->framebuffersToDestroyCount -= 1;
        }
    }

    SDL_UnlockMutex(renderer->disposeLock);
}

static void VULKAN_INTERNAL_CleanCommandBuffer(
    VulkanRenderer *renderer,
    VulkanCommandBuffer *commandBuffer,
    bool cancel)
{
    if (commandBuffer->autoReleaseFence) {
        VULKAN_ReleaseFence(
            (SDL_GPURenderer *)renderer,
            (SDL_GPUFence *)commandBuffer->inFlightFence);

        commandBuffer->inFlightFence = NULL;
    }

    // Uniform buffers are now available

    SDL_LockMutex(renderer->acquireUniformBufferLock);

    for (Sint32 i = 0; i < commandBuffer->usedUniformBufferCount; i += 1) {
        VULKAN_INTERNAL_ReturnUniformBufferToPool(
            renderer,
            commandBuffer->usedUniformBuffers[i]);
    }
    commandBuffer->usedUniformBufferCount = 0;

    SDL_UnlockMutex(renderer->acquireUniformBufferLock);

    // Decrement reference counts

    for (Sint32 i = 0; i < commandBuffer->usedBufferCount; i += 1) {
        (void)SDL_AtomicDecRef(&commandBuffer->usedBuffers[i]->referenceCount);
    }
    commandBuffer->usedBufferCount = 0;

    for (Sint32 i = 0; i < commandBuffer->usedTextureCount; i += 1) {
        (void)SDL_AtomicDecRef(&commandBuffer->usedTextures[i]->referenceCount);
    }
    commandBuffer->usedTextureCount = 0;

    for (Sint32 i = 0; i < commandBuffer->usedSamplerCount; i += 1) {
        (void)SDL_AtomicDecRef(&commandBuffer->usedSamplers[i]->referenceCount);
    }
    commandBuffer->usedSamplerCount = 0;

    for (Sint32 i = 0; i < commandBuffer->usedGraphicsPipelineCount; i += 1) {
        (void)SDL_AtomicDecRef(&commandBuffer->usedGraphicsPipelines[i]->referenceCount);
    }
    commandBuffer->usedGraphicsPipelineCount = 0;

    for (Sint32 i = 0; i < commandBuffer->usedComputePipelineCount; i += 1) {
        (void)SDL_AtomicDecRef(&commandBuffer->usedComputePipelines[i]->referenceCount);
    }
    commandBuffer->usedComputePipelineCount = 0;

    for (Sint32 i = 0; i < commandBuffer->usedFramebufferCount; i += 1) {
        (void)SDL_AtomicDecRef(&commandBuffer->usedFramebuffers[i]->referenceCount);
    }
    commandBuffer->usedFramebufferCount = 0;

    // Reset presentation data

    commandBuffer->presentDataCount = 0;
    commandBuffer->waitSemaphoreCount = 0;
    commandBuffer->signalSemaphoreCount = 0;

    // Reset defrag state

    if (commandBuffer->isDefrag) {
        renderer->defragInProgress = 0;
    }

    // Return command buffer to pool

    SDL_LockMutex(renderer->acquireCommandBufferLock);

    if (commandBuffer->commandPool->inactiveCommandBufferCount == commandBuffer->commandPool->inactiveCommandBufferCapacity) {
        commandBuffer->commandPool->inactiveCommandBufferCapacity += 1;
        commandBuffer->commandPool->inactiveCommandBuffers = SDL_realloc(
            commandBuffer->commandPool->inactiveCommandBuffers,
            commandBuffer->commandPool->inactiveCommandBufferCapacity * sizeof(VulkanCommandBuffer *));
    }

    commandBuffer->commandPool->inactiveCommandBuffers[commandBuffer->commandPool->inactiveCommandBufferCount] = commandBuffer;
    commandBuffer->commandPool->inactiveCommandBufferCount += 1;

    // Release descriptor set cache

    VULKAN_INTERNAL_ReturnDescriptorSetCacheToPool(
        renderer,
        commandBuffer->descriptorSetCache);

    commandBuffer->descriptorSetCache = NULL;

    SDL_UnlockMutex(renderer->acquireCommandBufferLock);

    // Remove this command buffer from the submitted list
    if (!cancel) {
        for (Uint32 i = 0; i < renderer->submittedCommandBufferCount; i += 1) {
            if (renderer->submittedCommandBuffers[i] == commandBuffer) {
                renderer->submittedCommandBuffers[i] = renderer->submittedCommandBuffers[renderer->submittedCommandBufferCount - 1];
                renderer->submittedCommandBufferCount -= 1;
            }
        }
    }
}

static bool VULKAN_WaitForFences(
    SDL_GPURenderer *driverData,
    bool waitAll,
    SDL_GPUFence *const *fences,
    Uint32 numFences)
{
    VulkanRenderer *renderer = (VulkanRenderer *)driverData;
    VkFence *vkFences = SDL_stack_alloc(VkFence, numFences);
    VkResult result;

    for (Uint32 i = 0; i < numFences; i += 1) {
        vkFences[i] = ((VulkanFenceHandle *)fences[i])->fence;
    }

    result = renderer->vkWaitForFences(
        renderer->logicalDevice,
        numFences,
        vkFences,
        waitAll,
        SDL_MAX_UINT64);

    CHECK_VULKAN_ERROR_AND_RETURN(result, vkWaitForFences, false);

    SDL_stack_free(vkFences);

    SDL_LockMutex(renderer->submitLock);

    for (Sint32 i = renderer->submittedCommandBufferCount - 1; i >= 0; i -= 1) {
        result = renderer->vkGetFenceStatus(
            renderer->logicalDevice,
            renderer->submittedCommandBuffers[i]->inFlightFence->fence);

        if (result == VK_SUCCESS) {
            VULKAN_INTERNAL_CleanCommandBuffer(
                renderer,
                renderer->submittedCommandBuffers[i],
                false);
        }
    }

    VULKAN_INTERNAL_PerformPendingDestroys(renderer);

    SDL_UnlockMutex(renderer->submitLock);

    return true;
}

static bool VULKAN_Wait(
    SDL_GPURenderer *driverData)
{
    VulkanRenderer *renderer = (VulkanRenderer *)driverData;
    VulkanCommandBuffer *commandBuffer;
    VkResult result;
    Sint32 i;

    result = renderer->vkDeviceWaitIdle(renderer->logicalDevice);

    CHECK_VULKAN_ERROR_AND_RETURN(result, vkDeviceWaitIdle, false);

    SDL_LockMutex(renderer->submitLock);

    for (i = renderer->submittedCommandBufferCount - 1; i >= 0; i -= 1) {
        commandBuffer = renderer->submittedCommandBuffers[i];
        VULKAN_INTERNAL_CleanCommandBuffer(renderer, commandBuffer, false);
    }

    VULKAN_INTERNAL_PerformPendingDestroys(renderer);

    SDL_UnlockMutex(renderer->submitLock);

    return true;
}

static SDL_GPUFence *VULKAN_SubmitAndAcquireFence(
    SDL_GPUCommandBuffer *commandBuffer)
{
    VulkanCommandBuffer *vulkanCommandBuffer = (VulkanCommandBuffer *)commandBuffer;
    vulkanCommandBuffer->autoReleaseFence = false;
    if (!VULKAN_Submit(commandBuffer)) {
        return NULL;
    }
    return (SDL_GPUFence *)vulkanCommandBuffer->inFlightFence;
}

static void VULKAN_INTERNAL_ReleaseCommandBuffer(VulkanCommandBuffer *vulkanCommandBuffer)
{
    VulkanRenderer *renderer = vulkanCommandBuffer->renderer;

    if (renderer->submittedCommandBufferCount + 1 >= renderer->submittedCommandBufferCapacity) {
        renderer->submittedCommandBufferCapacity = renderer->submittedCommandBufferCount + 1;

        renderer->submittedCommandBuffers = SDL_realloc(
            renderer->submittedCommandBuffers,
            sizeof(VulkanCommandBuffer *) * renderer->submittedCommandBufferCapacity);
    }

    renderer->submittedCommandBuffers[renderer->submittedCommandBufferCount] = vulkanCommandBuffer;
    renderer->submittedCommandBufferCount += 1;
}

static bool VULKAN_Submit(
    SDL_GPUCommandBuffer *commandBuffer)
{
    VulkanCommandBuffer *vulkanCommandBuffer = (VulkanCommandBuffer *)commandBuffer;
    VulkanRenderer *renderer = vulkanCommandBuffer->renderer;
    VkSubmitInfo submitInfo;
    VkPresentInfoKHR presentInfo;
    VulkanPresentData *presentData;
    VkResult vulkanResult, presentResult = VK_SUCCESS;
    VkPipelineStageFlags waitStages[MAX_PRESENT_COUNT];
    Uint32 swapchainImageIndex;
    VulkanTextureSubresource *swapchainTextureSubresource;
    VulkanMemorySubAllocator *allocator;
    bool performCleanups =
        (renderer->claimedWindowCount > 0 && vulkanCommandBuffer->presentDataCount > 0) ||
        renderer->claimedWindowCount == 0;

    SDL_LockMutex(renderer->submitLock);

    // FIXME: Can this just be permanent?
    for (Uint32 i = 0; i < MAX_PRESENT_COUNT; i += 1) {
        waitStages[i] = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    }

    for (Uint32 j = 0; j < vulkanCommandBuffer->presentDataCount; j += 1) {
        swapchainImageIndex = vulkanCommandBuffer->presentDatas[j].swapchainImageIndex;
        swapchainTextureSubresource = VULKAN_INTERNAL_FetchTextureSubresource(
            &vulkanCommandBuffer->presentDatas[j].windowData->textureContainers[swapchainImageIndex],
            0,
            0);

        VULKAN_INTERNAL_TextureSubresourceTransitionFromDefaultUsage(
            renderer,
            vulkanCommandBuffer,
            VULKAN_TEXTURE_USAGE_MODE_PRESENT,
            swapchainTextureSubresource);
    }

    if (performCleanups &&
        renderer->allocationsToDefragCount > 0 &&
        !renderer->defragInProgress) {
        if (!VULKAN_INTERNAL_DefragmentMemory(renderer, vulkanCommandBuffer))
        {
            SDL_LogError(SDL_LOG_CATEGORY_GPU, "%s", "Failed to defragment memory, likely OOM!");
        }
    }

    if (!VULKAN_INTERNAL_EndCommandBuffer(renderer, vulkanCommandBuffer)) {
        SDL_UnlockMutex(renderer->submitLock);
        return false;
    }

    vulkanCommandBuffer->inFlightFence = VULKAN_INTERNAL_AcquireFenceFromPool(renderer);
    if (vulkanCommandBuffer->inFlightFence == NULL) {
        SDL_UnlockMutex(renderer->submitLock);
        return false;
    }

    // Command buffer has a reference to the in-flight fence
    (void)SDL_AtomicIncRef(&vulkanCommandBuffer->inFlightFence->referenceCount);

    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.pNext = NULL;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &vulkanCommandBuffer->commandBuffer;

    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.pWaitSemaphores = vulkanCommandBuffer->waitSemaphores;
    submitInfo.waitSemaphoreCount = vulkanCommandBuffer->waitSemaphoreCount;
    submitInfo.pSignalSemaphores = vulkanCommandBuffer->signalSemaphores;
    submitInfo.signalSemaphoreCount = vulkanCommandBuffer->signalSemaphoreCount;

    vulkanResult = renderer->vkQueueSubmit(
        renderer->unifiedQueue,
        1,
        &submitInfo,
        vulkanCommandBuffer->inFlightFence->fence);

    if (vulkanResult != VK_SUCCESS) {
        SDL_UnlockMutex(renderer->submitLock);
        CHECK_VULKAN_ERROR_AND_RETURN(vulkanResult, vkQueueSubmit, false);
    }

    // Present, if applicable
    for (Uint32 j = 0; j < vulkanCommandBuffer->presentDataCount; j += 1) {
        presentData = &vulkanCommandBuffer->presentDatas[j];

        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.pNext = NULL;
        presentInfo.pWaitSemaphores =
            &presentData->windowData->renderFinishedSemaphore[presentData->windowData->frameCounter];
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pSwapchains = &presentData->windowData->swapchain;
        presentInfo.swapchainCount = 1;
        presentInfo.pImageIndices = &presentData->swapchainImageIndex;
        presentInfo.pResults = NULL;

        presentResult = renderer->vkQueuePresentKHR(
            renderer->unifiedQueue,
            &presentInfo);

        if (presentResult == VK_SUCCESS || presentResult == VK_SUBOPTIMAL_KHR || presentResult == VK_ERROR_OUT_OF_DATE_KHR) {
            // If presenting, the swapchain is using the in-flight fence
            presentData->windowData->inFlightFences[presentData->windowData->frameCounter] = (SDL_GPUFence*)vulkanCommandBuffer->inFlightFence;
            (void)SDL_AtomicIncRef(&vulkanCommandBuffer->inFlightFence->referenceCount);

            if (presentResult == VK_SUBOPTIMAL_KHR || presentResult == VK_ERROR_OUT_OF_DATE_KHR) {
                presentData->windowData->needsSwapchainRecreate = true;
            }
        } else {
            if (presentResult != VK_SUCCESS) {
                VULKAN_INTERNAL_ReleaseCommandBuffer(vulkanCommandBuffer);
                SDL_UnlockMutex(renderer->submitLock);
            }

            CHECK_VULKAN_ERROR_AND_RETURN(presentResult, vkQueuePresentKHR, false);
        }

        presentData->windowData->frameCounter =
            (presentData->windowData->frameCounter + 1) % renderer->allowedFramesInFlight;
    }

    if (performCleanups) {
        for (Sint32 i = renderer->submittedCommandBufferCount - 1; i >= 0; i -= 1) {
            vulkanResult = renderer->vkGetFenceStatus(
                renderer->logicalDevice,
                renderer->submittedCommandBuffers[i]->inFlightFence->fence);

            if (vulkanResult == VK_SUCCESS) {
                VULKAN_INTERNAL_CleanCommandBuffer(
                    renderer,
                    renderer->submittedCommandBuffers[i],
                    false);
            }
        }

        if (renderer->checkEmptyAllocations) {
            SDL_LockMutex(renderer->allocatorLock);

            for (Uint32 i = 0; i < VK_MAX_MEMORY_TYPES; i += 1) {
                allocator = &renderer->memoryAllocator->subAllocators[i];

                for (Sint32 j = allocator->allocationCount - 1; j >= 0; j -= 1) {
                    if (allocator->allocations[j]->usedRegionCount == 0) {
                        VULKAN_INTERNAL_DeallocateMemory(
                            renderer,
                            allocator,
                            j);
                    }
                }
            }

            renderer->checkEmptyAllocations = false;

            SDL_UnlockMutex(renderer->allocatorLock);
        }

        VULKAN_INTERNAL_PerformPendingDestroys(renderer);
    }

    // Mark command buffer as submitted
    VULKAN_INTERNAL_ReleaseCommandBuffer(vulkanCommandBuffer);

    SDL_UnlockMutex(renderer->submitLock);

    return true;
}

static bool VULKAN_Cancel(
    SDL_GPUCommandBuffer *commandBuffer)
{
    VulkanRenderer *renderer;
    VulkanCommandBuffer *vulkanCommandBuffer;
    VkResult result;

    vulkanCommandBuffer = (VulkanCommandBuffer *)commandBuffer;
    renderer = vulkanCommandBuffer->renderer;

    result = renderer->vkResetCommandBuffer(
        vulkanCommandBuffer->commandBuffer,
        VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT);
    CHECK_VULKAN_ERROR_AND_RETURN(result, vkResetCommandBuffer, false);

    vulkanCommandBuffer->autoReleaseFence = false;
    SDL_LockMutex(renderer->submitLock);
    VULKAN_INTERNAL_CleanCommandBuffer(renderer, vulkanCommandBuffer, true);
    SDL_UnlockMutex(renderer->submitLock);

    return true;
}

static bool VULKAN_INTERNAL_DefragmentMemory(
    VulkanRenderer *renderer,
    VulkanCommandBuffer *commandBuffer)
{
    renderer->defragInProgress = 1;
    commandBuffer->isDefrag = 1;

    SDL_LockMutex(renderer->allocatorLock);

    VulkanMemoryAllocation *allocation = renderer->allocationsToDefrag[renderer->allocationsToDefragCount - 1];
    renderer->allocationsToDefragCount -= 1;

    /* For each used region in the allocation
     * create a new resource, copy the data
     * and re-point the resource containers
     */
    for (Uint32 i = 0; i < allocation->usedRegionCount; i += 1) {
        VulkanMemoryUsedRegion *currentRegion = allocation->usedRegions[i];

        if (currentRegion->isBuffer && !currentRegion->vulkanBuffer->markedForDestroy) {
            currentRegion->vulkanBuffer->usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;

            VulkanBuffer *newBuffer = VULKAN_INTERNAL_CreateBuffer(
                renderer,
                currentRegion->vulkanBuffer->size,
                currentRegion->vulkanBuffer->usage,
                currentRegion->vulkanBuffer->type,
                false,
                currentRegion->vulkanBuffer->container != NULL ? currentRegion->vulkanBuffer->container->debugName : NULL);

            if (newBuffer == NULL) {
                SDL_UnlockMutex(renderer->allocatorLock);
                SDL_LogError(SDL_LOG_CATEGORY_GPU, "%s", "Failed to allocate defrag buffer!");
                return false;
            }

            // Copy buffer contents if necessary
            if (
                currentRegion->vulkanBuffer->type == VULKAN_BUFFER_TYPE_GPU && currentRegion->vulkanBuffer->transitioned) {
                VULKAN_INTERNAL_BufferTransitionFromDefaultUsage(
                    renderer,
                    commandBuffer,
                    VULKAN_BUFFER_USAGE_MODE_COPY_SOURCE,
                    currentRegion->vulkanBuffer);

                VULKAN_INTERNAL_BufferTransitionFromDefaultUsage(
                    renderer,
                    commandBuffer,
                    VULKAN_BUFFER_USAGE_MODE_COPY_DESTINATION,
                    newBuffer);

                VkBufferCopy bufferCopy;
                bufferCopy.srcOffset = 0;
                bufferCopy.dstOffset = 0;
                bufferCopy.size = currentRegion->resourceSize;

                renderer->vkCmdCopyBuffer(
                    commandBuffer->commandBuffer,
                    currentRegion->vulkanBuffer->buffer,
                    newBuffer->buffer,
                    1,
                    &bufferCopy);

                VULKAN_INTERNAL_BufferTransitionToDefaultUsage(
                    renderer,
                    commandBuffer,
                    VULKAN_BUFFER_USAGE_MODE_COPY_DESTINATION,
                    newBuffer);

                VULKAN_INTERNAL_TrackBuffer(commandBuffer, currentRegion->vulkanBuffer);
                VULKAN_INTERNAL_TrackBuffer(commandBuffer, newBuffer);
            }

            // re-point original container to new buffer
            newBuffer->container = currentRegion->vulkanBuffer->container;
            newBuffer->containerIndex = currentRegion->vulkanBuffer->containerIndex;
            if (newBuffer->type == VULKAN_BUFFER_TYPE_UNIFORM) {
                currentRegion->vulkanBuffer->uniformBufferForDefrag->buffer = newBuffer;
            } else {
                newBuffer->container->buffers[newBuffer->containerIndex] = newBuffer;
                if (newBuffer->container->activeBuffer == currentRegion->vulkanBuffer) {
                    newBuffer->container->activeBuffer = newBuffer;
                }
            }

            if (currentRegion->vulkanBuffer->uniformBufferForDefrag) {
                newBuffer->uniformBufferForDefrag = currentRegion->vulkanBuffer->uniformBufferForDefrag;
            }

            VULKAN_INTERNAL_ReleaseBuffer(renderer, currentRegion->vulkanBuffer);
        } else if (!currentRegion->isBuffer && !currentRegion->vulkanTexture->markedForDestroy) {
            VulkanTexture *newTexture = VULKAN_INTERNAL_CreateTexture(
                renderer,
                false,
                &currentRegion->vulkanTexture->container->header.info);

            if (newTexture == NULL) {
                SDL_UnlockMutex(renderer->allocatorLock);
                SDL_LogError(SDL_LOG_CATEGORY_GPU, "%s", "Failed to allocate defrag buffer!");
                return false;
            }

            SDL_GPUTextureCreateInfo info = currentRegion->vulkanTexture->container->header.info;
            for (Uint32 subresourceIndex = 0; subresourceIndex < currentRegion->vulkanTexture->subresourceCount; subresourceIndex += 1) {
                // copy subresource if necessary
                VulkanTextureSubresource *srcSubresource = &currentRegion->vulkanTexture->subresources[subresourceIndex];
                VulkanTextureSubresource *dstSubresource = &newTexture->subresources[subresourceIndex];

                VULKAN_INTERNAL_TextureSubresourceTransitionFromDefaultUsage(
                    renderer,
                    commandBuffer,
                    VULKAN_TEXTURE_USAGE_MODE_COPY_SOURCE,
                    srcSubresource);

                VULKAN_INTERNAL_TextureSubresourceMemoryBarrier(
                    renderer,
                    commandBuffer,
                    VULKAN_TEXTURE_USAGE_MODE_UNINITIALIZED,
                    VULKAN_TEXTURE_USAGE_MODE_COPY_DESTINATION,
                    dstSubresource);

                VkImageCopy imageCopy;
                imageCopy.srcOffset.x = 0;
                imageCopy.srcOffset.y = 0;
                imageCopy.srcOffset.z = 0;
                imageCopy.srcSubresource.aspectMask = srcSubresource->parent->aspectFlags;
                imageCopy.srcSubresource.baseArrayLayer = srcSubresource->layer;
                imageCopy.srcSubresource.layerCount = 1;
                imageCopy.srcSubresource.mipLevel = srcSubresource->level;
                imageCopy.extent.width = SDL_max(1, info.width >> srcSubresource->level);
                imageCopy.extent.height = SDL_max(1, info.height >> srcSubresource->level);
                imageCopy.extent.depth = info.type == SDL_GPU_TEXTURETYPE_3D ? info.layer_count_or_depth : 1;
                imageCopy.dstOffset.x = 0;
                imageCopy.dstOffset.y = 0;
                imageCopy.dstOffset.z = 0;
                imageCopy.dstSubresource.aspectMask = dstSubresource->parent->aspectFlags;
                imageCopy.dstSubresource.baseArrayLayer = dstSubresource->layer;
                imageCopy.dstSubresource.layerCount = 1;
                imageCopy.dstSubresource.mipLevel = dstSubresource->level;

                renderer->vkCmdCopyImage(
                    commandBuffer->commandBuffer,
                    currentRegion->vulkanTexture->image,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    newTexture->image,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    1,
                    &imageCopy);

                VULKAN_INTERNAL_TextureSubresourceTransitionToDefaultUsage(
                    renderer,
                    commandBuffer,
                    VULKAN_TEXTURE_USAGE_MODE_COPY_DESTINATION,
                    dstSubresource);

                VULKAN_INTERNAL_TrackTexture(commandBuffer, srcSubresource->parent);
                VULKAN_INTERNAL_TrackTexture(commandBuffer, dstSubresource->parent);
            }

            // re-point original container to new texture
            newTexture->container = currentRegion->vulkanTexture->container;
            newTexture->containerIndex = currentRegion->vulkanTexture->containerIndex;
            newTexture->container->textures[currentRegion->vulkanTexture->containerIndex] = newTexture;
            if (currentRegion->vulkanTexture == currentRegion->vulkanTexture->container->activeTexture) {
                newTexture->container->activeTexture = newTexture;
            }

            VULKAN_INTERNAL_ReleaseTexture(renderer, currentRegion->vulkanTexture);
        }
    }

    SDL_UnlockMutex(renderer->allocatorLock);

    return true;
}

// Format Info

static bool VULKAN_SupportsTextureFormat(
    SDL_GPURenderer *driverData,
    SDL_GPUTextureFormat format,
    SDL_GPUTextureType type,
    SDL_GPUTextureUsageFlags usage)
{
    VulkanRenderer *renderer = (VulkanRenderer *)driverData;
    VkFormat vulkanFormat = SDLToVK_TextureFormat[format];
    VkImageUsageFlags vulkanUsage = 0;
    VkImageCreateFlags createFlags = 0;
    VkImageFormatProperties properties;
    VkResult vulkanResult;

    if (usage & SDL_GPU_TEXTUREUSAGE_SAMPLER) {
        vulkanUsage |= VK_IMAGE_USAGE_SAMPLED_BIT;
    }
    if (usage & SDL_GPU_TEXTUREUSAGE_COLOR_TARGET) {
        vulkanUsage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    }
    if (usage & SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET) {
        vulkanUsage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    }
    if (usage & (SDL_GPU_TEXTUREUSAGE_GRAPHICS_STORAGE_READ |
                 SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_READ |
                 SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE |
                 SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_SIMULTANEOUS_READ_WRITE)) {
        vulkanUsage |= VK_IMAGE_USAGE_STORAGE_BIT;
    }

    if (type == SDL_GPU_TEXTURETYPE_CUBE || type == SDL_GPU_TEXTURETYPE_CUBE_ARRAY) {
        createFlags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    }

    vulkanResult = renderer->vkGetPhysicalDeviceImageFormatProperties(
        renderer->physicalDevice,
        vulkanFormat,
        (type == SDL_GPU_TEXTURETYPE_3D) ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D,
        VK_IMAGE_TILING_OPTIMAL,
        vulkanUsage,
        createFlags,
        &properties);

    return vulkanResult == VK_SUCCESS;
}

// Device instantiation

static inline Uint8 CheckDeviceExtensions(
    VkExtensionProperties *extensions,
    Uint32 numExtensions,
    VulkanExtensions *supports)
{
    Uint32 i;

    SDL_memset(supports, '\0', sizeof(VulkanExtensions));
    for (i = 0; i < numExtensions; i += 1) {
        const char *name = extensions[i].extensionName;
#define CHECK(ext)                           \
    if (SDL_strcmp(name, "VK_" #ext) == 0) { \
        supports->ext = 1;                   \
    }
        CHECK(KHR_swapchain)
        else CHECK(KHR_maintenance1) else CHECK(KHR_driver_properties) else CHECK(KHR_portability_subset) else CHECK(EXT_texture_compression_astc_hdr)
#undef CHECK
    }

    return (supports->KHR_swapchain &&
            supports->KHR_maintenance1);
}

static inline Uint32 GetDeviceExtensionCount(VulkanExtensions *supports)
{
    return (
        supports->KHR_swapchain +
        supports->KHR_maintenance1 +
        supports->KHR_driver_properties +
        supports->KHR_portability_subset +
        supports->EXT_texture_compression_astc_hdr);
}

static inline void CreateDeviceExtensionArray(
    VulkanExtensions *supports,
    const char **extensions)
{
    Uint8 cur = 0;
#define CHECK(ext)                      \
    if (supports->ext) {                \
        extensions[cur++] = "VK_" #ext; \
    }
    CHECK(KHR_swapchain)
    CHECK(KHR_maintenance1)
    CHECK(KHR_driver_properties)
    CHECK(KHR_portability_subset)
    CHECK(EXT_texture_compression_astc_hdr)
#undef CHECK
}

static inline Uint8 SupportsInstanceExtension(
    const char *ext,
    VkExtensionProperties *availableExtensions,
    Uint32 numAvailableExtensions)
{
    Uint32 i;
    for (i = 0; i < numAvailableExtensions; i += 1) {
        if (SDL_strcmp(ext, availableExtensions[i].extensionName) == 0) {
            return 1;
        }
    }
    return 0;
}

static Uint8 VULKAN_INTERNAL_CheckInstanceExtensions(
    const char **requiredExtensions,
    Uint32 requiredExtensionsLength,
    bool *supportsDebugUtils,
    bool *supportsColorspace)
{
    Uint32 extensionCount, i;
    VkExtensionProperties *availableExtensions;
    Uint8 allExtensionsSupported = 1;

    vkEnumerateInstanceExtensionProperties(
        NULL,
        &extensionCount,
        NULL);
    availableExtensions = SDL_malloc(
        extensionCount * sizeof(VkExtensionProperties));
    vkEnumerateInstanceExtensionProperties(
        NULL,
        &extensionCount,
        availableExtensions);

    for (i = 0; i < requiredExtensionsLength; i += 1) {
        if (!SupportsInstanceExtension(
                requiredExtensions[i],
                availableExtensions,
                extensionCount)) {
            allExtensionsSupported = 0;
            break;
        }
    }

    // This is optional, but nice to have!
    *supportsDebugUtils = SupportsInstanceExtension(
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
        availableExtensions,
        extensionCount);

    // Also optional and nice to have!
    *supportsColorspace = SupportsInstanceExtension(
        VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME,
        availableExtensions,
        extensionCount);

    SDL_free(availableExtensions);
    return allExtensionsSupported;
}

static Uint8 VULKAN_INTERNAL_CheckDeviceExtensions(
    VulkanRenderer *renderer,
    VkPhysicalDevice physicalDevice,
    VulkanExtensions *physicalDeviceExtensions)
{
    Uint32 extensionCount;
    VkExtensionProperties *availableExtensions;
    Uint8 allExtensionsSupported;

    renderer->vkEnumerateDeviceExtensionProperties(
        physicalDevice,
        NULL,
        &extensionCount,
        NULL);
    availableExtensions = (VkExtensionProperties *)SDL_malloc(
        extensionCount * sizeof(VkExtensionProperties));
    renderer->vkEnumerateDeviceExtensionProperties(
        physicalDevice,
        NULL,
        &extensionCount,
        availableExtensions);

    allExtensionsSupported = CheckDeviceExtensions(
        availableExtensions,
        extensionCount,
        physicalDeviceExtensions);

    SDL_free(availableExtensions);
    return allExtensionsSupported;
}

static Uint8 VULKAN_INTERNAL_CheckValidationLayers(
    const char **validationLayers,
    Uint32 validationLayersLength)
{
    Uint32 layerCount;
    VkLayerProperties *availableLayers;
    Uint32 i, j;
    Uint8 layerFound = 0;

    vkEnumerateInstanceLayerProperties(&layerCount, NULL);
    availableLayers = (VkLayerProperties *)SDL_malloc(
        layerCount * sizeof(VkLayerProperties));
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers);

    for (i = 0; i < validationLayersLength; i += 1) {
        layerFound = 0;

        for (j = 0; j < layerCount; j += 1) {
            if (SDL_strcmp(validationLayers[i], availableLayers[j].layerName) == 0) {
                layerFound = 1;
                break;
            }
        }

        if (!layerFound) {
            break;
        }
    }

    SDL_free(availableLayers);
    return layerFound;
}

static Uint8 VULKAN_INTERNAL_CreateInstance(VulkanRenderer *renderer)
{
    VkResult vulkanResult;
    VkApplicationInfo appInfo;
    VkInstanceCreateFlags createFlags;
    const char *const *originalInstanceExtensionNames;
    const char **instanceExtensionNames;
    Uint32 instanceExtensionCount;
    VkInstanceCreateInfo createInfo;
    static const char *layerNames[] = { "VK_LAYER_KHRONOS_validation" };

    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pNext = NULL;
    appInfo.pApplicationName = NULL;
    appInfo.applicationVersion = 0;
    appInfo.pEngineName = "SDLGPU";
    appInfo.engineVersion = SDL_VERSION;
    appInfo.apiVersion = VK_MAKE_VERSION(1, 0, 0);

    createFlags = 0;

    originalInstanceExtensionNames = SDL_Vulkan_GetInstanceExtensions(&instanceExtensionCount);
    if (!originalInstanceExtensionNames) {
        SDL_LogError(
            SDL_LOG_CATEGORY_GPU,
            "SDL_Vulkan_GetInstanceExtensions(): getExtensionCount: %s",
            SDL_GetError());

        return 0;
    }

    /* Extra space for the following extensions:
     * VK_KHR_get_physical_device_properties2
     * VK_EXT_swapchain_colorspace
     * VK_EXT_debug_utils
     * VK_KHR_portability_enumeration
     */
    instanceExtensionNames = SDL_stack_alloc(
        const char *,
        instanceExtensionCount + 4);
    SDL_memcpy((void *)instanceExtensionNames, originalInstanceExtensionNames, instanceExtensionCount * sizeof(const char *));

    // Core since 1.1
    instanceExtensionNames[instanceExtensionCount++] =
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME;

#ifdef SDL_PLATFORM_APPLE
    instanceExtensionNames[instanceExtensionCount++] =
        VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME;
    createFlags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif

    if (!VULKAN_INTERNAL_CheckInstanceExtensions(
            instanceExtensionNames,
            instanceExtensionCount,
            &renderer->supportsDebugUtils,
            &renderer->supportsColorspace)) {
        SDL_stack_free((char *)instanceExtensionNames);
        SET_STRING_ERROR_AND_RETURN("Required Vulkan instance extensions not supported", false);
    }

    if (renderer->supportsDebugUtils) {
        // Append the debug extension
        instanceExtensionNames[instanceExtensionCount++] =
            VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
    } else {
        SDL_LogWarn(
            SDL_LOG_CATEGORY_GPU,
            "%s is not supported!",
            VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    if (renderer->supportsColorspace) {
        // Append colorspace extension
        instanceExtensionNames[instanceExtensionCount++] =
            VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME;
    }

    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pNext = NULL;
    createInfo.flags = createFlags;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.ppEnabledLayerNames = layerNames;
    createInfo.enabledExtensionCount = instanceExtensionCount;
    createInfo.ppEnabledExtensionNames = instanceExtensionNames;
    if (renderer->debugMode) {
        createInfo.enabledLayerCount = SDL_arraysize(layerNames);
        if (!VULKAN_INTERNAL_CheckValidationLayers(
                layerNames,
                createInfo.enabledLayerCount)) {
            SDL_LogWarn(SDL_LOG_CATEGORY_GPU, "Validation layers not found, continuing without validation");
            createInfo.enabledLayerCount = 0;
        } else {
            SDL_LogInfo(SDL_LOG_CATEGORY_GPU, "Validation layers enabled, expect debug level performance!");
        }
    } else {
        createInfo.enabledLayerCount = 0;
    }

    vulkanResult = vkCreateInstance(&createInfo, NULL, &renderer->instance);
    SDL_stack_free((char *)instanceExtensionNames);

    if (vulkanResult != VK_SUCCESS) {
        CHECK_VULKAN_ERROR_AND_RETURN(vulkanResult, vkCreateInstance, 0);
    }

    return 1;
}

static Uint8 VULKAN_INTERNAL_IsDeviceSuitable(
    VulkanRenderer *renderer,
    VkPhysicalDevice physicalDevice,
    VulkanExtensions *physicalDeviceExtensions,
    Uint32 *queueFamilyIndex,
    Uint8 *deviceRank)
{
    Uint32 queueFamilyCount, queueFamilyRank, queueFamilyBest;
    VkQueueFamilyProperties *queueProps;
    bool supportsPresent;
    VkPhysicalDeviceProperties deviceProperties;
    VkPhysicalDeviceFeatures deviceFeatures;
    Uint32 i;

    const Uint8 *devicePriority = renderer->preferLowPower ? DEVICE_PRIORITY_LOWPOWER : DEVICE_PRIORITY_HIGHPERFORMANCE;

    /* Get the device rank before doing any checks, in case one fails.
     * Note: If no dedicated device exists, one that supports our features
     * would be fine
     */
    renderer->vkGetPhysicalDeviceProperties(
        physicalDevice,
        &deviceProperties);
    if (*deviceRank < devicePriority[deviceProperties.deviceType]) {
        /* This device outranks the best device we've found so far!
         * This includes a dedicated GPU that has less features than an
         * integrated GPU, because this is a freak case that is almost
         * never intentionally desired by the end user
         */
        *deviceRank = devicePriority[deviceProperties.deviceType];
    } else if (*deviceRank > devicePriority[deviceProperties.deviceType]) {
        /* Device is outranked by a previous device, don't even try to
         * run a query and reset the rank to avoid overwrites
         */
        *deviceRank = 0;
        return 0;
    }

    renderer->vkGetPhysicalDeviceFeatures(
        physicalDevice,
        &deviceFeatures);
    if (!deviceFeatures.independentBlend ||
        !deviceFeatures.imageCubeArray ||
        !deviceFeatures.depthClamp ||
        !deviceFeatures.shaderClipDistance ||
        !deviceFeatures.drawIndirectFirstInstance ||
        !deviceFeatures.sampleRateShading) {
        return 0;
    }

    if (!VULKAN_INTERNAL_CheckDeviceExtensions(
            renderer,
            physicalDevice,
            physicalDeviceExtensions)) {
        return 0;
    }

    renderer->vkGetPhysicalDeviceQueueFamilyProperties(
        physicalDevice,
        &queueFamilyCount,
        NULL);

    queueProps = SDL_stack_alloc(
        VkQueueFamilyProperties,
        queueFamilyCount);
    renderer->vkGetPhysicalDeviceQueueFamilyProperties(
        physicalDevice,
        &queueFamilyCount,
        queueProps);

    queueFamilyBest = 0;
    *queueFamilyIndex = SDL_MAX_UINT32;
    for (i = 0; i < queueFamilyCount; i += 1) {
        supportsPresent = SDL_Vulkan_GetPresentationSupport(
            renderer->instance,
            physicalDevice,
            i);
        if (!supportsPresent ||
            !(queueProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
            // Not a graphics family, ignore.
            continue;
        }

        /* The queue family bitflags are kind of annoying.
         *
         * We of course need a graphics family, but we ideally want the
         * _primary_ graphics family. The spec states that at least one
         * graphics family must also be a compute family, so generally
         * drivers make that the first one. But hey, maybe something
         * genuinely can't do compute or something, and FNA doesn't
         * need it, so we'll be open to a non-compute queue family.
         *
         * Additionally, it's common to see the primary queue family
         * have the transfer bit set, which is great! But this is
         * actually optional; it's impossible to NOT have transfers in
         * graphics/compute but it _is_ possible for a graphics/compute
         * family, even the primary one, to just decide not to set the
         * bitflag. Admittedly, a driver may want to isolate transfer
         * queues to a dedicated family so that queues made solely for
         * transfers can have an optimized DMA queue.
         *
         * That, or the driver author got lazy and decided not to set
         * the bit. Looking at you, Android.
         *
         * -flibit
         */
        if (queueProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            if (queueProps[i].queueFlags & VK_QUEUE_TRANSFER_BIT) {
                // Has all attribs!
                queueFamilyRank = 3;
            } else {
                // Probably has a DMA transfer queue family
                queueFamilyRank = 2;
            }
        } else {
            // Just a graphics family, probably has something better
            queueFamilyRank = 1;
        }
        if (queueFamilyRank > queueFamilyBest) {
            *queueFamilyIndex = i;
            queueFamilyBest = queueFamilyRank;
        }
    }

    SDL_stack_free(queueProps);

    if (*queueFamilyIndex == SDL_MAX_UINT32) {
        // Somehow no graphics queues existed. Compute-only device?
        return 0;
    }

    // FIXME: Need better structure for checking vs storing swapchain support details
    return 1;
}

static Uint8 VULKAN_INTERNAL_DeterminePhysicalDevice(VulkanRenderer *renderer)
{
    VkResult vulkanResult;
    VkPhysicalDevice *physicalDevices;
    VulkanExtensions *physicalDeviceExtensions;
    Uint32 i, physicalDeviceCount;
    Sint32 suitableIndex;
    Uint32 queueFamilyIndex, suitableQueueFamilyIndex;
    Uint8 deviceRank, highestRank;

    vulkanResult = renderer->vkEnumeratePhysicalDevices(
        renderer->instance,
        &physicalDeviceCount,
        NULL);
    CHECK_VULKAN_ERROR_AND_RETURN(vulkanResult, vkEnumeratePhysicalDevices, 0);

    if (physicalDeviceCount == 0) {
        SDL_LogInfo(SDL_LOG_CATEGORY_GPU, "Failed to find any GPUs with Vulkan support");
        return 0;
    }

    physicalDevices = SDL_stack_alloc(VkPhysicalDevice, physicalDeviceCount);
    physicalDeviceExtensions = SDL_stack_alloc(VulkanExtensions, physicalDeviceCount);

    vulkanResult = renderer->vkEnumeratePhysicalDevices(
        renderer->instance,
        &physicalDeviceCount,
        physicalDevices);

    /* This should be impossible to hit, but from what I can tell this can
     * be triggered not because the array is too small, but because there
     * were drivers that turned out to be bogus, so this is the loader's way
     * of telling us that the list is now smaller than expected :shrug:
     */
    if (vulkanResult == VK_INCOMPLETE) {
        SDL_LogWarn(SDL_LOG_CATEGORY_GPU, "vkEnumeratePhysicalDevices returned VK_INCOMPLETE, will keep trying anyway...");
        vulkanResult = VK_SUCCESS;
    }

    if (vulkanResult != VK_SUCCESS) {
        SDL_LogWarn(
            SDL_LOG_CATEGORY_GPU,
            "vkEnumeratePhysicalDevices failed: %s",
            VkErrorMessages(vulkanResult));
        SDL_stack_free(physicalDevices);
        SDL_stack_free(physicalDeviceExtensions);
        return 0;
    }

    // Any suitable device will do, but we'd like the best
    suitableIndex = -1;
    suitableQueueFamilyIndex = 0;
    highestRank = 0;
    for (i = 0; i < physicalDeviceCount; i += 1) {
        deviceRank = highestRank;
        if (VULKAN_INTERNAL_IsDeviceSuitable(
                renderer,
                physicalDevices[i],
                &physicalDeviceExtensions[i],
                &queueFamilyIndex,
                &deviceRank)) {
            /* Use this for rendering.
             * Note that this may override a previous device that
             * supports rendering, but shares the same device rank.
             */
            suitableIndex = i;
            suitableQueueFamilyIndex = queueFamilyIndex;
            highestRank = deviceRank;
        } else if (deviceRank > highestRank) {
            /* In this case, we found a... "realer?" GPU,
             * but it doesn't actually support our Vulkan.
             * We should disqualify all devices below as a
             * result, because if we don't we end up
             * ignoring real hardware and risk using
             * something like LLVMpipe instead!
             * -flibit
             */
            suitableIndex = -1;
            highestRank = deviceRank;
        }
    }

    if (suitableIndex != -1) {
        renderer->supports = physicalDeviceExtensions[suitableIndex];
        renderer->physicalDevice = physicalDevices[suitableIndex];
        renderer->queueFamilyIndex = suitableQueueFamilyIndex;
    } else {
        SDL_stack_free(physicalDevices);
        SDL_stack_free(physicalDeviceExtensions);
        return 0;
    }

    renderer->physicalDeviceProperties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    if (renderer->supports.KHR_driver_properties) {
        renderer->physicalDeviceDriverProperties.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES_KHR;
        renderer->physicalDeviceDriverProperties.pNext = NULL;

        renderer->physicalDeviceProperties.pNext =
            &renderer->physicalDeviceDriverProperties;

        renderer->vkGetPhysicalDeviceProperties2KHR(
            renderer->physicalDevice,
            &renderer->physicalDeviceProperties);
    } else {
        renderer->physicalDeviceProperties.pNext = NULL;

        renderer->vkGetPhysicalDeviceProperties(
            renderer->physicalDevice,
            &renderer->physicalDeviceProperties.properties);
    }

    renderer->vkGetPhysicalDeviceMemoryProperties(
        renderer->physicalDevice,
        &renderer->memoryProperties);

    SDL_stack_free(physicalDevices);
    SDL_stack_free(physicalDeviceExtensions);
    return 1;
}

static Uint8 VULKAN_INTERNAL_CreateLogicalDevice(
    VulkanRenderer *renderer)
{
    VkResult vulkanResult;
    VkDeviceCreateInfo deviceCreateInfo;
    VkPhysicalDeviceFeatures desiredDeviceFeatures;
    VkPhysicalDeviceFeatures haveDeviceFeatures;
    VkPhysicalDevicePortabilitySubsetFeaturesKHR portabilityFeatures;
    const char **deviceExtensions;

    VkDeviceQueueCreateInfo queueCreateInfo;
    float queuePriority = 1.0f;

    queueCreateInfo.sType =
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.pNext = NULL;
    queueCreateInfo.flags = 0;
    queueCreateInfo.queueFamilyIndex = renderer->queueFamilyIndex;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    // check feature support

    renderer->vkGetPhysicalDeviceFeatures(
        renderer->physicalDevice,
        &haveDeviceFeatures);

    // specifying used device features

    SDL_zero(desiredDeviceFeatures);
    desiredDeviceFeatures.independentBlend = VK_TRUE;
    desiredDeviceFeatures.samplerAnisotropy = VK_TRUE;
    desiredDeviceFeatures.imageCubeArray = VK_TRUE;
    desiredDeviceFeatures.depthClamp = VK_TRUE;
    desiredDeviceFeatures.shaderClipDistance = VK_TRUE;
    desiredDeviceFeatures.drawIndirectFirstInstance = VK_TRUE;
    desiredDeviceFeatures.sampleRateShading = VK_TRUE;

    if (haveDeviceFeatures.fillModeNonSolid) {
        desiredDeviceFeatures.fillModeNonSolid = VK_TRUE;
        renderer->supportsFillModeNonSolid = true;
    }

    if (haveDeviceFeatures.multiDrawIndirect) {
        desiredDeviceFeatures.multiDrawIndirect = VK_TRUE;
        renderer->supportsMultiDrawIndirect = true;
    }

    // creating the logical device

    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    if (renderer->supports.KHR_portability_subset) {
        portabilityFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PORTABILITY_SUBSET_FEATURES_KHR;
        portabilityFeatures.pNext = NULL;
        portabilityFeatures.constantAlphaColorBlendFactors = VK_FALSE;
        portabilityFeatures.events = VK_FALSE;
        portabilityFeatures.imageViewFormatReinterpretation = VK_FALSE;
        portabilityFeatures.imageViewFormatSwizzle = VK_TRUE;
        portabilityFeatures.imageView2DOn3DImage = VK_FALSE;
        portabilityFeatures.multisampleArrayImage = VK_FALSE;
        portabilityFeatures.mutableComparisonSamplers = VK_FALSE;
        portabilityFeatures.pointPolygons = VK_FALSE;
        portabilityFeatures.samplerMipLodBias = VK_FALSE; // Technically should be true, but eh
        portabilityFeatures.separateStencilMaskRef = VK_FALSE;
        portabilityFeatures.shaderSampleRateInterpolationFunctions = VK_FALSE;
        portabilityFeatures.tessellationIsolines = VK_FALSE;
        portabilityFeatures.tessellationPointMode = VK_FALSE;
        portabilityFeatures.triangleFans = VK_FALSE;
        portabilityFeatures.vertexAttributeAccessBeyondStride = VK_FALSE;
        deviceCreateInfo.pNext = &portabilityFeatures;
    } else {
        deviceCreateInfo.pNext = NULL;
    }
    deviceCreateInfo.flags = 0;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    deviceCreateInfo.enabledLayerCount = 0;
    deviceCreateInfo.ppEnabledLayerNames = NULL;
    deviceCreateInfo.enabledExtensionCount = GetDeviceExtensionCount(
        &renderer->supports);
    deviceExtensions = SDL_stack_alloc(
        const char *,
        deviceCreateInfo.enabledExtensionCount);
    CreateDeviceExtensionArray(&renderer->supports, deviceExtensions);
    deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions;
    deviceCreateInfo.pEnabledFeatures = &desiredDeviceFeatures;

    vulkanResult = renderer->vkCreateDevice(
        renderer->physicalDevice,
        &deviceCreateInfo,
        NULL,
        &renderer->logicalDevice);
    SDL_stack_free((void *)deviceExtensions);
    CHECK_VULKAN_ERROR_AND_RETURN(vulkanResult, vkCreateDevice, 0);

    // Load vkDevice entry points

#define VULKAN_DEVICE_FUNCTION(func)                    \
    renderer->func = (PFN_##func)                       \
                         renderer->vkGetDeviceProcAddr( \
                             renderer->logicalDevice,   \
                             #func);
#include "SDL_gpu_vulkan_vkfuncs.h"

    renderer->vkGetDeviceQueue(
        renderer->logicalDevice,
        renderer->queueFamilyIndex,
        0,
        &renderer->unifiedQueue);

    return 1;
}

static void VULKAN_INTERNAL_LoadEntryPoints(void)
{
    // Required for MoltenVK support
    SDL_setenv_unsafe("MVK_CONFIG_FULL_IMAGE_VIEW_SWIZZLE", "1", 1);

    // Load Vulkan entry points
    if (!SDL_Vulkan_LoadLibrary(NULL)) {
        SDL_LogWarn(SDL_LOG_CATEGORY_GPU, "Vulkan: SDL_Vulkan_LoadLibrary failed!");
        return;
    }

#ifdef HAVE_GCC_DIAGNOSTIC_PRAGMA
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
    vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)SDL_Vulkan_GetVkGetInstanceProcAddr();
#ifdef HAVE_GCC_DIAGNOSTIC_PRAGMA
#pragma GCC diagnostic pop
#endif
    if (vkGetInstanceProcAddr == NULL) {
        SDL_LogWarn(
            SDL_LOG_CATEGORY_GPU,
            "SDL_Vulkan_GetVkGetInstanceProcAddr(): %s",
            SDL_GetError());
        return;
    }

#define VULKAN_GLOBAL_FUNCTION(name)                                                                      \
    name = (PFN_##name)vkGetInstanceProcAddr(VK_NULL_HANDLE, #name);                                      \
    if (name == NULL) {                                                                                   \
        SDL_LogWarn(SDL_LOG_CATEGORY_GPU, "vkGetInstanceProcAddr(VK_NULL_HANDLE, \"" #name "\") failed"); \
        return;                                                                                           \
    }
#include "SDL_gpu_vulkan_vkfuncs.h"
}

static bool VULKAN_INTERNAL_PrepareVulkan(
    VulkanRenderer *renderer)
{
    VULKAN_INTERNAL_LoadEntryPoints();

    if (!VULKAN_INTERNAL_CreateInstance(renderer)) {
        SDL_LogWarn(SDL_LOG_CATEGORY_GPU, "Vulkan: Could not create Vulkan instance");
        return false;
    }

#define VULKAN_INSTANCE_FUNCTION(func) \
    renderer->func = (PFN_##func)vkGetInstanceProcAddr(renderer->instance, #func);
#include "SDL_gpu_vulkan_vkfuncs.h"

    if (!VULKAN_INTERNAL_DeterminePhysicalDevice(renderer)) {
        SDL_LogWarn(SDL_LOG_CATEGORY_GPU, "Vulkan: Failed to determine a suitable physical device");
        return false;
    }
    return true;
}

static bool VULKAN_PrepareDriver(SDL_VideoDevice *_this)
{
    // Set up dummy VulkanRenderer
    VulkanRenderer *renderer;
    Uint8 result;

    if (_this->Vulkan_CreateSurface == NULL) {
        return false;
    }

    if (!SDL_Vulkan_LoadLibrary(NULL)) {
        return false;
    }

    renderer = (VulkanRenderer *)SDL_malloc(sizeof(VulkanRenderer));
    SDL_memset(renderer, '\0', sizeof(VulkanRenderer));

    result = VULKAN_INTERNAL_PrepareVulkan(renderer);

    if (result) {
        renderer->vkDestroyInstance(renderer->instance, NULL);
    }
    SDL_free(renderer);
    SDL_Vulkan_UnloadLibrary();
    return result;
}

static SDL_GPUDevice *VULKAN_CreateDevice(bool debugMode, bool preferLowPower, SDL_PropertiesID props)
{
    VulkanRenderer *renderer;

    SDL_GPUDevice *result;
    Uint32 i;

    bool verboseLogs = SDL_GetBooleanProperty(
        props,
        SDL_PROP_GPU_DEVICE_CREATE_VERBOSE_BOOLEAN,
        true);

    if (!SDL_Vulkan_LoadLibrary(NULL)) {
        SDL_assert(!"This should have failed in PrepareDevice first!");
        return NULL;
    }

    renderer = (VulkanRenderer *)SDL_malloc(sizeof(VulkanRenderer));
    SDL_memset(renderer, '\0', sizeof(VulkanRenderer));
    renderer->debugMode = debugMode;
    renderer->preferLowPower = preferLowPower;
    renderer->allowedFramesInFlight = 2;

    if (!VULKAN_INTERNAL_PrepareVulkan(renderer)) {
        SDL_free(renderer);
        SDL_Vulkan_UnloadLibrary();
        SET_STRING_ERROR_AND_RETURN("Failed to initialize Vulkan!", NULL);
    }

    renderer->props = SDL_CreateProperties();
    if (verboseLogs) {
        SDL_LogInfo(SDL_LOG_CATEGORY_GPU, "SDL_GPU Driver: Vulkan");
    }

    // Record device name
    const char *deviceName = renderer->physicalDeviceProperties.properties.deviceName;
    SDL_SetStringProperty(
        renderer->props,
        SDL_PROP_GPU_DEVICE_NAME_STRING,
        deviceName);
    if (verboseLogs) {
        SDL_LogInfo(SDL_LOG_CATEGORY_GPU, "Vulkan Device: %s", deviceName);
    }

    // Record driver version. This is provided as a backup if
    // VK_KHR_driver_properties is not available but as most drivers support it
    // this property should be rarely used.
    //
    // This uses a vendor-specific encoding and it isn't well documented. The
    // vendor ID is the registered PCI ID of the vendor and can be found in
    // online databases.
    char driverVer[64];
    Uint32 rawDriverVer = renderer->physicalDeviceProperties.properties.driverVersion;
    Uint32 vendorId = renderer->physicalDeviceProperties.properties.vendorID;
    if (vendorId == 0x10de) {
        // Nvidia uses 10|8|8|6 encoding.
        (void)SDL_snprintf(
            driverVer,
            SDL_arraysize(driverVer),
            "%d.%d.%d.%d",
            (rawDriverVer >> 22) & 0x3ff,
            (rawDriverVer >> 14) & 0xff,
            (rawDriverVer >> 6) & 0xff,
            rawDriverVer & 0x3f);
    }
#ifdef SDL_PLATFORM_WINDOWS
    else if (vendorId == 0x8086) {
        // Intel uses 18|14 encoding on Windows only.
        (void)SDL_snprintf(
            driverVer,
            SDL_arraysize(driverVer),
            "%d.%d",
            (rawDriverVer >> 14) & 0x3ffff,
            rawDriverVer & 0x3fff);
    }
#endif
    else {
        // Assume standard Vulkan 10|10|12 encoding for everything else. AMD and
        // Mesa are known to use this encoding.
        (void)SDL_snprintf(
            driverVer,
            SDL_arraysize(driverVer),
            "%d.%d.%d",
            (rawDriverVer >> 22) & 0x3ff,
            (rawDriverVer >> 12) & 0x3ff,
            rawDriverVer & 0xfff);
    }
    SDL_SetStringProperty(
        renderer->props,
        SDL_PROP_GPU_DEVICE_DRIVER_VERSION_STRING,
        driverVer);
    // Log this only if VK_KHR_driver_properties is not available.

    if (renderer->supports.KHR_driver_properties) {
        // Record driver name and version
        const char *driverName = renderer->physicalDeviceDriverProperties.driverName;
        const char *driverInfo = renderer->physicalDeviceDriverProperties.driverInfo;
        SDL_SetStringProperty(
            renderer->props,
            SDL_PROP_GPU_DEVICE_DRIVER_NAME_STRING,
            driverName);
        SDL_SetStringProperty(
            renderer->props,
            SDL_PROP_GPU_DEVICE_DRIVER_INFO_STRING,
            driverInfo);
        if (verboseLogs) {
            // FIXME: driverInfo can be a multiline string.
            SDL_LogInfo(SDL_LOG_CATEGORY_GPU, "Vulkan Driver: %s %s", driverName, driverInfo);
        }

        // Record conformance level
        if (verboseLogs) {
            char conformance[64];
            (void)SDL_snprintf(
                conformance,
                SDL_arraysize(conformance),
                "%u.%u.%u.%u",
                renderer->physicalDeviceDriverProperties.conformanceVersion.major,
                renderer->physicalDeviceDriverProperties.conformanceVersion.minor,
                renderer->physicalDeviceDriverProperties.conformanceVersion.subminor,
                renderer->physicalDeviceDriverProperties.conformanceVersion.patch);
            SDL_LogInfo(SDL_LOG_CATEGORY_GPU, "Vulkan Conformance: %s", conformance);
        }
    } else {
        if (verboseLogs) {
            SDL_LogInfo(SDL_LOG_CATEGORY_GPU, "Vulkan Driver: %s", driverVer);
        }
    }

    if (!VULKAN_INTERNAL_CreateLogicalDevice(
            renderer)) {
        SDL_free(renderer);
        SDL_Vulkan_UnloadLibrary();
        SET_STRING_ERROR_AND_RETURN("Failed to create logical device!", NULL);
    }

    // FIXME: just move this into this function
    result = (SDL_GPUDevice *)SDL_malloc(sizeof(SDL_GPUDevice));
    ASSIGN_DRIVER(VULKAN)

    result->driverData = (SDL_GPURenderer *)renderer;

    /*
     * Create initial swapchain array
     */

    renderer->claimedWindowCapacity = 1;
    renderer->claimedWindowCount = 0;
    renderer->claimedWindows = SDL_malloc(
        renderer->claimedWindowCapacity * sizeof(WindowData *));

    // Threading

    renderer->allocatorLock = SDL_CreateMutex();
    renderer->disposeLock = SDL_CreateMutex();
    renderer->submitLock = SDL_CreateMutex();
    renderer->acquireCommandBufferLock = SDL_CreateMutex();
    renderer->acquireUniformBufferLock = SDL_CreateMutex();
    renderer->renderPassFetchLock = SDL_CreateMutex();
    renderer->framebufferFetchLock = SDL_CreateMutex();
    renderer->graphicsPipelineLayoutFetchLock = SDL_CreateMutex();
    renderer->computePipelineLayoutFetchLock = SDL_CreateMutex();
    renderer->descriptorSetLayoutFetchLock = SDL_CreateMutex();
    renderer->windowLock = SDL_CreateMutex();

    /*
     * Create submitted command buffer list
     */

    renderer->submittedCommandBufferCapacity = 16;
    renderer->submittedCommandBufferCount = 0;
    renderer->submittedCommandBuffers = SDL_malloc(sizeof(VulkanCommandBuffer *) * renderer->submittedCommandBufferCapacity);

    // Memory Allocator

    renderer->memoryAllocator = (VulkanMemoryAllocator *)SDL_malloc(
        sizeof(VulkanMemoryAllocator));

    for (i = 0; i < VK_MAX_MEMORY_TYPES; i += 1) {
        renderer->memoryAllocator->subAllocators[i].memoryTypeIndex = i;
        renderer->memoryAllocator->subAllocators[i].allocations = NULL;
        renderer->memoryAllocator->subAllocators[i].allocationCount = 0;
        renderer->memoryAllocator->subAllocators[i].sortedFreeRegions = SDL_malloc(
            sizeof(VulkanMemoryFreeRegion *) * 4);
        renderer->memoryAllocator->subAllocators[i].sortedFreeRegionCount = 0;
        renderer->memoryAllocator->subAllocators[i].sortedFreeRegionCapacity = 4;
    }

    // Create uniform buffer pool

    renderer->uniformBufferPoolCount = 32;
    renderer->uniformBufferPoolCapacity = 32;
    renderer->uniformBufferPool = SDL_malloc(
        renderer->uniformBufferPoolCapacity * sizeof(VulkanUniformBuffer *));

    for (i = 0; i < renderer->uniformBufferPoolCount; i += 1) {
        renderer->uniformBufferPool[i] = VULKAN_INTERNAL_CreateUniformBuffer(
            renderer,
            UNIFORM_BUFFER_SIZE);
    }

    renderer->descriptorSetCachePoolCapacity = 8;
    renderer->descriptorSetCachePoolCount = 0;
    renderer->descriptorSetCachePool = SDL_calloc(renderer->descriptorSetCachePoolCapacity, sizeof(DescriptorSetCache *));

    SDL_SetAtomicInt(&renderer->layoutResourceID, 0);

    // Device limits

    renderer->minUBOAlignment = (Uint32)renderer->physicalDeviceProperties.properties.limits.minUniformBufferOffsetAlignment;

    // Initialize caches

    renderer->commandPoolHashTable = SDL_CreateHashTable(
        0,  // !!! FIXME: a real guess here, for a _minimum_ if not a maximum, could be useful.
        false,  // manually synchronized due to submission timing
        VULKAN_INTERNAL_CommandPoolHashFunction,
        VULKAN_INTERNAL_CommandPoolHashKeyMatch,
        VULKAN_INTERNAL_CommandPoolHashDestroy,
        (void *)renderer);

    renderer->renderPassHashTable = SDL_CreateHashTable(
        0,  // !!! FIXME: a real guess here, for a _minimum_ if not a maximum, could be useful.
        false,  // manually synchronized due to lookup timing
        VULKAN_INTERNAL_RenderPassHashFunction,
        VULKAN_INTERNAL_RenderPassHashKeyMatch,
        VULKAN_INTERNAL_RenderPassHashDestroy,
        (void *)renderer);

    renderer->framebufferHashTable = SDL_CreateHashTable(
        0,  // !!! FIXME: a real guess here, for a _minimum_ if not a maximum, could be useful.
        false,  // manually synchronized due to iteration
        VULKAN_INTERNAL_FramebufferHashFunction,
        VULKAN_INTERNAL_FramebufferHashKeyMatch,
        VULKAN_INTERNAL_FramebufferHashDestroy,
        (void *)renderer);

    renderer->graphicsPipelineResourceLayoutHashTable = SDL_CreateHashTable(
        0,  // !!! FIXME: a real guess here, for a _minimum_ if not a maximum, could be useful.
        false,  // manually synchronized due to lookup timing
        VULKAN_INTERNAL_GraphicsPipelineResourceLayoutHashFunction,
        VULKAN_INTERNAL_GraphicsPipelineResourceLayoutHashKeyMatch,
        VULKAN_INTERNAL_GraphicsPipelineResourceLayoutHashDestroy,
        (void *)renderer);

    renderer->computePipelineResourceLayoutHashTable = SDL_CreateHashTable(
        0,  // !!! FIXME: a real guess here, for a _minimum_ if not a maximum, could be useful.
        false,  // manually synchronized due to lookup timing
        VULKAN_INTERNAL_ComputePipelineResourceLayoutHashFunction,
        VULKAN_INTERNAL_ComputePipelineResourceLayoutHashKeyMatch,
        VULKAN_INTERNAL_ComputePipelineResourceLayoutHashDestroy,
        (void *)renderer);

    renderer->descriptorSetLayoutHashTable = SDL_CreateHashTable(
        0,  // !!! FIXME: a real guess here, for a _minimum_ if not a maximum, could be useful.
        false,  // manually synchronized due to lookup timing
        VULKAN_INTERNAL_DescriptorSetLayoutHashFunction,
        VULKAN_INTERNAL_DescriptorSetLayoutHashKeyMatch,
        VULKAN_INTERNAL_DescriptorSetLayoutHashDestroy,
        (void *)renderer);

    // Initialize fence pool

    renderer->fencePool.lock = SDL_CreateMutex();

    renderer->fencePool.availableFenceCapacity = 4;
    renderer->fencePool.availableFenceCount = 0;
    renderer->fencePool.availableFences = SDL_malloc(
        renderer->fencePool.availableFenceCapacity * sizeof(VulkanFenceHandle *));

    // Deferred destroy storage

    renderer->texturesToDestroyCapacity = 16;
    renderer->texturesToDestroyCount = 0;

    renderer->texturesToDestroy = (VulkanTexture **)SDL_malloc(
        sizeof(VulkanTexture *) *
        renderer->texturesToDestroyCapacity);

    renderer->buffersToDestroyCapacity = 16;
    renderer->buffersToDestroyCount = 0;

    renderer->buffersToDestroy = SDL_malloc(
        sizeof(VulkanBuffer *) *
        renderer->buffersToDestroyCapacity);

    renderer->samplersToDestroyCapacity = 16;
    renderer->samplersToDestroyCount = 0;

    renderer->samplersToDestroy = SDL_malloc(
        sizeof(VulkanSampler *) *
        renderer->samplersToDestroyCapacity);

    renderer->graphicsPipelinesToDestroyCapacity = 16;
    renderer->graphicsPipelinesToDestroyCount = 0;

    renderer->graphicsPipelinesToDestroy = SDL_malloc(
        sizeof(VulkanGraphicsPipeline *) *
        renderer->graphicsPipelinesToDestroyCapacity);

    renderer->computePipelinesToDestroyCapacity = 16;
    renderer->computePipelinesToDestroyCount = 0;

    renderer->computePipelinesToDestroy = SDL_malloc(
        sizeof(VulkanComputePipeline *) *
        renderer->computePipelinesToDestroyCapacity);

    renderer->shadersToDestroyCapacity = 16;
    renderer->shadersToDestroyCount = 0;

    renderer->shadersToDestroy = SDL_malloc(
        sizeof(VulkanShader *) *
        renderer->shadersToDestroyCapacity);

    renderer->framebuffersToDestroyCapacity = 16;
    renderer->framebuffersToDestroyCount = 0;
    renderer->framebuffersToDestroy = SDL_malloc(
        sizeof(VulkanFramebuffer *) *
        renderer->framebuffersToDestroyCapacity);

    // Defrag state

    renderer->defragInProgress = 0;

    renderer->allocationsToDefragCount = 0;
    renderer->allocationsToDefragCapacity = 4;
    renderer->allocationsToDefrag = SDL_malloc(
        renderer->allocationsToDefragCapacity * sizeof(VulkanMemoryAllocation *));

    return result;
}

SDL_GPUBootstrap VulkanDriver = {
    "vulkan",
    SDL_GPU_SHADERFORMAT_SPIRV,
    VULKAN_PrepareDriver,
    VULKAN_CreateDevice
};

#endif // SDL_GPU_VULKAN
