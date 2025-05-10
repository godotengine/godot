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

#ifdef SDL_GPU_D3D12

#include "../../core/windows/SDL_windows.h"
#include "../../video/directx/SDL_d3d12.h"
#include "../SDL_sysgpu.h"

#ifdef __IDXGIInfoQueue_INTERFACE_DEFINED__
#define HAVE_IDXGIINFOQUEUE
#endif

// Built-in shaders, compiled with compile_shaders.bat

#define g_FullscreenVert    D3D12_FullscreenVert
#define g_BlitFrom2D        D3D12_BlitFrom2D
#define g_BlitFrom2DArray   D3D12_BlitFrom2DArray
#define g_BlitFrom3D        D3D12_BlitFrom3D
#define g_BlitFromCube      D3D12_BlitFromCube
#define g_BlitFromCubeArray D3D12_BlitFromCubeArray
#if defined(SDL_PLATFORM_XBOXSERIES)
#include "D3D12_Blit_Series.h"
#elif defined(SDL_PLATFORM_XBOXONE)
#include "D3D12_Blit_One.h"
#else
#include "D3D12_Blit.h"
#endif
#undef g_FullscreenVert
#undef g_BlitFrom2D
#undef g_BlitFrom2DArray
#undef g_BlitFrom3D
#undef g_BlitFromCube
#undef g_BlitFromCubeArray

// Macros

#define SET_ERROR(fmt, msg)                               \
    do {                                                  \
        if (renderer->debug_mode) {                       \
            SDL_LogError(SDL_LOG_CATEGORY_GPU, fmt, msg); \
        }                                                 \
        SDL_SetError(fmt, msg);                           \
    } while (0)

#define SET_ERROR_AND_RETURN(fmt, msg, ret)               \
    do {                                                  \
        if (renderer->debug_mode) {                       \
            SDL_LogError(SDL_LOG_CATEGORY_GPU, fmt, msg); \
        }                                                 \
        SDL_SetError(fmt, msg);                           \
        return ret;                                       \
    } while (0)

#define SET_STRING_ERROR_AND_RETURN(msg, ret) SET_ERROR_AND_RETURN("%s", msg, ret)

#define CHECK_D3D12_ERROR_AND_RETURN(msg, ret)           \
    do {                                                 \
        if (FAILED(res)) {                               \
            D3D12_INTERNAL_SetError(renderer, msg, res); \
            return (ret);                                \
        }                                                \
    } while (0)

// Defines
#if defined(_WIN32)
#if defined(SDL_PLATFORM_XBOXSERIES)
#define D3D12_DLL "d3d12_xs.dll"
#elif defined(SDL_PLATFORM_XBOXONE)
#define D3D12_DLL "d3d12_x.dll"
#else
#define D3D12_DLL "d3d12.dll"
#endif
#define DXGI_DLL      "dxgi.dll"
#define DXGIDEBUG_DLL "dxgidebug.dll"
#elif defined(__APPLE__)
#define D3D12_DLL     "libdxvk_d3d12.dylib"
#define DXGI_DLL      "libdxvk_dxgi.dylib"
#define DXGIDEBUG_DLL "libdxvk_dxgidebug.dylib"
#else
#define D3D12_DLL     "libdxvk_d3d12.so"
#define DXGI_DLL      "libdxvk_dxgi.so"
#define DXGIDEBUG_DLL "libdxvk_dxgidebug.so"
#endif

#define D3D12_CREATE_DEVICE_FUNC            "D3D12CreateDevice"
#define D3D12_SERIALIZE_ROOT_SIGNATURE_FUNC "D3D12SerializeRootSignature"
#define CREATE_DXGI_FACTORY1_FUNC           "CreateDXGIFactory1"
#define DXGI_GET_DEBUG_INTERFACE_FUNC       "DXGIGetDebugInterface"
#define D3D12_GET_DEBUG_INTERFACE_FUNC      "D3D12GetDebugInterface"
#define WINDOW_PROPERTY_DATA                "SDL_GPUD3D12WindowPropertyData"
#define D3D_FEATURE_LEVEL_CHOICE            D3D_FEATURE_LEVEL_11_1
#define D3D_FEATURE_LEVEL_CHOICE_STR        "11_1"
#define MAX_ROOT_SIGNATURE_PARAMETERS         64
#define D3D12_FENCE_UNSIGNALED_VALUE          0
#define D3D12_FENCE_SIGNAL_VALUE              1
// TODO: do these need to be tuned?
#define VIEW_GPU_DESCRIPTOR_COUNT             65536
#define SAMPLER_GPU_DESCRIPTOR_COUNT          2048
#define STAGING_HEAP_DESCRIPTOR_COUNT         1024

#define SDL_GPU_SHADERSTAGE_COMPUTE (SDL_GPUShaderStage)2

#ifdef _WIN32
#define HRESULT_FMT "(0x%08lX)"
#else
#define HRESULT_FMT "(0x%08X)"
#endif

// Function Pointer Signatures
typedef HRESULT(WINAPI *PFN_CREATE_DXGI_FACTORY1)(const GUID *riid, void **ppFactory);
typedef HRESULT(WINAPI *PFN_DXGI_GET_DEBUG_INTERFACE)(const GUID *riid, void **ppDebug);

// IIDs (from https://www.magnumdb.com/)
static const IID D3D_IID_IDXGIFactory1 = { 0x770aae78, 0xf26f, 0x4dba, { 0xa8, 0x29, 0x25, 0x3c, 0x83, 0xd1, 0xb3, 0x87 } };
static const IID D3D_IID_IDXGIFactory4 = { 0x1bc6ea02, 0xef36, 0x464f, { 0xbf, 0x0c, 0x21, 0xca, 0x39, 0xe5, 0x16, 0x8a } };
static const IID D3D_IID_IDXGIFactory5 = { 0x7632e1f5, 0xee65, 0x4dca, { 0x87, 0xfd, 0x84, 0xcd, 0x75, 0xf8, 0x83, 0x8d } };
static const IID D3D_IID_IDXGIFactory6 = { 0xc1b6694f, 0xff09, 0x44a9, { 0xb0, 0x3c, 0x77, 0x90, 0x0a, 0x0a, 0x1d, 0x17 } };
static const IID D3D_IID_IDXGIAdapter1 = { 0x29038f61, 0x3839, 0x4626, { 0x91, 0xfd, 0x08, 0x68, 0x79, 0x01, 0x1a, 0x05 } };
#if (defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES))
static const IID D3D_IID_IDXGIDevice1 = { 0x77db970f, 0x6276, 0x48ba, { 0xba, 0x28, 0x07, 0x01, 0x43, 0xb4, 0x39, 0x2c } };
#else
static const IID D3D_IID_IDXGIDevice = { 0x54ec77fa, 0x1377, 0x44e6, { 0x8c, 0x32, 0x88, 0xfd, 0x5f, 0x44, 0xc8, 0x4c } };
#endif
static const IID D3D_IID_IDXGISwapChain3 = { 0x94d99bdb, 0xf1f8, 0x4ab0, { 0xb2, 0x36, 0x7d, 0xa0, 0x17, 0x0e, 0xda, 0xb1 } };
#ifdef HAVE_IDXGIINFOQUEUE
static const IID D3D_IID_IDXGIDebug = { 0x119e7452, 0xde9e, 0x40fe, { 0x88, 0x06, 0x88, 0xf9, 0x0c, 0x12, 0xb4, 0x41 } };
static const IID D3D_IID_IDXGIInfoQueue = { 0xd67441c7, 0x672a, 0x476f, { 0x9e, 0x82, 0xcd, 0x55, 0xb4, 0x49, 0x49, 0xce } };
#endif
static const GUID D3D_IID_DXGI_DEBUG_ALL = { 0xe48ae283, 0xda80, 0x490b, { 0x87, 0xe6, 0x43, 0xe9, 0xa9, 0xcf, 0xda, 0x08 } };

static const IID D3D_IID_ID3D12Device = { 0x189819f1, 0x1db6, 0x4b57, { 0xbe, 0x54, 0x18, 0x21, 0x33, 0x9b, 0x85, 0xf7 } };
static const IID D3D_IID_ID3D12CommandQueue = { 0x0ec870a6, 0x5d7e, 0x4c22, { 0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed } };
static const IID D3D_IID_ID3D12DescriptorHeap = { 0x8efb471d, 0x616c, 0x4f49, { 0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51 } };
static const IID D3D_IID_ID3D12Resource = { 0x696442be, 0xa72e, 0x4059, { 0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad } };
static const IID D3D_IID_ID3D12CommandAllocator = { 0x6102dee4, 0xaf59, 0x4b09, { 0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24 } };
static const IID D3D_IID_ID3D12CommandList = { 0x7116d91c, 0xe7e4, 0x47ce, { 0xb8, 0xc6, 0xec, 0x81, 0x68, 0xf4, 0x37, 0xe5 } };
static const IID D3D_IID_ID3D12GraphicsCommandList = { 0x5b160d0f, 0xac1b, 0x4185, { 0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55 } };
static const IID D3D_IID_ID3D12Fence = { 0x0a753dcf, 0xc4d8, 0x4b91, { 0xad, 0xf6, 0xbe, 0x5a, 0x60, 0xd9, 0x5a, 0x76 } };
static const IID D3D_IID_ID3D12RootSignature = { 0xc54a6b66, 0x72df, 0x4ee8, { 0x8b, 0xe5, 0xa9, 0x46, 0xa1, 0x42, 0x92, 0x14 } };
static const IID D3D_IID_ID3D12CommandSignature = { 0xc36a797c, 0xec80, 0x4f0a, { 0x89, 0x85, 0xa7, 0xb2, 0x47, 0x50, 0x82, 0xd1 } };
static const IID D3D_IID_ID3D12PipelineState = { 0x765a30f3, 0xf624, 0x4c6f, { 0xa8, 0x28, 0xac, 0xe9, 0x48, 0x62, 0x24, 0x45 } };
static const IID D3D_IID_ID3D12Debug = { 0x344488b7, 0x6846, 0x474b, { 0xb9, 0x89, 0xf0, 0x27, 0x44, 0x82, 0x45, 0xe0 } };
static const IID D3D_IID_ID3D12InfoQueue = { 0x0742a90b, 0xc387, 0x483f, { 0xb9, 0x46, 0x30, 0xa7, 0xe4, 0xe6, 0x14, 0x58 } };
static const IID D3D_IID_ID3D12InfoQueue1 = { 0x2852dd88, 0xb484, 0x4c0c, { 0xb6, 0xb1, 0x67, 0x16, 0x85, 0x00, 0xe6, 0x00 } };

// Enums

typedef enum D3D12BufferType
{
    D3D12_BUFFER_TYPE_GPU,
    D3D12_BUFFER_TYPE_UNIFORM,
    D3D12_BUFFER_TYPE_UPLOAD,
    D3D12_BUFFER_TYPE_DOWNLOAD
} D3D12BufferType;

// Conversions

static SDL_GPUTextureFormat SwapchainCompositionToSDLTextureFormat[] = {
    SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM,      // SDR
    SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM_SRGB, // SDR_LINEAR
    SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT,  // HDR_EXTENDED_LINEAR
    SDL_GPU_TEXTUREFORMAT_R10G10B10A2_UNORM,   // HDR10_ST2084
};

static DXGI_FORMAT SwapchainCompositionToTextureFormat[] = {
    DXGI_FORMAT_B8G8R8A8_UNORM,     // SDR
    DXGI_FORMAT_B8G8R8A8_UNORM,     // SDR_LINEAR (NOTE: The RTV uses the sRGB format)
    DXGI_FORMAT_R16G16B16A16_FLOAT, // HDR_EXTENDED_LINEAR
    DXGI_FORMAT_R10G10B10A2_UNORM,  // HDR10_ST2084
};

static DXGI_COLOR_SPACE_TYPE SwapchainCompositionToColorSpace[] = {
    DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709,   // SDR
    DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709,   // SDR_LINEAR
    DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709,   // HDR_EXTENDED_LINEAR
    DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020 // HDR10_ST2084
};

static D3D12_BLEND SDLToD3D12_BlendFactor[] = {
    D3D12_BLEND_ZERO,             // INVALID
    D3D12_BLEND_ZERO,             // ZERO
    D3D12_BLEND_ONE,              // ONE
    D3D12_BLEND_SRC_COLOR,        // SRC_COLOR
    D3D12_BLEND_INV_SRC_COLOR,    // ONE_MINUS_SRC_COLOR
    D3D12_BLEND_DEST_COLOR,       // DST_COLOR
    D3D12_BLEND_INV_DEST_COLOR,   // ONE_MINUS_DST_COLOR
    D3D12_BLEND_SRC_ALPHA,        // SRC_ALPHA
    D3D12_BLEND_INV_SRC_ALPHA,    // ONE_MINUS_SRC_ALPHA
    D3D12_BLEND_DEST_ALPHA,       // DST_ALPHA
    D3D12_BLEND_INV_DEST_ALPHA,   // ONE_MINUS_DST_ALPHA
    D3D12_BLEND_BLEND_FACTOR,     // CONSTANT_COLOR
    D3D12_BLEND_INV_BLEND_FACTOR, // ONE_MINUS_CONSTANT_COLOR
    D3D12_BLEND_SRC_ALPHA_SAT,    // SRC_ALPHA_SATURATE
};
SDL_COMPILE_TIME_ASSERT(SDLToD3D12_BlendFactor, SDL_arraysize(SDLToD3D12_BlendFactor) == SDL_GPU_BLENDFACTOR_MAX_ENUM_VALUE);

static D3D12_BLEND SDLToD3D12_BlendFactorAlpha[] = {
    D3D12_BLEND_ZERO,             // INVALID
    D3D12_BLEND_ZERO,             // ZERO
    D3D12_BLEND_ONE,              // ONE
    D3D12_BLEND_SRC_ALPHA,        // SRC_COLOR
    D3D12_BLEND_INV_SRC_ALPHA,    // ONE_MINUS_SRC_COLOR
    D3D12_BLEND_DEST_ALPHA,       // DST_COLOR
    D3D12_BLEND_INV_DEST_ALPHA,   // ONE_MINUS_DST_COLOR
    D3D12_BLEND_SRC_ALPHA,        // SRC_ALPHA
    D3D12_BLEND_INV_SRC_ALPHA,    // ONE_MINUS_SRC_ALPHA
    D3D12_BLEND_DEST_ALPHA,       // DST_ALPHA
    D3D12_BLEND_INV_DEST_ALPHA,   // ONE_MINUS_DST_ALPHA
    D3D12_BLEND_BLEND_FACTOR,     // CONSTANT_COLOR
    D3D12_BLEND_INV_BLEND_FACTOR, // ONE_MINUS_CONSTANT_COLOR
    D3D12_BLEND_SRC_ALPHA_SAT,    // SRC_ALPHA_SATURATE
};
SDL_COMPILE_TIME_ASSERT(SDLToD3D12_BlendFactorAlpha, SDL_arraysize(SDLToD3D12_BlendFactorAlpha) == SDL_GPU_BLENDFACTOR_MAX_ENUM_VALUE);

static D3D12_BLEND_OP SDLToD3D12_BlendOp[] = {
    D3D12_BLEND_OP_ADD,          // INVALID
    D3D12_BLEND_OP_ADD,          // ADD
    D3D12_BLEND_OP_SUBTRACT,     // SUBTRACT
    D3D12_BLEND_OP_REV_SUBTRACT, // REVERSE_SUBTRACT
    D3D12_BLEND_OP_MIN,          // MIN
    D3D12_BLEND_OP_MAX           // MAX
};
SDL_COMPILE_TIME_ASSERT(SDLToD3D12_BlendOp, SDL_arraysize(SDLToD3D12_BlendOp) == SDL_GPU_BLENDOP_MAX_ENUM_VALUE);

// These are actually color formats.
// For some genius reason, D3D12 splits format capabilites for depth-stencil views.
static DXGI_FORMAT SDLToD3D12_TextureFormat[] = {
    DXGI_FORMAT_UNKNOWN,              // INVALID
    DXGI_FORMAT_A8_UNORM,             // A8_UNORM
    DXGI_FORMAT_R8_UNORM,             // R8_UNORM
    DXGI_FORMAT_R8G8_UNORM,           // R8G8_UNORM
    DXGI_FORMAT_R8G8B8A8_UNORM,       // R8G8B8A8_UNORM
    DXGI_FORMAT_R16_UNORM,            // R16_UNORM
    DXGI_FORMAT_R16G16_UNORM,         // R16G16_UNORM
    DXGI_FORMAT_R16G16B16A16_UNORM,   // R16G16B16A16_UNORM
    DXGI_FORMAT_R10G10B10A2_UNORM,    // R10G10B10A2_UNORM
    DXGI_FORMAT_B5G6R5_UNORM,         // B5G6R5_UNORM
    DXGI_FORMAT_B5G5R5A1_UNORM,       // B5G5R5A1_UNORM
    DXGI_FORMAT_B4G4R4A4_UNORM,       // B4G4R4A4_UNORM
    DXGI_FORMAT_B8G8R8A8_UNORM,       // B8G8R8A8_UNORM
    DXGI_FORMAT_BC1_UNORM,            // BC1_UNORM
    DXGI_FORMAT_BC2_UNORM,            // BC2_UNORM
    DXGI_FORMAT_BC3_UNORM,            // BC3_UNORM
    DXGI_FORMAT_BC4_UNORM,            // BC4_UNORM
    DXGI_FORMAT_BC5_UNORM,            // BC5_UNORM
    DXGI_FORMAT_BC7_UNORM,            // BC7_UNORM
    DXGI_FORMAT_BC6H_SF16,            // BC6H_FLOAT
    DXGI_FORMAT_BC6H_UF16,            // BC6H_UFLOAT
    DXGI_FORMAT_R8_SNORM,             // R8_SNORM
    DXGI_FORMAT_R8G8_SNORM,           // R8G8_SNORM
    DXGI_FORMAT_R8G8B8A8_SNORM,       // R8G8B8A8_SNORM
    DXGI_FORMAT_R16_SNORM,            // R16_SNORM
    DXGI_FORMAT_R16G16_SNORM,         // R16G16_SNORM
    DXGI_FORMAT_R16G16B16A16_SNORM,   // R16G16B16A16_SNORM
    DXGI_FORMAT_R16_FLOAT,            // R16_FLOAT
    DXGI_FORMAT_R16G16_FLOAT,         // R16G16_FLOAT
    DXGI_FORMAT_R16G16B16A16_FLOAT,   // R16G16B16A16_FLOAT
    DXGI_FORMAT_R32_FLOAT,            // R32_FLOAT
    DXGI_FORMAT_R32G32_FLOAT,         // R32G32_FLOAT
    DXGI_FORMAT_R32G32B32A32_FLOAT,   // R32G32B32A32_FLOAT
    DXGI_FORMAT_R11G11B10_FLOAT,      // R11G11B10_UFLOAT
    DXGI_FORMAT_R8_UINT,              // R8_UINT
    DXGI_FORMAT_R8G8_UINT,            // R8G8_UINT
    DXGI_FORMAT_R8G8B8A8_UINT,        // R8G8B8A8_UINT
    DXGI_FORMAT_R16_UINT,             // R16_UINT
    DXGI_FORMAT_R16G16_UINT,          // R16G16_UINT
    DXGI_FORMAT_R16G16B16A16_UINT,    // R16G16B16A16_UINT
    DXGI_FORMAT_R32_UINT,             // R32_UINT
    DXGI_FORMAT_R32G32_UINT,          // R32G32_UINT
    DXGI_FORMAT_R32G32B32A32_UINT,    // R32G32B32A32_UINT
    DXGI_FORMAT_R8_SINT,              // R8_INT
    DXGI_FORMAT_R8G8_SINT,            // R8G8_INT
    DXGI_FORMAT_R8G8B8A8_SINT,        // R8G8B8A8_INT
    DXGI_FORMAT_R16_SINT,             // R16_INT
    DXGI_FORMAT_R16G16_SINT,          // R16G16_INT
    DXGI_FORMAT_R16G16B16A16_SINT,    // R16G16B16A16_INT
    DXGI_FORMAT_R32_SINT,             // R32_INT
    DXGI_FORMAT_R32G32_SINT,          // R32G32_INT
    DXGI_FORMAT_R32G32B32A32_SINT,    // R32G32B32A32_INT
    DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,  // R8G8B8A8_UNORM_SRGB
    DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,  // B8G8R8A8_UNORM_SRGB
    DXGI_FORMAT_BC1_UNORM_SRGB,       // BC1_UNORM_SRGB
    DXGI_FORMAT_BC2_UNORM_SRGB,       // BC2_UNORM_SRGB
    DXGI_FORMAT_BC3_UNORM_SRGB,       // BC3_UNORM_SRGB
    DXGI_FORMAT_BC7_UNORM_SRGB,       // BC7_UNORM_SRGB
    DXGI_FORMAT_R16_UNORM,                // D16_UNORM
    DXGI_FORMAT_R24_UNORM_X8_TYPELESS,    // D24_UNORM
    DXGI_FORMAT_R32_FLOAT,                // D32_FLOAT
    DXGI_FORMAT_R24_UNORM_X8_TYPELESS,    // D24_UNORM_S8_UINT
    DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS, // D32_FLOAT_S8_UINT
    DXGI_FORMAT_UNKNOWN,              // ASTC_4x4_UNORM
    DXGI_FORMAT_UNKNOWN,              // ASTC_5x4_UNORM
    DXGI_FORMAT_UNKNOWN,              // ASTC_5x5_UNORM
    DXGI_FORMAT_UNKNOWN,              // ASTC_6x5_UNORM
    DXGI_FORMAT_UNKNOWN,              // ASTC_6x6_UNORM
    DXGI_FORMAT_UNKNOWN,              // ASTC_8x5_UNORM
    DXGI_FORMAT_UNKNOWN,              // ASTC_8x6_UNORM
    DXGI_FORMAT_UNKNOWN,              // ASTC_8x8_UNORM
    DXGI_FORMAT_UNKNOWN,              // ASTC_10x5_UNORM
    DXGI_FORMAT_UNKNOWN,              // ASTC_10x6_UNORM
    DXGI_FORMAT_UNKNOWN,              // ASTC_10x8_UNORM
    DXGI_FORMAT_UNKNOWN,              // ASTC_10x10_UNORM
    DXGI_FORMAT_UNKNOWN,              // ASTC_12x10_UNORM
    DXGI_FORMAT_UNKNOWN,              // ASTC_12x12_UNORM
    DXGI_FORMAT_UNKNOWN,              // ASTC_4x4_UNORM_SRGB
    DXGI_FORMAT_UNKNOWN,              // ASTC_5x4_UNORM_SRGB
    DXGI_FORMAT_UNKNOWN,              // ASTC_5x5_UNORM_SRGB
    DXGI_FORMAT_UNKNOWN,              // ASTC_6x5_UNORM_SRGB
    DXGI_FORMAT_UNKNOWN,              // ASTC_6x6_UNORM_SRGB
    DXGI_FORMAT_UNKNOWN,              // ASTC_8x5_UNORM_SRGB
    DXGI_FORMAT_UNKNOWN,              // ASTC_8x6_UNORM_SRGB
    DXGI_FORMAT_UNKNOWN,              // ASTC_8x8_UNORM_SRGB
    DXGI_FORMAT_UNKNOWN,              // ASTC_10x5_UNORM_SRGB
    DXGI_FORMAT_UNKNOWN,              // ASTC_10x6_UNORM_SRGB
    DXGI_FORMAT_UNKNOWN,              // ASTC_10x8_UNORM_SRGB
    DXGI_FORMAT_UNKNOWN,              // ASTC_10x10_UNORM_SRGB
    DXGI_FORMAT_UNKNOWN,              // ASTC_12x10_UNORM_SRGB
    DXGI_FORMAT_UNKNOWN,              // ASTC_12x12_UNORM_SRGB
    DXGI_FORMAT_UNKNOWN,              // ASTC_4x4_FLOAT
    DXGI_FORMAT_UNKNOWN,              // ASTC_5x4_FLOAT
    DXGI_FORMAT_UNKNOWN,              // ASTC_5x5_FLOAT
    DXGI_FORMAT_UNKNOWN,              // ASTC_6x5_FLOAT
    DXGI_FORMAT_UNKNOWN,              // ASTC_6x6_FLOAT
    DXGI_FORMAT_UNKNOWN,              // ASTC_8x5_FLOAT
    DXGI_FORMAT_UNKNOWN,              // ASTC_8x6_FLOAT
    DXGI_FORMAT_UNKNOWN,              // ASTC_8x8_FLOAT
    DXGI_FORMAT_UNKNOWN,              // ASTC_10x5_FLOAT
    DXGI_FORMAT_UNKNOWN,              // ASTC_10x6_FLOAT
    DXGI_FORMAT_UNKNOWN,              // ASTC_10x8_FLOAT
    DXGI_FORMAT_UNKNOWN,              // ASTC_10x10_FLOAT
    DXGI_FORMAT_UNKNOWN,              // ASTC_12x10_FLOAT
    DXGI_FORMAT_UNKNOWN,              // ASTC_12x12_FLOAT
};
SDL_COMPILE_TIME_ASSERT(SDLToD3D12_TextureFormat, SDL_arraysize(SDLToD3D12_TextureFormat) == SDL_GPU_TEXTUREFORMAT_MAX_ENUM_VALUE);

static DXGI_FORMAT SDLToD3D12_DepthFormat[] = {
    DXGI_FORMAT_UNKNOWN,              // INVALID
    DXGI_FORMAT_UNKNOWN,              // A8_UNORM
    DXGI_FORMAT_UNKNOWN,              // R8_UNORM
    DXGI_FORMAT_UNKNOWN,              // R8G8_UNORM
    DXGI_FORMAT_UNKNOWN,              // R8G8B8A8_UNORM
    DXGI_FORMAT_UNKNOWN,              // R16_UNORM
    DXGI_FORMAT_UNKNOWN,              // R16G16_UNORM
    DXGI_FORMAT_UNKNOWN,              // R16G16B16A16_UNORM
    DXGI_FORMAT_UNKNOWN,              // R10G10B10A2_UNORM
    DXGI_FORMAT_UNKNOWN,              // B5G6R5_UNORM
    DXGI_FORMAT_UNKNOWN,              // B5G5R5A1_UNORM
    DXGI_FORMAT_UNKNOWN,              // B4G4R4A4_UNORM
    DXGI_FORMAT_UNKNOWN,              // B8G8R8A8_UNORM
    DXGI_FORMAT_UNKNOWN,              // BC1_UNORM
    DXGI_FORMAT_UNKNOWN,              // BC2_UNORM
    DXGI_FORMAT_UNKNOWN,              // BC3_UNORM
    DXGI_FORMAT_UNKNOWN,              // BC4_UNORM
    DXGI_FORMAT_UNKNOWN,              // BC5_UNORM
    DXGI_FORMAT_UNKNOWN,              // BC7_UNORM
    DXGI_FORMAT_UNKNOWN,              // BC6H_FLOAT
    DXGI_FORMAT_UNKNOWN,              // BC6H_UFLOAT
    DXGI_FORMAT_UNKNOWN,              // R8_SNORM
    DXGI_FORMAT_UNKNOWN,              // R8G8_SNORM
    DXGI_FORMAT_UNKNOWN,              // R8G8B8A8_SNORM
    DXGI_FORMAT_UNKNOWN,              // R16_SNORM
    DXGI_FORMAT_UNKNOWN,              // R16G16_SNORM
    DXGI_FORMAT_UNKNOWN,              // R16G16B16A16_SNORM
    DXGI_FORMAT_UNKNOWN,              // R16_FLOAT
    DXGI_FORMAT_UNKNOWN,              // R16G16_FLOAT
    DXGI_FORMAT_UNKNOWN,              // R16G16B16A16_FLOAT
    DXGI_FORMAT_UNKNOWN,              // R32_FLOAT
    DXGI_FORMAT_UNKNOWN,              // R32G32_FLOAT
    DXGI_FORMAT_UNKNOWN,              // R32G32B32A32_FLOAT
    DXGI_FORMAT_UNKNOWN,              // R11G11B10_UFLOAT
    DXGI_FORMAT_UNKNOWN,              // R8_UINT
    DXGI_FORMAT_UNKNOWN,              // R8G8_UINT
    DXGI_FORMAT_UNKNOWN,              // R8G8B8A8_UINT
    DXGI_FORMAT_UNKNOWN,              // R16_UINT
    DXGI_FORMAT_UNKNOWN,              // R16G16_UINT
    DXGI_FORMAT_UNKNOWN,              // R16G16B16A16_UINT
    DXGI_FORMAT_UNKNOWN,              // R32_UINT
    DXGI_FORMAT_UNKNOWN,              // R32G32_UINT
    DXGI_FORMAT_UNKNOWN,              // R32G32B32A32_UINT
    DXGI_FORMAT_UNKNOWN,              // R8_INT
    DXGI_FORMAT_UNKNOWN,              // R8G8_INT
    DXGI_FORMAT_UNKNOWN,              // R8G8B8A8_INT
    DXGI_FORMAT_UNKNOWN,              // R16_INT
    DXGI_FORMAT_UNKNOWN,              // R16G16_INT
    DXGI_FORMAT_UNKNOWN,              // R16G16B16A16_INT
    DXGI_FORMAT_UNKNOWN,              // R32_INT
    DXGI_FORMAT_UNKNOWN,              // R32G32_INT
    DXGI_FORMAT_UNKNOWN,              // R32G32B32A32_INT
    DXGI_FORMAT_UNKNOWN,              // R8G8B8A8_UNORM_SRGB
    DXGI_FORMAT_UNKNOWN,              // B8G8R8A8_UNORM_SRGB
    DXGI_FORMAT_UNKNOWN,              // BC1_UNORM_SRGB
    DXGI_FORMAT_UNKNOWN,              // BC2_UNORM_SRGB
    DXGI_FORMAT_UNKNOWN,              // BC3_UNORM_SRGB
    DXGI_FORMAT_UNKNOWN,              // BC7_UNORM_SRGB
    DXGI_FORMAT_D16_UNORM,            // D16_UNORM
    DXGI_FORMAT_D24_UNORM_S8_UINT,    // D24_UNORM
    DXGI_FORMAT_D32_FLOAT,            // D32_FLOAT
    DXGI_FORMAT_D24_UNORM_S8_UINT,    // D24_UNORM_S8_UINT
    DXGI_FORMAT_D32_FLOAT_S8X24_UINT, // D32_FLOAT_S8_UINT
    DXGI_FORMAT_UNKNOWN,              // ASTC_4x4_UNORM
    DXGI_FORMAT_UNKNOWN,              // ASTC_5x4_UNORM
    DXGI_FORMAT_UNKNOWN,              // ASTC_5x5_UNORM
    DXGI_FORMAT_UNKNOWN,              // ASTC_6x5_UNORM
    DXGI_FORMAT_UNKNOWN,              // ASTC_6x6_UNORM
    DXGI_FORMAT_UNKNOWN,              // ASTC_8x5_UNORM
    DXGI_FORMAT_UNKNOWN,              // ASTC_8x6_UNORM
    DXGI_FORMAT_UNKNOWN,              // ASTC_8x8_UNORM
    DXGI_FORMAT_UNKNOWN,              // ASTC_10x5_UNORM
    DXGI_FORMAT_UNKNOWN,              // ASTC_10x6_UNORM
    DXGI_FORMAT_UNKNOWN,              // ASTC_10x8_UNORM
    DXGI_FORMAT_UNKNOWN,              // ASTC_10x10_UNORM
    DXGI_FORMAT_UNKNOWN,              // ASTC_12x10_UNORM
    DXGI_FORMAT_UNKNOWN,              // ASTC_12x12_UNORM
    DXGI_FORMAT_UNKNOWN,              // ASTC_4x4_UNORM_SRGB
    DXGI_FORMAT_UNKNOWN,              // ASTC_5x4_UNORM_SRGB
    DXGI_FORMAT_UNKNOWN,              // ASTC_5x5_UNORM_SRGB
    DXGI_FORMAT_UNKNOWN,              // ASTC_6x5_UNORM_SRGB
    DXGI_FORMAT_UNKNOWN,              // ASTC_6x6_UNORM_SRGB
    DXGI_FORMAT_UNKNOWN,              // ASTC_8x5_UNORM_SRGB
    DXGI_FORMAT_UNKNOWN,              // ASTC_8x6_UNORM_SRGB
    DXGI_FORMAT_UNKNOWN,              // ASTC_8x8_UNORM_SRGB
    DXGI_FORMAT_UNKNOWN,              // ASTC_10x5_UNORM_SRGB
    DXGI_FORMAT_UNKNOWN,              // ASTC_10x6_UNORM_SRGB
    DXGI_FORMAT_UNKNOWN,              // ASTC_10x8_UNORM_SRGB
    DXGI_FORMAT_UNKNOWN,              // ASTC_10x10_UNORM_SRGB
    DXGI_FORMAT_UNKNOWN,              // ASTC_12x10_UNORM_SRGB
    DXGI_FORMAT_UNKNOWN,              // ASTC_12x12_UNORM_SRGB
    DXGI_FORMAT_UNKNOWN,              // ASTC_4x4_FLOAT
    DXGI_FORMAT_UNKNOWN,              // ASTC_5x4_FLOAT
    DXGI_FORMAT_UNKNOWN,              // ASTC_5x5_FLOAT
    DXGI_FORMAT_UNKNOWN,              // ASTC_6x5_FLOAT
    DXGI_FORMAT_UNKNOWN,              // ASTC_6x6_FLOAT
    DXGI_FORMAT_UNKNOWN,              // ASTC_8x5_FLOAT
    DXGI_FORMAT_UNKNOWN,              // ASTC_8x6_FLOAT
    DXGI_FORMAT_UNKNOWN,              // ASTC_8x8_FLOAT
    DXGI_FORMAT_UNKNOWN,              // ASTC_10x5_FLOAT
    DXGI_FORMAT_UNKNOWN,              // ASTC_10x6_FLOAT
    DXGI_FORMAT_UNKNOWN,              // ASTC_10x8_FLOAT
    DXGI_FORMAT_UNKNOWN,              // ASTC_10x10_FLOAT
    DXGI_FORMAT_UNKNOWN,              // ASTC_12x10_FLOAT
    DXGI_FORMAT_UNKNOWN,              // ASTC_12x12_FLOAT
};
SDL_COMPILE_TIME_ASSERT(SDLToD3D12_DepthFormat, SDL_arraysize(SDLToD3D12_DepthFormat) == SDL_GPU_TEXTUREFORMAT_MAX_ENUM_VALUE);

static DXGI_FORMAT SDLToD3D12_TypelessFormat[] = {
    DXGI_FORMAT_UNKNOWN,              // INVALID
    DXGI_FORMAT_UNKNOWN,              // A8_UNORM
    DXGI_FORMAT_UNKNOWN,              // R8_UNORM
    DXGI_FORMAT_UNKNOWN,              // R8G8_UNORM
    DXGI_FORMAT_UNKNOWN,              // R8G8B8A8_UNORM
    DXGI_FORMAT_UNKNOWN,              // R16_UNORM
    DXGI_FORMAT_UNKNOWN,              // R16G16_UNORM
    DXGI_FORMAT_UNKNOWN,              // R16G16B16A16_UNORM
    DXGI_FORMAT_UNKNOWN,              // R10G10B10A2_UNORM
    DXGI_FORMAT_UNKNOWN,              // B5G6R5_UNORM
    DXGI_FORMAT_UNKNOWN,              // B5G5R5A1_UNORM
    DXGI_FORMAT_UNKNOWN,              // B4G4R4A4_UNORM
    DXGI_FORMAT_UNKNOWN,              // B8G8R8A8_UNORM
    DXGI_FORMAT_UNKNOWN,              // BC1_UNORM
    DXGI_FORMAT_UNKNOWN,              // BC2_UNORM
    DXGI_FORMAT_UNKNOWN,              // BC3_UNORM
    DXGI_FORMAT_UNKNOWN,              // BC4_UNORM
    DXGI_FORMAT_UNKNOWN,              // BC5_UNORM
    DXGI_FORMAT_UNKNOWN,              // BC7_UNORM
    DXGI_FORMAT_UNKNOWN,              // BC6H_FLOAT
    DXGI_FORMAT_UNKNOWN,              // BC6H_UFLOAT
    DXGI_FORMAT_UNKNOWN,              // R8_SNORM
    DXGI_FORMAT_UNKNOWN,              // R8G8_SNORM
    DXGI_FORMAT_UNKNOWN,              // R8G8B8A8_SNORM
    DXGI_FORMAT_UNKNOWN,              // R16_SNORM
    DXGI_FORMAT_UNKNOWN,              // R16G16_SNORM
    DXGI_FORMAT_UNKNOWN,              // R16G16B16A16_SNORM
    DXGI_FORMAT_UNKNOWN,              // R16_FLOAT
    DXGI_FORMAT_UNKNOWN,              // R16G16_FLOAT
    DXGI_FORMAT_UNKNOWN,              // R16G16B16A16_FLOAT
    DXGI_FORMAT_UNKNOWN,              // R32_FLOAT
    DXGI_FORMAT_UNKNOWN,              // R32G32_FLOAT
    DXGI_FORMAT_UNKNOWN,              // R32G32B32A32_FLOAT
    DXGI_FORMAT_UNKNOWN,              // R11G11B10_UFLOAT
    DXGI_FORMAT_UNKNOWN,              // R8_UINT
    DXGI_FORMAT_UNKNOWN,              // R8G8_UINT
    DXGI_FORMAT_UNKNOWN,              // R8G8B8A8_UINT
    DXGI_FORMAT_UNKNOWN,              // R16_UINT
    DXGI_FORMAT_UNKNOWN,              // R16G16_UINT
    DXGI_FORMAT_UNKNOWN,              // R16G16B16A16_UINT
    DXGI_FORMAT_UNKNOWN,              // R32_UINT
    DXGI_FORMAT_UNKNOWN,              // R32G32_UINT
    DXGI_FORMAT_UNKNOWN,              // R32G32B32A32_UINT
    DXGI_FORMAT_UNKNOWN,              // R8_INT
    DXGI_FORMAT_UNKNOWN,              // R8G8_INT
    DXGI_FORMAT_UNKNOWN,              // R8G8B8A8_INT
    DXGI_FORMAT_UNKNOWN,              // R16_INT
    DXGI_FORMAT_UNKNOWN,              // R16G16_INT
    DXGI_FORMAT_UNKNOWN,              // R16G16B16A16_INT
    DXGI_FORMAT_UNKNOWN,              // R32_INT
    DXGI_FORMAT_UNKNOWN,              // R32G32_INT
    DXGI_FORMAT_UNKNOWN,              // R32G32B32A32_INT
    DXGI_FORMAT_UNKNOWN,              // R8G8B8A8_UNORM_SRGB
    DXGI_FORMAT_UNKNOWN,              // B8G8R8A8_UNORM_SRGB
    DXGI_FORMAT_UNKNOWN,              // BC1_UNORM_SRGB
    DXGI_FORMAT_UNKNOWN,              // BC2_UNORM_SRGB
    DXGI_FORMAT_UNKNOWN,              // BC3_UNORM_SRGB
    DXGI_FORMAT_UNKNOWN,              // BC7_UNORM_SRGB
    DXGI_FORMAT_R16_TYPELESS,         // D16_UNORM
    DXGI_FORMAT_R24G8_TYPELESS,       // D24_UNORM
    DXGI_FORMAT_R32_TYPELESS,         // D32_FLOAT
    DXGI_FORMAT_R24G8_TYPELESS,       // D24_UNORM_S8_UINT
    DXGI_FORMAT_R32G8X24_TYPELESS,    // D32_FLOAT_S8_UINT
    DXGI_FORMAT_UNKNOWN,              // ASTC_4x4_UNORM
    DXGI_FORMAT_UNKNOWN,              // ASTC_5x4_UNORM
    DXGI_FORMAT_UNKNOWN,              // ASTC_5x5_UNORM
    DXGI_FORMAT_UNKNOWN,              // ASTC_6x5_UNORM
    DXGI_FORMAT_UNKNOWN,              // ASTC_6x6_UNORM
    DXGI_FORMAT_UNKNOWN,              // ASTC_8x5_UNORM
    DXGI_FORMAT_UNKNOWN,              // ASTC_8x6_UNORM
    DXGI_FORMAT_UNKNOWN,              // ASTC_8x8_UNORM
    DXGI_FORMAT_UNKNOWN,              // ASTC_10x5_UNORM
    DXGI_FORMAT_UNKNOWN,              // ASTC_10x6_UNORM
    DXGI_FORMAT_UNKNOWN,              // ASTC_10x8_UNORM
    DXGI_FORMAT_UNKNOWN,              // ASTC_10x10_UNORM
    DXGI_FORMAT_UNKNOWN,              // ASTC_12x10_UNORM
    DXGI_FORMAT_UNKNOWN,              // ASTC_12x12_UNORM
    DXGI_FORMAT_UNKNOWN,              // ASTC_4x4_UNORM_SRGB
    DXGI_FORMAT_UNKNOWN,              // ASTC_5x4_UNORM_SRGB
    DXGI_FORMAT_UNKNOWN,              // ASTC_5x5_UNORM_SRGB
    DXGI_FORMAT_UNKNOWN,              // ASTC_6x5_UNORM_SRGB
    DXGI_FORMAT_UNKNOWN,              // ASTC_6x6_UNORM_SRGB
    DXGI_FORMAT_UNKNOWN,              // ASTC_8x5_UNORM_SRGB
    DXGI_FORMAT_UNKNOWN,              // ASTC_8x6_UNORM_SRGB
    DXGI_FORMAT_UNKNOWN,              // ASTC_8x8_UNORM_SRGB
    DXGI_FORMAT_UNKNOWN,              // ASTC_10x5_UNORM_SRGB
    DXGI_FORMAT_UNKNOWN,              // ASTC_10x6_UNORM_SRGB
    DXGI_FORMAT_UNKNOWN,              // ASTC_10x8_UNORM_SRGB
    DXGI_FORMAT_UNKNOWN,              // ASTC_10x10_UNORM_SRGB
    DXGI_FORMAT_UNKNOWN,              // ASTC_12x10_UNORM_SRGB
    DXGI_FORMAT_UNKNOWN,              // ASTC_12x12_UNORM_SRGB
    DXGI_FORMAT_UNKNOWN,              // ASTC_4x4_FLOAT
    DXGI_FORMAT_UNKNOWN,              // ASTC_5x4_FLOAT
    DXGI_FORMAT_UNKNOWN,              // ASTC_5x5_FLOAT
    DXGI_FORMAT_UNKNOWN,              // ASTC_6x5_FLOAT
    DXGI_FORMAT_UNKNOWN,              // ASTC_6x6_FLOAT
    DXGI_FORMAT_UNKNOWN,              // ASTC_8x5_FLOAT
    DXGI_FORMAT_UNKNOWN,              // ASTC_8x6_FLOAT
    DXGI_FORMAT_UNKNOWN,              // ASTC_8x8_FLOAT
    DXGI_FORMAT_UNKNOWN,              // ASTC_10x5_FLOAT
    DXGI_FORMAT_UNKNOWN,              // ASTC_10x6_FLOAT
    DXGI_FORMAT_UNKNOWN,              // ASTC_10x8_FLOAT
    DXGI_FORMAT_UNKNOWN,              // ASTC_10x10_FLOAT
    DXGI_FORMAT_UNKNOWN,              // ASTC_12x10_FLOAT
    DXGI_FORMAT_UNKNOWN,              // ASTC_12x12_FLOAT
};
SDL_COMPILE_TIME_ASSERT(SDLToD3D12_TypelessFormat, SDL_arraysize(SDLToD3D12_TypelessFormat) == SDL_GPU_TEXTUREFORMAT_MAX_ENUM_VALUE);

static D3D12_COMPARISON_FUNC SDLToD3D12_CompareOp[] = {
    D3D12_COMPARISON_FUNC_NEVER,         // INVALID
    D3D12_COMPARISON_FUNC_NEVER,         // NEVER
    D3D12_COMPARISON_FUNC_LESS,          // LESS
    D3D12_COMPARISON_FUNC_EQUAL,         // EQUAL
    D3D12_COMPARISON_FUNC_LESS_EQUAL,    // LESS_OR_EQUAL
    D3D12_COMPARISON_FUNC_GREATER,       // GREATER
    D3D12_COMPARISON_FUNC_NOT_EQUAL,     // NOT_EQUAL
    D3D12_COMPARISON_FUNC_GREATER_EQUAL, // GREATER_OR_EQUAL
    D3D12_COMPARISON_FUNC_ALWAYS         // ALWAYS
};
SDL_COMPILE_TIME_ASSERT(SDLToD3D12_CompareOp, SDL_arraysize(SDLToD3D12_CompareOp) == SDL_GPU_COMPAREOP_MAX_ENUM_VALUE);

static D3D12_STENCIL_OP SDLToD3D12_StencilOp[] = {
    D3D12_STENCIL_OP_KEEP,     // INVALID
    D3D12_STENCIL_OP_KEEP,     // KEEP
    D3D12_STENCIL_OP_ZERO,     // ZERO
    D3D12_STENCIL_OP_REPLACE,  // REPLACE
    D3D12_STENCIL_OP_INCR_SAT, // INCREMENT_AND_CLAMP
    D3D12_STENCIL_OP_DECR_SAT, // DECREMENT_AND_CLAMP
    D3D12_STENCIL_OP_INVERT,   // INVERT
    D3D12_STENCIL_OP_INCR,     // INCREMENT_AND_WRAP
    D3D12_STENCIL_OP_DECR      // DECREMENT_AND_WRAP
};
SDL_COMPILE_TIME_ASSERT(SDLToD3D12_StencilOp, SDL_arraysize(SDLToD3D12_StencilOp) == SDL_GPU_STENCILOP_MAX_ENUM_VALUE);

static D3D12_CULL_MODE SDLToD3D12_CullMode[] = {
    D3D12_CULL_MODE_NONE,  // NONE
    D3D12_CULL_MODE_FRONT, // FRONT
    D3D12_CULL_MODE_BACK   // BACK
};

static D3D12_FILL_MODE SDLToD3D12_FillMode[] = {
    D3D12_FILL_MODE_SOLID,    // FILL
    D3D12_FILL_MODE_WIREFRAME // LINE
};

static D3D12_INPUT_CLASSIFICATION SDLToD3D12_InputRate[] = {
    D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,  // VERTEX
    D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA // INSTANCE
};

static DXGI_FORMAT SDLToD3D12_VertexFormat[] = {
    DXGI_FORMAT_UNKNOWN,            // UNKNOWN
    DXGI_FORMAT_R32_SINT,           // INT
    DXGI_FORMAT_R32G32_SINT,        // INT2
    DXGI_FORMAT_R32G32B32_SINT,     // INT3
    DXGI_FORMAT_R32G32B32A32_SINT,  // INT4
    DXGI_FORMAT_R32_UINT,           // UINT
    DXGI_FORMAT_R32G32_UINT,        // UINT2
    DXGI_FORMAT_R32G32B32_UINT,     // UINT3
    DXGI_FORMAT_R32G32B32A32_UINT,  // UINT4
    DXGI_FORMAT_R32_FLOAT,          // FLOAT
    DXGI_FORMAT_R32G32_FLOAT,       // FLOAT2
    DXGI_FORMAT_R32G32B32_FLOAT,    // FLOAT3
    DXGI_FORMAT_R32G32B32A32_FLOAT, // FLOAT4
    DXGI_FORMAT_R8G8_SINT,          // BYTE2
    DXGI_FORMAT_R8G8B8A8_SINT,      // BYTE4
    DXGI_FORMAT_R8G8_UINT,          // UBYTE2
    DXGI_FORMAT_R8G8B8A8_UINT,      // UBYTE4
    DXGI_FORMAT_R8G8_SNORM,         // BYTE2_NORM
    DXGI_FORMAT_R8G8B8A8_SNORM,     // BYTE4_NORM
    DXGI_FORMAT_R8G8_UNORM,         // UBYTE2_NORM
    DXGI_FORMAT_R8G8B8A8_UNORM,     // UBYTE4_NORM
    DXGI_FORMAT_R16G16_SINT,        // SHORT2
    DXGI_FORMAT_R16G16B16A16_SINT,  // SHORT4
    DXGI_FORMAT_R16G16_UINT,        // USHORT2
    DXGI_FORMAT_R16G16B16A16_UINT,  // USHORT4
    DXGI_FORMAT_R16G16_SNORM,       // SHORT2_NORM
    DXGI_FORMAT_R16G16B16A16_SNORM, // SHORT4_NORM
    DXGI_FORMAT_R16G16_UNORM,       // USHORT2_NORM
    DXGI_FORMAT_R16G16B16A16_UNORM, // USHORT4_NORM
    DXGI_FORMAT_R16G16_FLOAT,       // HALF2
    DXGI_FORMAT_R16G16B16A16_FLOAT  // HALF4
};
SDL_COMPILE_TIME_ASSERT(SDLToD3D12_VertexFormat, SDL_arraysize(SDLToD3D12_VertexFormat) == SDL_GPU_VERTEXELEMENTFORMAT_MAX_ENUM_VALUE);

static Uint32 SDLToD3D12_SampleCount[] = {
    1, // SDL_GPU_SAMPLECOUNT_1
    2, // SDL_GPU_SAMPLECOUNT_2
    4, // SDL_GPU_SAMPLECOUNT_4
    8, // SDL_GPU_SAMPLECOUNT_8
};

static D3D12_PRIMITIVE_TOPOLOGY SDLToD3D12_PrimitiveType[] = {
    D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,  // TRIANGLELIST
    D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP, // TRIANGLESTRIP
    D3D_PRIMITIVE_TOPOLOGY_LINELIST,      // LINELIST
    D3D_PRIMITIVE_TOPOLOGY_LINESTRIP,     // LINESTRIP
    D3D_PRIMITIVE_TOPOLOGY_POINTLIST      // POINTLIST
};

static D3D12_PRIMITIVE_TOPOLOGY_TYPE SDLToD3D12_PrimitiveTopologyType[] = {
    D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE, // TRIANGLELIST
    D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE, // TRIANGLESTRIP
    D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE,     // LINELIST
    D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE,     // LINESTRIP
    D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT     // POINTLIST
};

static D3D12_TEXTURE_ADDRESS_MODE SDLToD3D12_SamplerAddressMode[] = {
    D3D12_TEXTURE_ADDRESS_MODE_WRAP,   // REPEAT
    D3D12_TEXTURE_ADDRESS_MODE_MIRROR, // MIRRORED_REPEAT
    D3D12_TEXTURE_ADDRESS_MODE_CLAMP   // CLAMP_TO_EDGE
};

static D3D12_FILTER SDLToD3D12_Filter(
    SDL_GPUFilter minFilter,
    SDL_GPUFilter magFilter,
    SDL_GPUSamplerMipmapMode mipmapMode,
    bool comparisonEnabled,
    bool anisotropyEnabled)
{
    D3D12_FILTER result = D3D12_ENCODE_BASIC_FILTER(
        (minFilter == SDL_GPU_FILTER_LINEAR) ? 1 : 0,
        (magFilter == SDL_GPU_FILTER_LINEAR) ? 1 : 0,
        (mipmapMode == SDL_GPU_SAMPLERMIPMAPMODE_LINEAR) ? 1 : 0,
        comparisonEnabled ? 1 : 0);

    if (anisotropyEnabled) {
        result = (D3D12_FILTER)(result | D3D12_ANISOTROPIC_FILTERING_BIT);
    }

    return result;
}

// Structures
typedef struct D3D12Renderer D3D12Renderer;
typedef struct D3D12CommandBufferPool D3D12CommandBufferPool;
typedef struct D3D12CommandBuffer D3D12CommandBuffer;
typedef struct D3D12Texture D3D12Texture;
typedef struct D3D12Shader D3D12Shader;
typedef struct D3D12GraphicsPipeline D3D12GraphicsPipeline;
typedef struct D3D12ComputePipeline D3D12ComputePipeline;
typedef struct D3D12Buffer D3D12Buffer;
typedef struct D3D12BufferContainer D3D12BufferContainer;
typedef struct D3D12UniformBuffer D3D12UniformBuffer;
typedef struct D3D12DescriptorHeap D3D12DescriptorHeap;
typedef struct D3D12StagingDescriptor D3D12StagingDescriptor;
typedef struct D3D12TextureDownload D3D12TextureDownload;

typedef struct D3D12Fence
{
    ID3D12Fence *handle;
    HANDLE event; // used for blocking
    SDL_AtomicInt referenceCount;
} D3D12Fence;

struct D3D12DescriptorHeap
{
    ID3D12DescriptorHeap *handle;
    D3D12_DESCRIPTOR_HEAP_TYPE heapType;
    D3D12_CPU_DESCRIPTOR_HANDLE descriptorHeapCPUStart;
    D3D12_GPU_DESCRIPTOR_HANDLE descriptorHeapGPUStart; // only used by GPU heaps
    Uint32 maxDescriptors;
    Uint32 descriptorSize;
    bool staging;

    Uint32 currentDescriptorIndex; // only used by GPU heaps
};

typedef struct D3D12GPUDescriptorHeapPool
{
    Uint32 capacity;
    Uint32 count;
    D3D12DescriptorHeap **heaps;
    SDL_Mutex *lock;
} D3D12GPUDescriptorHeapPool;

// The only thing we care about with staging descriptors is being able to grab a free descriptor.
typedef struct D3D12StagingDescriptorPool
{
    Uint32 heapCount;
    D3D12DescriptorHeap **heaps;

    // Descriptor handles are owned by resources, so these can be thought of as descriptions of a free index within a heap.
    Uint32 freeDescriptorCapacity;
    Uint32 freeDescriptorCount;
    D3D12StagingDescriptor *freeDescriptors;

    SDL_Mutex *lock;
} D3D12StagingDescriptorPool;

struct D3D12StagingDescriptor
{
    D3D12StagingDescriptorPool *pool;
    D3D12DescriptorHeap *heap;
    D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle;
    Uint32 cpuHandleIndex;
};

typedef struct D3D12TextureContainer
{
    TextureCommonHeader header;

    D3D12Texture *activeTexture;

    D3D12Texture **textures;
    Uint32 textureCapacity;
    Uint32 textureCount;

    // Swapchain images cannot be cycled
    bool canBeCycled;

    char *debugName;
} D3D12TextureContainer;

// Null views represent by heap = NULL
typedef struct D3D12TextureSubresource
{
    D3D12Texture *parent;
    Uint32 layer;
    Uint32 level;
    Uint32 depth;
    Uint32 index;

    // One per depth slice
    D3D12StagingDescriptor *rtvHandles; // NULL if not a color target

    D3D12StagingDescriptor uavHandle; // NULL if not a compute storage write texture
    D3D12StagingDescriptor dsvHandle; // NULL if not a depth stencil target
} D3D12TextureSubresource;

struct D3D12Texture
{
    D3D12TextureContainer *container;
    Uint32 containerIndex;

    D3D12TextureSubresource *subresources;
    Uint32 subresourceCount; /* layerCount * num_levels */

    ID3D12Resource *resource;
    D3D12StagingDescriptor srvHandle;

    SDL_AtomicInt referenceCount;
};

typedef struct D3D12Sampler
{
    SDL_GPUSamplerCreateInfo createInfo;
    D3D12StagingDescriptor handle;
    SDL_AtomicInt referenceCount;
} D3D12Sampler;

typedef struct D3D12WindowData
{
    SDL_Window *window;
#if (defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES))
    D3D12XBOX_FRAME_PIPELINE_TOKEN frameToken;
#else
    IDXGISwapChain3 *swapchain;
#endif
    SDL_GPUPresentMode present_mode;
    SDL_GPUSwapchainComposition swapchainComposition;
    DXGI_COLOR_SPACE_TYPE swapchainColorSpace;
    Uint32 frameCounter;

    D3D12TextureContainer textureContainers[MAX_FRAMES_IN_FLIGHT];
    Uint32 swapchainTextureCount;

    SDL_GPUFence *inFlightFences[MAX_FRAMES_IN_FLIGHT];
    Uint32 width;
    Uint32 height;
    bool needsSwapchainRecreate;
} D3D12WindowData;

typedef struct D3D12PresentData
{
    D3D12WindowData *windowData;
    Uint32 swapchainImageIndex;
} D3D12PresentData;

struct D3D12Renderer
{
    // Reference to the parent device
    SDL_GPUDevice *sdlGPUDevice;

#if !(defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES))
    IDXGIDebug *dxgiDebug;
    IDXGIFactory4 *factory;
#ifdef HAVE_IDXGIINFOQUEUE
    IDXGIInfoQueue *dxgiInfoQueue;
#endif
    IDXGIAdapter1 *adapter;
    SDL_SharedObject *dxgi_dll;
    SDL_SharedObject *dxgidebug_dll;
#endif
    ID3D12Debug *d3d12Debug;
    BOOL supportsTearing;
    SDL_SharedObject *d3d12_dll;
    ID3D12Device *device;
    PFN_D3D12_SERIALIZE_ROOT_SIGNATURE D3D12SerializeRootSignature_func;
    const char *semantic;
    SDL_iconv_t iconv;

    ID3D12CommandQueue *commandQueue;

    bool debug_mode;
    bool GPUUploadHeapSupported;
    // FIXME: these might not be necessary since we're not using custom heaps
    bool UMA;
    bool UMACacheCoherent;
    SDL_PropertiesID props;
    Uint32 allowedFramesInFlight;

    // Indirect command signatures
    ID3D12CommandSignature *indirectDrawCommandSignature;
    ID3D12CommandSignature *indirectIndexedDrawCommandSignature;
    ID3D12CommandSignature *indirectDispatchCommandSignature;

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

    // Resources

    D3D12CommandBuffer **availableCommandBuffers;
    Uint32 availableCommandBufferCount;
    Uint32 availableCommandBufferCapacity;

    D3D12CommandBuffer **submittedCommandBuffers;
    Uint32 submittedCommandBufferCount;
    Uint32 submittedCommandBufferCapacity;

    D3D12UniformBuffer **uniformBufferPool;
    Uint32 uniformBufferPoolCount;
    Uint32 uniformBufferPoolCapacity;

    D3D12WindowData **claimedWindows;
    Uint32 claimedWindowCount;
    Uint32 claimedWindowCapacity;

    D3D12Fence **availableFences;
    Uint32 availableFenceCount;
    Uint32 availableFenceCapacity;

    D3D12StagingDescriptorPool *stagingDescriptorPools[D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES];
    D3D12GPUDescriptorHeapPool gpuDescriptorHeapPools[2];

    // Deferred resource releasing

    D3D12Buffer **buffersToDestroy;
    Uint32 buffersToDestroyCount;
    Uint32 buffersToDestroyCapacity;

    D3D12Texture **texturesToDestroy;
    Uint32 texturesToDestroyCount;
    Uint32 texturesToDestroyCapacity;

    D3D12Sampler **samplersToDestroy;
    Uint32 samplersToDestroyCount;
    Uint32 samplersToDestroyCapacity;

    D3D12GraphicsPipeline **graphicsPipelinesToDestroy;
    Uint32 graphicsPipelinesToDestroyCount;
    Uint32 graphicsPipelinesToDestroyCapacity;

    D3D12ComputePipeline **computePipelinesToDestroy;
    Uint32 computePipelinesToDestroyCount;
    Uint32 computePipelinesToDestroyCapacity;

    // Locks
    SDL_Mutex *acquireCommandBufferLock;
    SDL_Mutex *acquireUniformBufferLock;
    SDL_Mutex *submitLock;
    SDL_Mutex *windowLock;
    SDL_Mutex *fenceLock;
    SDL_Mutex *disposeLock;
};

struct D3D12CommandBuffer
{
    // reserved for SDL_gpu
    CommandBufferCommonHeader common;

    // non owning parent reference
    D3D12Renderer *renderer;

    ID3D12CommandAllocator *commandAllocator;
    ID3D12GraphicsCommandList *graphicsCommandList;
    D3D12Fence *inFlightFence;
    bool autoReleaseFence;

    // Presentation data
    D3D12PresentData *presentDatas;
    Uint32 presentDataCount;
    Uint32 presentDataCapacity;

    D3D12TextureSubresource *colorTargetSubresources[MAX_COLOR_TARGET_BINDINGS];
    D3D12TextureSubresource *colorResolveSubresources[MAX_COLOR_TARGET_BINDINGS];
    D3D12TextureSubresource *depthStencilTextureSubresource;
    D3D12GraphicsPipeline *currentGraphicsPipeline;
    D3D12ComputePipeline *currentComputePipeline;

    // Set at acquire time
    D3D12DescriptorHeap *gpuDescriptorHeaps[D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER + 1];

    D3D12UniformBuffer **usedUniformBuffers;
    Uint32 usedUniformBufferCount;
    Uint32 usedUniformBufferCapacity;

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

    D3D12Buffer *vertexBuffers[MAX_VERTEX_BUFFERS];
    Uint32 vertexBufferOffsets[MAX_VERTEX_BUFFERS];
    Uint32 vertexBufferCount;

    D3D12Texture *vertexSamplerTextures[MAX_TEXTURE_SAMPLERS_PER_STAGE];
    D3D12Sampler *vertexSamplers[MAX_TEXTURE_SAMPLERS_PER_STAGE];
    D3D12Texture *vertexStorageTextures[MAX_STORAGE_TEXTURES_PER_STAGE];
    D3D12Buffer *vertexStorageBuffers[MAX_STORAGE_BUFFERS_PER_STAGE];
    D3D12UniformBuffer *vertexUniformBuffers[MAX_UNIFORM_BUFFERS_PER_STAGE];

    D3D12Texture *fragmentSamplerTextures[MAX_TEXTURE_SAMPLERS_PER_STAGE];
    D3D12Sampler *fragmentSamplers[MAX_TEXTURE_SAMPLERS_PER_STAGE];
    D3D12Texture *fragmentStorageTextures[MAX_STORAGE_TEXTURES_PER_STAGE];
    D3D12Buffer *fragmentStorageBuffers[MAX_STORAGE_BUFFERS_PER_STAGE];
    D3D12UniformBuffer *fragmentUniformBuffers[MAX_UNIFORM_BUFFERS_PER_STAGE];

    D3D12Texture *computeSamplerTextures[MAX_TEXTURE_SAMPLERS_PER_STAGE];
    D3D12Sampler *computeSamplers[MAX_TEXTURE_SAMPLERS_PER_STAGE];
    D3D12Texture *computeReadOnlyStorageTextures[MAX_STORAGE_TEXTURES_PER_STAGE];
    D3D12Buffer *computeReadOnlyStorageBuffers[MAX_STORAGE_BUFFERS_PER_STAGE];
    D3D12TextureSubresource *computeReadWriteStorageTextureSubresources[MAX_COMPUTE_WRITE_TEXTURES];
    Uint32 computeReadWriteStorageTextureSubresourceCount;
    D3D12Buffer *computeReadWriteStorageBuffers[MAX_COMPUTE_WRITE_BUFFERS];
    Uint32 computeReadWriteStorageBufferCount;
    D3D12UniformBuffer *computeUniformBuffers[MAX_UNIFORM_BUFFERS_PER_STAGE];

    // Resource tracking
    D3D12Texture **usedTextures;
    Uint32 usedTextureCount;
    Uint32 usedTextureCapacity;

    D3D12Buffer **usedBuffers;
    Uint32 usedBufferCount;
    Uint32 usedBufferCapacity;

    D3D12Sampler **usedSamplers;
    Uint32 usedSamplerCount;
    Uint32 usedSamplerCapacity;

    D3D12GraphicsPipeline **usedGraphicsPipelines;
    Uint32 usedGraphicsPipelineCount;
    Uint32 usedGraphicsPipelineCapacity;

    D3D12ComputePipeline **usedComputePipelines;
    Uint32 usedComputePipelineCount;
    Uint32 usedComputePipelineCapacity;

    // Used for texture pitch hack
    D3D12TextureDownload **textureDownloads;
    Uint32 textureDownloadCount;
    Uint32 textureDownloadCapacity;
};

struct D3D12Shader
{
    // todo cleanup
    void *bytecode;
    size_t bytecodeSize;

    SDL_GPUShaderStage stage;
    Uint32 num_samplers;
    Uint32 numUniformBuffers;
    Uint32 numStorageBuffers;
    Uint32 numStorageTextures;
};

typedef struct D3D12GraphicsRootSignature
{
    ID3D12RootSignature *handle;

    Sint32 vertexSamplerRootIndex;
    Sint32 vertexSamplerTextureRootIndex;
    Sint32 vertexStorageTextureRootIndex;
    Sint32 vertexStorageBufferRootIndex;

    Sint32 vertexUniformBufferRootIndex[MAX_UNIFORM_BUFFERS_PER_STAGE];

    Sint32 fragmentSamplerRootIndex;
    Sint32 fragmentSamplerTextureRootIndex;
    Sint32 fragmentStorageTextureRootIndex;
    Sint32 fragmentStorageBufferRootIndex;

    Sint32 fragmentUniformBufferRootIndex[MAX_UNIFORM_BUFFERS_PER_STAGE];
} D3D12GraphicsRootSignature;

struct D3D12GraphicsPipeline
{
    ID3D12PipelineState *pipelineState;
    D3D12GraphicsRootSignature *rootSignature;
    SDL_GPUPrimitiveType primitiveType;

    Uint32 vertexStrides[MAX_VERTEX_BUFFERS];

    Uint32 vertexSamplerCount;
    Uint32 vertexUniformBufferCount;
    Uint32 vertexStorageBufferCount;
    Uint32 vertexStorageTextureCount;

    Uint32 fragmentSamplerCount;
    Uint32 fragmentUniformBufferCount;
    Uint32 fragmentStorageBufferCount;
    Uint32 fragmentStorageTextureCount;

    SDL_AtomicInt referenceCount;
};

typedef struct D3D12ComputeRootSignature
{
    ID3D12RootSignature *handle;

    Sint32 samplerRootIndex;
    Sint32 samplerTextureRootIndex;
    Sint32 readOnlyStorageTextureRootIndex;
    Sint32 readOnlyStorageBufferRootIndex;
    Sint32 readWriteStorageTextureRootIndex;
    Sint32 readWriteStorageBufferRootIndex;
    Sint32 uniformBufferRootIndex[MAX_UNIFORM_BUFFERS_PER_STAGE];
} D3D12ComputeRootSignature;

struct D3D12ComputePipeline
{
    ID3D12PipelineState *pipelineState;
    D3D12ComputeRootSignature *rootSignature;

    Uint32 numSamplers;
    Uint32 numReadOnlyStorageTextures;
    Uint32 numReadOnlyStorageBuffers;
    Uint32 numReadWriteStorageTextures;
    Uint32 numReadWriteStorageBuffers;
    Uint32 numUniformBuffers;

    SDL_AtomicInt referenceCount;
};

struct D3D12TextureDownload
{
    D3D12Buffer *destinationBuffer;
    D3D12Buffer *temporaryBuffer;
    Uint32 width;
    Uint32 height;
    Uint32 depth;
    Uint32 bufferOffset;
    Uint32 bytesPerRow;
    Uint32 bytesPerDepthSlice;
    Uint32 alignedBytesPerRow;
};

struct D3D12Buffer
{
    D3D12BufferContainer *container;
    Uint32 containerIndex;

    ID3D12Resource *handle;
    D3D12StagingDescriptor uavDescriptor;
    D3D12StagingDescriptor srvDescriptor;
    D3D12StagingDescriptor cbvDescriptor;
    D3D12_GPU_VIRTUAL_ADDRESS virtualAddress;
    Uint8 *mapPointer; // NULL except for upload buffers and fast uniform buffers
    SDL_AtomicInt referenceCount;
    bool transitioned; // used for initial resource barrier
};

struct D3D12BufferContainer
{
    SDL_GPUBufferUsageFlags usage;
    Uint32 size;
    D3D12BufferType type;

    D3D12Buffer *activeBuffer;

    D3D12Buffer **buffers;
    Uint32 bufferCapacity;
    Uint32 bufferCount;

    D3D12_RESOURCE_DESC bufferDesc;

    char *debugName;
};

struct D3D12UniformBuffer
{
    D3D12Buffer *buffer;
    Uint32 writeOffset;
    Uint32 drawOffset;
    Uint32 currentBlockSize;
};

// Forward function declarations

static void D3D12_ReleaseWindow(SDL_GPURenderer *driverData, SDL_Window *window);
static bool D3D12_Wait(SDL_GPURenderer *driverData);
static bool D3D12_WaitForFences(SDL_GPURenderer *driverData, bool waitAll, SDL_GPUFence *const *fences, Uint32 numFences);
static void D3D12_INTERNAL_ReleaseBlitPipelines(SDL_GPURenderer *driverData);

// Helpers

static Uint32 D3D12_INTERNAL_Align(Uint32 location, Uint32 alignment)
{
    return (location + (alignment - 1)) & ~(alignment - 1);
}

// Xbox Hack

#if (defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES))
// FIXME: This is purely to work around a presentation bug when recreating the device/command queue.
static ID3D12Device *s_Device;
static ID3D12CommandQueue *s_CommandQueue;
#endif

#if defined(SDL_PLATFORM_XBOXONE)
// These are not defined in d3d12_x.h.
typedef HRESULT (D3DAPI* PFN_D3D12_XBOX_CREATE_DEVICE)(_In_opt_ IGraphicsUnknown*, _In_ const D3D12XBOX_CREATE_DEVICE_PARAMETERS*, _In_ REFIID, _Outptr_opt_ void**);
#define D3D12_STANDARD_MULTISAMPLE_PATTERN DXGI_STANDARD_MULTISAMPLE_QUALITY_PATTERN
#endif

// Logging

static void D3D12_INTERNAL_SetError(
    D3D12Renderer *renderer,
    const char *msg,
    HRESULT res)
{
    #define MAX_ERROR_LEN 1024 // FIXME: Arbitrary!

    // Buffer for text, ensure space for \0 terminator after buffer
    char wszMsgBuff[MAX_ERROR_LEN + 1];
    DWORD dwChars; // Number of chars returned.

    if (res == DXGI_ERROR_DEVICE_REMOVED) {
        if (renderer->device) {
            res = ID3D12Device_GetDeviceRemovedReason(renderer->device);
        }
    }

    // Try to get the message from the system errors.
    dwChars = FormatMessageA(
        FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL,
        res,
        0,
        wszMsgBuff,
        MAX_ERROR_LEN,
        NULL);

    // No message? Screw it, just post the code.
    if (dwChars == 0) {
        if (renderer->debug_mode) {
            SDL_LogError(SDL_LOG_CATEGORY_GPU, "%s! Error Code: " HRESULT_FMT, msg, res);
        }
        SDL_SetError("%s! Error Code: " HRESULT_FMT, msg, res);
        return;
    }

    // Ensure valid range
    dwChars = SDL_min(dwChars, MAX_ERROR_LEN);

    // Trim whitespace from tail of message
    while (dwChars > 0) {
        if (wszMsgBuff[dwChars - 1] <= ' ') {
            dwChars--;
        } else {
            break;
        }
    }

    // Ensure null-terminated string
    wszMsgBuff[dwChars] = '\0';

    if (renderer->debug_mode) {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "%s! Error Code: %s " HRESULT_FMT, msg, wszMsgBuff, res);
    }
    SDL_SetError("%s! Error Code: %s " HRESULT_FMT, msg, wszMsgBuff, res);
}

// Release / Cleanup

static void D3D12_INTERNAL_ReleaseStagingDescriptorHandle(
    D3D12StagingDescriptor *cpuDescriptor)
{
    D3D12StagingDescriptorPool *pool = cpuDescriptor->pool;

    if (pool != NULL) {
        SDL_LockMutex(pool->lock);
        SDL_memcpy(&pool->freeDescriptors[pool->freeDescriptorCount], cpuDescriptor, sizeof(D3D12StagingDescriptor));
        pool->freeDescriptorCount += 1;
        SDL_UnlockMutex(pool->lock);
    }
}

static void D3D12_INTERNAL_DestroyBuffer(
    D3D12Buffer *buffer)
{
    if (!buffer) {
        return;
    }

    if (buffer->mapPointer != NULL) {
        ID3D12Resource_Unmap(
            buffer->handle,
            0,
            NULL);
    }
    D3D12_INTERNAL_ReleaseStagingDescriptorHandle(
        &buffer->srvDescriptor);
    D3D12_INTERNAL_ReleaseStagingDescriptorHandle(
        &buffer->uavDescriptor);
    D3D12_INTERNAL_ReleaseStagingDescriptorHandle(
        &buffer->cbvDescriptor);

    if (buffer->handle) {
        ID3D12Resource_Release(buffer->handle);
    }
    SDL_free(buffer);
}

static void D3D12_INTERNAL_ReleaseBuffer(
    D3D12Renderer *renderer,
    D3D12Buffer *buffer)
{
    SDL_LockMutex(renderer->disposeLock);

    EXPAND_ARRAY_IF_NEEDED(
        renderer->buffersToDestroy,
        D3D12Buffer *,
        renderer->buffersToDestroyCount + 1,
        renderer->buffersToDestroyCapacity,
        renderer->buffersToDestroyCapacity * 2);

    renderer->buffersToDestroy[renderer->buffersToDestroyCount] = buffer;
    renderer->buffersToDestroyCount += 1;

    SDL_UnlockMutex(renderer->disposeLock);
}

static void D3D12_INTERNAL_ReleaseBufferContainer(
    D3D12Renderer *renderer,
    D3D12BufferContainer *container)
{
    SDL_LockMutex(renderer->disposeLock);

    for (Uint32 i = 0; i < container->bufferCount; i += 1) {
        D3D12_INTERNAL_ReleaseBuffer(
            renderer,
            container->buffers[i]);
    }

    // Containers are just client handles, so we can free immediately
    if (container->debugName) {
        SDL_free(container->debugName);
    }
    SDL_free(container->buffers);
    SDL_free(container);

    SDL_UnlockMutex(renderer->disposeLock);
}

static void D3D12_INTERNAL_DestroyTexture(
    D3D12Texture *texture)
{
    if (!texture) {
        return;
    }
    for (Uint32 i = 0; i < texture->subresourceCount; i += 1) {
        D3D12TextureSubresource *subresource = &texture->subresources[i];
        if (subresource->rtvHandles) {
            for (Uint32 depthIndex = 0; depthIndex < subresource->depth; depthIndex += 1) {
                D3D12_INTERNAL_ReleaseStagingDescriptorHandle(
                    &subresource->rtvHandles[depthIndex]);
            }
            SDL_free(subresource->rtvHandles);
        }

        D3D12_INTERNAL_ReleaseStagingDescriptorHandle(
            &subresource->uavHandle);

        D3D12_INTERNAL_ReleaseStagingDescriptorHandle(
            &subresource->dsvHandle);
    }
    SDL_free(texture->subresources);

    D3D12_INTERNAL_ReleaseStagingDescriptorHandle(
        &texture->srvHandle);

    if (texture->resource) {
        ID3D12Resource_Release(texture->resource);
    }

    SDL_free(texture);
}

static void D3D12_INTERNAL_ReleaseTexture(
    D3D12Renderer *renderer,
    D3D12Texture *texture)
{
    SDL_LockMutex(renderer->disposeLock);

    EXPAND_ARRAY_IF_NEEDED(
        renderer->texturesToDestroy,
        D3D12Texture *,
        renderer->texturesToDestroyCount + 1,
        renderer->texturesToDestroyCapacity,
        renderer->texturesToDestroyCapacity * 2);

    renderer->texturesToDestroy[renderer->texturesToDestroyCount] = texture;
    renderer->texturesToDestroyCount += 1;

    SDL_UnlockMutex(renderer->disposeLock);
}

static void D3D12_INTERNAL_ReleaseTextureContainer(
    D3D12Renderer *renderer,
    D3D12TextureContainer *container)
{
    SDL_LockMutex(renderer->disposeLock);

    for (Uint32 i = 0; i < container->textureCount; i += 1) {
        D3D12_INTERNAL_ReleaseTexture(
            renderer,
            container->textures[i]);
    }

    // Containers are just client handles, so we can destroy immediately
    if (container->debugName) {
        SDL_free(container->debugName);
    }
    SDL_free(container->textures);
    SDL_free(container);

    SDL_UnlockMutex(renderer->disposeLock);
}

static void D3D12_INTERNAL_DestroySampler(
    D3D12Sampler *sampler)
{
    D3D12_INTERNAL_ReleaseStagingDescriptorHandle(
        &sampler->handle);

    SDL_free(sampler);
}

static void D3D12_INTERNAL_DestroyGraphicsRootSignature(
    D3D12GraphicsRootSignature *rootSignature)
{
    if (!rootSignature) {
        return;
    }
    if (rootSignature->handle) {
        ID3D12RootSignature_Release(rootSignature->handle);
    }
    SDL_free(rootSignature);
}

static void D3D12_INTERNAL_DestroyGraphicsPipeline(
    D3D12GraphicsPipeline *graphicsPipeline)
{
    if (graphicsPipeline->pipelineState) {
        ID3D12PipelineState_Release(graphicsPipeline->pipelineState);
    }
    D3D12_INTERNAL_DestroyGraphicsRootSignature(graphicsPipeline->rootSignature);
    SDL_free(graphicsPipeline);
}

static void D3D12_INTERNAL_DestroyComputeRootSignature(
    D3D12ComputeRootSignature *rootSignature)
{
    if (!rootSignature) {
        return;
    }
    if (rootSignature->handle) {
        ID3D12RootSignature_Release(rootSignature->handle);
    }
    SDL_free(rootSignature);
}

static void D3D12_INTERNAL_DestroyComputePipeline(
    D3D12ComputePipeline *computePipeline)
{
    if (computePipeline->pipelineState) {
        ID3D12PipelineState_Release(computePipeline->pipelineState);
    }
    D3D12_INTERNAL_DestroyComputeRootSignature(computePipeline->rootSignature);
    SDL_free(computePipeline);
}

static void D3D12_INTERNAL_ReleaseFenceToPool(
    D3D12Renderer *renderer,
    D3D12Fence *fence)
{
    SDL_LockMutex(renderer->fenceLock);

    EXPAND_ARRAY_IF_NEEDED(
        renderer->availableFences,
        D3D12Fence *,
        renderer->availableFenceCount + 1,
        renderer->availableFenceCapacity,
        renderer->availableFenceCapacity * 2);

    renderer->availableFences[renderer->availableFenceCount] = fence;
    renderer->availableFenceCount += 1;

    SDL_UnlockMutex(renderer->fenceLock);
}

static void D3D12_ReleaseFence(
    SDL_GPURenderer *driverData,
    SDL_GPUFence *fence)
{
    D3D12Fence *d3d12Fence = (D3D12Fence *)fence;

    if (SDL_AtomicDecRef(&d3d12Fence->referenceCount)) {
        D3D12_INTERNAL_ReleaseFenceToPool(
            (D3D12Renderer *)driverData,
            d3d12Fence);
    }
}

static bool D3D12_QueryFence(
    SDL_GPURenderer *driverData,
    SDL_GPUFence *fence)
{
    D3D12Fence *d3d12Fence = (D3D12Fence *)fence;
    return ID3D12Fence_GetCompletedValue(d3d12Fence->handle) == D3D12_FENCE_SIGNAL_VALUE;
}

static void D3D12_INTERNAL_DestroyDescriptorHeap(D3D12DescriptorHeap *descriptorHeap)
{
    if (!descriptorHeap) {
        return;
    }
    if (descriptorHeap->handle) {
        ID3D12DescriptorHeap_Release(descriptorHeap->handle);
    }
    SDL_free(descriptorHeap);
}

static void D3D12_INTERNAL_DestroyStagingDescriptorPool(
    D3D12StagingDescriptorPool *pool)
{
    for (Uint32 i = 0; i < pool->heapCount; i += 1) {
        D3D12_INTERNAL_DestroyDescriptorHeap(pool->heaps[i]);
    }
    SDL_free(pool->heaps);
    SDL_free(pool->freeDescriptors);
    SDL_DestroyMutex(pool->lock);

    SDL_free(pool);
}

static void D3D12_INTERNAL_DestroyCommandBuffer(D3D12CommandBuffer *commandBuffer)
{
    if (!commandBuffer) {
        return;
    }
    if (commandBuffer->graphicsCommandList) {
        ID3D12GraphicsCommandList_Release(commandBuffer->graphicsCommandList);
    }
    if (commandBuffer->commandAllocator) {
        ID3D12CommandAllocator_Release(commandBuffer->commandAllocator);
    }
    SDL_free(commandBuffer->presentDatas);
    SDL_free(commandBuffer->usedTextures);
    SDL_free(commandBuffer->usedBuffers);
    SDL_free(commandBuffer->usedSamplers);
    SDL_free(commandBuffer->usedGraphicsPipelines);
    SDL_free(commandBuffer->usedComputePipelines);
    SDL_free(commandBuffer->usedUniformBuffers);
    SDL_free(commandBuffer->textureDownloads);
    SDL_free(commandBuffer);
}

static void D3D12_INTERNAL_DestroyFence(D3D12Fence *fence)
{
    if (!fence) {
        return;
    }
    if (fence->handle) {
        ID3D12Fence_Release(fence->handle);
    }
    if (fence->event) {
        CloseHandle(fence->event);
    }
    SDL_free(fence);
}

static void D3D12_INTERNAL_DestroyRenderer(D3D12Renderer *renderer)
{
    // Release uniform buffers
    for (Uint32 i = 0; i < renderer->uniformBufferPoolCount; i += 1) {
        D3D12_INTERNAL_DestroyBuffer(
            renderer->uniformBufferPool[i]->buffer);
        SDL_free(renderer->uniformBufferPool[i]);
    }

    // Clean up descriptor heaps
    for (Uint32 i = 0; i < D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES; i += 1) {
        if (renderer->stagingDescriptorPools[i]) {
            D3D12_INTERNAL_DestroyStagingDescriptorPool(renderer->stagingDescriptorPools[i]);
            renderer->stagingDescriptorPools[i] = NULL;
        }
    }

    for (Uint32 i = 0; i < 2; i += 1) {
        if (renderer->gpuDescriptorHeapPools[i].heaps) {
            for (Uint32 j = 0; j < renderer->gpuDescriptorHeapPools[i].count; j += 1) {
                if (renderer->gpuDescriptorHeapPools[i].heaps[j]) {
                    D3D12_INTERNAL_DestroyDescriptorHeap(renderer->gpuDescriptorHeapPools[i].heaps[j]);
                    renderer->gpuDescriptorHeapPools[i].heaps[j] = NULL;
                }
            }
            SDL_free(renderer->gpuDescriptorHeapPools[i].heaps);
        }
        if (renderer->gpuDescriptorHeapPools[i].lock) {
            SDL_DestroyMutex(renderer->gpuDescriptorHeapPools[i].lock);
            renderer->gpuDescriptorHeapPools[i].lock = NULL;
        }
    }

    // Release command buffers
    for (Uint32 i = 0; i < renderer->availableCommandBufferCount; i += 1) {
        if (renderer->availableCommandBuffers[i]) {
            D3D12_INTERNAL_DestroyCommandBuffer(renderer->availableCommandBuffers[i]);
            renderer->availableCommandBuffers[i] = NULL;
        }
    }

    // Release fences
    for (Uint32 i = 0; i < renderer->availableFenceCount; i += 1) {
        if (renderer->availableFences[i]) {
            D3D12_INTERNAL_DestroyFence(renderer->availableFences[i]);
            renderer->availableFences[i] = NULL;
        }
    }

    // Clean up allocations
    SDL_free(renderer->availableCommandBuffers);
    SDL_free(renderer->submittedCommandBuffers);
    SDL_free(renderer->uniformBufferPool);
    SDL_free(renderer->claimedWindows);
    SDL_free(renderer->availableFences);
    SDL_free(renderer->buffersToDestroy);
    SDL_free(renderer->texturesToDestroy);
    SDL_free(renderer->samplersToDestroy);
    SDL_free(renderer->graphicsPipelinesToDestroy);
    SDL_free(renderer->computePipelinesToDestroy);

    SDL_DestroyProperties(renderer->props);

    // Tear down D3D12 objects
    if (renderer->indirectDrawCommandSignature) {
        ID3D12CommandSignature_Release(renderer->indirectDrawCommandSignature);
        renderer->indirectDrawCommandSignature = NULL;
    }
    if (renderer->indirectIndexedDrawCommandSignature) {
        ID3D12CommandSignature_Release(renderer->indirectIndexedDrawCommandSignature);
        renderer->indirectIndexedDrawCommandSignature = NULL;
    }
    if (renderer->indirectDispatchCommandSignature) {
        ID3D12CommandSignature_Release(renderer->indirectDispatchCommandSignature);
        renderer->indirectDispatchCommandSignature = NULL;
    }
#if !(defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES))
    if (renderer->commandQueue) {
        ID3D12CommandQueue_Release(renderer->commandQueue);
        renderer->commandQueue = NULL;
    }
    if (renderer->device) {
        ID3D12Device_Release(renderer->device);
        renderer->device = NULL;
    }
    if (renderer->adapter) {
        IDXGIAdapter1_Release(renderer->adapter);
        renderer->adapter = NULL;
    }
    if (renderer->factory) {
        IDXGIFactory4_Release(renderer->factory);
        renderer->factory = NULL;
    }
    if (renderer->dxgiDebug) {
        IDXGIDebug_ReportLiveObjects(
            renderer->dxgiDebug,
            D3D_IID_DXGI_DEBUG_ALL,
            (DXGI_DEBUG_RLO_FLAGS)(DXGI_DEBUG_RLO_SUMMARY | DXGI_DEBUG_RLO_DETAIL));
        IDXGIDebug_Release(renderer->dxgiDebug);
        renderer->dxgiDebug = NULL;
    }
#endif
    if (renderer->d3d12_dll) {
        SDL_UnloadObject(renderer->d3d12_dll);
        renderer->d3d12_dll = NULL;
    }
#if !(defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES))
    if (renderer->dxgi_dll) {
        SDL_UnloadObject(renderer->dxgi_dll);
        renderer->dxgi_dll = NULL;
    }
    if (renderer->dxgidebug_dll) {
        SDL_UnloadObject(renderer->dxgidebug_dll);
        renderer->dxgidebug_dll = NULL;
    }
#endif
    renderer->D3D12SerializeRootSignature_func = NULL;

    if (renderer->iconv) {
        SDL_iconv_close(renderer->iconv);
    }

    SDL_DestroyMutex(renderer->acquireCommandBufferLock);
    SDL_DestroyMutex(renderer->acquireUniformBufferLock);
    SDL_DestroyMutex(renderer->submitLock);
    SDL_DestroyMutex(renderer->windowLock);
    SDL_DestroyMutex(renderer->fenceLock);
    SDL_DestroyMutex(renderer->disposeLock);
    SDL_free(renderer);
}

static void D3D12_DestroyDevice(SDL_GPUDevice *device)
{
    D3D12Renderer *renderer = (D3D12Renderer *)device->driverData;

    // Release blit pipeline structures
    D3D12_INTERNAL_ReleaseBlitPipelines((SDL_GPURenderer *)renderer);

    // Flush any remaining GPU work...
    D3D12_Wait((SDL_GPURenderer *)renderer);

    // Release window data
    for (Sint32 i = renderer->claimedWindowCount - 1; i >= 0; i -= 1) {
        D3D12_ReleaseWindow((SDL_GPURenderer *)renderer, renderer->claimedWindows[i]->window);
    }

    D3D12_INTERNAL_DestroyRenderer(renderer);
    SDL_free(device);
}

static SDL_PropertiesID D3D12_GetDeviceProperties(SDL_GPUDevice *device)
{
    D3D12Renderer *renderer = (D3D12Renderer *)device->driverData;
    return renderer->props;
}

// Barriers

static inline Uint32 D3D12_INTERNAL_CalcSubresource(
    Uint32 mipLevel,
    Uint32 layer,
    Uint32 numLevels)
{
    return mipLevel + (layer * numLevels);
}

static void D3D12_INTERNAL_ResourceBarrier(
    D3D12CommandBuffer *commandBuffer,
    D3D12_RESOURCE_STATES sourceState,
    D3D12_RESOURCE_STATES destinationState,
    ID3D12Resource *resource,
    Uint32 subresourceIndex,
    bool needsUavBarrier)
{
    D3D12_RESOURCE_BARRIER barrierDesc[2];
    Uint32 numBarriers = 0;

    // No transition barrier is needed if the state is not changing.
    if (sourceState != destinationState) {
        barrierDesc[numBarriers].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrierDesc[numBarriers].Flags = (D3D12_RESOURCE_BARRIER_FLAGS)0;
        barrierDesc[numBarriers].Transition.StateBefore = sourceState;
        barrierDesc[numBarriers].Transition.StateAfter = destinationState;
        barrierDesc[numBarriers].Transition.pResource = resource;
        barrierDesc[numBarriers].Transition.Subresource = subresourceIndex;

        numBarriers += 1;
    }

    if (needsUavBarrier) {
        barrierDesc[numBarriers].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        barrierDesc[numBarriers].Flags = (D3D12_RESOURCE_BARRIER_FLAGS)0;
        barrierDesc[numBarriers].UAV.pResource = resource;

        numBarriers += 1;
    }

    if (numBarriers > 0) {
        ID3D12GraphicsCommandList_ResourceBarrier(
            commandBuffer->graphicsCommandList,
            numBarriers,
            barrierDesc);
    }
}

static void D3D12_INTERNAL_TextureSubresourceBarrier(
    D3D12CommandBuffer *commandBuffer,
    D3D12_RESOURCE_STATES sourceState,
    D3D12_RESOURCE_STATES destinationState,
    D3D12TextureSubresource *textureSubresource)
{
    bool needsUAVBarrier =
        (textureSubresource->parent->container->header.info.usage & SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE) ||
        (textureSubresource->parent->container->header.info.usage & SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_SIMULTANEOUS_READ_WRITE);

    D3D12_INTERNAL_ResourceBarrier(
        commandBuffer,
        sourceState,
        destinationState,
        textureSubresource->parent->resource,
        textureSubresource->index,
        needsUAVBarrier);
}

static D3D12_RESOURCE_STATES D3D12_INTERNAL_DefaultTextureResourceState(
    SDL_GPUTextureUsageFlags usageFlags)
{
    // NOTE: order matters here!

    if (usageFlags & SDL_GPU_TEXTUREUSAGE_SAMPLER) {
        return D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE;
    } else if (usageFlags & SDL_GPU_TEXTUREUSAGE_GRAPHICS_STORAGE_READ) {
        return D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE;
    } else if (usageFlags & SDL_GPU_TEXTUREUSAGE_COLOR_TARGET) {
        return D3D12_RESOURCE_STATE_RENDER_TARGET;
    } else if (usageFlags & SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET) {
        return D3D12_RESOURCE_STATE_DEPTH_WRITE;
    } else if (usageFlags & SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_READ) {
        return D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    } else if (usageFlags & SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE) {
        return D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    } else if (usageFlags & SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_SIMULTANEOUS_READ_WRITE) {
        return D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    } else {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Texture has no default usage mode!");
        return D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE;
    }
}

static void D3D12_INTERNAL_TextureSubresourceTransitionFromDefaultUsage(
    D3D12CommandBuffer *commandBuffer,
    D3D12_RESOURCE_STATES destinationUsageMode,
    D3D12TextureSubresource *textureSubresource)
{
    D3D12_INTERNAL_TextureSubresourceBarrier(
        commandBuffer,
        D3D12_INTERNAL_DefaultTextureResourceState(textureSubresource->parent->container->header.info.usage),
        destinationUsageMode,
        textureSubresource);
}

static void D3D12_INTERNAL_TextureTransitionFromDefaultUsage(
    D3D12CommandBuffer *commandBuffer,
    D3D12_RESOURCE_STATES destinationUsageMode,
    D3D12Texture *texture)
{
    for (Uint32 i = 0; i < texture->subresourceCount; i += 1) {
        D3D12_INTERNAL_TextureSubresourceTransitionFromDefaultUsage(
            commandBuffer,
            destinationUsageMode,
            &texture->subresources[i]);
    }
}

static void D3D12_INTERNAL_TextureSubresourceTransitionToDefaultUsage(
    D3D12CommandBuffer *commandBuffer,
    D3D12_RESOURCE_STATES sourceUsageMode,
    D3D12TextureSubresource *textureSubresource)
{
    D3D12_INTERNAL_TextureSubresourceBarrier(
        commandBuffer,
        sourceUsageMode,
        D3D12_INTERNAL_DefaultTextureResourceState(textureSubresource->parent->container->header.info.usage),
        textureSubresource);
}

static void D3D12_INTERNAL_TextureTransitionToDefaultUsage(
    D3D12CommandBuffer *commandBuffer,
    D3D12_RESOURCE_STATES sourceUsageMode,
    D3D12Texture *texture)
{
    for (Uint32 i = 0; i < texture->subresourceCount; i += 1) {
        D3D12_INTERNAL_TextureSubresourceTransitionToDefaultUsage(
            commandBuffer,
            sourceUsageMode,
            &texture->subresources[i]);
    }
}

static D3D12_RESOURCE_STATES D3D12_INTERNAL_DefaultBufferResourceState(
    D3D12Buffer *buffer)
{
    if (buffer->container->usage & SDL_GPU_BUFFERUSAGE_VERTEX) {
        return D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
    } else if (buffer->container->usage & SDL_GPU_BUFFERUSAGE_INDEX) {
        return D3D12_RESOURCE_STATE_INDEX_BUFFER;
    } else if (buffer->container->usage & SDL_GPU_BUFFERUSAGE_INDIRECT) {
        return D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
    } else if (buffer->container->usage & SDL_GPU_BUFFERUSAGE_GRAPHICS_STORAGE_READ) {
        return D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE;
    } else if (buffer->container->usage & SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ) {
        return D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    } else if (buffer->container->usage & SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE) {
        return D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    } else {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Buffer has no default usage mode!");
        return D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
    }
}

static void D3D12_INTERNAL_BufferBarrier(
    D3D12CommandBuffer *commandBuffer,
    D3D12_RESOURCE_STATES sourceState,
    D3D12_RESOURCE_STATES destinationState,
    D3D12Buffer *buffer)
{
    D3D12_INTERNAL_ResourceBarrier(
        commandBuffer,
        buffer->transitioned ? sourceState : D3D12_RESOURCE_STATE_COMMON,
        destinationState,
        buffer->handle,
        0,
        buffer->container->usage & SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE);

    buffer->transitioned = true;
}

static void D3D12_INTERNAL_BufferTransitionFromDefaultUsage(
    D3D12CommandBuffer *commandBuffer,
    D3D12_RESOURCE_STATES destinationState,
    D3D12Buffer *buffer)
{
    D3D12_INTERNAL_BufferBarrier(
        commandBuffer,
        D3D12_INTERNAL_DefaultBufferResourceState(buffer),
        destinationState,
        buffer);
}

static void D3D12_INTERNAL_BufferTransitionToDefaultUsage(
    D3D12CommandBuffer *commandBuffer,
    D3D12_RESOURCE_STATES sourceState,
    D3D12Buffer *buffer)
{
    D3D12_INTERNAL_BufferBarrier(
        commandBuffer,
        sourceState,
        D3D12_INTERNAL_DefaultBufferResourceState(buffer),
        buffer);
}

// Resource tracking

#define TRACK_RESOURCE(resource, type, array, count, capacity) \
    Uint32 i;                                                  \
                                                               \
    for (i = 0; i < commandBuffer->count; i += 1) {            \
        if (commandBuffer->array[i] == resource) {             \
            return;                                            \
        }                                                      \
    }                                                          \
                                                               \
    if (commandBuffer->count == commandBuffer->capacity) {     \
        commandBuffer->capacity += 1;                          \
        commandBuffer->array = (type *)SDL_realloc(            \
            commandBuffer->array,                              \
            commandBuffer->capacity * sizeof(type));           \
    }                                                          \
    commandBuffer->array[commandBuffer->count] = resource;     \
    commandBuffer->count += 1;                                 \
    SDL_AtomicIncRef(&resource->referenceCount);

static void D3D12_INTERNAL_TrackTexture(
    D3D12CommandBuffer *commandBuffer,
    D3D12Texture *texture)
{
    TRACK_RESOURCE(
        texture,
        D3D12Texture *,
        usedTextures,
        usedTextureCount,
        usedTextureCapacity)
}

static void D3D12_INTERNAL_TrackBuffer(
    D3D12CommandBuffer *commandBuffer,
    D3D12Buffer *buffer)
{
    TRACK_RESOURCE(
        buffer,
        D3D12Buffer *,
        usedBuffers,
        usedBufferCount,
        usedBufferCapacity)
}

static void D3D12_INTERNAL_TrackSampler(
    D3D12CommandBuffer *commandBuffer,
    D3D12Sampler *sampler)
{
    TRACK_RESOURCE(
        sampler,
        D3D12Sampler *,
        usedSamplers,
        usedSamplerCount,
        usedSamplerCapacity)
}

static void D3D12_INTERNAL_TrackGraphicsPipeline(
    D3D12CommandBuffer *commandBuffer,
    D3D12GraphicsPipeline *graphicsPipeline)
{
    TRACK_RESOURCE(
        graphicsPipeline,
        D3D12GraphicsPipeline *,
        usedGraphicsPipelines,
        usedGraphicsPipelineCount,
        usedGraphicsPipelineCapacity)
}

static void D3D12_INTERNAL_TrackComputePipeline(
    D3D12CommandBuffer *commandBuffer,
    D3D12ComputePipeline *computePipeline)
{
    TRACK_RESOURCE(
        computePipeline,
        D3D12ComputePipeline *,
        usedComputePipelines,
        usedComputePipelineCount,
        usedComputePipelineCapacity)
}

#undef TRACK_RESOURCE

// Debug Naming

static void D3D12_INTERNAL_SetPipelineStateName(
    D3D12Renderer *renderer,
    ID3D12PipelineState *pipelineState,
    const char *text
) {
    if (renderer->debug_mode && text != NULL) {
        WCHAR *wchar_text = WIN_UTF8ToStringW(text);
        ID3D12PipelineState_SetName(
            pipelineState,
            wchar_text);
        SDL_free(wchar_text);
    }
}

static void D3D12_INTERNAL_SetResourceName(
    D3D12Renderer *renderer,
    ID3D12Resource *resource,
    const char *text
) {
    if (renderer->debug_mode && text != NULL) {
        WCHAR *wchar_text = WIN_UTF8ToStringW(text);
        ID3D12Resource_SetName(
            resource,
            wchar_text);
        SDL_free(wchar_text);
    }
}

static void D3D12_SetBufferName(
    SDL_GPURenderer *driverData,
    SDL_GPUBuffer *buffer,
    const char *text)
{
    D3D12Renderer *renderer = (D3D12Renderer *)driverData;
    D3D12BufferContainer *container = (D3D12BufferContainer *)buffer;

    if (renderer->debug_mode && text != NULL) {
        if (container->debugName != NULL) {
            SDL_free(container->debugName);
        }

        container->debugName = SDL_strdup(text);

        for (Uint32 i = 0; i < container->bufferCount; i += 1) {
            D3D12_INTERNAL_SetResourceName(
                renderer,
                container->buffers[i]->handle,
                text);
        }
    }
}

static void D3D12_SetTextureName(
    SDL_GPURenderer *driverData,
    SDL_GPUTexture *texture,
    const char *text)
{
    D3D12Renderer *renderer = (D3D12Renderer *)driverData;
    D3D12TextureContainer *container = (D3D12TextureContainer *)texture;

    if (renderer->debug_mode && text != NULL) {
        if (container->debugName != NULL) {
            SDL_free(container->debugName);
        }

        container->debugName = SDL_strdup(text);

        for (Uint32 i = 0; i < container->textureCount; i += 1) {
            D3D12_INTERNAL_SetResourceName(
                renderer,
                container->textures[i]->resource,
                text);
        }
    }
}

/* These debug functions are all marked as "for internal usage only"
 * on D3D12... works on renderdoc!
 */

static void D3D12_InsertDebugLabel(
    SDL_GPUCommandBuffer *commandBuffer,
    const char *text)
{
    D3D12CommandBuffer *d3d12CommandBuffer = (D3D12CommandBuffer *)commandBuffer;
    WCHAR *wchar_text = WIN_UTF8ToStringW(text);

    ID3D12GraphicsCommandList_SetMarker(
        d3d12CommandBuffer->graphicsCommandList,
        0,
        wchar_text,
        (UINT)SDL_wcslen(wchar_text) * sizeof(WCHAR));

    SDL_free(wchar_text);
}

static void D3D12_PushDebugGroup(
    SDL_GPUCommandBuffer *commandBuffer,
    const char *name)
{
    D3D12CommandBuffer *d3d12CommandBuffer = (D3D12CommandBuffer *)commandBuffer;
    WCHAR *wchar_text = WIN_UTF8ToStringW(name);

    ID3D12GraphicsCommandList_BeginEvent(
        d3d12CommandBuffer->graphicsCommandList,
        0,
        wchar_text,
        (UINT)SDL_wcslen(wchar_text) * sizeof(WCHAR));

    SDL_free(wchar_text);
}

static void D3D12_PopDebugGroup(
    SDL_GPUCommandBuffer *commandBuffer)
{
    D3D12CommandBuffer *d3d12CommandBuffer = (D3D12CommandBuffer *)commandBuffer;
    ID3D12GraphicsCommandList_EndEvent(d3d12CommandBuffer->graphicsCommandList);
}

// State Creation

static D3D12DescriptorHeap *D3D12_INTERNAL_CreateDescriptorHeap(
    D3D12Renderer *renderer,
    D3D12_DESCRIPTOR_HEAP_TYPE type,
    Uint32 descriptorCount,
    bool staging)
{
    D3D12DescriptorHeap *heap;
    ID3D12DescriptorHeap *handle;
    D3D12_DESCRIPTOR_HEAP_DESC heapDesc;
    HRESULT res;

    heap = (D3D12DescriptorHeap *)SDL_calloc(1, sizeof(D3D12DescriptorHeap));
    if (!heap) {
        return NULL;
    }

    heap->currentDescriptorIndex = 0;

    heapDesc.NumDescriptors = descriptorCount;
    heapDesc.Type = type;
    heapDesc.Flags = staging ? D3D12_DESCRIPTOR_HEAP_FLAG_NONE : D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    heapDesc.NodeMask = 0;

    res = ID3D12Device_CreateDescriptorHeap(
        renderer->device,
        &heapDesc,
        D3D_GUID(D3D_IID_ID3D12DescriptorHeap),
        (void **)&handle);

    if (FAILED(res)) {
        D3D12_INTERNAL_SetError(renderer, "Failed to create descriptor heap!", res);
        D3D12_INTERNAL_DestroyDescriptorHeap(heap);
        return NULL;
    }

    heap->handle = handle;
    heap->heapType = type;
    heap->maxDescriptors = descriptorCount;
    heap->staging = staging;
    heap->descriptorSize = ID3D12Device_GetDescriptorHandleIncrementSize(renderer->device, type);
    D3D_CALL_RET(handle, GetCPUDescriptorHandleForHeapStart, &heap->descriptorHeapCPUStart);
    if (!staging) {
        D3D_CALL_RET(handle, GetGPUDescriptorHandleForHeapStart, &heap->descriptorHeapGPUStart);
    }

    return heap;
}

static D3D12StagingDescriptorPool *D3D12_INTERNAL_CreateStagingDescriptorPool(
    D3D12Renderer *renderer,
    D3D12_DESCRIPTOR_HEAP_TYPE heapType
) {
    D3D12DescriptorHeap *heap = D3D12_INTERNAL_CreateDescriptorHeap(
        renderer,
        heapType,
        STAGING_HEAP_DESCRIPTOR_COUNT,
        true);

    if (!heap) {
        return NULL;
    }

    D3D12StagingDescriptorPool *pool = (D3D12StagingDescriptorPool*) SDL_calloc(1, sizeof(D3D12StagingDescriptorPool));

    pool->heapCount = 1;
    pool->heaps = (D3D12DescriptorHeap**) SDL_malloc(sizeof(D3D12DescriptorHeap*));
    pool->heaps[0] = heap;

    pool->freeDescriptorCapacity = STAGING_HEAP_DESCRIPTOR_COUNT;
    pool->freeDescriptorCount = STAGING_HEAP_DESCRIPTOR_COUNT;
    pool->freeDescriptors = (D3D12StagingDescriptor*) SDL_malloc(STAGING_HEAP_DESCRIPTOR_COUNT * sizeof(D3D12StagingDescriptor));

    for (Uint32 i = 0; i < STAGING_HEAP_DESCRIPTOR_COUNT; i += 1) {
        pool->freeDescriptors[i].pool = pool;
        pool->freeDescriptors[i].heap = heap;
        pool->freeDescriptors[i].cpuHandleIndex = i;
        pool->freeDescriptors[i].cpuHandle.ptr = heap->descriptorHeapCPUStart.ptr + (i * heap->descriptorSize);
    }

    pool->lock = SDL_CreateMutex();

    return pool;
}

/* If the pool is empty, we need to refill it! */
static bool D3D12_INTERNAL_ExpandStagingDescriptorPool(
    D3D12Renderer *renderer,
    D3D12StagingDescriptorPool *pool
) {
    D3D12DescriptorHeap *heap = D3D12_INTERNAL_CreateDescriptorHeap(
        renderer,
        pool->heaps[0]->heapType,
        STAGING_HEAP_DESCRIPTOR_COUNT,
        true);

    if (!heap) {
        return false;
    }

    pool->heapCount += 1;
    pool->heaps = (D3D12DescriptorHeap**) SDL_realloc(pool->heaps, pool->heapCount * sizeof(D3D12DescriptorHeap*));
    pool->heaps[pool->heapCount - 1] = heap;

    pool->freeDescriptorCapacity += STAGING_HEAP_DESCRIPTOR_COUNT;
    pool->freeDescriptorCount += STAGING_HEAP_DESCRIPTOR_COUNT;
    pool->freeDescriptors = (D3D12StagingDescriptor*) SDL_realloc(pool->freeDescriptors, pool->freeDescriptorCapacity * sizeof(D3D12StagingDescriptor));

    for (Uint32 i = 0; i < STAGING_HEAP_DESCRIPTOR_COUNT; i += 1) {
        pool->freeDescriptors[i].pool = pool;
        pool->freeDescriptors[i].heap = heap;
        pool->freeDescriptors[i].cpuHandleIndex = i;
        pool->freeDescriptors[i].cpuHandle.ptr = heap->descriptorHeapCPUStart.ptr + (i * heap->descriptorSize);
    }

    return true;
}

static D3D12DescriptorHeap *D3D12_INTERNAL_AcquireGPUDescriptorHeapFromPool(
    D3D12CommandBuffer *commandBuffer,
    D3D12_DESCRIPTOR_HEAP_TYPE descriptorHeapType)
{
    D3D12DescriptorHeap *result;
    D3D12Renderer *renderer = commandBuffer->renderer;
    D3D12GPUDescriptorHeapPool *pool = &renderer->gpuDescriptorHeapPools[descriptorHeapType];

    SDL_LockMutex(pool->lock);
    if (pool->count > 0) {
        result = pool->heaps[pool->count - 1];
        pool->count -= 1;
    } else {
        result = D3D12_INTERNAL_CreateDescriptorHeap(
            renderer,
            descriptorHeapType,
            descriptorHeapType == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV ? VIEW_GPU_DESCRIPTOR_COUNT : SAMPLER_GPU_DESCRIPTOR_COUNT,
            false);
    }
    SDL_UnlockMutex(pool->lock);

    return result;
}

static void D3D12_INTERNAL_ReturnGPUDescriptorHeapToPool(
    D3D12Renderer *renderer,
    D3D12DescriptorHeap *heap)
{
    if (heap == NULL) {
        return;
    }

    D3D12GPUDescriptorHeapPool *pool = &renderer->gpuDescriptorHeapPools[heap->heapType];

    heap->currentDescriptorIndex = 0;

    SDL_LockMutex(pool->lock);
    if (pool->count >= pool->capacity) {
        pool->capacity *= 2;
        pool->heaps = (D3D12DescriptorHeap **)SDL_realloc(
            pool->heaps,
            pool->capacity * sizeof(D3D12DescriptorHeap *));
    }

    pool->heaps[pool->count] = heap;
    pool->count += 1;
    SDL_UnlockMutex(pool->lock);
}

/*
 * The root signature lets us define "root parameters" which are essentially bind points for resources.
 * These let us define the register ranges as well as the register "space".
 * The register space is akin to the descriptor set index in Vulkan, which allows us to group resources
 * by stage so that the registers from the vertex and fragment shaders don't clobber each other.
 *
 * Most of our root parameters are implemented as "descriptor tables" so we can
 * copy and then point to contiguous descriptor regions.
 * Uniform buffers are the exception - these have to be implemented as raw "root descriptors" so
 * that we can dynamically update the address that the constant buffer view points to.
 *
 * The root signature has a maximum size of 64 DWORDs.
 * A descriptor table uses 1 DWORD.
 * A root descriptor uses 2 DWORDS.
 * This means our biggest root signature uses 24 DWORDs total, well under the limit.
 *
 * The root parameter indices are created dynamically and stored in the D3D12GraphicsRootSignature struct.
 */
static D3D12GraphicsRootSignature *D3D12_INTERNAL_CreateGraphicsRootSignature(
    D3D12Renderer *renderer,
    D3D12Shader *vertexShader,
    D3D12Shader *fragmentShader)
{
    // FIXME: I think the max can be smaller...
    D3D12_ROOT_PARAMETER rootParameters[MAX_ROOT_SIGNATURE_PARAMETERS];
    D3D12_DESCRIPTOR_RANGE descriptorRanges[MAX_ROOT_SIGNATURE_PARAMETERS];
    Uint32 parameterCount = 0;
    Uint32 rangeCount = 0;
    D3D12_DESCRIPTOR_RANGE descriptorRange;
    D3D12_ROOT_PARAMETER rootParameter;
    D3D12GraphicsRootSignature *d3d12GraphicsRootSignature =
        (D3D12GraphicsRootSignature *)SDL_calloc(1, sizeof(D3D12GraphicsRootSignature));
    if (!d3d12GraphicsRootSignature) {
        return NULL;
    }

    SDL_zeroa(rootParameters);
    SDL_zeroa(descriptorRanges);
    SDL_zero(rootParameter);

    d3d12GraphicsRootSignature->vertexSamplerRootIndex = -1;
    d3d12GraphicsRootSignature->vertexSamplerTextureRootIndex = -1;
    d3d12GraphicsRootSignature->vertexStorageTextureRootIndex = -1;
    d3d12GraphicsRootSignature->vertexStorageBufferRootIndex = -1;

    d3d12GraphicsRootSignature->fragmentSamplerRootIndex = -1;
    d3d12GraphicsRootSignature->fragmentSamplerTextureRootIndex = -1;
    d3d12GraphicsRootSignature->fragmentStorageTextureRootIndex = -1;
    d3d12GraphicsRootSignature->fragmentStorageBufferRootIndex = -1;

    for (Uint32 i = 0; i < MAX_UNIFORM_BUFFERS_PER_STAGE; i += 1) {
        d3d12GraphicsRootSignature->vertexUniformBufferRootIndex[i] = -1;
        d3d12GraphicsRootSignature->fragmentUniformBufferRootIndex[i] = -1;
    }

    if (vertexShader->num_samplers > 0) {
        // Vertex Samplers
        descriptorRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
        descriptorRange.NumDescriptors = vertexShader->num_samplers;
        descriptorRange.BaseShaderRegister = 0;
        descriptorRange.RegisterSpace = 0;
        descriptorRange.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
        descriptorRanges[rangeCount] = descriptorRange;

        rootParameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        rootParameter.DescriptorTable.NumDescriptorRanges = 1;
        rootParameter.DescriptorTable.pDescriptorRanges = &descriptorRanges[rangeCount];
        rootParameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
        rootParameters[parameterCount] = rootParameter;
        d3d12GraphicsRootSignature->vertexSamplerRootIndex = parameterCount;
        rangeCount += 1;
        parameterCount += 1;

        descriptorRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        descriptorRange.NumDescriptors = vertexShader->num_samplers;
        descriptorRange.BaseShaderRegister = 0;
        descriptorRange.RegisterSpace = 0;
        descriptorRange.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
        descriptorRanges[rangeCount] = descriptorRange;

        rootParameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        rootParameter.DescriptorTable.NumDescriptorRanges = 1;
        rootParameter.DescriptorTable.pDescriptorRanges = &descriptorRanges[rangeCount];
        rootParameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
        rootParameters[parameterCount] = rootParameter;
        d3d12GraphicsRootSignature->vertexSamplerTextureRootIndex = parameterCount;
        rangeCount += 1;
        parameterCount += 1;
    }

    if (vertexShader->numStorageTextures) {
        // Vertex storage textures
        descriptorRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        descriptorRange.NumDescriptors = vertexShader->numStorageTextures;
        descriptorRange.BaseShaderRegister = vertexShader->num_samplers;
        descriptorRange.RegisterSpace = 0;
        descriptorRange.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
        descriptorRanges[rangeCount] = descriptorRange;

        rootParameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        rootParameter.DescriptorTable.NumDescriptorRanges = 1;
        rootParameter.DescriptorTable.pDescriptorRanges = &descriptorRanges[rangeCount];
        rootParameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
        rootParameters[parameterCount] = rootParameter;
        d3d12GraphicsRootSignature->vertexStorageTextureRootIndex = parameterCount;
        rangeCount += 1;
        parameterCount += 1;
    }

    if (vertexShader->numStorageBuffers) {

        // Vertex storage buffers
        descriptorRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        descriptorRange.NumDescriptors = vertexShader->numStorageBuffers;
        descriptorRange.BaseShaderRegister = vertexShader->num_samplers + vertexShader->numStorageTextures;
        descriptorRange.RegisterSpace = 0;
        descriptorRange.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
        descriptorRanges[rangeCount] = descriptorRange;

        rootParameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        rootParameter.DescriptorTable.NumDescriptorRanges = 1;
        rootParameter.DescriptorTable.pDescriptorRanges = &descriptorRanges[rangeCount];
        rootParameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
        rootParameters[parameterCount] = rootParameter;
        d3d12GraphicsRootSignature->vertexStorageBufferRootIndex = parameterCount;
        rangeCount += 1;
        parameterCount += 1;
    }

    // Vertex Uniforms
    for (Uint32 i = 0; i < vertexShader->numUniformBuffers; i += 1) {
        rootParameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        rootParameter.Descriptor.ShaderRegister = i;
        rootParameter.Descriptor.RegisterSpace = 1;
        rootParameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
        rootParameters[parameterCount] = rootParameter;
        d3d12GraphicsRootSignature->vertexUniformBufferRootIndex[i] = parameterCount;
        parameterCount += 1;
    }

    if (fragmentShader->num_samplers) {
        // Fragment Samplers
        descriptorRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
        descriptorRange.NumDescriptors = fragmentShader->num_samplers;
        descriptorRange.BaseShaderRegister = 0;
        descriptorRange.RegisterSpace = 2;
        descriptorRange.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
        descriptorRanges[rangeCount] = descriptorRange;

        rootParameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        rootParameter.DescriptorTable.NumDescriptorRanges = 1;
        rootParameter.DescriptorTable.pDescriptorRanges = &descriptorRanges[rangeCount];
        rootParameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        rootParameters[parameterCount] = rootParameter;
        d3d12GraphicsRootSignature->fragmentSamplerRootIndex = parameterCount;
        rangeCount += 1;
        parameterCount += 1;

        descriptorRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        descriptorRange.NumDescriptors = fragmentShader->num_samplers;
        descriptorRange.BaseShaderRegister = 0;
        descriptorRange.RegisterSpace = 2;
        descriptorRange.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
        descriptorRanges[rangeCount] = descriptorRange;

        rootParameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        rootParameter.DescriptorTable.NumDescriptorRanges = 1;
        rootParameter.DescriptorTable.pDescriptorRanges = &descriptorRanges[rangeCount];
        rootParameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        rootParameters[parameterCount] = rootParameter;
        d3d12GraphicsRootSignature->fragmentSamplerTextureRootIndex = parameterCount;
        rangeCount += 1;
        parameterCount += 1;
    }

    if (fragmentShader->numStorageTextures) {
        // Fragment Storage Textures
        descriptorRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        descriptorRange.NumDescriptors = fragmentShader->numStorageTextures;
        descriptorRange.BaseShaderRegister = fragmentShader->num_samplers;
        descriptorRange.RegisterSpace = 2;
        descriptorRange.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
        descriptorRanges[rangeCount] = descriptorRange;

        rootParameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        rootParameter.DescriptorTable.NumDescriptorRanges = 1;
        rootParameter.DescriptorTable.pDescriptorRanges = &descriptorRanges[rangeCount];
        rootParameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        rootParameters[parameterCount] = rootParameter;
        d3d12GraphicsRootSignature->fragmentStorageTextureRootIndex = parameterCount;
        rangeCount += 1;
        parameterCount += 1;
    }

    if (fragmentShader->numStorageBuffers) {
        // Fragment Storage Buffers
        descriptorRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        descriptorRange.NumDescriptors = fragmentShader->numStorageBuffers;
        descriptorRange.BaseShaderRegister = fragmentShader->num_samplers + fragmentShader->numStorageTextures;
        descriptorRange.RegisterSpace = 2;
        descriptorRange.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
        descriptorRanges[rangeCount] = descriptorRange;

        rootParameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        rootParameter.DescriptorTable.NumDescriptorRanges = 1;
        rootParameter.DescriptorTable.pDescriptorRanges = &descriptorRanges[rangeCount];
        rootParameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        rootParameters[parameterCount] = rootParameter;
        d3d12GraphicsRootSignature->fragmentStorageBufferRootIndex = parameterCount;
        rangeCount += 1;
        parameterCount += 1;
    }

    // Fragment Uniforms
    for (Uint32 i = 0; i < fragmentShader->numUniformBuffers; i += 1) {
        rootParameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        rootParameter.Descriptor.ShaderRegister = i;
        rootParameter.Descriptor.RegisterSpace = 3;
        rootParameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        rootParameters[parameterCount] = rootParameter;
        d3d12GraphicsRootSignature->fragmentUniformBufferRootIndex[i] = parameterCount;
        parameterCount += 1;
    }

    // FIXME: shouldn't have to assert here
    SDL_assert(parameterCount <= MAX_ROOT_SIGNATURE_PARAMETERS);
    SDL_assert(rangeCount <= MAX_ROOT_SIGNATURE_PARAMETERS);

    // Create the root signature description
    D3D12_ROOT_SIGNATURE_DESC rootSignatureDesc;
    rootSignatureDesc.NumParameters = parameterCount;
    rootSignatureDesc.pParameters = rootParameters;
    rootSignatureDesc.NumStaticSamplers = 0;
    rootSignatureDesc.pStaticSamplers = NULL;
    rootSignatureDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

    // Serialize the root signature
    ID3DBlob *serializedRootSignature;
    ID3DBlob *errorBlob;
    HRESULT res = renderer->D3D12SerializeRootSignature_func(&rootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, &serializedRootSignature, &errorBlob);

    if (FAILED(res)) {
        if (errorBlob) {
            SET_ERROR("Failed to serialize RootSignature: %s", (const char *)ID3D10Blob_GetBufferPointer(errorBlob));
            ID3D10Blob_Release(errorBlob);
        }
        D3D12_INTERNAL_DestroyGraphicsRootSignature(d3d12GraphicsRootSignature);
        return NULL;
    }

    // Create the root signature
    ID3D12RootSignature *rootSignature;

    res = ID3D12Device_CreateRootSignature(
        renderer->device,
        0,
        ID3D10Blob_GetBufferPointer(serializedRootSignature),
        ID3D10Blob_GetBufferSize(serializedRootSignature),
        D3D_GUID(D3D_IID_ID3D12RootSignature),
        (void **)&rootSignature);

    if (FAILED(res)) {
        if (errorBlob) {
            SET_ERROR("Failed to create RootSignature: %s", (const char *)ID3D10Blob_GetBufferPointer(errorBlob));
            ID3D10Blob_Release(errorBlob);
        }
        D3D12_INTERNAL_DestroyGraphicsRootSignature(d3d12GraphicsRootSignature);
        return NULL;
    }

    d3d12GraphicsRootSignature->handle = rootSignature;
    return d3d12GraphicsRootSignature;
}

static bool D3D12_INTERNAL_IsValidShaderBytecode(
    const Uint8 *code,
    size_t codeSize)
{
    // Both DXIL and DXBC bytecode have a 4 byte header containing `DXBC`.
    if (codeSize < 4 || code == NULL) {
        return false;
    }
    return SDL_memcmp(code, "DXBC", 4) == 0;
}

static bool D3D12_INTERNAL_CreateShaderBytecode(
    D3D12Renderer *renderer,
    const Uint8 *code,
    size_t codeSize,
    SDL_GPUShaderFormat format,
    void **pBytecode,
    size_t *pBytecodeSize)
{
    if (!D3D12_INTERNAL_IsValidShaderBytecode(code, codeSize)) {
        if (format == SDL_GPU_SHADERFORMAT_DXBC) {
            SET_STRING_ERROR_AND_RETURN("The provided shader code is not valid DXBC!", false);
        }
        SET_STRING_ERROR_AND_RETURN("The provided shader code is not valid DXIL!", false);
    }

    if (pBytecode != NULL) {
        *pBytecode = SDL_malloc(codeSize);
        if (!*pBytecode) {
            return false;
        }
        SDL_memcpy(*pBytecode, code, codeSize);
        *pBytecodeSize = codeSize;
    }

    return true;
}

static D3D12ComputeRootSignature *D3D12_INTERNAL_CreateComputeRootSignature(
    D3D12Renderer *renderer,
    const SDL_GPUComputePipelineCreateInfo *createInfo)
{
    // FIXME: I think the max can be smaller...
    D3D12_ROOT_PARAMETER rootParameters[MAX_ROOT_SIGNATURE_PARAMETERS];
    D3D12_DESCRIPTOR_RANGE descriptorRanges[MAX_ROOT_SIGNATURE_PARAMETERS];
    Uint32 parameterCount = 0;
    Uint32 rangeCount = 0;
    D3D12_DESCRIPTOR_RANGE descriptorRange;
    D3D12_ROOT_PARAMETER rootParameter;
    D3D12ComputeRootSignature *d3d12ComputeRootSignature =
        (D3D12ComputeRootSignature *)SDL_calloc(1, sizeof(D3D12ComputeRootSignature));
    if (!d3d12ComputeRootSignature) {
        return NULL;
    }

    SDL_zeroa(rootParameters);
    SDL_zeroa(descriptorRanges);
    SDL_zero(rootParameter);

    d3d12ComputeRootSignature->samplerRootIndex = -1;
    d3d12ComputeRootSignature->samplerTextureRootIndex = -1;
    d3d12ComputeRootSignature->readOnlyStorageTextureRootIndex = -1;
    d3d12ComputeRootSignature->readOnlyStorageBufferRootIndex = -1;
    d3d12ComputeRootSignature->readWriteStorageTextureRootIndex = -1;
    d3d12ComputeRootSignature->readWriteStorageBufferRootIndex = -1;

    for (Uint32 i = 0; i < MAX_UNIFORM_BUFFERS_PER_STAGE; i += 1) {
        d3d12ComputeRootSignature->uniformBufferRootIndex[i] = -1;
    }

    if (createInfo->num_samplers) {
        descriptorRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
        descriptorRange.NumDescriptors = createInfo->num_samplers;
        descriptorRange.BaseShaderRegister = 0;
        descriptorRange.RegisterSpace = 0;
        descriptorRange.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
        descriptorRanges[rangeCount] = descriptorRange;

        rootParameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        rootParameter.DescriptorTable.NumDescriptorRanges = 1;
        rootParameter.DescriptorTable.pDescriptorRanges = &descriptorRanges[rangeCount];
        rootParameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL; // ALL is used for compute
        rootParameters[parameterCount] = rootParameter;
        d3d12ComputeRootSignature->samplerRootIndex = parameterCount;
        rangeCount += 1;
        parameterCount += 1;

        descriptorRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        descriptorRange.NumDescriptors = createInfo->num_samplers;
        descriptorRange.BaseShaderRegister = 0;
        descriptorRange.RegisterSpace = 0;
        descriptorRange.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
        descriptorRanges[rangeCount] = descriptorRange;

        rootParameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        rootParameter.DescriptorTable.NumDescriptorRanges = 1;
        rootParameter.DescriptorTable.pDescriptorRanges = &descriptorRanges[rangeCount];
        rootParameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL; // ALL is used for compute
        rootParameters[parameterCount] = rootParameter;
        d3d12ComputeRootSignature->samplerTextureRootIndex = parameterCount;
        rangeCount += 1;
        parameterCount += 1;
    }

    if (createInfo->num_readonly_storage_textures) {
        descriptorRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        descriptorRange.NumDescriptors = createInfo->num_readonly_storage_textures;
        descriptorRange.BaseShaderRegister = createInfo->num_samplers;
        descriptorRange.RegisterSpace = 0;
        descriptorRange.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
        descriptorRanges[rangeCount] = descriptorRange;

        rootParameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        rootParameter.DescriptorTable.NumDescriptorRanges = 1;
        rootParameter.DescriptorTable.pDescriptorRanges = &descriptorRanges[rangeCount];
        rootParameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL; // ALL is used for compute
        rootParameters[parameterCount] = rootParameter;
        d3d12ComputeRootSignature->readOnlyStorageTextureRootIndex = parameterCount;
        rangeCount += 1;
        parameterCount += 1;
    }

    if (createInfo->num_readonly_storage_buffers) {
        descriptorRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        descriptorRange.NumDescriptors = createInfo->num_readonly_storage_buffers;
        descriptorRange.BaseShaderRegister = createInfo->num_samplers + createInfo->num_readonly_storage_textures;
        descriptorRange.RegisterSpace = 0;
        descriptorRange.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
        descriptorRanges[rangeCount] = descriptorRange;

        rootParameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        rootParameter.DescriptorTable.NumDescriptorRanges = 1;
        rootParameter.DescriptorTable.pDescriptorRanges = &descriptorRanges[rangeCount];
        rootParameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL; // ALL is used for compute
        rootParameters[parameterCount] = rootParameter;
        d3d12ComputeRootSignature->readOnlyStorageBufferRootIndex = parameterCount;
        rangeCount += 1;
        parameterCount += 1;
    }

    if (createInfo->num_readwrite_storage_textures) {
        descriptorRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        descriptorRange.NumDescriptors = createInfo->num_readwrite_storage_textures;
        descriptorRange.BaseShaderRegister = 0;
        descriptorRange.RegisterSpace = 1;
        descriptorRange.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
        descriptorRanges[rangeCount] = descriptorRange;

        rootParameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        rootParameter.DescriptorTable.NumDescriptorRanges = 1;
        rootParameter.DescriptorTable.pDescriptorRanges = &descriptorRanges[rangeCount];
        rootParameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL; // ALL is used for compute
        rootParameters[parameterCount] = rootParameter;
        d3d12ComputeRootSignature->readWriteStorageTextureRootIndex = parameterCount;
        rangeCount += 1;
        parameterCount += 1;
    }

    if (createInfo->num_readwrite_storage_buffers) {
        descriptorRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        descriptorRange.NumDescriptors = createInfo->num_readwrite_storage_buffers;
        descriptorRange.BaseShaderRegister = createInfo->num_readwrite_storage_textures;
        descriptorRange.RegisterSpace = 1;
        descriptorRange.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
        descriptorRanges[rangeCount] = descriptorRange;

        rootParameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        rootParameter.DescriptorTable.NumDescriptorRanges = 1;
        rootParameter.DescriptorTable.pDescriptorRanges = &descriptorRanges[rangeCount];
        rootParameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL; // ALL is used for compute
        rootParameters[parameterCount] = rootParameter;
        d3d12ComputeRootSignature->readWriteStorageBufferRootIndex = parameterCount;
        rangeCount += 1;
        parameterCount += 1;
    }

    for (Uint32 i = 0; i < createInfo->num_uniform_buffers; i += 1) {
        rootParameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        rootParameter.Descriptor.ShaderRegister = i;
        rootParameter.Descriptor.RegisterSpace = 2;
        rootParameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL; // ALL is used for compute
        rootParameters[parameterCount] = rootParameter;
        d3d12ComputeRootSignature->uniformBufferRootIndex[i] = parameterCount;
        parameterCount += 1;
    }

    D3D12_ROOT_SIGNATURE_DESC rootSignatureDesc;
    rootSignatureDesc.NumParameters = parameterCount;
    rootSignatureDesc.pParameters = rootParameters;
    rootSignatureDesc.NumStaticSamplers = 0;
    rootSignatureDesc.pStaticSamplers = NULL;
    rootSignatureDesc.Flags = (D3D12_ROOT_SIGNATURE_FLAGS)0;

    ID3DBlob *serializedRootSignature;
    ID3DBlob *errorBlob;
    HRESULT res = renderer->D3D12SerializeRootSignature_func(
        &rootSignatureDesc,
        D3D_ROOT_SIGNATURE_VERSION_1,
        &serializedRootSignature,
        &errorBlob);

    if (FAILED(res)) {
        if (errorBlob) {
            SET_ERROR("Failed to serialize RootSignature: %s", (const char *)ID3D10Blob_GetBufferPointer(errorBlob));
            ID3D10Blob_Release(errorBlob);
        }
        D3D12_INTERNAL_DestroyComputeRootSignature(d3d12ComputeRootSignature);
        return NULL;
    }

    ID3D12RootSignature *rootSignature;

    res = ID3D12Device_CreateRootSignature(
        renderer->device,
        0,
        ID3D10Blob_GetBufferPointer(serializedRootSignature),
        ID3D10Blob_GetBufferSize(serializedRootSignature),
        D3D_GUID(D3D_IID_ID3D12RootSignature),
        (void **)&rootSignature);

    if (FAILED(res)) {
        if (errorBlob) {
            SET_ERROR("Failed to create RootSignature: %s", (const char *)ID3D10Blob_GetBufferPointer(errorBlob));
            ID3D10Blob_Release(errorBlob);
        }
        D3D12_INTERNAL_DestroyComputeRootSignature(d3d12ComputeRootSignature);
        return NULL;
    }

    d3d12ComputeRootSignature->handle = rootSignature;
    return d3d12ComputeRootSignature;
}

static SDL_GPUComputePipeline *D3D12_CreateComputePipeline(
    SDL_GPURenderer *driverData,
    const SDL_GPUComputePipelineCreateInfo *createinfo)
{
    D3D12Renderer *renderer = (D3D12Renderer *)driverData;
    void *bytecode;
    size_t bytecodeSize;
    ID3D12PipelineState *pipelineState;

    if (!D3D12_INTERNAL_CreateShaderBytecode(
            renderer,
            createinfo->code,
            createinfo->code_size,
            createinfo->format,
            &bytecode,
            &bytecodeSize)) {
        return NULL;
    }

    D3D12ComputeRootSignature *rootSignature = D3D12_INTERNAL_CreateComputeRootSignature(
        renderer,
        createinfo);

    if (rootSignature == NULL) {
        SDL_free(bytecode);
        SET_STRING_ERROR_AND_RETURN("Could not create root signature!", NULL);
    }

    D3D12_COMPUTE_PIPELINE_STATE_DESC pipelineDesc;
    pipelineDesc.CS.pShaderBytecode = bytecode;
    pipelineDesc.CS.BytecodeLength = bytecodeSize;
    pipelineDesc.pRootSignature = rootSignature->handle;
    pipelineDesc.CachedPSO.CachedBlobSizeInBytes = 0;
    pipelineDesc.CachedPSO.pCachedBlob = NULL;
    pipelineDesc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;
    pipelineDesc.NodeMask = 0;

    HRESULT res = ID3D12Device_CreateComputePipelineState(
        renderer->device,
        &pipelineDesc,
        D3D_GUID(D3D_IID_ID3D12PipelineState),
        (void **)&pipelineState);

    if (FAILED(res)) {
        D3D12_INTERNAL_SetError(renderer, "Could not create compute pipeline state", res);
        SDL_free(bytecode);
        return NULL;
    }

    D3D12ComputePipeline *computePipeline =
        (D3D12ComputePipeline *)SDL_calloc(1, sizeof(D3D12ComputePipeline));

    if (!computePipeline) {
        ID3D12PipelineState_Release(pipelineState);
        SDL_free(bytecode);
        return NULL;
    }

    computePipeline->pipelineState = pipelineState;
    computePipeline->rootSignature = rootSignature;
    computePipeline->numSamplers = createinfo->num_samplers;
    computePipeline->numReadOnlyStorageTextures = createinfo->num_readonly_storage_textures;
    computePipeline->numReadOnlyStorageBuffers = createinfo->num_readonly_storage_buffers;
    computePipeline->numReadWriteStorageTextures = createinfo->num_readwrite_storage_textures;
    computePipeline->numReadWriteStorageBuffers = createinfo->num_readwrite_storage_buffers;
    computePipeline->numUniformBuffers = createinfo->num_uniform_buffers;
    SDL_SetAtomicInt(&computePipeline->referenceCount, 0);

    if (renderer->debug_mode && SDL_HasProperty(createinfo->props, SDL_PROP_GPU_COMPUTEPIPELINE_CREATE_NAME_STRING)) {
        D3D12_INTERNAL_SetPipelineStateName(
            renderer,
            computePipeline->pipelineState,
            SDL_GetStringProperty(createinfo->props, SDL_PROP_GPU_COMPUTEPIPELINE_CREATE_NAME_STRING, NULL));
    }

    return (SDL_GPUComputePipeline *)computePipeline;
}

static bool D3D12_INTERNAL_ConvertRasterizerState(SDL_GPURasterizerState rasterizerState, D3D12_RASTERIZER_DESC *desc)
{
    if (!desc) {
        return false;
    }

    desc->FillMode = SDLToD3D12_FillMode[rasterizerState.fill_mode];
    desc->CullMode = SDLToD3D12_CullMode[rasterizerState.cull_mode];

    switch (rasterizerState.front_face) {
    case SDL_GPU_FRONTFACE_COUNTER_CLOCKWISE:
        desc->FrontCounterClockwise = TRUE;
        break;
    case SDL_GPU_FRONTFACE_CLOCKWISE:
        desc->FrontCounterClockwise = FALSE;
        break;
    default:
        return false;
    }

    if (rasterizerState.enable_depth_bias) {
        desc->DepthBias = SDL_lroundf(rasterizerState.depth_bias_constant_factor);
        desc->DepthBiasClamp = rasterizerState.depth_bias_clamp;
        desc->SlopeScaledDepthBias = rasterizerState.depth_bias_slope_factor;
    } else {
        desc->DepthBias = 0;
        desc->DepthBiasClamp = 0.0f;
        desc->SlopeScaledDepthBias = 0.0f;
    }

    desc->DepthClipEnable = rasterizerState.enable_depth_clip;
    desc->MultisampleEnable = FALSE;
    desc->AntialiasedLineEnable = FALSE;
    desc->ForcedSampleCount = 0;
    desc->ConservativeRaster = D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;

    return true;
}

static bool D3D12_INTERNAL_ConvertBlendState(
    const SDL_GPUGraphicsPipelineCreateInfo *pipelineInfo,
    D3D12_BLEND_DESC *blendDesc)
{
    if (!blendDesc) {
        return false;
    }

    SDL_zerop(blendDesc);
    blendDesc->AlphaToCoverageEnable = pipelineInfo->multisample_state.enable_alpha_to_coverage;
    blendDesc->IndependentBlendEnable = FALSE;

    for (UINT i = 0; i < MAX_COLOR_TARGET_BINDINGS; i += 1) {
        D3D12_RENDER_TARGET_BLEND_DESC rtBlendDesc;
        rtBlendDesc.BlendEnable = FALSE;
        rtBlendDesc.LogicOpEnable = FALSE;
        rtBlendDesc.SrcBlend = D3D12_BLEND_ONE;
        rtBlendDesc.DestBlend = D3D12_BLEND_ZERO;
        rtBlendDesc.BlendOp = D3D12_BLEND_OP_ADD;
        rtBlendDesc.SrcBlendAlpha = D3D12_BLEND_ONE;
        rtBlendDesc.DestBlendAlpha = D3D12_BLEND_ZERO;
        rtBlendDesc.BlendOpAlpha = D3D12_BLEND_OP_ADD;
        rtBlendDesc.LogicOp = D3D12_LOGIC_OP_NOOP;
        rtBlendDesc.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;

        // If target_info has more blend states, you can set IndependentBlendEnable to TRUE and assign different blend states to each render target slot
        if (i < pipelineInfo->target_info.num_color_targets) {
            SDL_GPUColorTargetBlendState sdlBlendState = pipelineInfo->target_info.color_target_descriptions[i].blend_state;
            SDL_GPUColorComponentFlags colorWriteMask = sdlBlendState.enable_color_write_mask ?
                sdlBlendState.color_write_mask :
                0xF;

            rtBlendDesc.BlendEnable = sdlBlendState.enable_blend;
            rtBlendDesc.SrcBlend = SDLToD3D12_BlendFactor[sdlBlendState.src_color_blendfactor];
            rtBlendDesc.DestBlend = SDLToD3D12_BlendFactor[sdlBlendState.dst_color_blendfactor];
            rtBlendDesc.BlendOp = SDLToD3D12_BlendOp[sdlBlendState.color_blend_op];
            rtBlendDesc.SrcBlendAlpha = SDLToD3D12_BlendFactorAlpha[sdlBlendState.src_alpha_blendfactor];
            rtBlendDesc.DestBlendAlpha = SDLToD3D12_BlendFactorAlpha[sdlBlendState.dst_alpha_blendfactor];
            rtBlendDesc.BlendOpAlpha = SDLToD3D12_BlendOp[sdlBlendState.alpha_blend_op];
            rtBlendDesc.RenderTargetWriteMask = colorWriteMask;

            if (i > 0) {
                blendDesc->IndependentBlendEnable = TRUE;
            }
        }

        blendDesc->RenderTarget[i] = rtBlendDesc;
    }

    return true;
}

static bool D3D12_INTERNAL_ConvertDepthStencilState(SDL_GPUDepthStencilState depthStencilState, D3D12_DEPTH_STENCIL_DESC *desc)
{
    if (desc == NULL) {
        return false;
    }

    desc->DepthEnable = depthStencilState.enable_depth_test == true ? TRUE : FALSE;
    desc->DepthWriteMask = depthStencilState.enable_depth_write == true ? D3D12_DEPTH_WRITE_MASK_ALL : D3D12_DEPTH_WRITE_MASK_ZERO;
    desc->DepthFunc = SDLToD3D12_CompareOp[depthStencilState.compare_op];
    desc->StencilEnable = depthStencilState.enable_stencil_test == true ? TRUE : FALSE;
    desc->StencilReadMask = depthStencilState.compare_mask;
    desc->StencilWriteMask = depthStencilState.write_mask;

    desc->FrontFace.StencilFailOp = SDLToD3D12_StencilOp[depthStencilState.front_stencil_state.fail_op];
    desc->FrontFace.StencilDepthFailOp = SDLToD3D12_StencilOp[depthStencilState.front_stencil_state.depth_fail_op];
    desc->FrontFace.StencilPassOp = SDLToD3D12_StencilOp[depthStencilState.front_stencil_state.pass_op];
    desc->FrontFace.StencilFunc = SDLToD3D12_CompareOp[depthStencilState.front_stencil_state.compare_op];

    desc->BackFace.StencilFailOp = SDLToD3D12_StencilOp[depthStencilState.back_stencil_state.fail_op];
    desc->BackFace.StencilDepthFailOp = SDLToD3D12_StencilOp[depthStencilState.back_stencil_state.depth_fail_op];
    desc->BackFace.StencilPassOp = SDLToD3D12_StencilOp[depthStencilState.back_stencil_state.pass_op];
    desc->BackFace.StencilFunc = SDLToD3D12_CompareOp[depthStencilState.back_stencil_state.compare_op];

    return true;
}

static bool D3D12_INTERNAL_ConvertVertexInputState(SDL_GPUVertexInputState vertexInputState, D3D12_INPUT_ELEMENT_DESC *desc, const char *semantic)
{
    if (desc == NULL || vertexInputState.num_vertex_attributes == 0) {
        return false;
    }

    for (Uint32 i = 0; i < vertexInputState.num_vertex_attributes; i += 1) {
        SDL_GPUVertexAttribute attribute = vertexInputState.vertex_attributes[i];

        desc[i].SemanticName = semantic;
        desc[i].SemanticIndex = attribute.location;
        desc[i].Format = SDLToD3D12_VertexFormat[attribute.format];
        desc[i].InputSlot = attribute.buffer_slot;
        desc[i].AlignedByteOffset = attribute.offset;
        desc[i].InputSlotClass = SDLToD3D12_InputRate[vertexInputState.vertex_buffer_descriptions[attribute.buffer_slot].input_rate];
        desc[i].InstanceDataStepRate = (vertexInputState.vertex_buffer_descriptions[attribute.buffer_slot].input_rate == SDL_GPU_VERTEXINPUTRATE_INSTANCE)
            ? 1
            : 0;
    }

    return true;
}

static bool D3D12_INTERNAL_AssignStagingDescriptorHandle(
    D3D12Renderer *renderer,
    D3D12_DESCRIPTOR_HEAP_TYPE heapType,
    D3D12StagingDescriptor *cpuDescriptor)
{
    D3D12StagingDescriptor *descriptor;
    D3D12StagingDescriptorPool *pool = renderer->stagingDescriptorPools[heapType];

    SDL_LockMutex(pool->lock);

    if (pool->freeDescriptorCount == 0) {
        if (!D3D12_INTERNAL_ExpandStagingDescriptorPool(renderer, pool))
        {
            SDL_UnlockMutex(pool->lock);
            return false;
        }
    }

    descriptor = &pool->freeDescriptors[pool->freeDescriptorCount - 1];
    SDL_memcpy(cpuDescriptor, descriptor, sizeof(D3D12StagingDescriptor));
    pool->freeDescriptorCount -= 1;

    SDL_UnlockMutex(pool->lock);

    return true;
}

static SDL_GPUGraphicsPipeline *D3D12_CreateGraphicsPipeline(
    SDL_GPURenderer *driverData,
    const SDL_GPUGraphicsPipelineCreateInfo *createinfo)
{
    D3D12Renderer *renderer = (D3D12Renderer *)driverData;
    D3D12Shader *vertShader = (D3D12Shader *)createinfo->vertex_shader;
    D3D12Shader *fragShader = (D3D12Shader *)createinfo->fragment_shader;

    if (renderer->debug_mode) {
        if (vertShader->stage != SDL_GPU_SHADERSTAGE_VERTEX) {
            SDL_assert_release(!"CreateGraphicsPipeline was passed a fragment shader for the vertex stage");
        }
        if (fragShader->stage != SDL_GPU_SHADERSTAGE_FRAGMENT) {
            SDL_assert_release(!"CreateGraphicsPipeline was passed a vertex shader for the fragment stage");
        }
    }

    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc;
    SDL_zero(psoDesc);
    psoDesc.VS.pShaderBytecode = vertShader->bytecode;
    psoDesc.VS.BytecodeLength = vertShader->bytecodeSize;
    psoDesc.PS.pShaderBytecode = fragShader->bytecode;
    psoDesc.PS.BytecodeLength = fragShader->bytecodeSize;

    D3D12_INPUT_ELEMENT_DESC inputElementDescs[D3D12_IA_VERTEX_INPUT_STRUCTURE_ELEMENT_COUNT];
    if (createinfo->vertex_input_state.num_vertex_attributes > 0) {
        psoDesc.InputLayout.pInputElementDescs = inputElementDescs;
        psoDesc.InputLayout.NumElements = createinfo->vertex_input_state.num_vertex_attributes;
        D3D12_INTERNAL_ConvertVertexInputState(createinfo->vertex_input_state, inputElementDescs, renderer->semantic);
    }

    psoDesc.PrimitiveTopologyType = SDLToD3D12_PrimitiveTopologyType[createinfo->primitive_type];

    if (!D3D12_INTERNAL_ConvertRasterizerState(createinfo->rasterizer_state, &psoDesc.RasterizerState)) {
        return NULL;
    }
    if (!D3D12_INTERNAL_ConvertBlendState(createinfo, &psoDesc.BlendState)) {
        return NULL;
    }
    if (!D3D12_INTERNAL_ConvertDepthStencilState(createinfo->depth_stencil_state, &psoDesc.DepthStencilState)) {
        return NULL;
    }

    D3D12GraphicsPipeline *pipeline = (D3D12GraphicsPipeline *)SDL_calloc(1, sizeof(D3D12GraphicsPipeline));
    if (!pipeline) {
        return NULL;
    }

    psoDesc.SampleMask = 0xFFFFFFFF;
    psoDesc.SampleDesc.Count = SDLToD3D12_SampleCount[createinfo->multisample_state.sample_count];
    psoDesc.SampleDesc.Quality = (createinfo->multisample_state.sample_count > SDL_GPU_SAMPLECOUNT_1) ? D3D12_STANDARD_MULTISAMPLE_PATTERN : 0;

    if (createinfo->target_info.has_depth_stencil_target) {
        psoDesc.DSVFormat = SDLToD3D12_DepthFormat[createinfo->target_info.depth_stencil_format];
    }
    psoDesc.NumRenderTargets = createinfo->target_info.num_color_targets;
    for (uint32_t i = 0; i < createinfo->target_info.num_color_targets; i += 1) {
        psoDesc.RTVFormats[i] = SDLToD3D12_TextureFormat[createinfo->target_info.color_target_descriptions[i].format];
    }

    // Assuming some default values or further initialization
    psoDesc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;
    psoDesc.CachedPSO.CachedBlobSizeInBytes = 0;
    psoDesc.CachedPSO.pCachedBlob = NULL;

    psoDesc.NodeMask = 0;

    D3D12GraphicsRootSignature *rootSignature = D3D12_INTERNAL_CreateGraphicsRootSignature(
        renderer,
        vertShader,
        fragShader);

    if (rootSignature == NULL) {
        D3D12_INTERNAL_DestroyGraphicsPipeline(pipeline);
        return NULL;
    }
    pipeline->rootSignature = rootSignature;

    psoDesc.pRootSignature = rootSignature->handle;
    ID3D12PipelineState *pipelineState;

    HRESULT res = ID3D12Device_CreateGraphicsPipelineState(
        renderer->device,
        &psoDesc,
        D3D_GUID(D3D_IID_ID3D12PipelineState),
        (void **)&pipelineState);
    if (FAILED(res)) {
        D3D12_INTERNAL_SetError(renderer, "Could not create graphics pipeline state", res);
        D3D12_INTERNAL_DestroyGraphicsPipeline(pipeline);
        return NULL;
    }

    pipeline->pipelineState = pipelineState;

    for (Uint32 i = 0; i < createinfo->vertex_input_state.num_vertex_buffers; i += 1) {
        pipeline->vertexStrides[createinfo->vertex_input_state.vertex_buffer_descriptions[i].slot] =
            createinfo->vertex_input_state.vertex_buffer_descriptions[i].pitch;
    }

    pipeline->primitiveType = createinfo->primitive_type;

    pipeline->vertexSamplerCount = vertShader->num_samplers;
    pipeline->vertexStorageTextureCount = vertShader->numStorageTextures;
    pipeline->vertexStorageBufferCount = vertShader->numStorageBuffers;
    pipeline->vertexUniformBufferCount = vertShader->numUniformBuffers;

    pipeline->fragmentSamplerCount = fragShader->num_samplers;
    pipeline->fragmentStorageTextureCount = fragShader->numStorageTextures;
    pipeline->fragmentStorageBufferCount = fragShader->numStorageBuffers;
    pipeline->fragmentUniformBufferCount = fragShader->numUniformBuffers;

    SDL_SetAtomicInt(&pipeline->referenceCount, 0);

    if (renderer->debug_mode && SDL_HasProperty(createinfo->props, SDL_PROP_GPU_GRAPHICSPIPELINE_CREATE_NAME_STRING)) {
        D3D12_INTERNAL_SetPipelineStateName(
            renderer,
            pipeline->pipelineState,
            SDL_GetStringProperty(createinfo->props, SDL_PROP_GPU_GRAPHICSPIPELINE_CREATE_NAME_STRING, NULL));
    }

    return (SDL_GPUGraphicsPipeline *)pipeline;
}

static SDL_GPUSampler *D3D12_CreateSampler(
    SDL_GPURenderer *driverData,
    const SDL_GPUSamplerCreateInfo *createinfo)
{
    D3D12Renderer *renderer = (D3D12Renderer *)driverData;
    D3D12Sampler *sampler = (D3D12Sampler *)SDL_calloc(1, sizeof(D3D12Sampler));
    if (!sampler) {
        return NULL;
    }
    D3D12_SAMPLER_DESC samplerDesc;

    samplerDesc.Filter = SDLToD3D12_Filter(
        createinfo->min_filter,
        createinfo->mag_filter,
        createinfo->mipmap_mode,
        createinfo->enable_compare,
        createinfo->enable_anisotropy);
    samplerDesc.AddressU = SDLToD3D12_SamplerAddressMode[createinfo->address_mode_u];
    samplerDesc.AddressV = SDLToD3D12_SamplerAddressMode[createinfo->address_mode_v];
    samplerDesc.AddressW = SDLToD3D12_SamplerAddressMode[createinfo->address_mode_w];
    samplerDesc.MaxAnisotropy = (Uint32)createinfo->max_anisotropy;
    samplerDesc.ComparisonFunc = SDLToD3D12_CompareOp[createinfo->compare_op];
    samplerDesc.MinLOD = createinfo->min_lod;
    samplerDesc.MaxLOD = createinfo->max_lod;
    samplerDesc.MipLODBias = createinfo->mip_lod_bias;
    samplerDesc.BorderColor[0] = 0;
    samplerDesc.BorderColor[1] = 0;
    samplerDesc.BorderColor[2] = 0;
    samplerDesc.BorderColor[3] = 0;

    D3D12_INTERNAL_AssignStagingDescriptorHandle(
        renderer,
        D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
        &sampler->handle);

    ID3D12Device_CreateSampler(
        renderer->device,
        &samplerDesc,
        sampler->handle.cpuHandle);

    sampler->createInfo = *createinfo;
    SDL_SetAtomicInt(&sampler->referenceCount, 0);

    // Ignore name property because it is not applicable to D3D12.

    return (SDL_GPUSampler *)sampler;
}

static SDL_GPUShader *D3D12_CreateShader(
    SDL_GPURenderer *driverData,
    const SDL_GPUShaderCreateInfo *createinfo)
{
    D3D12Renderer *renderer = (D3D12Renderer *)driverData;
    void *bytecode;
    size_t bytecodeSize;
    D3D12Shader *shader;

    if (!D3D12_INTERNAL_CreateShaderBytecode(
            renderer,
            createinfo->code,
            createinfo->code_size,
            createinfo->format,
            &bytecode,
            &bytecodeSize)) {
        return NULL;
    }
    shader = (D3D12Shader *)SDL_calloc(1, sizeof(D3D12Shader));
    if (!shader) {
        SDL_free(bytecode);
        return NULL;
    }
    shader->stage = createinfo->stage;
    shader->num_samplers = createinfo->num_samplers;
    shader->numStorageBuffers = createinfo->num_storage_buffers;
    shader->numStorageTextures = createinfo->num_storage_textures;
    shader->numUniformBuffers = createinfo->num_uniform_buffers;

    shader->bytecode = bytecode;
    shader->bytecodeSize = bytecodeSize;

    // Ignore name property because it is not applicable to D3D12.

    return (SDL_GPUShader *)shader;
}

static D3D12Texture *D3D12_INTERNAL_CreateTexture(
    D3D12Renderer *renderer,
    const SDL_GPUTextureCreateInfo *createinfo,
    bool isSwapchainTexture,
    const char *debugName)
{
    D3D12Texture *texture;
    ID3D12Resource *handle;
    D3D12_HEAP_PROPERTIES heapProperties;
    D3D12_HEAP_FLAGS heapFlags = (D3D12_HEAP_FLAGS)0;
    D3D12_RESOURCE_DESC desc;
    D3D12_RESOURCE_FLAGS resourceFlags = (D3D12_RESOURCE_FLAGS)0;
    D3D12_RESOURCE_STATES initialState = (D3D12_RESOURCE_STATES)0;
    D3D12_CLEAR_VALUE clearValue;
    DXGI_FORMAT format;
    bool useClearValue = false;
    bool needsSRV =
        (createinfo->usage & SDL_GPU_TEXTUREUSAGE_SAMPLER) ||
        (createinfo->usage & SDL_GPU_TEXTUREUSAGE_GRAPHICS_STORAGE_READ) ||
        (createinfo->usage & SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_READ);
    bool needsUAV =
        (createinfo->usage & SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE) ||
        (createinfo->usage & SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_SIMULTANEOUS_READ_WRITE);
    HRESULT res;

    texture = (D3D12Texture *)SDL_calloc(1, sizeof(D3D12Texture));
    if (!texture) {
        return NULL;
    }

    Uint32 layerCount = createinfo->type == SDL_GPU_TEXTURETYPE_3D ? 1 : createinfo->layer_count_or_depth;
    Uint32 depth = createinfo->type == SDL_GPU_TEXTURETYPE_3D ? createinfo->layer_count_or_depth : 1;
    bool isMultisample = createinfo->sample_count > SDL_GPU_SAMPLECOUNT_1;

    format = SDLToD3D12_TextureFormat[createinfo->format];

    if (createinfo->usage & SDL_GPU_TEXTUREUSAGE_COLOR_TARGET) {
        resourceFlags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        useClearValue = true;
        clearValue.Format = format;
        clearValue.Color[0] = SDL_GetFloatProperty(createinfo->props, SDL_PROP_GPU_TEXTURE_CREATE_D3D12_CLEAR_R_FLOAT, 0);
        clearValue.Color[1] = SDL_GetFloatProperty(createinfo->props, SDL_PROP_GPU_TEXTURE_CREATE_D3D12_CLEAR_G_FLOAT, 0);
        clearValue.Color[2] = SDL_GetFloatProperty(createinfo->props, SDL_PROP_GPU_TEXTURE_CREATE_D3D12_CLEAR_B_FLOAT, 0);
        clearValue.Color[3] = SDL_GetFloatProperty(createinfo->props, SDL_PROP_GPU_TEXTURE_CREATE_D3D12_CLEAR_A_FLOAT, 0);
    }

    if (createinfo->usage & SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET) {
        resourceFlags |= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
        useClearValue = true;
        clearValue.Format = SDLToD3D12_DepthFormat[createinfo->format];
        clearValue.DepthStencil.Depth = SDL_GetFloatProperty(createinfo->props, SDL_PROP_GPU_TEXTURE_CREATE_D3D12_CLEAR_DEPTH_FLOAT, 0);
        clearValue.DepthStencil.Stencil = (UINT8)SDL_GetNumberProperty(createinfo->props, SDL_PROP_GPU_TEXTURE_CREATE_D3D12_CLEAR_STENCIL_NUMBER, 0);
        format = needsSRV ? SDLToD3D12_TypelessFormat[createinfo->format] : SDLToD3D12_DepthFormat[createinfo->format];
    }

    if (needsUAV) {
        resourceFlags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    }

    heapProperties.Type = D3D12_HEAP_TYPE_DEFAULT;
    heapProperties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heapProperties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heapProperties.CreationNodeMask = 0; // We don't do multi-adapter operation
    heapProperties.VisibleNodeMask = 0;  // We don't do multi-adapter operation

    heapFlags = isSwapchainTexture ? D3D12_HEAP_FLAG_ALLOW_DISPLAY : D3D12_HEAP_FLAG_NONE;

    if (createinfo->type != SDL_GPU_TEXTURETYPE_3D) {
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Alignment = isSwapchainTexture ? 0 : D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
        desc.Width = createinfo->width;
        desc.Height = createinfo->height;
        desc.DepthOrArraySize = (UINT16)createinfo->layer_count_or_depth;
        desc.MipLevels = (UINT16)createinfo->num_levels;
        desc.Format = format;
        desc.SampleDesc.Count = SDLToD3D12_SampleCount[createinfo->sample_count];
        desc.SampleDesc.Quality = isMultisample ? D3D12_STANDARD_MULTISAMPLE_PATTERN : 0;
        desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN; // Apparently this is the most efficient choice
        desc.Flags = resourceFlags;
    } else {
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
        desc.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
        desc.Width = createinfo->width;
        desc.Height = createinfo->height;
        desc.DepthOrArraySize = (UINT16)createinfo->layer_count_or_depth;
        desc.MipLevels = (UINT16)createinfo->num_levels;
        desc.Format = format;
        desc.SampleDesc.Count = 1;
        desc.SampleDesc.Quality = 0;
        desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        desc.Flags = resourceFlags;
    }

    initialState = isSwapchainTexture ? D3D12_RESOURCE_STATE_PRESENT : D3D12_INTERNAL_DefaultTextureResourceState(createinfo->usage);

    res = ID3D12Device_CreateCommittedResource(
        renderer->device,
        &heapProperties,
        heapFlags,
        &desc,
        initialState,
        useClearValue ? &clearValue : NULL,
        D3D_GUID(D3D_IID_ID3D12Resource),
        (void **)&handle);
    if (FAILED(res)) {
        D3D12_INTERNAL_SetError(renderer, "Failed to create texture!", res);
        D3D12_INTERNAL_DestroyTexture(texture);
        return NULL;
    }

    texture->resource = handle;

    // Create the SRV if applicable
    if (needsSRV) {
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc;

        D3D12_INTERNAL_AssignStagingDescriptorHandle(
            renderer,
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            &texture->srvHandle);

        srvDesc.Format = SDLToD3D12_TextureFormat[createinfo->format];
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

        if (createinfo->type == SDL_GPU_TEXTURETYPE_CUBE) {
            srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
            srvDesc.TextureCube.MipLevels = createinfo->num_levels;
            srvDesc.TextureCube.MostDetailedMip = 0;
            srvDesc.TextureCube.ResourceMinLODClamp = 0;
        } else if (createinfo->type == SDL_GPU_TEXTURETYPE_CUBE_ARRAY) {
            srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBEARRAY;
            srvDesc.TextureCubeArray.MipLevels = createinfo->num_levels;
            srvDesc.TextureCubeArray.MostDetailedMip = 0;
            srvDesc.TextureCubeArray.First2DArrayFace = 0;
            srvDesc.TextureCubeArray.NumCubes = createinfo->layer_count_or_depth / 6;
            srvDesc.TextureCubeArray.ResourceMinLODClamp = 0;
        } else if (createinfo->type == SDL_GPU_TEXTURETYPE_2D_ARRAY) {
            srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
            srvDesc.Texture2DArray.MipLevels = createinfo->num_levels;
            srvDesc.Texture2DArray.MostDetailedMip = 0;
            srvDesc.Texture2DArray.FirstArraySlice = 0;
            srvDesc.Texture2DArray.ArraySize = layerCount;
            srvDesc.Texture2DArray.ResourceMinLODClamp = 0;
            srvDesc.Texture2DArray.PlaneSlice = 0;
        } else if (createinfo->type == SDL_GPU_TEXTURETYPE_3D) {
            srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
            srvDesc.Texture3D.MipLevels = createinfo->num_levels;
            srvDesc.Texture3D.MostDetailedMip = 0;
            srvDesc.Texture3D.ResourceMinLODClamp = 0; // default behavior
        } else {
            srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            srvDesc.Texture2D.MipLevels = createinfo->num_levels;
            srvDesc.Texture2D.MostDetailedMip = 0;
            srvDesc.Texture2D.PlaneSlice = 0;
            srvDesc.Texture2D.ResourceMinLODClamp = 0; // default behavior
        }

        ID3D12Device_CreateShaderResourceView(
            renderer->device,
            handle,
            &srvDesc,
            texture->srvHandle.cpuHandle);
    }

    SDL_SetAtomicInt(&texture->referenceCount, 0);

    texture->subresourceCount = createinfo->num_levels * layerCount;
    texture->subresources = (D3D12TextureSubresource *)SDL_calloc(
        texture->subresourceCount, sizeof(D3D12TextureSubresource));
    if (!texture->subresources) {
        D3D12_INTERNAL_DestroyTexture(texture);
        return NULL;
    }
    for (Uint32 layerIndex = 0; layerIndex < layerCount; layerIndex += 1) {
        for (Uint32 levelIndex = 0; levelIndex < createinfo->num_levels; levelIndex += 1) {
            Uint32 subresourceIndex = D3D12_INTERNAL_CalcSubresource(
                levelIndex,
                layerIndex,
                createinfo->num_levels);

            texture->subresources[subresourceIndex].parent = texture;
            texture->subresources[subresourceIndex].layer = layerIndex;
            texture->subresources[subresourceIndex].level = levelIndex;
            texture->subresources[subresourceIndex].depth = depth;
            texture->subresources[subresourceIndex].index = subresourceIndex;

            texture->subresources[subresourceIndex].rtvHandles = NULL;
            texture->subresources[subresourceIndex].uavHandle.heap = NULL;
            texture->subresources[subresourceIndex].dsvHandle.heap = NULL;

            // Create RTV if needed
            if (createinfo->usage & SDL_GPU_TEXTUREUSAGE_COLOR_TARGET) {
                texture->subresources[subresourceIndex].rtvHandles = (D3D12StagingDescriptor *)SDL_calloc(depth, sizeof(D3D12StagingDescriptor));

                for (Uint32 depthIndex = 0; depthIndex < depth; depthIndex += 1) {
                    D3D12_RENDER_TARGET_VIEW_DESC rtvDesc;

                    D3D12_INTERNAL_AssignStagingDescriptorHandle(
                        renderer,
                        D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                        &texture->subresources[subresourceIndex].rtvHandles[depthIndex]);

                    rtvDesc.Format = SDLToD3D12_TextureFormat[createinfo->format];

                    if (createinfo->type == SDL_GPU_TEXTURETYPE_2D_ARRAY || createinfo->type == SDL_GPU_TEXTURETYPE_CUBE || createinfo->type == SDL_GPU_TEXTURETYPE_CUBE_ARRAY) {
                        rtvDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DARRAY;
                        rtvDesc.Texture2DArray.MipSlice = levelIndex;
                        rtvDesc.Texture2DArray.FirstArraySlice = layerIndex;
                        rtvDesc.Texture2DArray.ArraySize = 1;
                        rtvDesc.Texture2DArray.PlaneSlice = 0;
                    } else if (createinfo->type == SDL_GPU_TEXTURETYPE_3D) {
                        rtvDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE3D;
                        rtvDesc.Texture3D.MipSlice = levelIndex;
                        rtvDesc.Texture3D.FirstWSlice = depthIndex;
                        rtvDesc.Texture3D.WSize = 1;
                    } else if (isMultisample) {
                        rtvDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DMS;
                    } else {
                        rtvDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
                        rtvDesc.Texture2D.MipSlice = levelIndex;
                        rtvDesc.Texture2D.PlaneSlice = 0;
                    }

                    ID3D12Device_CreateRenderTargetView(
                        renderer->device,
                        texture->resource,
                        &rtvDesc,
                        texture->subresources[subresourceIndex].rtvHandles[depthIndex].cpuHandle);
                }
            }

            // Create DSV if needed
            if (createinfo->usage & SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET) {
                D3D12_DEPTH_STENCIL_VIEW_DESC dsvDesc;

                D3D12_INTERNAL_AssignStagingDescriptorHandle(
                    renderer,
                    D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
                    &texture->subresources[subresourceIndex].dsvHandle);

                dsvDesc.Format = SDLToD3D12_DepthFormat[createinfo->format];
                dsvDesc.Flags = (D3D12_DSV_FLAGS)0;

                if (isMultisample) {
                    dsvDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DMS;
                } else {
                    dsvDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
                    dsvDesc.Texture2D.MipSlice = levelIndex;
                }

                ID3D12Device_CreateDepthStencilView(
                    renderer->device,
                    texture->resource,
                    &dsvDesc,
                    texture->subresources[subresourceIndex].dsvHandle.cpuHandle);
            }

            // Create subresource UAV if necessary
            if (needsUAV) {
                D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc;

                D3D12_INTERNAL_AssignStagingDescriptorHandle(
                    renderer,
                    D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    &texture->subresources[subresourceIndex].uavHandle);

                uavDesc.Format = SDLToD3D12_TextureFormat[createinfo->format];

                if (createinfo->type == SDL_GPU_TEXTURETYPE_2D_ARRAY || createinfo->type == SDL_GPU_TEXTURETYPE_CUBE || createinfo->type == SDL_GPU_TEXTURETYPE_CUBE_ARRAY) {
                    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
                    uavDesc.Texture2DArray.MipSlice = levelIndex;
                    uavDesc.Texture2DArray.FirstArraySlice = layerIndex;
                    uavDesc.Texture2DArray.ArraySize = 1;
                } else if (createinfo->type == SDL_GPU_TEXTURETYPE_3D) {
                    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
                    uavDesc.Texture3D.MipSlice = levelIndex;
                    uavDesc.Texture3D.FirstWSlice = 0;
                    uavDesc.Texture3D.WSize = depth;
                } else {
                    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
                    uavDesc.Texture2D.MipSlice = levelIndex;
                    uavDesc.Texture2D.PlaneSlice = 0;
                }

                ID3D12Device_CreateUnorderedAccessView(
                    renderer->device,
                    texture->resource,
                    NULL,
                    &uavDesc,
                    texture->subresources[subresourceIndex].uavHandle.cpuHandle);
            }
        }
    }

    D3D12_INTERNAL_SetResourceName(
        renderer,
        texture->resource,
        debugName);

    return texture;
}

static SDL_GPUTexture *D3D12_CreateTexture(
    SDL_GPURenderer *driverData,
    const SDL_GPUTextureCreateInfo *createinfo)
{
    D3D12TextureContainer *container = (D3D12TextureContainer *)SDL_calloc(1, sizeof(D3D12TextureContainer));
    if (!container) {
        return NULL;
    }

    // Copy properties so we don't lose information when the client destroys them
    container->header.info = *createinfo;
    container->header.info.props = SDL_CreateProperties();
    if (createinfo->props) {
        SDL_CopyProperties(createinfo->props, container->header.info.props);
    }

    container->textureCapacity = 1;
    container->textureCount = 1;
    container->textures = (D3D12Texture **)SDL_calloc(
        container->textureCapacity, sizeof(D3D12Texture *));

    if (!container->textures) {
        SDL_free(container);
        return NULL;
    }

    container->debugName = NULL;
    if (SDL_HasProperty(createinfo->props, SDL_PROP_GPU_TEXTURE_CREATE_NAME_STRING)) {
        container->debugName = SDL_strdup(SDL_GetStringProperty(createinfo->props, SDL_PROP_GPU_TEXTURE_CREATE_NAME_STRING, NULL));
    }

    container->canBeCycled = true;

    D3D12Texture *texture = D3D12_INTERNAL_CreateTexture(
        (D3D12Renderer *)driverData,
        createinfo,
        false,
        container->debugName);

    if (!texture) {
        SDL_free(container->textures);
        SDL_free(container);
        return NULL;
    }

    container->textures[0] = texture;
    container->activeTexture = texture;

    texture->container = container;
    texture->containerIndex = 0;

    return (SDL_GPUTexture *)container;
}

static D3D12Buffer *D3D12_INTERNAL_CreateBuffer(
    D3D12Renderer *renderer,
    SDL_GPUBufferUsageFlags usageFlags,
    Uint32 size,
    D3D12BufferType type,
    const char *debugName)
{
    D3D12Buffer *buffer;
    ID3D12Resource *handle;
    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc;
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc;
    D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc;
    D3D12_HEAP_PROPERTIES heapProperties;
    D3D12_RESOURCE_DESC desc;
    D3D12_HEAP_FLAGS heapFlags = (D3D12_HEAP_FLAGS)0;
    D3D12_RESOURCE_FLAGS resourceFlags = (D3D12_RESOURCE_FLAGS)0;
    D3D12_RESOURCE_STATES initialState = D3D12_RESOURCE_STATE_COMMON;
    HRESULT res;

    buffer = (D3D12Buffer *)SDL_calloc(1, sizeof(D3D12Buffer));

    if (!buffer) {
        return NULL;
    }

    if (usageFlags & SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE) {
        resourceFlags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    }
#if (defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES))
    if (usageFlags & SDL_GPU_BUFFERUSAGE_INDIRECT) {
        resourceFlags |= D3D12XBOX_RESOURCE_FLAG_ALLOW_INDIRECT_BUFFER;
    }
#endif

    heapProperties.CreationNodeMask = 0; // We don't do multi-adapter operation
    heapProperties.VisibleNodeMask = 0;  // We don't do multi-adapter operation

    if (type == D3D12_BUFFER_TYPE_GPU) {
        heapProperties.Type = D3D12_HEAP_TYPE_DEFAULT;
        heapProperties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        heapProperties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
        heapFlags = D3D12_HEAP_FLAG_NONE;
    } else if (type == D3D12_BUFFER_TYPE_UPLOAD) {
        heapProperties.Type = D3D12_HEAP_TYPE_UPLOAD;
        heapProperties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        heapProperties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
        heapFlags = D3D12_HEAP_FLAG_NONE;
        initialState = D3D12_RESOURCE_STATE_GENERIC_READ;
    } else if (type == D3D12_BUFFER_TYPE_DOWNLOAD) {
        heapProperties.Type = D3D12_HEAP_TYPE_READBACK;
        heapProperties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        heapProperties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
        heapFlags = D3D12_HEAP_FLAG_NONE;
        initialState = D3D12_RESOURCE_STATE_COPY_DEST;
    } else if (type == D3D12_BUFFER_TYPE_UNIFORM) {
        // D3D12 is badly designed, so we have to check if the fast path for uniform buffers is enabled
        if (renderer->GPUUploadHeapSupported) {
            heapProperties.Type = D3D12_HEAP_TYPE_GPU_UPLOAD;
            heapProperties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
            heapProperties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
        } else {
            heapProperties.Type = D3D12_HEAP_TYPE_UPLOAD;
            heapProperties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
            heapProperties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
            initialState = D3D12_RESOURCE_STATE_GENERIC_READ;
        }
        heapFlags = D3D12_HEAP_FLAG_NONE;
    } else {
        SET_STRING_ERROR_AND_RETURN("Unrecognized buffer type!", NULL);
    }

    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
    desc.Width = size;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = resourceFlags;

    res = ID3D12Device_CreateCommittedResource(
        renderer->device,
        &heapProperties,
        heapFlags,
        &desc,
        initialState,
        NULL,
        D3D_GUID(D3D_IID_ID3D12Resource),
        (void **)&handle);
    if (FAILED(res)) {
        D3D12_INTERNAL_SetError(renderer, "Could not create buffer!", res);
        D3D12_INTERNAL_DestroyBuffer(buffer);
        return NULL;
    }

    buffer->handle = handle;
    SDL_SetAtomicInt(&buffer->referenceCount, 0);

    buffer->uavDescriptor.heap = NULL;
    buffer->srvDescriptor.heap = NULL;
    buffer->cbvDescriptor.heap = NULL;

    if (usageFlags & SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE) {
        D3D12_INTERNAL_AssignStagingDescriptorHandle(
            renderer,
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            &buffer->uavDescriptor);

        uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uavDesc.Format = DXGI_FORMAT_R32_TYPELESS;
        uavDesc.Buffer.FirstElement = 0;
        uavDesc.Buffer.NumElements = size / sizeof(Uint32);
        uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        uavDesc.Buffer.CounterOffsetInBytes = 0; // TODO: support counters?
        uavDesc.Buffer.StructureByteStride = 0;

        // Create UAV
        ID3D12Device_CreateUnorderedAccessView(
            renderer->device,
            handle,
            NULL, // TODO: support counters?
            &uavDesc,
            buffer->uavDescriptor.cpuHandle);
    }

    if (
        (usageFlags & SDL_GPU_BUFFERUSAGE_GRAPHICS_STORAGE_READ) ||
        (usageFlags & SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ)) {
        D3D12_INTERNAL_AssignStagingDescriptorHandle(
            renderer,
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            &buffer->srvDescriptor);

        srvDesc.Format = DXGI_FORMAT_R32_TYPELESS;
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srvDesc.Buffer.FirstElement = 0;
        srvDesc.Buffer.NumElements = size / sizeof(Uint32);
        srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
        srvDesc.Buffer.StructureByteStride = 0;

        // Create SRV
        ID3D12Device_CreateShaderResourceView(
            renderer->device,
            handle,
            &srvDesc,
            buffer->srvDescriptor.cpuHandle);
    }

    // FIXME: we may not need a CBV since we use root descriptors
    if (type == D3D12_BUFFER_TYPE_UNIFORM) {
        D3D12_INTERNAL_AssignStagingDescriptorHandle(
            renderer,
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            &buffer->cbvDescriptor);

        cbvDesc.BufferLocation = ID3D12Resource_GetGPUVirtualAddress(handle);
        cbvDesc.SizeInBytes = size;

        // Create CBV
        ID3D12Device_CreateConstantBufferView(
            renderer->device,
            &cbvDesc,
            buffer->cbvDescriptor.cpuHandle);
    }

    buffer->virtualAddress = 0;
    if (type == D3D12_BUFFER_TYPE_GPU || type == D3D12_BUFFER_TYPE_UNIFORM) {
        buffer->virtualAddress = ID3D12Resource_GetGPUVirtualAddress(buffer->handle);
    }

    buffer->mapPointer = NULL;
    // Persistently map upload buffers
    if (type == D3D12_BUFFER_TYPE_UPLOAD) {
        res = ID3D12Resource_Map(
            buffer->handle,
            0,
            NULL,
            (void **)&buffer->mapPointer);
        if (FAILED(res)) {
            D3D12_INTERNAL_SetError(renderer, "Failed to map upload buffer!", res);
            D3D12_INTERNAL_DestroyBuffer(buffer);
            return NULL;
        }
    }

    buffer->container = NULL;
    buffer->containerIndex = 0;

    buffer->transitioned = initialState != D3D12_RESOURCE_STATE_COMMON;
    SDL_SetAtomicInt(&buffer->referenceCount, 0);

    D3D12_INTERNAL_SetResourceName(
        renderer,
        buffer->handle,
        debugName);

    return buffer;
}

static D3D12BufferContainer *D3D12_INTERNAL_CreateBufferContainer(
    D3D12Renderer *renderer,
    SDL_GPUBufferUsageFlags usageFlags,
    Uint32 size,
    D3D12BufferType type,
    const char *debugName)
{
    D3D12BufferContainer *container;
    D3D12Buffer *buffer;

    container = (D3D12BufferContainer *)SDL_calloc(1, sizeof(D3D12BufferContainer));
    if (!container) {
        return NULL;
    }

    container->usage = usageFlags;
    container->size = size;
    container->type = type;

    container->bufferCapacity = 1;
    container->bufferCount = 1;
    container->buffers = (D3D12Buffer **)SDL_calloc(
        container->bufferCapacity, sizeof(D3D12Buffer *));
    if (!container->buffers) {
        SDL_free(container);
        return NULL;
    }
    container->debugName = NULL;

    buffer = D3D12_INTERNAL_CreateBuffer(
        renderer,
        usageFlags,
        size,
        type,
        debugName);

    if (buffer == NULL) {
        SDL_free(container->buffers);
        SDL_free(container);
        return NULL;
    }

    container->activeBuffer = buffer;
    container->buffers[0] = buffer;
    buffer->container = container;
    buffer->containerIndex = 0;

    if (debugName != NULL) {
        container->debugName = SDL_strdup(debugName);
    }

    return container;
}

static SDL_GPUBuffer *D3D12_CreateBuffer(
    SDL_GPURenderer *driverData,
    SDL_GPUBufferUsageFlags usageFlags,
    Uint32 size,
    const char *debugName)
{
    return (SDL_GPUBuffer *)D3D12_INTERNAL_CreateBufferContainer(
        (D3D12Renderer *)driverData,
        usageFlags,
        size,
        D3D12_BUFFER_TYPE_GPU,
        debugName);
}

static SDL_GPUTransferBuffer *D3D12_CreateTransferBuffer(
    SDL_GPURenderer *driverData,
    SDL_GPUTransferBufferUsage usage,
    Uint32 size,
    const char *debugName)
{
    return (SDL_GPUTransferBuffer *)D3D12_INTERNAL_CreateBufferContainer(
        (D3D12Renderer *)driverData,
        0,
        size,
        usage == SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD ? D3D12_BUFFER_TYPE_UPLOAD : D3D12_BUFFER_TYPE_DOWNLOAD,
        debugName);
}

// Disposal

static void D3D12_ReleaseTexture(
    SDL_GPURenderer *driverData,
    SDL_GPUTexture *texture)
{
    D3D12Renderer *renderer = (D3D12Renderer *)driverData;
    D3D12TextureContainer *container = (D3D12TextureContainer *)texture;

    D3D12_INTERNAL_ReleaseTextureContainer(
        renderer,
        container);
}

static void D3D12_ReleaseSampler(
    SDL_GPURenderer *driverData,
    SDL_GPUSampler *sampler)
{
    D3D12Renderer *renderer = (D3D12Renderer *)driverData;
    D3D12Sampler *d3d12Sampler = (D3D12Sampler *)sampler;

    SDL_LockMutex(renderer->disposeLock);

    EXPAND_ARRAY_IF_NEEDED(
        renderer->samplersToDestroy,
        D3D12Sampler *,
        renderer->samplersToDestroyCount + 1,
        renderer->samplersToDestroyCapacity,
        renderer->samplersToDestroyCapacity * 2);

    renderer->samplersToDestroy[renderer->samplersToDestroyCount] = d3d12Sampler;
    renderer->samplersToDestroyCount += 1;

    SDL_UnlockMutex(renderer->disposeLock);
}

static void D3D12_ReleaseBuffer(
    SDL_GPURenderer *driverData,
    SDL_GPUBuffer *buffer)
{
    D3D12Renderer *renderer = (D3D12Renderer *)driverData;
    D3D12BufferContainer *bufferContainer = (D3D12BufferContainer *)buffer;

    D3D12_INTERNAL_ReleaseBufferContainer(
        renderer,
        bufferContainer);
}

static void D3D12_ReleaseTransferBuffer(
    SDL_GPURenderer *driverData,
    SDL_GPUTransferBuffer *transferBuffer)
{
    D3D12Renderer *renderer = (D3D12Renderer *)driverData;
    D3D12BufferContainer *transferBufferContainer = (D3D12BufferContainer *)transferBuffer;

    D3D12_INTERNAL_ReleaseBufferContainer(
        renderer,
        transferBufferContainer);
}

static void D3D12_ReleaseShader(
    SDL_GPURenderer *driverData,
    SDL_GPUShader *shader)
{
    /* D3D12Renderer *renderer = (D3D12Renderer *)driverData; */
    D3D12Shader *d3d12shader = (D3D12Shader *)shader;

    if (d3d12shader->bytecode) {
        SDL_free(d3d12shader->bytecode);
        d3d12shader->bytecode = NULL;
    }
    SDL_free(d3d12shader);
}

static void D3D12_ReleaseComputePipeline(
    SDL_GPURenderer *driverData,
    SDL_GPUComputePipeline *computePipeline)
{
    D3D12Renderer *renderer = (D3D12Renderer *)driverData;
    D3D12ComputePipeline *d3d12ComputePipeline = (D3D12ComputePipeline *)computePipeline;

    SDL_LockMutex(renderer->disposeLock);

    EXPAND_ARRAY_IF_NEEDED(
        renderer->computePipelinesToDestroy,
        D3D12ComputePipeline *,
        renderer->computePipelinesToDestroyCount + 1,
        renderer->computePipelinesToDestroyCapacity,
        renderer->computePipelinesToDestroyCapacity * 2);

    renderer->computePipelinesToDestroy[renderer->computePipelinesToDestroyCount] = d3d12ComputePipeline;
    renderer->computePipelinesToDestroyCount += 1;

    SDL_UnlockMutex(renderer->disposeLock);
}

static void D3D12_ReleaseGraphicsPipeline(
    SDL_GPURenderer *driverData,
    SDL_GPUGraphicsPipeline *graphicsPipeline)
{
    D3D12Renderer *renderer = (D3D12Renderer *)driverData;
    D3D12GraphicsPipeline *d3d12GraphicsPipeline = (D3D12GraphicsPipeline *)graphicsPipeline;

    SDL_LockMutex(renderer->disposeLock);

    EXPAND_ARRAY_IF_NEEDED(
        renderer->graphicsPipelinesToDestroy,
        D3D12GraphicsPipeline *,
        renderer->graphicsPipelinesToDestroyCount + 1,
        renderer->graphicsPipelinesToDestroyCapacity,
        renderer->graphicsPipelinesToDestroyCapacity * 2);

    renderer->graphicsPipelinesToDestroy[renderer->graphicsPipelinesToDestroyCount] = d3d12GraphicsPipeline;
    renderer->graphicsPipelinesToDestroyCount += 1;

    SDL_UnlockMutex(renderer->disposeLock);
}

static void D3D12_INTERNAL_ReleaseBlitPipelines(SDL_GPURenderer *driverData)
{
    D3D12Renderer *renderer = (D3D12Renderer *)driverData;
    D3D12_ReleaseSampler(driverData, renderer->blitLinearSampler);
    D3D12_ReleaseSampler(driverData, renderer->blitNearestSampler);
    D3D12_ReleaseShader(driverData, renderer->blitVertexShader);
    D3D12_ReleaseShader(driverData, renderer->blitFrom2DShader);
    D3D12_ReleaseShader(driverData, renderer->blitFrom2DArrayShader);
    D3D12_ReleaseShader(driverData, renderer->blitFrom3DShader);
    D3D12_ReleaseShader(driverData, renderer->blitFromCubeShader);
    D3D12_ReleaseShader(driverData, renderer->blitFromCubeArrayShader);

    for (Uint32 i = 0; i < renderer->blitPipelineCount; i += 1) {
        D3D12_ReleaseGraphicsPipeline(driverData, renderer->blitPipelines[i].pipeline);
    }
    SDL_free(renderer->blitPipelines);
}

// Render Pass

static void D3D12_SetViewport(
    SDL_GPUCommandBuffer *commandBuffer,
    const SDL_GPUViewport *viewport)
{
    D3D12CommandBuffer *d3d12CommandBuffer = (D3D12CommandBuffer *)commandBuffer;
    D3D12_VIEWPORT d3d12Viewport;
    d3d12Viewport.TopLeftX = viewport->x;
    d3d12Viewport.TopLeftY = viewport->y;
    d3d12Viewport.Width = viewport->w;
    d3d12Viewport.Height = viewport->h;
    d3d12Viewport.MinDepth = viewport->min_depth;
    d3d12Viewport.MaxDepth = viewport->max_depth;
    ID3D12GraphicsCommandList_RSSetViewports(d3d12CommandBuffer->graphicsCommandList, 1, &d3d12Viewport);
}

static void D3D12_SetScissor(
    SDL_GPUCommandBuffer *commandBuffer,
    const SDL_Rect *scissor)
{
    D3D12CommandBuffer *d3d12CommandBuffer = (D3D12CommandBuffer *)commandBuffer;
    D3D12_RECT scissorRect;
    scissorRect.left = scissor->x;
    scissorRect.top = scissor->y;
    scissorRect.right = scissor->x + scissor->w;
    scissorRect.bottom = scissor->y + scissor->h;
    ID3D12GraphicsCommandList_RSSetScissorRects(d3d12CommandBuffer->graphicsCommandList, 1, &scissorRect);
}

static void D3D12_SetBlendConstants(
    SDL_GPUCommandBuffer *commandBuffer,
    SDL_FColor blendConstants)
{
    D3D12CommandBuffer *d3d12CommandBuffer = (D3D12CommandBuffer *)commandBuffer;
    FLOAT blendFactor[4] = { blendConstants.r, blendConstants.g, blendConstants.b, blendConstants.a };
    ID3D12GraphicsCommandList_OMSetBlendFactor(d3d12CommandBuffer->graphicsCommandList, blendFactor);
}

static void D3D12_SetStencilReference(
    SDL_GPUCommandBuffer *commandBuffer,
    Uint8 reference
) {
    D3D12CommandBuffer *d3d12CommandBuffer = (D3D12CommandBuffer *)commandBuffer;
    ID3D12GraphicsCommandList_OMSetStencilRef(d3d12CommandBuffer->graphicsCommandList, reference);
}

static D3D12TextureSubresource *D3D12_INTERNAL_FetchTextureSubresource(
    D3D12TextureContainer *container,
    Uint32 layer,
    Uint32 level)
{
    Uint32 index = D3D12_INTERNAL_CalcSubresource(
        level,
        layer,
        container->header.info.num_levels);
    return &container->activeTexture->subresources[index];
}

static void D3D12_INTERNAL_CycleActiveTexture(
    D3D12Renderer *renderer,
    D3D12TextureContainer *container)
{
    D3D12Texture *texture;

    // If a previously-cycled texture is available, we can use that.
    for (Uint32 i = 0; i < container->textureCount; i += 1) {
        texture = container->textures[i];

        if (SDL_GetAtomicInt(&texture->referenceCount) == 0) {
            container->activeTexture = texture;
            return;
        }
    }

    // No texture is available, generate a new one.
    texture = D3D12_INTERNAL_CreateTexture(
        renderer,
        &container->header.info,
        false,
        container->debugName);

    if (!texture) {
        return;
    }

    EXPAND_ARRAY_IF_NEEDED(
        container->textures,
        D3D12Texture *,
        container->textureCount + 1,
        container->textureCapacity,
        container->textureCapacity * 2);

    container->textures[container->textureCount] = texture;
    texture->container = container;
    texture->containerIndex = container->textureCount;
    container->textureCount += 1;

    container->activeTexture = texture;
}

static D3D12TextureSubresource *D3D12_INTERNAL_PrepareTextureSubresourceForWrite(
    D3D12CommandBuffer *commandBuffer,
    D3D12TextureContainer *container,
    Uint32 layer,
    Uint32 level,
    bool cycle,
    D3D12_RESOURCE_STATES destinationUsageMode)
{
    D3D12TextureSubresource *subresource = D3D12_INTERNAL_FetchTextureSubresource(
        container,
        layer,
        level);

    if (
        container->canBeCycled &&
        cycle &&
        SDL_GetAtomicInt(&subresource->parent->referenceCount) > 0) {
        D3D12_INTERNAL_CycleActiveTexture(
            commandBuffer->renderer,
            container);

        subresource = D3D12_INTERNAL_FetchTextureSubresource(
            container,
            layer,
            level);
    }

    D3D12_INTERNAL_TextureSubresourceTransitionFromDefaultUsage(
        commandBuffer,
        destinationUsageMode,
        subresource);

    return subresource;
}

static void D3D12_INTERNAL_CycleActiveBuffer(
    D3D12Renderer *renderer,
    D3D12BufferContainer *container)
{
    // If a previously-cycled buffer is available, we can use that.
    for (Uint32 i = 0; i < container->bufferCount; i += 1) {
        D3D12Buffer *buffer = container->buffers[i];
        if (SDL_GetAtomicInt(&buffer->referenceCount) == 0) {
            container->activeBuffer = buffer;
            return;
        }
    }

    // No buffer handle is available, create a new one.
    D3D12Buffer *buffer = D3D12_INTERNAL_CreateBuffer(
        renderer,
        container->usage,
        container->size,
        container->type,
        container->debugName);

    if (!buffer) {
        return;
    }

    EXPAND_ARRAY_IF_NEEDED(
        container->buffers,
        D3D12Buffer *,
        container->bufferCount + 1,
        container->bufferCapacity,
        container->bufferCapacity * 2);

    container->buffers[container->bufferCount] = buffer;
    buffer->container = container;
    buffer->containerIndex = container->bufferCount;
    container->bufferCount += 1;

    container->activeBuffer = buffer;

    if (renderer->debug_mode && container->debugName != NULL) {
        D3D12_INTERNAL_SetResourceName(
            renderer,
            container->activeBuffer->handle,
            container->debugName);
    }
}

static D3D12Buffer *D3D12_INTERNAL_PrepareBufferForWrite(
    D3D12CommandBuffer *commandBuffer,
    D3D12BufferContainer *container,
    bool cycle,
    D3D12_RESOURCE_STATES destinationState)
{
    if (
        cycle &&
        SDL_GetAtomicInt(&container->activeBuffer->referenceCount) > 0) {
        D3D12_INTERNAL_CycleActiveBuffer(
            commandBuffer->renderer,
            container);
    }

    D3D12_INTERNAL_BufferTransitionFromDefaultUsage(
        commandBuffer,
        destinationState,
        container->activeBuffer);

    return container->activeBuffer;
}

static void D3D12_BeginRenderPass(
    SDL_GPUCommandBuffer *commandBuffer,
    const SDL_GPUColorTargetInfo *colorTargetInfos,
    Uint32 numColorTargets,
    const SDL_GPUDepthStencilTargetInfo *depthStencilTargetInfo)
{
    D3D12CommandBuffer *d3d12CommandBuffer = (D3D12CommandBuffer *)commandBuffer;

    Uint32 framebufferWidth = SDL_MAX_UINT32;
    Uint32 framebufferHeight = SDL_MAX_UINT32;

    for (Uint32 i = 0; i < numColorTargets; i += 1) {
        D3D12TextureContainer *container = (D3D12TextureContainer *)colorTargetInfos[i].texture;
        Uint32 h = container->header.info.height >> colorTargetInfos[i].mip_level;
        Uint32 w = container->header.info.width >> colorTargetInfos[i].mip_level;

        // The framebuffer cannot be larger than the smallest target.

        if (w < framebufferWidth) {
            framebufferWidth = w;
        }

        if (h < framebufferHeight) {
            framebufferHeight = h;
        }
    }

    if (depthStencilTargetInfo != NULL) {
        D3D12TextureContainer *container = (D3D12TextureContainer *)depthStencilTargetInfo->texture;

        Uint32 h = container->header.info.height;
        Uint32 w = container->header.info.width;

        // The framebuffer cannot be larger than the smallest target.

        if (w < framebufferWidth) {
            framebufferWidth = w;
        }

        if (h < framebufferHeight) {
            framebufferHeight = h;
        }
    }

    D3D12_CPU_DESCRIPTOR_HANDLE rtvs[MAX_COLOR_TARGET_BINDINGS];

    for (Uint32 i = 0; i < numColorTargets; i += 1) {
        D3D12TextureContainer *container = (D3D12TextureContainer *)colorTargetInfos[i].texture;
        D3D12TextureSubresource *subresource = D3D12_INTERNAL_PrepareTextureSubresourceForWrite(
            d3d12CommandBuffer,
            container,
            container->header.info.type == SDL_GPU_TEXTURETYPE_3D ? 0 : colorTargetInfos[i].layer_or_depth_plane,
            colorTargetInfos[i].mip_level,
            colorTargetInfos[i].cycle,
            D3D12_RESOURCE_STATE_RENDER_TARGET);

        Uint32 rtvIndex = container->header.info.type == SDL_GPU_TEXTURETYPE_3D ? colorTargetInfos[i].layer_or_depth_plane : 0;
        D3D12_CPU_DESCRIPTOR_HANDLE rtv = subresource->rtvHandles[rtvIndex].cpuHandle;

        if (colorTargetInfos[i].load_op == SDL_GPU_LOADOP_CLEAR) {
            float clearColor[4];
            clearColor[0] = colorTargetInfos[i].clear_color.r;
            clearColor[1] = colorTargetInfos[i].clear_color.g;
            clearColor[2] = colorTargetInfos[i].clear_color.b;
            clearColor[3] = colorTargetInfos[i].clear_color.a;

            ID3D12GraphicsCommandList_ClearRenderTargetView(
                d3d12CommandBuffer->graphicsCommandList,
                rtv,
                clearColor,
                0,
                NULL);
        }

        rtvs[i] = rtv;
        d3d12CommandBuffer->colorTargetSubresources[i] = subresource;

        D3D12_INTERNAL_TrackTexture(d3d12CommandBuffer, subresource->parent);

        if (colorTargetInfos[i].store_op == SDL_GPU_STOREOP_RESOLVE || colorTargetInfos[i].store_op == SDL_GPU_STOREOP_RESOLVE_AND_STORE) {
            D3D12TextureContainer *resolveContainer = (D3D12TextureContainer *)colorTargetInfos[i].resolve_texture;
            D3D12TextureSubresource *resolveSubresource = D3D12_INTERNAL_PrepareTextureSubresourceForWrite(
                d3d12CommandBuffer,
                resolveContainer,
                colorTargetInfos[i].resolve_layer,
                colorTargetInfos[i].resolve_mip_level,
                colorTargetInfos[i].cycle_resolve_texture,
                D3D12_RESOURCE_STATE_RESOLVE_DEST);

            d3d12CommandBuffer->colorResolveSubresources[i] = resolveSubresource;
            D3D12_INTERNAL_TrackTexture(d3d12CommandBuffer, resolveSubresource->parent);
        }
    }

    D3D12_CPU_DESCRIPTOR_HANDLE dsv;
    if (depthStencilTargetInfo != NULL) {
        D3D12TextureContainer *container = (D3D12TextureContainer *)depthStencilTargetInfo->texture;
        D3D12TextureSubresource *subresource = D3D12_INTERNAL_PrepareTextureSubresourceForWrite(
            d3d12CommandBuffer,
            container,
            0,
            0,
            depthStencilTargetInfo->cycle,
            D3D12_RESOURCE_STATE_DEPTH_WRITE);

        if (
            depthStencilTargetInfo->load_op == SDL_GPU_LOADOP_CLEAR ||
            depthStencilTargetInfo->stencil_load_op == SDL_GPU_LOADOP_CLEAR) {
            D3D12_CLEAR_FLAGS clearFlags = (D3D12_CLEAR_FLAGS)0;
            if (depthStencilTargetInfo->load_op == SDL_GPU_LOADOP_CLEAR) {
                clearFlags |= D3D12_CLEAR_FLAG_DEPTH;
            }
            if (depthStencilTargetInfo->stencil_load_op == SDL_GPU_LOADOP_CLEAR) {
                clearFlags |= D3D12_CLEAR_FLAG_STENCIL;
            }

            ID3D12GraphicsCommandList_ClearDepthStencilView(
                d3d12CommandBuffer->graphicsCommandList,
                subresource->dsvHandle.cpuHandle,
                clearFlags,
                depthStencilTargetInfo->clear_depth,
                depthStencilTargetInfo->clear_stencil,
                0,
                NULL);
        }

        dsv = subresource->dsvHandle.cpuHandle;
        d3d12CommandBuffer->depthStencilTextureSubresource = subresource;
        D3D12_INTERNAL_TrackTexture(d3d12CommandBuffer, subresource->parent);
    }

    ID3D12GraphicsCommandList_OMSetRenderTargets(
        d3d12CommandBuffer->graphicsCommandList,
        numColorTargets,
        rtvs,
        false,
        (depthStencilTargetInfo == NULL) ? NULL : &dsv);

    // Set sensible default states
    SDL_GPUViewport defaultViewport;
    defaultViewport.x = 0;
    defaultViewport.y = 0;
    defaultViewport.w = (float)framebufferWidth;
    defaultViewport.h = (float)framebufferHeight;
    defaultViewport.min_depth = 0;
    defaultViewport.max_depth = 1;

    D3D12_SetViewport(
        commandBuffer,
        &defaultViewport);

    SDL_Rect defaultScissor;
    defaultScissor.x = 0;
    defaultScissor.y = 0;
    defaultScissor.w = (Sint32)framebufferWidth;
    defaultScissor.h = (Sint32)framebufferHeight;

    D3D12_SetScissor(
        commandBuffer,
        &defaultScissor);

    D3D12_SetStencilReference(
        commandBuffer,
        0);

    SDL_FColor blendConstants;
    blendConstants.r = 1.0f;
    blendConstants.g = 1.0f;
    blendConstants.b = 1.0f;
    blendConstants.a = 1.0f;

    D3D12_SetBlendConstants(
        commandBuffer,
        blendConstants);
}

static void D3D12_INTERNAL_TrackUniformBuffer(
    D3D12CommandBuffer *commandBuffer,
    D3D12UniformBuffer *uniformBuffer)
{
    Uint32 i;
    for (i = 0; i < commandBuffer->usedUniformBufferCount; i += 1) {
        if (commandBuffer->usedUniformBuffers[i] == uniformBuffer) {
            return;
        }
    }

    if (commandBuffer->usedUniformBufferCount == commandBuffer->usedUniformBufferCapacity) {
        commandBuffer->usedUniformBufferCapacity += 1;
        commandBuffer->usedUniformBuffers = (D3D12UniformBuffer **)SDL_realloc(
            commandBuffer->usedUniformBuffers,
            commandBuffer->usedUniformBufferCapacity * sizeof(D3D12UniformBuffer *));
    }

    commandBuffer->usedUniformBuffers[commandBuffer->usedUniformBufferCount] = uniformBuffer;
    commandBuffer->usedUniformBufferCount += 1;

    D3D12_INTERNAL_TrackBuffer(
        commandBuffer,
        uniformBuffer->buffer);
}

static D3D12UniformBuffer *D3D12_INTERNAL_AcquireUniformBufferFromPool(
    D3D12CommandBuffer *commandBuffer)
{
    D3D12Renderer *renderer = commandBuffer->renderer;
    D3D12UniformBuffer *uniformBuffer;

    SDL_LockMutex(renderer->acquireUniformBufferLock);

    if (renderer->uniformBufferPoolCount > 0) {
        uniformBuffer = renderer->uniformBufferPool[renderer->uniformBufferPoolCount - 1];
        renderer->uniformBufferPoolCount -= 1;
    } else {
        uniformBuffer = (D3D12UniformBuffer *)SDL_calloc(1, sizeof(D3D12UniformBuffer));
        if (!uniformBuffer) {
            SDL_UnlockMutex(renderer->acquireUniformBufferLock);
            return NULL;
        }

        uniformBuffer->buffer = D3D12_INTERNAL_CreateBuffer(
            renderer,
            0,
            UNIFORM_BUFFER_SIZE,
            D3D12_BUFFER_TYPE_UNIFORM,
            NULL);
        if (!uniformBuffer->buffer) {
            SDL_UnlockMutex(renderer->acquireUniformBufferLock);
            return NULL;
        }
    }

    SDL_UnlockMutex(renderer->acquireUniformBufferLock);

    uniformBuffer->currentBlockSize = 0;
    uniformBuffer->drawOffset = 0;
    uniformBuffer->writeOffset = 0;

    HRESULT res = ID3D12Resource_Map(
        uniformBuffer->buffer->handle,
        0,
        NULL,
        (void **)&uniformBuffer->buffer->mapPointer);
    CHECK_D3D12_ERROR_AND_RETURN("Failed to map buffer pool!", NULL);

    D3D12_INTERNAL_TrackUniformBuffer(commandBuffer, uniformBuffer);

    return uniformBuffer;
}

static void D3D12_INTERNAL_ReturnUniformBufferToPool(
    D3D12Renderer *renderer,
    D3D12UniformBuffer *uniformBuffer)
{
    if (renderer->uniformBufferPoolCount >= renderer->uniformBufferPoolCapacity) {
        renderer->uniformBufferPoolCapacity *= 2;
        renderer->uniformBufferPool = (D3D12UniformBuffer **)SDL_realloc(
            renderer->uniformBufferPool,
            renderer->uniformBufferPoolCapacity * sizeof(D3D12UniformBuffer *));
    }

    renderer->uniformBufferPool[renderer->uniformBufferPoolCount] = uniformBuffer;
    renderer->uniformBufferPoolCount += 1;
}

static void D3D12_INTERNAL_PushUniformData(
    D3D12CommandBuffer *commandBuffer,
    SDL_GPUShaderStage shaderStage,
    Uint32 slotIndex,
    const void *data,
    Uint32 length)
{
    D3D12UniformBuffer *uniformBuffer;

    if (shaderStage == SDL_GPU_SHADERSTAGE_VERTEX) {
        if (commandBuffer->vertexUniformBuffers[slotIndex] == NULL) {
            commandBuffer->vertexUniformBuffers[slotIndex] = D3D12_INTERNAL_AcquireUniformBufferFromPool(
                commandBuffer);
        }
        uniformBuffer = commandBuffer->vertexUniformBuffers[slotIndex];
    } else if (shaderStage == SDL_GPU_SHADERSTAGE_FRAGMENT) {
        if (commandBuffer->fragmentUniformBuffers[slotIndex] == NULL) {
            commandBuffer->fragmentUniformBuffers[slotIndex] = D3D12_INTERNAL_AcquireUniformBufferFromPool(
                commandBuffer);
        }
        uniformBuffer = commandBuffer->fragmentUniformBuffers[slotIndex];
    } else if (shaderStage == SDL_GPU_SHADERSTAGE_COMPUTE) {
        if (commandBuffer->computeUniformBuffers[slotIndex] == NULL) {
            commandBuffer->computeUniformBuffers[slotIndex] = D3D12_INTERNAL_AcquireUniformBufferFromPool(
                commandBuffer);
        }
        uniformBuffer = commandBuffer->computeUniformBuffers[slotIndex];
    } else {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Unrecognized shader stage!");
        return;
    }

    uniformBuffer->currentBlockSize =
        D3D12_INTERNAL_Align(
            length,
            256);

    // If there is no more room, acquire a new uniform buffer
    if (uniformBuffer->writeOffset + uniformBuffer->currentBlockSize >= UNIFORM_BUFFER_SIZE) {
        ID3D12Resource_Unmap(
            uniformBuffer->buffer->handle,
            0,
            NULL);
        uniformBuffer->buffer->mapPointer = NULL;

        uniformBuffer = D3D12_INTERNAL_AcquireUniformBufferFromPool(commandBuffer);

        uniformBuffer->drawOffset = 0;
        uniformBuffer->writeOffset = 0;

        if (shaderStage == SDL_GPU_SHADERSTAGE_VERTEX) {
            commandBuffer->vertexUniformBuffers[slotIndex] = uniformBuffer;
        } else if (shaderStage == SDL_GPU_SHADERSTAGE_FRAGMENT) {
            commandBuffer->fragmentUniformBuffers[slotIndex] = uniformBuffer;
        } else if (shaderStage == SDL_GPU_SHADERSTAGE_COMPUTE) {
            commandBuffer->computeUniformBuffers[slotIndex] = uniformBuffer;
        } else {
            SDL_LogError(SDL_LOG_CATEGORY_GPU, "Unrecognized shader stage!");
        }
    }

    uniformBuffer->drawOffset = uniformBuffer->writeOffset;

    SDL_memcpy(
        (Uint8 *)uniformBuffer->buffer->mapPointer + uniformBuffer->writeOffset,
        data,
        length);

    uniformBuffer->writeOffset += uniformBuffer->currentBlockSize;

    if (shaderStage == SDL_GPU_SHADERSTAGE_VERTEX) {
        commandBuffer->needVertexUniformBufferBind[slotIndex] = true;
    } else if (shaderStage == SDL_GPU_SHADERSTAGE_FRAGMENT) {
        commandBuffer->needFragmentUniformBufferBind[slotIndex] = true;
    } else if (shaderStage == SDL_GPU_SHADERSTAGE_COMPUTE) {
        commandBuffer->needComputeUniformBufferBind[slotIndex] = true;
    } else {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Unrecognized shader stage!");
    }
}

static void D3D12_BindGraphicsPipeline(
    SDL_GPUCommandBuffer *commandBuffer,
    SDL_GPUGraphicsPipeline *graphicsPipeline)
{
    D3D12CommandBuffer *d3d12CommandBuffer = (D3D12CommandBuffer *)commandBuffer;
    D3D12GraphicsPipeline *pipeline = (D3D12GraphicsPipeline *)graphicsPipeline;
    Uint32 i;

    d3d12CommandBuffer->currentGraphicsPipeline = pipeline;

    // Set the pipeline state
    ID3D12GraphicsCommandList_SetPipelineState(d3d12CommandBuffer->graphicsCommandList, pipeline->pipelineState);
    ID3D12GraphicsCommandList_SetGraphicsRootSignature(d3d12CommandBuffer->graphicsCommandList, pipeline->rootSignature->handle);
    ID3D12GraphicsCommandList_IASetPrimitiveTopology(d3d12CommandBuffer->graphicsCommandList, SDLToD3D12_PrimitiveType[pipeline->primitiveType]);

    // Mark that bindings are needed
    d3d12CommandBuffer->needVertexSamplerBind = true;
    d3d12CommandBuffer->needVertexStorageTextureBind = true;
    d3d12CommandBuffer->needVertexStorageBufferBind = true;
    d3d12CommandBuffer->needFragmentSamplerBind = true;
    d3d12CommandBuffer->needFragmentStorageTextureBind = true;
    d3d12CommandBuffer->needFragmentStorageBufferBind = true;

    for (i = 0; i < MAX_UNIFORM_BUFFERS_PER_STAGE; i += 1) {
        d3d12CommandBuffer->needVertexUniformBufferBind[i] = true;
        d3d12CommandBuffer->needFragmentUniformBufferBind[i] = true;
    }

    for (i = 0; i < pipeline->vertexUniformBufferCount; i += 1) {
        if (d3d12CommandBuffer->vertexUniformBuffers[i] == NULL) {
            d3d12CommandBuffer->vertexUniformBuffers[i] = D3D12_INTERNAL_AcquireUniformBufferFromPool(
                d3d12CommandBuffer);
        }
    }

    for (i = 0; i < pipeline->fragmentUniformBufferCount; i += 1) {
        if (d3d12CommandBuffer->fragmentUniformBuffers[i] == NULL) {
            d3d12CommandBuffer->fragmentUniformBuffers[i] = D3D12_INTERNAL_AcquireUniformBufferFromPool(
                d3d12CommandBuffer);
        }
    }

    D3D12_INTERNAL_TrackGraphicsPipeline(d3d12CommandBuffer, pipeline);
}

static void D3D12_BindVertexBuffers(
    SDL_GPUCommandBuffer *commandBuffer,
    Uint32 firstSlot,
    const SDL_GPUBufferBinding *bindings,
    Uint32 numBindings)
{
    D3D12CommandBuffer *d3d12CommandBuffer = (D3D12CommandBuffer *)commandBuffer;

    for (Uint32 i = 0; i < numBindings; i += 1) {
        D3D12Buffer *currentBuffer = ((D3D12BufferContainer *)bindings[i].buffer)->activeBuffer;

        if (d3d12CommandBuffer->vertexBuffers[firstSlot + i] != currentBuffer || d3d12CommandBuffer->vertexBufferOffsets[firstSlot + i] != bindings[i].offset) {
            D3D12_INTERNAL_TrackBuffer(d3d12CommandBuffer, currentBuffer);

            d3d12CommandBuffer->vertexBuffers[firstSlot + i] = currentBuffer;
            d3d12CommandBuffer->vertexBufferOffsets[firstSlot + i] = bindings[i].offset;
            d3d12CommandBuffer->needVertexBufferBind = true;
        }
    }

    d3d12CommandBuffer->vertexBufferCount =
        SDL_max(d3d12CommandBuffer->vertexBufferCount, firstSlot + numBindings);
}

static void D3D12_BindIndexBuffer(
    SDL_GPUCommandBuffer *commandBuffer,
    const SDL_GPUBufferBinding *binding,
    SDL_GPUIndexElementSize indexElementSize)
{
    D3D12CommandBuffer *d3d12CommandBuffer = (D3D12CommandBuffer *)commandBuffer;
    D3D12Buffer *buffer = ((D3D12BufferContainer *)binding->buffer)->activeBuffer;
    D3D12_INDEX_BUFFER_VIEW view;

    D3D12_INTERNAL_TrackBuffer(d3d12CommandBuffer, buffer);

    view.BufferLocation = buffer->virtualAddress + binding->offset;
    view.SizeInBytes = buffer->container->size - binding->offset;
    view.Format =
        indexElementSize == SDL_GPU_INDEXELEMENTSIZE_16BIT ? DXGI_FORMAT_R16_UINT : DXGI_FORMAT_R32_UINT;

    ID3D12GraphicsCommandList_IASetIndexBuffer(
        d3d12CommandBuffer->graphicsCommandList,
        &view);
}

static void D3D12_BindVertexSamplers(
    SDL_GPUCommandBuffer *commandBuffer,
    Uint32 firstSlot,
    const SDL_GPUTextureSamplerBinding *textureSamplerBindings,
    Uint32 numBindings)
{
    D3D12CommandBuffer *d3d12CommandBuffer = (D3D12CommandBuffer *)commandBuffer;

    for (Uint32 i = 0; i < numBindings; i += 1) {
        D3D12TextureContainer *container = (D3D12TextureContainer *)textureSamplerBindings[i].texture;
        D3D12Sampler *sampler = (D3D12Sampler *)textureSamplerBindings[i].sampler;

        if (d3d12CommandBuffer->vertexSamplers[firstSlot + i] != sampler) {
            D3D12_INTERNAL_TrackSampler(
                d3d12CommandBuffer,
                sampler);

            d3d12CommandBuffer->vertexSamplers[firstSlot + i] = sampler;
            d3d12CommandBuffer->needVertexSamplerBind = true;
        }

        if (d3d12CommandBuffer->vertexSamplerTextures[firstSlot + i] != container->activeTexture) {
            D3D12_INTERNAL_TrackTexture(
                d3d12CommandBuffer,
                container->activeTexture);

            d3d12CommandBuffer->vertexSamplerTextures[firstSlot + i] = container->activeTexture;
            d3d12CommandBuffer->needVertexSamplerBind = true;
        }
    }
}

static void D3D12_BindVertexStorageTextures(
    SDL_GPUCommandBuffer *commandBuffer,
    Uint32 firstSlot,
    SDL_GPUTexture *const *storageTextures,
    Uint32 numBindings)
{
    D3D12CommandBuffer *d3d12CommandBuffer = (D3D12CommandBuffer *)commandBuffer;

    for (Uint32 i = 0; i < numBindings; i += 1) {
        D3D12TextureContainer *container = (D3D12TextureContainer *)storageTextures[i];
        D3D12Texture *texture = container->activeTexture;

        if (d3d12CommandBuffer->vertexStorageTextures[firstSlot + i] != texture) {
            D3D12_INTERNAL_TrackTexture(d3d12CommandBuffer, texture);

            d3d12CommandBuffer->vertexStorageTextures[firstSlot + i] = texture;
            d3d12CommandBuffer->needVertexStorageTextureBind = true;
        }
    }
}

static void D3D12_BindVertexStorageBuffers(
    SDL_GPUCommandBuffer *commandBuffer,
    Uint32 firstSlot,
    SDL_GPUBuffer *const *storageBuffers,
    Uint32 numBindings)
{
    D3D12CommandBuffer *d3d12CommandBuffer = (D3D12CommandBuffer *)commandBuffer;

    for (Uint32 i = 0; i < numBindings; i += 1) {
        D3D12BufferContainer *container = (D3D12BufferContainer *)storageBuffers[i];
        if (d3d12CommandBuffer->vertexStorageBuffers[firstSlot + i] != container->activeBuffer) {
            D3D12_INTERNAL_TrackBuffer(
                d3d12CommandBuffer,
                container->activeBuffer);

            d3d12CommandBuffer->vertexStorageBuffers[firstSlot + i] = container->activeBuffer;
            d3d12CommandBuffer->needVertexStorageBufferBind = true;
        }
    }
}

static void D3D12_BindFragmentSamplers(
    SDL_GPUCommandBuffer *commandBuffer,
    Uint32 firstSlot,
    const SDL_GPUTextureSamplerBinding *textureSamplerBindings,
    Uint32 numBindings)
{
    D3D12CommandBuffer *d3d12CommandBuffer = (D3D12CommandBuffer *)commandBuffer;

    for (Uint32 i = 0; i < numBindings; i += 1) {
        D3D12TextureContainer *container = (D3D12TextureContainer *)textureSamplerBindings[i].texture;
        D3D12Sampler *sampler = (D3D12Sampler *)textureSamplerBindings[i].sampler;

        if (d3d12CommandBuffer->fragmentSamplers[firstSlot + i] != sampler) {
            D3D12_INTERNAL_TrackSampler(
                d3d12CommandBuffer,
                sampler);

            d3d12CommandBuffer->fragmentSamplers[firstSlot + i] = sampler;
            d3d12CommandBuffer->needFragmentSamplerBind = true;
        }

        if (d3d12CommandBuffer->fragmentSamplerTextures[firstSlot + i] != container->activeTexture) {
            D3D12_INTERNAL_TrackTexture(
                d3d12CommandBuffer,
                container->activeTexture);

            d3d12CommandBuffer->fragmentSamplerTextures[firstSlot + i] = container->activeTexture;
            d3d12CommandBuffer->needFragmentSamplerBind = true;
        }
    }
}

static void D3D12_BindFragmentStorageTextures(
    SDL_GPUCommandBuffer *commandBuffer,
    Uint32 firstSlot,
    SDL_GPUTexture *const *storageTextures,
    Uint32 numBindings)
{
    D3D12CommandBuffer *d3d12CommandBuffer = (D3D12CommandBuffer *)commandBuffer;

    for (Uint32 i = 0; i < numBindings; i += 1) {
        D3D12TextureContainer *container = (D3D12TextureContainer *)storageTextures[i];
        D3D12Texture *texture = container->activeTexture;

        if (d3d12CommandBuffer->fragmentStorageTextures[firstSlot + i] != texture) {
            D3D12_INTERNAL_TrackTexture(d3d12CommandBuffer, texture);

            d3d12CommandBuffer->fragmentStorageTextures[firstSlot + i] = texture;
            d3d12CommandBuffer->needFragmentStorageTextureBind = true;
        }
    }
}

static void D3D12_BindFragmentStorageBuffers(
    SDL_GPUCommandBuffer *commandBuffer,
    Uint32 firstSlot,
    SDL_GPUBuffer *const *storageBuffers,
    Uint32 numBindings)
{
    D3D12CommandBuffer *d3d12CommandBuffer = (D3D12CommandBuffer *)commandBuffer;

    for (Uint32 i = 0; i < numBindings; i += 1) {
        D3D12BufferContainer *container = (D3D12BufferContainer *)storageBuffers[i];

        if (d3d12CommandBuffer->fragmentStorageBuffers[firstSlot + i] != container->activeBuffer) {
            D3D12_INTERNAL_TrackBuffer(
                d3d12CommandBuffer,
                container->activeBuffer);

            d3d12CommandBuffer->fragmentStorageBuffers[firstSlot + i] = container->activeBuffer;
            d3d12CommandBuffer->needFragmentStorageBufferBind = true;
        }
    }
}

static void D3D12_PushVertexUniformData(
    SDL_GPUCommandBuffer *commandBuffer,
    Uint32 slotIndex,
    const void *data,
    Uint32 length)
{
    D3D12CommandBuffer *d3d12CommandBuffer = (D3D12CommandBuffer *)commandBuffer;

    D3D12_INTERNAL_PushUniformData(
        d3d12CommandBuffer,
        SDL_GPU_SHADERSTAGE_VERTEX,
        slotIndex,
        data,
        length);
}

static void D3D12_PushFragmentUniformData(
    SDL_GPUCommandBuffer *commandBuffer,
    Uint32 slotIndex,
    const void *data,
    Uint32 length)
{
    D3D12CommandBuffer *d3d12CommandBuffer = (D3D12CommandBuffer *)commandBuffer;

    D3D12_INTERNAL_PushUniformData(
        d3d12CommandBuffer,
        SDL_GPU_SHADERSTAGE_FRAGMENT,
        slotIndex,
        data,
        length);
}

static void D3D12_INTERNAL_SetGPUDescriptorHeaps(D3D12CommandBuffer *commandBuffer)
{
    ID3D12DescriptorHeap *heaps[2];
    D3D12DescriptorHeap *viewHeap;
    D3D12DescriptorHeap *samplerHeap;

    viewHeap = D3D12_INTERNAL_AcquireGPUDescriptorHeapFromPool(commandBuffer, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    samplerHeap = D3D12_INTERNAL_AcquireGPUDescriptorHeapFromPool(commandBuffer, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);

    commandBuffer->gpuDescriptorHeaps[0] = viewHeap;
    commandBuffer->gpuDescriptorHeaps[1] = samplerHeap;

    heaps[0] = viewHeap->handle;
    heaps[1] = samplerHeap->handle;

    ID3D12GraphicsCommandList_SetDescriptorHeaps(
        commandBuffer->graphicsCommandList,
        2,
        heaps);
}

static void D3D12_INTERNAL_WriteGPUDescriptors(
    D3D12CommandBuffer *commandBuffer,
    D3D12_DESCRIPTOR_HEAP_TYPE heapType,
    D3D12_CPU_DESCRIPTOR_HANDLE *resourceDescriptorHandles,
    Uint32 resourceHandleCount,
    D3D12_GPU_DESCRIPTOR_HANDLE *gpuBaseDescriptor)
{
    D3D12DescriptorHeap *heap;
    D3D12_CPU_DESCRIPTOR_HANDLE gpuHeapCpuHandle;

    /* Descriptor overflow, acquire new heaps */
    if (commandBuffer->gpuDescriptorHeaps[heapType]->currentDescriptorIndex >= commandBuffer->gpuDescriptorHeaps[heapType]->maxDescriptors) {
        D3D12_INTERNAL_SetGPUDescriptorHeaps(commandBuffer);
    }

    heap = commandBuffer->gpuDescriptorHeaps[heapType];

    // FIXME: need to error on overflow
    gpuHeapCpuHandle.ptr = heap->descriptorHeapCPUStart.ptr + (heap->currentDescriptorIndex * heap->descriptorSize);
    gpuBaseDescriptor->ptr = heap->descriptorHeapGPUStart.ptr + (heap->currentDescriptorIndex * heap->descriptorSize);

    for (Uint32 i = 0; i < resourceHandleCount; i += 1) {
        ID3D12Device_CopyDescriptorsSimple(
            commandBuffer->renderer->device,
            1,
            gpuHeapCpuHandle,
            resourceDescriptorHandles[i],
            heapType);

        heap->currentDescriptorIndex += 1;
        gpuHeapCpuHandle.ptr += heap->descriptorSize;
    }
}

static void D3D12_INTERNAL_BindGraphicsResources(
    D3D12CommandBuffer *commandBuffer)
{
    D3D12GraphicsPipeline *graphicsPipeline = commandBuffer->currentGraphicsPipeline;

    /* Acquire GPU descriptor heaps if we haven't yet */
    if (commandBuffer->gpuDescriptorHeaps[0] == NULL) {
        D3D12_INTERNAL_SetGPUDescriptorHeaps(commandBuffer);
    }

    D3D12_CPU_DESCRIPTOR_HANDLE cpuHandles[MAX_TEXTURE_SAMPLERS_PER_STAGE];
    D3D12_GPU_DESCRIPTOR_HANDLE gpuDescriptorHandle;
    D3D12_VERTEX_BUFFER_VIEW vertexBufferViews[MAX_VERTEX_BUFFERS];

    if (commandBuffer->needVertexBufferBind) {
        for (Uint32 i = 0; i < commandBuffer->vertexBufferCount; i += 1) {
            vertexBufferViews[i].BufferLocation = commandBuffer->vertexBuffers[i]->virtualAddress + commandBuffer->vertexBufferOffsets[i];
            vertexBufferViews[i].SizeInBytes = commandBuffer->vertexBuffers[i]->container->size - commandBuffer->vertexBufferOffsets[i];
            vertexBufferViews[i].StrideInBytes = graphicsPipeline->vertexStrides[i];
        }

        ID3D12GraphicsCommandList_IASetVertexBuffers(
            commandBuffer->graphicsCommandList,
            0,
            commandBuffer->vertexBufferCount,
            vertexBufferViews);
    }

    if (commandBuffer->needVertexSamplerBind) {
        if (graphicsPipeline->vertexSamplerCount > 0) {
            for (Uint32 i = 0; i < graphicsPipeline->vertexSamplerCount; i += 1) {
                cpuHandles[i] = commandBuffer->vertexSamplers[i]->handle.cpuHandle;
            }

            D3D12_INTERNAL_WriteGPUDescriptors(
                commandBuffer,
                D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
                cpuHandles,
                graphicsPipeline->vertexSamplerCount,
                &gpuDescriptorHandle);

            ID3D12GraphicsCommandList_SetGraphicsRootDescriptorTable(
                commandBuffer->graphicsCommandList,
                graphicsPipeline->rootSignature->vertexSamplerRootIndex,
                gpuDescriptorHandle);

            for (Uint32 i = 0; i < graphicsPipeline->vertexSamplerCount; i += 1) {
                cpuHandles[i] = commandBuffer->vertexSamplerTextures[i]->srvHandle.cpuHandle;
            }

            D3D12_INTERNAL_WriteGPUDescriptors(
                commandBuffer,
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                cpuHandles,
                graphicsPipeline->vertexSamplerCount,
                &gpuDescriptorHandle);

            ID3D12GraphicsCommandList_SetGraphicsRootDescriptorTable(
                commandBuffer->graphicsCommandList,
                graphicsPipeline->rootSignature->vertexSamplerTextureRootIndex,
                gpuDescriptorHandle);
        }
        commandBuffer->needVertexSamplerBind = false;
    }

    if (commandBuffer->needVertexStorageTextureBind) {
        if (graphicsPipeline->vertexStorageTextureCount > 0) {
            for (Uint32 i = 0; i < graphicsPipeline->vertexStorageTextureCount; i += 1) {
                cpuHandles[i] = commandBuffer->vertexStorageTextures[i]->srvHandle.cpuHandle;
            }

            D3D12_INTERNAL_WriteGPUDescriptors(
                commandBuffer,
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                cpuHandles,
                graphicsPipeline->vertexStorageTextureCount,
                &gpuDescriptorHandle);

            ID3D12GraphicsCommandList_SetGraphicsRootDescriptorTable(
                commandBuffer->graphicsCommandList,
                graphicsPipeline->rootSignature->vertexStorageTextureRootIndex,
                gpuDescriptorHandle);
        }
        commandBuffer->needVertexStorageTextureBind = false;
    }

    if (commandBuffer->needVertexStorageBufferBind) {
        if (graphicsPipeline->vertexStorageBufferCount > 0) {
            for (Uint32 i = 0; i < graphicsPipeline->vertexStorageBufferCount; i += 1) {
                cpuHandles[i] = commandBuffer->vertexStorageBuffers[i]->srvDescriptor.cpuHandle;
            }

            D3D12_INTERNAL_WriteGPUDescriptors(
                commandBuffer,
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                cpuHandles,
                graphicsPipeline->vertexStorageBufferCount,
                &gpuDescriptorHandle);

            ID3D12GraphicsCommandList_SetGraphicsRootDescriptorTable(
                commandBuffer->graphicsCommandList,
                graphicsPipeline->rootSignature->vertexStorageBufferRootIndex,
                gpuDescriptorHandle);
        }
        commandBuffer->needVertexStorageBufferBind = false;
    }

    for (Uint32 i = 0; i < MAX_UNIFORM_BUFFERS_PER_STAGE; i += 1) {
        if (commandBuffer->needVertexUniformBufferBind[i]) {
            if (graphicsPipeline->vertexUniformBufferCount > i) {
                ID3D12GraphicsCommandList_SetGraphicsRootConstantBufferView(
                    commandBuffer->graphicsCommandList,
                    graphicsPipeline->rootSignature->vertexUniformBufferRootIndex[i],
                    commandBuffer->vertexUniformBuffers[i]->buffer->virtualAddress + commandBuffer->vertexUniformBuffers[i]->drawOffset);
            }
            commandBuffer->needVertexUniformBufferBind[i] = false;
        }
    }

    if (commandBuffer->needFragmentSamplerBind) {
        if (graphicsPipeline->fragmentSamplerCount > 0) {
            for (Uint32 i = 0; i < graphicsPipeline->fragmentSamplerCount; i += 1) {
                cpuHandles[i] = commandBuffer->fragmentSamplers[i]->handle.cpuHandle;
            }

            D3D12_INTERNAL_WriteGPUDescriptors(
                commandBuffer,
                D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
                cpuHandles,
                graphicsPipeline->fragmentSamplerCount,
                &gpuDescriptorHandle);

            ID3D12GraphicsCommandList_SetGraphicsRootDescriptorTable(
                commandBuffer->graphicsCommandList,
                graphicsPipeline->rootSignature->fragmentSamplerRootIndex,
                gpuDescriptorHandle);

            for (Uint32 i = 0; i < graphicsPipeline->fragmentSamplerCount; i += 1) {
                cpuHandles[i] = commandBuffer->fragmentSamplerTextures[i]->srvHandle.cpuHandle;
            }

            D3D12_INTERNAL_WriteGPUDescriptors(
                commandBuffer,
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                cpuHandles,
                graphicsPipeline->fragmentSamplerCount,
                &gpuDescriptorHandle);

            ID3D12GraphicsCommandList_SetGraphicsRootDescriptorTable(
                commandBuffer->graphicsCommandList,
                graphicsPipeline->rootSignature->fragmentSamplerTextureRootIndex,
                gpuDescriptorHandle);
        }
        commandBuffer->needFragmentSamplerBind = false;
    }

    if (commandBuffer->needFragmentStorageTextureBind) {
        if (graphicsPipeline->fragmentStorageTextureCount > 0) {
            for (Uint32 i = 0; i < graphicsPipeline->fragmentStorageTextureCount; i += 1) {
                cpuHandles[i] = commandBuffer->fragmentStorageTextures[i]->srvHandle.cpuHandle;
            }

            D3D12_INTERNAL_WriteGPUDescriptors(
                commandBuffer,
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                cpuHandles,
                graphicsPipeline->fragmentStorageTextureCount,
                &gpuDescriptorHandle);

            ID3D12GraphicsCommandList_SetGraphicsRootDescriptorTable(
                commandBuffer->graphicsCommandList,
                graphicsPipeline->rootSignature->fragmentStorageTextureRootIndex,
                gpuDescriptorHandle);
        }
        commandBuffer->needFragmentStorageTextureBind = false;
    }

    if (commandBuffer->needFragmentStorageBufferBind) {
        if (graphicsPipeline->fragmentStorageBufferCount > 0) {
            for (Uint32 i = 0; i < graphicsPipeline->fragmentStorageBufferCount; i += 1) {
                cpuHandles[i] = commandBuffer->fragmentStorageBuffers[i]->srvDescriptor.cpuHandle;
            }

            D3D12_INTERNAL_WriteGPUDescriptors(
                commandBuffer,
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                cpuHandles,
                graphicsPipeline->fragmentStorageBufferCount,
                &gpuDescriptorHandle);

            ID3D12GraphicsCommandList_SetGraphicsRootDescriptorTable(
                commandBuffer->graphicsCommandList,
                graphicsPipeline->rootSignature->fragmentStorageBufferRootIndex,
                gpuDescriptorHandle);
        }
        commandBuffer->needFragmentStorageBufferBind = false;
    }

    for (Uint32 i = 0; i < MAX_UNIFORM_BUFFERS_PER_STAGE; i += 1) {
        if (commandBuffer->needFragmentUniformBufferBind[i]) {
            if (graphicsPipeline->fragmentUniformBufferCount > i) {
                ID3D12GraphicsCommandList_SetGraphicsRootConstantBufferView(
                    commandBuffer->graphicsCommandList,
                    graphicsPipeline->rootSignature->fragmentUniformBufferRootIndex[i],
                    commandBuffer->fragmentUniformBuffers[i]->buffer->virtualAddress + commandBuffer->fragmentUniformBuffers[i]->drawOffset);
            }
            commandBuffer->needFragmentUniformBufferBind[i] = false;
        }
    }
}

static void D3D12_DrawIndexedPrimitives(
    SDL_GPUCommandBuffer *commandBuffer,
    Uint32 numIndices,
    Uint32 numInstances,
    Uint32 firstIndex,
    Sint32 vertexOffset,
    Uint32 firstInstance)
{
    D3D12CommandBuffer *d3d12CommandBuffer = (D3D12CommandBuffer *)commandBuffer;
    D3D12_INTERNAL_BindGraphicsResources(d3d12CommandBuffer);

    ID3D12GraphicsCommandList_DrawIndexedInstanced(
        d3d12CommandBuffer->graphicsCommandList,
        numIndices,
        numInstances,
        firstIndex,
        vertexOffset,
        firstInstance);
}

static void D3D12_DrawPrimitives(
    SDL_GPUCommandBuffer *commandBuffer,
    Uint32 numVertices,
    Uint32 numInstances,
    Uint32 firstVertex,
    Uint32 firstInstance)
{
    D3D12CommandBuffer *d3d12CommandBuffer = (D3D12CommandBuffer *)commandBuffer;
    D3D12_INTERNAL_BindGraphicsResources(d3d12CommandBuffer);

    ID3D12GraphicsCommandList_DrawInstanced(
        d3d12CommandBuffer->graphicsCommandList,
        numVertices,
        numInstances,
        firstVertex,
        firstInstance);
}

static void D3D12_DrawPrimitivesIndirect(
    SDL_GPUCommandBuffer *commandBuffer,
    SDL_GPUBuffer *buffer,
    Uint32 offset,
    Uint32 drawCount)
{
    D3D12CommandBuffer *d3d12CommandBuffer = (D3D12CommandBuffer *)commandBuffer;
    D3D12Buffer *d3d12Buffer = ((D3D12BufferContainer *)buffer)->activeBuffer;

    D3D12_INTERNAL_BindGraphicsResources(d3d12CommandBuffer);

    ID3D12GraphicsCommandList_ExecuteIndirect(
        d3d12CommandBuffer->graphicsCommandList,
        d3d12CommandBuffer->renderer->indirectDrawCommandSignature,
        drawCount,
        d3d12Buffer->handle,
        offset,
        NULL,
        0);

    D3D12_INTERNAL_TrackBuffer(d3d12CommandBuffer, d3d12Buffer);
}

static void D3D12_DrawIndexedPrimitivesIndirect(
    SDL_GPUCommandBuffer *commandBuffer,
    SDL_GPUBuffer *buffer,
    Uint32 offset,
    Uint32 drawCount)
{
    D3D12CommandBuffer *d3d12CommandBuffer = (D3D12CommandBuffer *)commandBuffer;
    D3D12Buffer *d3d12Buffer = ((D3D12BufferContainer *)buffer)->activeBuffer;

    D3D12_INTERNAL_BindGraphicsResources(d3d12CommandBuffer);

    ID3D12GraphicsCommandList_ExecuteIndirect(
        d3d12CommandBuffer->graphicsCommandList,
        d3d12CommandBuffer->renderer->indirectIndexedDrawCommandSignature,
        drawCount,
        d3d12Buffer->handle,
        offset,
        NULL,
        0);

    D3D12_INTERNAL_TrackBuffer(d3d12CommandBuffer, d3d12Buffer);
}

static void D3D12_EndRenderPass(
    SDL_GPUCommandBuffer *commandBuffer)
{
    D3D12CommandBuffer *d3d12CommandBuffer = (D3D12CommandBuffer *)commandBuffer;
    Uint32 i;

    for (i = 0; i < MAX_COLOR_TARGET_BINDINGS; i += 1) {
        if (d3d12CommandBuffer->colorTargetSubresources[i] != NULL) {
            if (d3d12CommandBuffer->colorResolveSubresources[i] != NULL) {
                // Resolving requires some extra barriers
                D3D12_INTERNAL_TextureSubresourceBarrier(
                    d3d12CommandBuffer,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    D3D12_RESOURCE_STATE_RESOLVE_SOURCE,
                    d3d12CommandBuffer->colorTargetSubresources[i]
                );

                ID3D12GraphicsCommandList_ResolveSubresource(
                    d3d12CommandBuffer->graphicsCommandList,
                    d3d12CommandBuffer->colorResolveSubresources[i]->parent->resource,
                    d3d12CommandBuffer->colorResolveSubresources[i]->index,
                    d3d12CommandBuffer->colorTargetSubresources[i]->parent->resource,
                    d3d12CommandBuffer->colorTargetSubresources[i]->index,
                    SDLToD3D12_TextureFormat[d3d12CommandBuffer->colorTargetSubresources[i]->parent->container->header.info.format]);

                D3D12_INTERNAL_TextureSubresourceTransitionToDefaultUsage(
                    d3d12CommandBuffer,
                    D3D12_RESOURCE_STATE_RESOLVE_SOURCE,
                    d3d12CommandBuffer->colorTargetSubresources[i]);

                D3D12_INTERNAL_TextureSubresourceTransitionToDefaultUsage(
                    d3d12CommandBuffer,
                    D3D12_RESOURCE_STATE_RESOLVE_DEST,
                    d3d12CommandBuffer->colorResolveSubresources[i]);
            } else {
                D3D12_INTERNAL_TextureSubresourceTransitionToDefaultUsage(
                    d3d12CommandBuffer,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    d3d12CommandBuffer->colorTargetSubresources[i]);
            }
        }
    }

    if (d3d12CommandBuffer->depthStencilTextureSubresource != NULL) {
        D3D12_INTERNAL_TextureSubresourceTransitionToDefaultUsage(
            d3d12CommandBuffer,
            D3D12_RESOURCE_STATE_DEPTH_WRITE,
            d3d12CommandBuffer->depthStencilTextureSubresource);

        d3d12CommandBuffer->depthStencilTextureSubresource = NULL;
    }

    d3d12CommandBuffer->currentGraphicsPipeline = NULL;

    ID3D12GraphicsCommandList_OMSetRenderTargets(
        d3d12CommandBuffer->graphicsCommandList,
        0,
        NULL,
        false,
        NULL);

    // Reset bind state
    SDL_zeroa(d3d12CommandBuffer->colorTargetSubresources);
    SDL_zeroa(d3d12CommandBuffer->colorResolveSubresources);
    d3d12CommandBuffer->depthStencilTextureSubresource = NULL;

    SDL_zeroa(d3d12CommandBuffer->vertexBuffers);
    SDL_zeroa(d3d12CommandBuffer->vertexBufferOffsets);
    d3d12CommandBuffer->vertexBufferCount = 0;

    SDL_zeroa(d3d12CommandBuffer->vertexSamplerTextures);
    SDL_zeroa(d3d12CommandBuffer->vertexSamplers);
    SDL_zeroa(d3d12CommandBuffer->vertexStorageTextures);
    SDL_zeroa(d3d12CommandBuffer->vertexStorageBuffers);

    SDL_zeroa(d3d12CommandBuffer->fragmentSamplerTextures);
    SDL_zeroa(d3d12CommandBuffer->fragmentSamplers);
    SDL_zeroa(d3d12CommandBuffer->fragmentStorageTextures);
    SDL_zeroa(d3d12CommandBuffer->fragmentStorageBuffers);
}

// Compute Pass

static void D3D12_BeginComputePass(
    SDL_GPUCommandBuffer *commandBuffer,
    const SDL_GPUStorageTextureReadWriteBinding *storageTextureBindings,
    Uint32 numStorageTextureBindings,
    const SDL_GPUStorageBufferReadWriteBinding *storageBufferBindings,
    Uint32 numStorageBufferBindings)
{
    D3D12CommandBuffer *d3d12CommandBuffer = (D3D12CommandBuffer *)commandBuffer;

    d3d12CommandBuffer->computeReadWriteStorageTextureSubresourceCount = numStorageTextureBindings;
    d3d12CommandBuffer->computeReadWriteStorageBufferCount = numStorageBufferBindings;

    /* Read-write resources will be actually bound in BindComputePipeline
     * after the root signature is set.
     * We also have to scan to see which barriers we actually need because depth slices aren't separate subresources
     */
    if (numStorageTextureBindings > 0) {
        for (Uint32 i = 0; i < numStorageTextureBindings; i += 1) {
            D3D12TextureContainer *container = (D3D12TextureContainer *)storageTextureBindings[i].texture;

            D3D12TextureSubresource *subresource = D3D12_INTERNAL_PrepareTextureSubresourceForWrite(
                d3d12CommandBuffer,
                container,
                storageTextureBindings[i].layer,
                storageTextureBindings[i].mip_level,
                storageTextureBindings[i].cycle,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

            d3d12CommandBuffer->computeReadWriteStorageTextureSubresources[i] = subresource;

            D3D12_INTERNAL_TrackTexture(
                d3d12CommandBuffer,
                subresource->parent);
        }
    }

    if (numStorageBufferBindings > 0) {
        for (Uint32 i = 0; i < numStorageBufferBindings; i += 1) {
            D3D12BufferContainer *container = (D3D12BufferContainer *)storageBufferBindings[i].buffer;

            D3D12Buffer *buffer = D3D12_INTERNAL_PrepareBufferForWrite(
                d3d12CommandBuffer,
                container,
                storageBufferBindings[i].cycle,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

            d3d12CommandBuffer->computeReadWriteStorageBuffers[i] = buffer;

            D3D12_INTERNAL_TrackBuffer(
                d3d12CommandBuffer,
                buffer);
        }
    }
}

static void D3D12_BindComputePipeline(
    SDL_GPUCommandBuffer *commandBuffer,
    SDL_GPUComputePipeline *computePipeline)
{
    D3D12CommandBuffer *d3d12CommandBuffer = (D3D12CommandBuffer *)commandBuffer;

    /* Acquire GPU descriptor heaps if we haven't yet */
    if (d3d12CommandBuffer->gpuDescriptorHeaps[0] == NULL) {
        D3D12_INTERNAL_SetGPUDescriptorHeaps(d3d12CommandBuffer);
    }

    D3D12ComputePipeline *pipeline = (D3D12ComputePipeline *)computePipeline;
    D3D12_CPU_DESCRIPTOR_HANDLE cpuHandles[MAX_TEXTURE_SAMPLERS_PER_STAGE];
    D3D12_GPU_DESCRIPTOR_HANDLE gpuDescriptorHandle;

    ID3D12GraphicsCommandList_SetPipelineState(
        d3d12CommandBuffer->graphicsCommandList,
        pipeline->pipelineState);

    ID3D12GraphicsCommandList_SetComputeRootSignature(
        d3d12CommandBuffer->graphicsCommandList,
        pipeline->rootSignature->handle);

    d3d12CommandBuffer->currentComputePipeline = pipeline;

    d3d12CommandBuffer->needComputeSamplerBind = true;
    d3d12CommandBuffer->needComputeReadOnlyStorageTextureBind = true;
    d3d12CommandBuffer->needComputeReadOnlyStorageBufferBind = true;

    for (Uint32 i = 0; i < MAX_UNIFORM_BUFFERS_PER_STAGE; i += 1) {
        d3d12CommandBuffer->needComputeUniformBufferBind[i] = true;
    }

    for (Uint32 i = 0; i < pipeline->numUniformBuffers; i += 1) {
        if (d3d12CommandBuffer->computeUniformBuffers[i] == NULL) {
            d3d12CommandBuffer->computeUniformBuffers[i] = D3D12_INTERNAL_AcquireUniformBufferFromPool(
                d3d12CommandBuffer);
        }
    }

    D3D12_INTERNAL_TrackComputePipeline(d3d12CommandBuffer, pipeline);

    // Bind write-only resources after setting root signature
    if (pipeline->numReadWriteStorageTextures > 0) {
        for (Uint32 i = 0; i < pipeline->numReadWriteStorageTextures; i += 1) {
            cpuHandles[i] = d3d12CommandBuffer->computeReadWriteStorageTextureSubresources[i]->uavHandle.cpuHandle;
        }

        D3D12_INTERNAL_WriteGPUDescriptors(
            d3d12CommandBuffer,
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            cpuHandles,
            d3d12CommandBuffer->computeReadWriteStorageTextureSubresourceCount,
            &gpuDescriptorHandle);

        ID3D12GraphicsCommandList_SetComputeRootDescriptorTable(
            d3d12CommandBuffer->graphicsCommandList,
            d3d12CommandBuffer->currentComputePipeline->rootSignature->readWriteStorageTextureRootIndex,
            gpuDescriptorHandle);
    }

    if (pipeline->numReadWriteStorageBuffers > 0) {
        for (Uint32 i = 0; i < pipeline->numReadWriteStorageBuffers; i += 1) {
            cpuHandles[i] = d3d12CommandBuffer->computeReadWriteStorageBuffers[i]->uavDescriptor.cpuHandle;
        }

        D3D12_INTERNAL_WriteGPUDescriptors(
            d3d12CommandBuffer,
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            cpuHandles,
            d3d12CommandBuffer->computeReadWriteStorageBufferCount,
            &gpuDescriptorHandle);

        ID3D12GraphicsCommandList_SetComputeRootDescriptorTable(
            d3d12CommandBuffer->graphicsCommandList,
            d3d12CommandBuffer->currentComputePipeline->rootSignature->readWriteStorageBufferRootIndex,
            gpuDescriptorHandle);
    }
}

static void D3D12_BindComputeSamplers(
    SDL_GPUCommandBuffer *commandBuffer,
    Uint32 firstSlot,
    const SDL_GPUTextureSamplerBinding *textureSamplerBindings,
    Uint32 numBindings)
{
    D3D12CommandBuffer *d3d12CommandBuffer = (D3D12CommandBuffer *)commandBuffer;

    for (Uint32 i = 0; i < numBindings; i += 1) {
        D3D12TextureContainer *container = (D3D12TextureContainer *)textureSamplerBindings[i].texture;
        D3D12Sampler *sampler = (D3D12Sampler *)textureSamplerBindings[i].sampler;

        if (d3d12CommandBuffer->computeSamplers[firstSlot + i] != sampler) {
            D3D12_INTERNAL_TrackSampler(
                d3d12CommandBuffer,
                (D3D12Sampler *)textureSamplerBindings[i].sampler);

            d3d12CommandBuffer->computeSamplers[firstSlot + i] = (D3D12Sampler *)textureSamplerBindings[i].sampler;
            d3d12CommandBuffer->needComputeSamplerBind = true;
        }

        if (d3d12CommandBuffer->computeSamplerTextures[firstSlot + i] != container->activeTexture) {
            D3D12_INTERNAL_TrackTexture(
                d3d12CommandBuffer,
                container->activeTexture);

            d3d12CommandBuffer->computeSamplerTextures[firstSlot + i] = container->activeTexture;
            d3d12CommandBuffer->needComputeSamplerBind = true;
        }
    }
}

static void D3D12_BindComputeStorageTextures(
    SDL_GPUCommandBuffer *commandBuffer,
    Uint32 firstSlot,
    SDL_GPUTexture *const *storageTextures,
    Uint32 numBindings)
{
    D3D12CommandBuffer *d3d12CommandBuffer = (D3D12CommandBuffer *)commandBuffer;

    for (Uint32 i = 0; i < numBindings; i += 1) {
        D3D12TextureContainer *container = (D3D12TextureContainer *)storageTextures[i];

        if (d3d12CommandBuffer->computeReadOnlyStorageTextures[firstSlot + i] != container->activeTexture) {
            /* If a different texture was in this slot, transition it back to its default usage */
            if (d3d12CommandBuffer->computeReadOnlyStorageTextures[firstSlot + i] != NULL) {
                D3D12_INTERNAL_TextureTransitionToDefaultUsage(
                    d3d12CommandBuffer,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    d3d12CommandBuffer->computeReadOnlyStorageTextures[firstSlot + i]);
            }

            /* Then transition the new texture and prepare it for binding */
            D3D12_INTERNAL_TextureTransitionFromDefaultUsage(
                d3d12CommandBuffer,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                container->activeTexture);

            D3D12_INTERNAL_TrackTexture(
                d3d12CommandBuffer,
                container->activeTexture);

            d3d12CommandBuffer->computeReadOnlyStorageTextures[firstSlot + i] = container->activeTexture;
            d3d12CommandBuffer->needComputeReadOnlyStorageTextureBind = true;
        }
    }
}

static void D3D12_BindComputeStorageBuffers(
    SDL_GPUCommandBuffer *commandBuffer,
    Uint32 firstSlot,
    SDL_GPUBuffer *const *storageBuffers,
    Uint32 numBindings)
{
    D3D12CommandBuffer *d3d12CommandBuffer = (D3D12CommandBuffer *)commandBuffer;

    for (Uint32 i = 0; i < numBindings; i += 1) {
        D3D12BufferContainer *container = (D3D12BufferContainer *)storageBuffers[i];
        D3D12Buffer *buffer = container->activeBuffer;

        if (d3d12CommandBuffer->computeReadOnlyStorageBuffers[firstSlot + i] != buffer) {
            /* If a different buffer was in this slot, transition it back to its default usage */
            if (d3d12CommandBuffer->computeReadOnlyStorageBuffers[firstSlot + i] != NULL) {
                D3D12_INTERNAL_BufferTransitionToDefaultUsage(
                    d3d12CommandBuffer,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    d3d12CommandBuffer->computeReadOnlyStorageBuffers[firstSlot + i]);
            }

            /* Then transition the new buffer and prepare it for binding */
            D3D12_INTERNAL_BufferTransitionFromDefaultUsage(
                d3d12CommandBuffer,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                buffer);

            D3D12_INTERNAL_TrackBuffer(
                d3d12CommandBuffer,
                buffer);

            d3d12CommandBuffer->computeReadOnlyStorageBuffers[firstSlot + i] = buffer;
            d3d12CommandBuffer->needComputeReadOnlyStorageBufferBind = true;
        }
    }
}

static void D3D12_PushComputeUniformData(
    SDL_GPUCommandBuffer *commandBuffer,
    Uint32 slotIndex,
    const void *data,
    Uint32 length)
{
    D3D12CommandBuffer *d3d12CommandBuffer = (D3D12CommandBuffer *)commandBuffer;

    D3D12_INTERNAL_PushUniformData(
        d3d12CommandBuffer,
        SDL_GPU_SHADERSTAGE_COMPUTE,
        slotIndex,
        data,
        length);
}

static void D3D12_INTERNAL_BindComputeResources(
    D3D12CommandBuffer *commandBuffer)
{
    D3D12ComputePipeline *computePipeline = commandBuffer->currentComputePipeline;

    /* Acquire GPU descriptor heaps if we haven't yet */
    if (commandBuffer->gpuDescriptorHeaps[0] == NULL) {
        D3D12_INTERNAL_SetGPUDescriptorHeaps(commandBuffer);
    }

    D3D12_CPU_DESCRIPTOR_HANDLE cpuHandles[MAX_TEXTURE_SAMPLERS_PER_STAGE];
    D3D12_GPU_DESCRIPTOR_HANDLE gpuDescriptorHandle;

    if (commandBuffer->needComputeSamplerBind) {
        if (computePipeline->numSamplers > 0) {
            for (Uint32 i = 0; i < computePipeline->numSamplers; i += 1) {
                cpuHandles[i] = commandBuffer->computeSamplers[i]->handle.cpuHandle;
            }

            D3D12_INTERNAL_WriteGPUDescriptors(
                commandBuffer,
                D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
                cpuHandles,
                computePipeline->numSamplers,
                &gpuDescriptorHandle);

            ID3D12GraphicsCommandList_SetComputeRootDescriptorTable(
                commandBuffer->graphicsCommandList,
                computePipeline->rootSignature->samplerRootIndex,
                gpuDescriptorHandle);

            for (Uint32 i = 0; i < computePipeline->numSamplers; i += 1) {
                cpuHandles[i] = commandBuffer->computeSamplerTextures[i]->srvHandle.cpuHandle;
            }

            D3D12_INTERNAL_WriteGPUDescriptors(
                commandBuffer,
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                cpuHandles,
                computePipeline->numSamplers,
                &gpuDescriptorHandle);

            ID3D12GraphicsCommandList_SetComputeRootDescriptorTable(
                commandBuffer->graphicsCommandList,
                computePipeline->rootSignature->samplerTextureRootIndex,
                gpuDescriptorHandle);
        }
        commandBuffer->needComputeSamplerBind = false;
    }

    if (commandBuffer->needComputeReadOnlyStorageTextureBind) {
        if (computePipeline->numReadOnlyStorageTextures > 0) {
            for (Uint32 i = 0; i < computePipeline->numReadOnlyStorageTextures; i += 1) {
                cpuHandles[i] = commandBuffer->computeReadOnlyStorageTextures[i]->srvHandle.cpuHandle;
            }

            D3D12_INTERNAL_WriteGPUDescriptors(
                commandBuffer,
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                cpuHandles,
                computePipeline->numReadOnlyStorageTextures,
                &gpuDescriptorHandle);

            ID3D12GraphicsCommandList_SetComputeRootDescriptorTable(
                commandBuffer->graphicsCommandList,
                computePipeline->rootSignature->readOnlyStorageTextureRootIndex,
                gpuDescriptorHandle);
        }
        commandBuffer->needComputeReadOnlyStorageTextureBind = false;
    }

    if (commandBuffer->needComputeReadOnlyStorageBufferBind) {
        if (computePipeline->numReadOnlyStorageBuffers > 0) {
            for (Uint32 i = 0; i < computePipeline->numReadOnlyStorageBuffers; i += 1) {
                cpuHandles[i] = commandBuffer->computeReadOnlyStorageBuffers[i]->srvDescriptor.cpuHandle;
            }

            D3D12_INTERNAL_WriteGPUDescriptors(
                commandBuffer,
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                cpuHandles,
                computePipeline->numReadOnlyStorageBuffers,
                &gpuDescriptorHandle);

            ID3D12GraphicsCommandList_SetComputeRootDescriptorTable(
                commandBuffer->graphicsCommandList,
                computePipeline->rootSignature->readOnlyStorageBufferRootIndex,
                gpuDescriptorHandle);
        }
        commandBuffer->needComputeReadOnlyStorageBufferBind = false;
    }

    for (Uint32 i = 0; i < MAX_UNIFORM_BUFFERS_PER_STAGE; i += 1) {
        if (commandBuffer->needComputeUniformBufferBind[i]) {
            if (computePipeline->numUniformBuffers > i) {
                ID3D12GraphicsCommandList_SetComputeRootConstantBufferView(
                    commandBuffer->graphicsCommandList,
                    computePipeline->rootSignature->uniformBufferRootIndex[i],
                    commandBuffer->computeUniformBuffers[i]->buffer->virtualAddress + commandBuffer->computeUniformBuffers[i]->drawOffset);
            }
        }
        commandBuffer->needComputeUniformBufferBind[i] = false;
    }
}

static void D3D12_DispatchCompute(
    SDL_GPUCommandBuffer *commandBuffer,
    Uint32 groupcountX,
    Uint32 groupcountY,
    Uint32 groupcountZ)
{
    D3D12CommandBuffer *d3d12CommandBuffer = (D3D12CommandBuffer *)commandBuffer;

    D3D12_INTERNAL_BindComputeResources(d3d12CommandBuffer);
    ID3D12GraphicsCommandList_Dispatch(
        d3d12CommandBuffer->graphicsCommandList,
        groupcountX,
        groupcountY,
        groupcountZ);
}

static void D3D12_DispatchComputeIndirect(
    SDL_GPUCommandBuffer *commandBuffer,
    SDL_GPUBuffer *buffer,
    Uint32 offset)
{
    D3D12CommandBuffer *d3d12CommandBuffer = (D3D12CommandBuffer *)commandBuffer;
    D3D12Buffer *d3d12Buffer = ((D3D12BufferContainer *)buffer)->activeBuffer;

    D3D12_INTERNAL_BindComputeResources(d3d12CommandBuffer);
    ID3D12GraphicsCommandList_ExecuteIndirect(
        d3d12CommandBuffer->graphicsCommandList,
        d3d12CommandBuffer->renderer->indirectDispatchCommandSignature,
        1,
        d3d12Buffer->handle,
        offset,
        NULL,
        0);

    D3D12_INTERNAL_TrackBuffer(d3d12CommandBuffer, d3d12Buffer);
}

static void D3D12_EndComputePass(
    SDL_GPUCommandBuffer *commandBuffer)
{
    D3D12CommandBuffer *d3d12CommandBuffer = (D3D12CommandBuffer *)commandBuffer;

    for (Uint32 i = 0; i < d3d12CommandBuffer->computeReadWriteStorageTextureSubresourceCount; i += 1) {
        if (d3d12CommandBuffer->computeReadWriteStorageTextureSubresources[i]) {
            D3D12_INTERNAL_TextureSubresourceTransitionToDefaultUsage(
                d3d12CommandBuffer,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                d3d12CommandBuffer->computeReadWriteStorageTextureSubresources[i]);

            d3d12CommandBuffer->computeReadWriteStorageTextureSubresources[i] = NULL;
        }
    }
    d3d12CommandBuffer->computeReadWriteStorageTextureSubresourceCount = 0;

    for (Uint32 i = 0; i < d3d12CommandBuffer->computeReadWriteStorageBufferCount; i += 1) {
        if (d3d12CommandBuffer->computeReadWriteStorageBuffers[i]) {
            D3D12_INTERNAL_BufferTransitionToDefaultUsage(
                d3d12CommandBuffer,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                d3d12CommandBuffer->computeReadWriteStorageBuffers[i]);

            d3d12CommandBuffer->computeReadWriteStorageBuffers[i] = NULL;
        }
    }
    d3d12CommandBuffer->computeReadWriteStorageBufferCount = 0;

    for (Uint32 i = 0; i < MAX_STORAGE_TEXTURES_PER_STAGE; i += 1) {
        if (d3d12CommandBuffer->computeReadOnlyStorageTextures[i]) {
            D3D12_INTERNAL_TextureTransitionToDefaultUsage(
                d3d12CommandBuffer,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                d3d12CommandBuffer->computeReadOnlyStorageTextures[i]);

            d3d12CommandBuffer->computeReadOnlyStorageTextures[i] = NULL;
        }
    }

    for (Uint32 i = 0; i < MAX_STORAGE_BUFFERS_PER_STAGE; i += 1) {
        if (d3d12CommandBuffer->computeReadOnlyStorageBuffers[i]) {
            D3D12_INTERNAL_BufferTransitionToDefaultUsage(
                d3d12CommandBuffer,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                d3d12CommandBuffer->computeReadOnlyStorageBuffers[i]);

            d3d12CommandBuffer->computeReadOnlyStorageBuffers[i] = NULL;
        }
    }

    SDL_zeroa(d3d12CommandBuffer->computeSamplerTextures);
    SDL_zeroa(d3d12CommandBuffer->computeSamplers);

    d3d12CommandBuffer->currentComputePipeline = NULL;
}

// TransferBuffer Data

static void *D3D12_MapTransferBuffer(
    SDL_GPURenderer *driverData,
    SDL_GPUTransferBuffer *transferBuffer,
    bool cycle)
{
    D3D12Renderer *renderer = (D3D12Renderer *)driverData;
    D3D12BufferContainer *container = (D3D12BufferContainer *)transferBuffer;
    void *dataPointer;

    if (
        cycle &&
        SDL_GetAtomicInt(&container->activeBuffer->referenceCount) > 0) {
        D3D12_INTERNAL_CycleActiveBuffer(
            renderer,
            container);
    }

    // Upload buffers are persistently mapped, download buffers are not
    if (container->type == D3D12_BUFFER_TYPE_UPLOAD) {
        dataPointer = container->activeBuffer->mapPointer;
    } else {
        ID3D12Resource_Map(
            container->activeBuffer->handle,
            0,
            NULL,
            (void **)&dataPointer);
    }

    return dataPointer;
}

static void D3D12_UnmapTransferBuffer(
    SDL_GPURenderer *driverData,
    SDL_GPUTransferBuffer *transferBuffer)
{
    (void)driverData;
    D3D12BufferContainer *container = (D3D12BufferContainer *)transferBuffer;

    // Upload buffers are persistently mapped, download buffers are not
    if (container->type == D3D12_BUFFER_TYPE_DOWNLOAD) {
        ID3D12Resource_Unmap(
            container->activeBuffer->handle,
            0,
            NULL);
    }
}

// Copy Pass

static void D3D12_BeginCopyPass(
    SDL_GPUCommandBuffer *commandBuffer)
{
    // no-op
    (void)commandBuffer;
}

static void D3D12_UploadToTexture(
    SDL_GPUCommandBuffer *commandBuffer,
    const SDL_GPUTextureTransferInfo *source,
    const SDL_GPUTextureRegion *destination,
    bool cycle)
{
    D3D12CommandBuffer *d3d12CommandBuffer = (D3D12CommandBuffer *)commandBuffer;
    D3D12BufferContainer *transferBufferContainer = (D3D12BufferContainer *)source->transfer_buffer;
    D3D12Buffer *temporaryBuffer = NULL;
    D3D12_TEXTURE_COPY_LOCATION sourceLocation;
    D3D12_TEXTURE_COPY_LOCATION destinationLocation;
    Uint32 pixelsPerRow = source->pixels_per_row;
    Uint32 rowPitch;
    Uint32 alignedRowPitch;
    Uint32 rowsPerSlice = source->rows_per_layer;
    Uint32 bytesPerSlice;
    bool needsRealignment;
    bool needsPlacementCopy;

    // Note that the transfer buffer does not need a barrier, as it is synced by the client.

    D3D12TextureContainer *textureContainer = (D3D12TextureContainer *)destination->texture;
    D3D12TextureSubresource *textureSubresource = D3D12_INTERNAL_PrepareTextureSubresourceForWrite(
        d3d12CommandBuffer,
        textureContainer,
        destination->layer,
        destination->mip_level,
        cycle,
        D3D12_RESOURCE_STATE_COPY_DEST);

    /* D3D12 requires texture data row pitch to be 256 byte aligned, which is obviously insane.
     * Instead of exposing that restriction to the client, which is a huge rake to step on,
     * and a restriction that no other backend requires, we're going to copy data to a temporary buffer,
     * copy THAT data to the texture, and then get rid of the temporary buffer ASAP.
     * If we're lucky and the row pitch and depth pitch are already aligned, we can skip all of that.
     *
     * D3D12 also requires offsets to be 512 byte aligned. We'll fix that for the client and warn them as well.
     *
     * And just for some extra fun, D3D12 doesn't actually support depth pitch, so we have to realign that too!
     */

    if (pixelsPerRow == 0) {
        pixelsPerRow = destination->w;
    }

    rowPitch = BytesPerRow(pixelsPerRow, textureContainer->header.info.format);

    if (rowsPerSlice == 0) {
        rowsPerSlice = destination->h;
    }

    bytesPerSlice = rowsPerSlice * rowPitch;

    alignedRowPitch = D3D12_INTERNAL_Align(rowPitch, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);
    needsRealignment = rowsPerSlice != destination->h || rowPitch != alignedRowPitch;
    needsPlacementCopy = source->offset % D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT != 0;

    sourceLocation.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    sourceLocation.PlacedFootprint.Footprint.Format = SDLToD3D12_TextureFormat[textureContainer->header.info.format];
    sourceLocation.PlacedFootprint.Footprint.RowPitch = alignedRowPitch;

    destinationLocation.pResource = textureContainer->activeTexture->resource;
    destinationLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    destinationLocation.SubresourceIndex = textureSubresource->index;

    if (needsRealignment) {
        temporaryBuffer = D3D12_INTERNAL_CreateBuffer(
            d3d12CommandBuffer->renderer,
            0,
            alignedRowPitch * destination->h * destination->d,
            D3D12_BUFFER_TYPE_UPLOAD,
            NULL);

        if (!temporaryBuffer) {
            return;
        }

        sourceLocation.pResource = temporaryBuffer->handle;

        for (Uint32 sliceIndex = 0; sliceIndex < destination->d; sliceIndex += 1) {
            // copy row count minus one to avoid overread
            for (Uint32 rowIndex = 0; rowIndex < rowsPerSlice - 1; rowIndex += 1) {
                SDL_memcpy(
                    temporaryBuffer->mapPointer + (sliceIndex * rowsPerSlice) + (rowIndex * alignedRowPitch),
                    transferBufferContainer->activeBuffer->mapPointer + source->offset + (sliceIndex * bytesPerSlice) + (rowIndex * rowPitch),
                    alignedRowPitch);
            }
            Uint32 offset = source->offset + (sliceIndex * bytesPerSlice) + ((rowsPerSlice - 1) * rowPitch);
            SDL_memcpy(
                temporaryBuffer->mapPointer + (sliceIndex * rowsPerSlice) + ((rowsPerSlice - 1) * alignedRowPitch),
                transferBufferContainer->activeBuffer->mapPointer + offset,
                SDL_min(alignedRowPitch, transferBufferContainer->size - offset));

            sourceLocation.PlacedFootprint.Footprint.Width = destination->w;
            sourceLocation.PlacedFootprint.Footprint.Height = rowsPerSlice;
            sourceLocation.PlacedFootprint.Footprint.Depth = 1;
            sourceLocation.PlacedFootprint.Offset = (sliceIndex * bytesPerSlice);

            ID3D12GraphicsCommandList_CopyTextureRegion(
                d3d12CommandBuffer->graphicsCommandList,
                &destinationLocation,
                destination->x,
                destination->y,
                destination->z + sliceIndex,
                &sourceLocation,
                NULL);
        }

        D3D12_INTERNAL_TrackBuffer(d3d12CommandBuffer, temporaryBuffer);
        D3D12_INTERNAL_ReleaseBuffer(
            d3d12CommandBuffer->renderer,
            temporaryBuffer);

        if (d3d12CommandBuffer->renderer->debug_mode) {
            SDL_LogWarn(SDL_LOG_CATEGORY_GPU, "Texture upload row pitch not aligned to 256 bytes! This is suboptimal on D3D12!");
        }
    } else if (needsPlacementCopy) {
        temporaryBuffer = D3D12_INTERNAL_CreateBuffer(
            d3d12CommandBuffer->renderer,
            0,
            alignedRowPitch * destination->h * destination->d,
            D3D12_BUFFER_TYPE_UPLOAD,
            NULL);

        if (!temporaryBuffer) {
            return;
        }

        SDL_memcpy(
            temporaryBuffer->mapPointer,
            transferBufferContainer->activeBuffer->mapPointer + source->offset,
            alignedRowPitch * destination->h * destination->d);

        sourceLocation.pResource = temporaryBuffer->handle;
        sourceLocation.PlacedFootprint.Offset = 0;
        sourceLocation.PlacedFootprint.Footprint.Width = destination->w;
        sourceLocation.PlacedFootprint.Footprint.Height = destination->h;
        sourceLocation.PlacedFootprint.Footprint.Depth = 1;

        ID3D12GraphicsCommandList_CopyTextureRegion(
            d3d12CommandBuffer->graphicsCommandList,
            &destinationLocation,
            destination->x,
            destination->y,
            destination->z,
            &sourceLocation,
            NULL);

        D3D12_INTERNAL_TrackBuffer(d3d12CommandBuffer, temporaryBuffer);
        D3D12_INTERNAL_ReleaseBuffer(
            d3d12CommandBuffer->renderer,
            temporaryBuffer);

        if (d3d12CommandBuffer->renderer->debug_mode) {
            SDL_LogWarn(SDL_LOG_CATEGORY_GPU, "Texture upload offset not aligned to 512 bytes! This is suboptimal on D3D12!");
        }
    } else {
        sourceLocation.pResource = transferBufferContainer->activeBuffer->handle;
        sourceLocation.PlacedFootprint.Offset = source->offset;
        sourceLocation.PlacedFootprint.Footprint.Width = destination->w;
        sourceLocation.PlacedFootprint.Footprint.Height = destination->h;
        sourceLocation.PlacedFootprint.Footprint.Depth = destination->d;

        ID3D12GraphicsCommandList_CopyTextureRegion(
            d3d12CommandBuffer->graphicsCommandList,
            &destinationLocation,
            destination->x,
            destination->y,
            destination->z,
            &sourceLocation,
            NULL);
    }

    D3D12_INTERNAL_TextureSubresourceTransitionToDefaultUsage(
        d3d12CommandBuffer,
        D3D12_RESOURCE_STATE_COPY_DEST,
        textureSubresource);

    D3D12_INTERNAL_TrackBuffer(d3d12CommandBuffer, transferBufferContainer->activeBuffer);
    D3D12_INTERNAL_TrackTexture(d3d12CommandBuffer, textureSubresource->parent);
}

static void D3D12_UploadToBuffer(
    SDL_GPUCommandBuffer *commandBuffer,
    const SDL_GPUTransferBufferLocation *source,
    const SDL_GPUBufferRegion *destination,
    bool cycle)
{
    D3D12CommandBuffer *d3d12CommandBuffer = (D3D12CommandBuffer *)commandBuffer;
    D3D12BufferContainer *transferBufferContainer = (D3D12BufferContainer *)source->transfer_buffer;
    D3D12BufferContainer *bufferContainer = (D3D12BufferContainer *)destination->buffer;

    // The transfer buffer does not need a barrier, it is synced by the client.

    D3D12Buffer *buffer = D3D12_INTERNAL_PrepareBufferForWrite(
        d3d12CommandBuffer,
        bufferContainer,
        cycle,
        D3D12_RESOURCE_STATE_COPY_DEST);

    ID3D12GraphicsCommandList_CopyBufferRegion(
        d3d12CommandBuffer->graphicsCommandList,
        buffer->handle,
        destination->offset,
        transferBufferContainer->activeBuffer->handle,
        source->offset,
        destination->size);

    D3D12_INTERNAL_BufferTransitionToDefaultUsage(
        d3d12CommandBuffer,
        D3D12_RESOURCE_STATE_COPY_DEST,
        buffer);

    D3D12_INTERNAL_TrackBuffer(d3d12CommandBuffer, transferBufferContainer->activeBuffer);
    D3D12_INTERNAL_TrackBuffer(d3d12CommandBuffer, buffer);
}

static void D3D12_CopyTextureToTexture(
    SDL_GPUCommandBuffer *commandBuffer,
    const SDL_GPUTextureLocation *source,
    const SDL_GPUTextureLocation *destination,
    Uint32 w,
    Uint32 h,
    Uint32 d,
    bool cycle)
{
    D3D12CommandBuffer *d3d12CommandBuffer = (D3D12CommandBuffer *)commandBuffer;
    D3D12_TEXTURE_COPY_LOCATION sourceLocation;
    D3D12_TEXTURE_COPY_LOCATION destinationLocation;

    D3D12TextureSubresource *sourceSubresource = D3D12_INTERNAL_FetchTextureSubresource(
        (D3D12TextureContainer *)source->texture,
        source->layer,
        source->mip_level);

    D3D12TextureSubresource *destinationSubresource = D3D12_INTERNAL_PrepareTextureSubresourceForWrite(
        d3d12CommandBuffer,
        (D3D12TextureContainer *)destination->texture,
        destination->layer,
        destination->mip_level,
        cycle,
        D3D12_RESOURCE_STATE_COPY_DEST);

    D3D12_INTERNAL_TextureSubresourceTransitionFromDefaultUsage(
        d3d12CommandBuffer,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        sourceSubresource);

    sourceLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    sourceLocation.SubresourceIndex = sourceSubresource->index;
    sourceLocation.pResource = sourceSubresource->parent->resource;

    destinationLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    destinationLocation.SubresourceIndex = destinationSubresource->index;
    destinationLocation.pResource = destinationSubresource->parent->resource;

    D3D12_BOX sourceBox = { source->x, source->y, source->z, source->x + w, source->y + h, source->z + d };

    ID3D12GraphicsCommandList_CopyTextureRegion(
        d3d12CommandBuffer->graphicsCommandList,
        &destinationLocation,
        destination->x,
        destination->y,
        destination->z,
        &sourceLocation,
        &sourceBox);

    D3D12_INTERNAL_TextureSubresourceTransitionToDefaultUsage(
        d3d12CommandBuffer,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        sourceSubresource);

    D3D12_INTERNAL_TextureSubresourceTransitionToDefaultUsage(
        d3d12CommandBuffer,
        D3D12_RESOURCE_STATE_COPY_DEST,
        destinationSubresource);

    D3D12_INTERNAL_TrackTexture(
        d3d12CommandBuffer,
        sourceSubresource->parent);

    D3D12_INTERNAL_TrackTexture(
        d3d12CommandBuffer,
        destinationSubresource->parent);
}

static void D3D12_CopyBufferToBuffer(
    SDL_GPUCommandBuffer *commandBuffer,
    const SDL_GPUBufferLocation *source,
    const SDL_GPUBufferLocation *destination,
    Uint32 size,
    bool cycle)
{
    D3D12CommandBuffer *d3d12CommandBuffer = (D3D12CommandBuffer *)commandBuffer;
    D3D12BufferContainer *sourceContainer = (D3D12BufferContainer *)source->buffer;
    D3D12BufferContainer *destinationContainer = (D3D12BufferContainer *)destination->buffer;

    D3D12Buffer *sourceBuffer = sourceContainer->activeBuffer;
    D3D12Buffer *destinationBuffer = D3D12_INTERNAL_PrepareBufferForWrite(
        d3d12CommandBuffer,
        destinationContainer,
        cycle,
        D3D12_RESOURCE_STATE_COPY_DEST);

    D3D12_INTERNAL_BufferTransitionFromDefaultUsage(
        d3d12CommandBuffer,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        sourceBuffer);

    ID3D12GraphicsCommandList_CopyBufferRegion(
        d3d12CommandBuffer->graphicsCommandList,
        destinationBuffer->handle,
        destination->offset,
        sourceBuffer->handle,
        source->offset,
        size);

    D3D12_INTERNAL_BufferTransitionToDefaultUsage(
        d3d12CommandBuffer,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        sourceBuffer);

    D3D12_INTERNAL_BufferTransitionToDefaultUsage(
        d3d12CommandBuffer,
        D3D12_RESOURCE_STATE_COPY_DEST,
        destinationBuffer);

    D3D12_INTERNAL_TrackBuffer(d3d12CommandBuffer, sourceBuffer);
    D3D12_INTERNAL_TrackBuffer(d3d12CommandBuffer, destinationBuffer);
}

static void D3D12_DownloadFromTexture(
    SDL_GPUCommandBuffer *commandBuffer,
    const SDL_GPUTextureRegion *source,
    const SDL_GPUTextureTransferInfo *destination)
{
    D3D12CommandBuffer *d3d12CommandBuffer = (D3D12CommandBuffer *)commandBuffer;
    D3D12_TEXTURE_COPY_LOCATION sourceLocation;
    D3D12_TEXTURE_COPY_LOCATION destinationLocation;
    Uint32 pixelsPerRow = destination->pixels_per_row;
    Uint32 rowPitch;
    Uint32 alignedRowPitch;
    Uint32 rowsPerSlice = destination->rows_per_layer;
    bool needsRealignment;
    bool needsPlacementCopy;
    D3D12TextureDownload *textureDownload = NULL;
    D3D12TextureContainer *sourceContainer = (D3D12TextureContainer *)source->texture;
    D3D12TextureSubresource *sourceSubresource = D3D12_INTERNAL_FetchTextureSubresource(
        sourceContainer,
        source->layer,
        source->mip_level);
    D3D12BufferContainer *destinationContainer = (D3D12BufferContainer *)destination->transfer_buffer;
    D3D12Buffer *destinationBuffer = destinationContainer->activeBuffer;

    /* D3D12 requires texture data row pitch to be 256 byte aligned, which is obviously insane.
     * Instead of exposing that restriction to the client, which is a huge rake to step on,
     * and a restriction that no other backend requires, we're going to copy data to a temporary buffer,
     * copy THAT data to the texture, and then get rid of the temporary buffer ASAP.
     * If we're lucky and the row pitch and depth pitch are already aligned, we can skip all of that.
     *
     * D3D12 also requires offsets to be 512 byte aligned. We'll fix that for the client and warn them as well.
     *
     * And just for some extra fun, D3D12 doesn't actually support depth pitch, so we have to realign that too!
     *
     * Since this is an async download we have to do all these fixups after the command is finished,
     * so we'll cache the metadata and map and copy it when the command buffer is cleaned.
     */

    if (pixelsPerRow == 0) {
        pixelsPerRow = source->w;
    }

    rowPitch = BytesPerRow(pixelsPerRow, sourceContainer->header.info.format);

    if (rowsPerSlice == 0) {
        rowsPerSlice = source->h;
    }

    alignedRowPitch = D3D12_INTERNAL_Align(rowPitch, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);
    needsRealignment = rowsPerSlice != source->h || rowPitch != alignedRowPitch;
    needsPlacementCopy = destination->offset % D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT != 0;

    sourceLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    sourceLocation.SubresourceIndex = sourceSubresource->index;
    sourceLocation.pResource = sourceSubresource->parent->resource;

    D3D12_BOX sourceBox = { source->x, source->y, source->z, source->x + source->w, source->y + rowsPerSlice, source->z + source->d };

    destinationLocation.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    destinationLocation.PlacedFootprint.Footprint.Format = SDLToD3D12_TextureFormat[sourceContainer->header.info.format];
    destinationLocation.PlacedFootprint.Footprint.Width = source->w;
    destinationLocation.PlacedFootprint.Footprint.Height = rowsPerSlice;
    destinationLocation.PlacedFootprint.Footprint.Depth = source->d;
    destinationLocation.PlacedFootprint.Footprint.RowPitch = alignedRowPitch;

    if (needsRealignment || needsPlacementCopy) {
        textureDownload = (D3D12TextureDownload *)SDL_malloc(sizeof(D3D12TextureDownload));

        if (!textureDownload) {
            SDL_LogError(SDL_LOG_CATEGORY_GPU, "Failed to create texture download structure!");
            return;
        }

        textureDownload->temporaryBuffer = D3D12_INTERNAL_CreateBuffer(
            d3d12CommandBuffer->renderer,
            0,
            alignedRowPitch * rowsPerSlice * source->d,
            D3D12_BUFFER_TYPE_DOWNLOAD,
            NULL);

        if (!textureDownload->temporaryBuffer) {
            SDL_free(textureDownload);
            return;
        }

        textureDownload->destinationBuffer = destinationBuffer;
        textureDownload->bufferOffset = destination->offset;
        textureDownload->width = source->w;
        textureDownload->height = rowsPerSlice;
        textureDownload->depth = source->d;
        textureDownload->bytesPerRow = rowPitch;
        textureDownload->bytesPerDepthSlice = rowPitch * rowsPerSlice;
        textureDownload->alignedBytesPerRow = alignedRowPitch;

        destinationLocation.pResource = textureDownload->temporaryBuffer->handle;
        destinationLocation.PlacedFootprint.Offset = 0;
        if (d3d12CommandBuffer->renderer->debug_mode) {
            SDL_LogWarn(SDL_LOG_CATEGORY_GPU, "Texture pitch or offset not aligned properly! This is suboptimal on D3D12!");
        }
    } else {
        destinationLocation.pResource = destinationBuffer->handle;
        destinationLocation.PlacedFootprint.Offset = destination->offset;
    }

    D3D12_INTERNAL_TextureSubresourceTransitionFromDefaultUsage(
        d3d12CommandBuffer,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        sourceSubresource);

    ID3D12GraphicsCommandList_CopyTextureRegion(
        d3d12CommandBuffer->graphicsCommandList,
        &destinationLocation,
        0,
        0,
        0,
        &sourceLocation,
        &sourceBox);

    D3D12_INTERNAL_TextureSubresourceTransitionToDefaultUsage(
        d3d12CommandBuffer,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        sourceSubresource);

    D3D12_INTERNAL_TrackBuffer(d3d12CommandBuffer, destinationBuffer);
    D3D12_INTERNAL_TrackTexture(d3d12CommandBuffer, sourceSubresource->parent);

    if (textureDownload != NULL) {
        D3D12_INTERNAL_TrackBuffer(d3d12CommandBuffer, textureDownload->temporaryBuffer);

        if (d3d12CommandBuffer->textureDownloadCount >= d3d12CommandBuffer->textureDownloadCapacity) {
            d3d12CommandBuffer->textureDownloadCapacity *= 2;
            d3d12CommandBuffer->textureDownloads = (D3D12TextureDownload **)SDL_realloc(
                d3d12CommandBuffer->textureDownloads,
                d3d12CommandBuffer->textureDownloadCapacity * sizeof(D3D12TextureDownload *));
        }

        d3d12CommandBuffer->textureDownloads[d3d12CommandBuffer->textureDownloadCount] = textureDownload;
        d3d12CommandBuffer->textureDownloadCount += 1;

        D3D12_INTERNAL_ReleaseBuffer(d3d12CommandBuffer->renderer, textureDownload->temporaryBuffer);
    }
}

static void D3D12_DownloadFromBuffer(
    SDL_GPUCommandBuffer *commandBuffer,
    const SDL_GPUBufferRegion *source,
    const SDL_GPUTransferBufferLocation *destination)
{
    D3D12CommandBuffer *d3d12CommandBuffer = (D3D12CommandBuffer *)commandBuffer;
    D3D12BufferContainer *sourceContainer = (D3D12BufferContainer *)source->buffer;
    D3D12BufferContainer *destinationContainer = (D3D12BufferContainer *)destination->transfer_buffer;

    D3D12Buffer *sourceBuffer = sourceContainer->activeBuffer;
    D3D12_INTERNAL_BufferTransitionFromDefaultUsage(
        d3d12CommandBuffer,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        sourceBuffer);

    D3D12Buffer *destinationBuffer = destinationContainer->activeBuffer;

    ID3D12GraphicsCommandList_CopyBufferRegion(
        d3d12CommandBuffer->graphicsCommandList,
        destinationBuffer->handle,
        destination->offset,
        sourceBuffer->handle,
        source->offset,
        source->size);

    D3D12_INTERNAL_BufferTransitionToDefaultUsage(
        d3d12CommandBuffer,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        sourceBuffer);

    D3D12_INTERNAL_TrackBuffer(d3d12CommandBuffer, sourceBuffer);
    D3D12_INTERNAL_TrackBuffer(d3d12CommandBuffer, destinationBuffer);
}

static void D3D12_EndCopyPass(
    SDL_GPUCommandBuffer *commandBuffer)
{
    // no-op
    (void)commandBuffer;
}

static void D3D12_GenerateMipmaps(
    SDL_GPUCommandBuffer *commandBuffer,
    SDL_GPUTexture *texture)
{
    D3D12CommandBuffer *d3d12CommandBuffer = (D3D12CommandBuffer *)commandBuffer;
    D3D12Renderer *renderer = d3d12CommandBuffer->renderer;
    D3D12TextureContainer *container = (D3D12TextureContainer *)texture;
    SDL_GPUGraphicsPipeline *blitPipeline;

    blitPipeline = SDL_GPU_FetchBlitPipeline(
        renderer->sdlGPUDevice,
        container->header.info.type,
        container->header.info.format,
        renderer->blitVertexShader,
        renderer->blitFrom2DShader,
        renderer->blitFrom2DArrayShader,
        renderer->blitFrom3DShader,
        renderer->blitFromCubeShader,
        renderer->blitFromCubeArrayShader,
        &renderer->blitPipelines,
        &renderer->blitPipelineCount,
        &renderer->blitPipelineCapacity);

    if (blitPipeline == NULL) {
        SDL_LogWarn(SDL_LOG_CATEGORY_GPU, "Could not fetch blit pipeline");
        return;
    }

    // We have to do this one subresource at a time
    for (Uint32 layerOrDepthIndex = 0; layerOrDepthIndex < container->header.info.layer_count_or_depth; layerOrDepthIndex += 1) {
        for (Uint32 levelIndex = 1; levelIndex < container->header.info.num_levels; levelIndex += 1) {
            SDL_GPUBlitInfo blitInfo;
            SDL_zero(blitInfo);

            blitInfo.source.texture = texture;
            blitInfo.source.mip_level = levelIndex - 1;
            blitInfo.source.layer_or_depth_plane = layerOrDepthIndex;
            blitInfo.source.x = 0;
            blitInfo.source.y = 0;
            blitInfo.source.w = SDL_max(container->header.info.width >> (levelIndex - 1), 1);
            blitInfo.source.h = SDL_max(container->header.info.height >> (levelIndex - 1), 1);

            blitInfo.destination.texture = texture;
            blitInfo.destination.mip_level = levelIndex;
            blitInfo.destination.layer_or_depth_plane = layerOrDepthIndex;
            blitInfo.destination.x = 0;
            blitInfo.destination.y = 0;
            blitInfo.destination.w = SDL_max(container->header.info.width >> levelIndex, 1);
            blitInfo.destination.h = SDL_max(container->header.info.height >> levelIndex, 1);

            blitInfo.load_op = SDL_GPU_LOADOP_DONT_CARE;
            blitInfo.filter = SDL_GPU_FILTER_LINEAR;

            SDL_BlitGPUTexture(
                commandBuffer,
                &blitInfo);
        }
    }

    D3D12_INTERNAL_TrackTexture(d3d12CommandBuffer, container->activeTexture);
}

static void D3D12_Blit(
    SDL_GPUCommandBuffer *commandBuffer,
    const SDL_GPUBlitInfo *info)
{
    D3D12CommandBuffer *d3d12CommandBuffer = (D3D12CommandBuffer *)commandBuffer;
    D3D12Renderer *renderer = (D3D12Renderer *)d3d12CommandBuffer->renderer;

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

// Submission/Presentation

static D3D12WindowData *D3D12_INTERNAL_FetchWindowData(
    SDL_Window *window)
{
    SDL_PropertiesID properties = SDL_GetWindowProperties(window);
    return (D3D12WindowData *)SDL_GetPointerProperty(properties, WINDOW_PROPERTY_DATA, NULL);
}

static bool D3D12_INTERNAL_OnWindowResize(void *userdata, SDL_Event *e)
{
    SDL_Window *w = (SDL_Window *)userdata;
    D3D12WindowData *data;
    if (e->type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED && e->window.windowID == SDL_GetWindowID(w)) {
        data = D3D12_INTERNAL_FetchWindowData(w);
        data->needsSwapchainRecreate = true;
    }

    return true;
}

static bool D3D12_SupportsSwapchainComposition(
    SDL_GPURenderer *driverData,
    SDL_Window *window,
    SDL_GPUSwapchainComposition swapchainComposition)
{
#if defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES)
    // FIXME: HDR support would be nice to add, but it seems complicated...
    return swapchainComposition == SDL_GPU_SWAPCHAINCOMPOSITION_SDR ||
           swapchainComposition == SDL_GPU_SWAPCHAINCOMPOSITION_SDR_LINEAR;
#else
    D3D12Renderer *renderer = (D3D12Renderer *)driverData;
    DXGI_FORMAT format;
    D3D12_FEATURE_DATA_FORMAT_SUPPORT formatSupport;
    Uint32 colorSpaceSupport;
    HRESULT res;

    format = SwapchainCompositionToTextureFormat[swapchainComposition];

    formatSupport.Format = format;
    res = ID3D12Device_CheckFeatureSupport(
        renderer->device,
        D3D12_FEATURE_FORMAT_SUPPORT,
        &formatSupport,
        sizeof(formatSupport));
    if (FAILED(res)) {
        // Format is apparently unknown
        return false;
    }

    if (!(formatSupport.Support1 & D3D12_FORMAT_SUPPORT1_DISPLAY)) {
        return false;
    }

    D3D12WindowData *windowData = D3D12_INTERNAL_FetchWindowData(window);
    if (windowData == NULL) {
        SET_STRING_ERROR_AND_RETURN("Must claim window before querying swapchain composition support!", false);
    }

    // Check the color space support if necessary
    if (swapchainComposition != SDL_GPU_SWAPCHAINCOMPOSITION_SDR) {
        IDXGISwapChain3_CheckColorSpaceSupport(
            windowData->swapchain,
            SwapchainCompositionToColorSpace[swapchainComposition],
            &colorSpaceSupport);

        if (!(colorSpaceSupport & DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT)) {
            return false;
        }
    }
#endif

    return true;
}

static bool D3D12_SupportsPresentMode(
    SDL_GPURenderer *driverData,
    SDL_Window *window,
    SDL_GPUPresentMode presentMode)
{
    (void)driverData;
    (void)window;

    switch (presentMode) {
    case SDL_GPU_PRESENTMODE_IMMEDIATE:
    case SDL_GPU_PRESENTMODE_VSYNC:
        return true;
    case SDL_GPU_PRESENTMODE_MAILBOX:
#if (defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES))
        return false;
#else
        return true;
#endif
    default:
        SDL_assert(!"Unrecognized present mode");
        return false;
    }
}

#if defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES)
static bool D3D12_INTERNAL_CreateSwapchain(
    D3D12Renderer *renderer,
    D3D12WindowData *windowData,
    SDL_GPUSwapchainComposition swapchain_composition,
    SDL_GPUPresentMode present_mode)
{
    int width, height;
    SDL_GPUTextureCreateInfo createInfo;
    D3D12Texture *texture;

    // Get the swapchain size
    SDL_SyncWindow(windowData->window);
    SDL_GetWindowSizeInPixels(windowData->window, &width, &height);

    // Min swapchain image count is 2
    windowData->swapchainTextureCount = SDL_clamp(renderer->allowedFramesInFlight, 2, 3);

    // Create the swapchain textures
    SDL_zero(createInfo);
    createInfo.type = SDL_GPU_TEXTURETYPE_2D;
    createInfo.width = width;
    createInfo.height = height;
    createInfo.format = SwapchainCompositionToSDLTextureFormat[swapchain_composition];
    createInfo.usage = SDL_GPU_TEXTUREUSAGE_COLOR_TARGET;
    createInfo.layer_count_or_depth = 1;
    createInfo.num_levels = 1;

    for (Uint32 i = 0; i < windowData->swapchainTextureCount; i += 1) {
        texture = D3D12_INTERNAL_CreateTexture(renderer, &createInfo, true, "Swapchain");
        texture->container = &windowData->textureContainers[i];
        windowData->textureContainers[i].activeTexture = texture;
        windowData->textureContainers[i].canBeCycled = false;
        windowData->textureContainers[i].header.info = createInfo;
        windowData->textureContainers[i].textureCapacity = 1;
        windowData->textureContainers[i].textureCount = 1;
        windowData->textureContainers[i].textures = &windowData->textureContainers[i].activeTexture;
    }

    // Initialize the swapchain data
    windowData->present_mode = present_mode;
    windowData->swapchainComposition = swapchain_composition;
    windowData->swapchainColorSpace = DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;
    windowData->frameCounter = 0;
    windowData->width = width;
    windowData->height = height;

    // Precache blit pipelines for the swapchain format
    for (Uint32 i = 0; i < 5; i += 1) {
        SDL_GPU_FetchBlitPipeline(
            renderer->sdlGPUDevice,
            (SDL_GPUTextureType)i,
            createInfo.format,
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

    return true;
}

static void D3D12_INTERNAL_DestroySwapchain(
    D3D12Renderer *renderer,
    D3D12WindowData *windowData)
{
    renderer->commandQueue->PresentX(0, NULL, NULL);
    for (Uint32 i = 0; i < windowData->swapchainTextureCount; i += 1) {
        D3D12_INTERNAL_DestroyTexture(windowData->textureContainers[i].activeTexture);
    }
}

static bool D3D12_INTERNAL_ResizeSwapchain(
    D3D12Renderer *renderer,
    D3D12WindowData *windowData)
{
    // Wait so we don't release in-flight views
    D3D12_Wait((SDL_GPURenderer *)renderer);

    // Present a black screen
    renderer->commandQueue->PresentX(0, NULL, NULL);

    // Clean up the previous swapchain textures
    for (Uint32 i = 0; i < windowData->swapchainTextureCount; i += 1) {
        D3D12_INTERNAL_DestroyTexture(windowData->textureContainers[i].activeTexture);
    }

    // Create a new swapchain
    D3D12_INTERNAL_CreateSwapchain(
        renderer,
        windowData,
        windowData->swapchainComposition,
        windowData->present_mode);

    windowData->needsSwapchainRecreate = false;
    return true;
}
#else
static bool D3D12_INTERNAL_InitializeSwapchainTexture(
    D3D12Renderer *renderer,
    IDXGISwapChain3 *swapchain,
    SDL_GPUSwapchainComposition composition,
    Uint32 index,
    D3D12TextureContainer *pTextureContainer)
{
    D3D12Texture *pTexture;
    ID3D12Resource *swapchainTexture;
    D3D12_RESOURCE_DESC textureDesc;
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc;
    D3D12_RENDER_TARGET_VIEW_DESC rtvDesc;
    DXGI_FORMAT swapchainFormat = SwapchainCompositionToTextureFormat[composition];
    HRESULT res;

    res = IDXGISwapChain_GetBuffer(
        swapchain,
        index,
        D3D_GUID(D3D_IID_ID3D12Resource),
        (void **)&swapchainTexture);
    CHECK_D3D12_ERROR_AND_RETURN("Could not get buffer from swapchain!", false);

    pTexture = (D3D12Texture *)SDL_calloc(1, sizeof(D3D12Texture));
    if (!pTexture) {
        ID3D12Resource_Release(swapchainTexture);
        return false;
    }
    pTexture->resource = NULL; // This will be set in AcquireSwapchainTexture
    SDL_SetAtomicInt(&pTexture->referenceCount, 0);
    pTexture->subresourceCount = 1;
    pTexture->subresources = (D3D12TextureSubresource *)SDL_calloc(1, sizeof(D3D12TextureSubresource));
    if (!pTexture->subresources) {
        SDL_free(pTexture);
        ID3D12Resource_Release(swapchainTexture);
        return false;
    }
    pTexture->subresources[0].rtvHandles = SDL_calloc(1, sizeof(D3D12StagingDescriptor));
    pTexture->subresources[0].uavHandle.heap = NULL;
    pTexture->subresources[0].dsvHandle.heap = NULL;
    pTexture->subresources[0].parent = pTexture;
    pTexture->subresources[0].index = 0;
    pTexture->subresources[0].layer = 0;
    pTexture->subresources[0].depth = 1;
    pTexture->subresources[0].level = 0;

    ID3D12Resource_GetDesc(swapchainTexture, &textureDesc);
    pTextureContainer->header.info.width = (Uint32)textureDesc.Width;
    pTextureContainer->header.info.height = (Uint32)textureDesc.Height;
    pTextureContainer->header.info.layer_count_or_depth = 1;
    pTextureContainer->header.info.num_levels = 1;
    pTextureContainer->header.info.type = SDL_GPU_TEXTURETYPE_2D;
    pTextureContainer->header.info.usage = SDL_GPU_TEXTUREUSAGE_COLOR_TARGET;
    pTextureContainer->header.info.sample_count = SDL_GPU_SAMPLECOUNT_1;
    pTextureContainer->header.info.format = SwapchainCompositionToSDLTextureFormat[composition];

    pTextureContainer->debugName = NULL;
    pTextureContainer->textures = (D3D12Texture **)SDL_calloc(1, sizeof(D3D12Texture *));
    if (!pTextureContainer->textures) {
        SDL_free(pTexture->subresources);
        SDL_free(pTexture);
        ID3D12Resource_Release(swapchainTexture);
        return false;
    }

    pTextureContainer->textureCapacity = 1;
    pTextureContainer->textureCount = 1;
    pTextureContainer->textures[0] = pTexture;
    pTextureContainer->activeTexture = pTexture;
    pTextureContainer->canBeCycled = false;

    pTexture->container = pTextureContainer;
    pTexture->containerIndex = 0;

    // Create the SRV for the swapchain
    D3D12_INTERNAL_AssignStagingDescriptorHandle(
        renderer,
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
        &pTexture->srvHandle);

    srvDesc.Format = SwapchainCompositionToTextureFormat[composition];
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MipLevels = 1;
    srvDesc.Texture2D.MostDetailedMip = 0;
    srvDesc.Texture2D.ResourceMinLODClamp = 0;
    srvDesc.Texture2D.PlaneSlice = 0;

    ID3D12Device_CreateShaderResourceView(
        renderer->device,
        swapchainTexture,
        &srvDesc,
        pTexture->srvHandle.cpuHandle);

    // Create the RTV for the swapchain
    D3D12_INTERNAL_AssignStagingDescriptorHandle(
        renderer,
        D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
        &pTexture->subresources[0].rtvHandles[0]);

    rtvDesc.Format = (composition == SDL_GPU_SWAPCHAINCOMPOSITION_SDR_LINEAR) ? DXGI_FORMAT_B8G8R8A8_UNORM_SRGB : swapchainFormat;
    rtvDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
    rtvDesc.Texture2D.MipSlice = 0;
    rtvDesc.Texture2D.PlaneSlice = 0;

    ID3D12Device_CreateRenderTargetView(
        renderer->device,
        swapchainTexture,
        &rtvDesc,
        pTexture->subresources[0].rtvHandles[0].cpuHandle);

    ID3D12Resource_Release(swapchainTexture);

    return true;
}

static bool D3D12_INTERNAL_ResizeSwapchain(
    D3D12Renderer *renderer,
    D3D12WindowData *windowData)
{
    // Wait so we don't release in-flight views
    D3D12_Wait((SDL_GPURenderer *)renderer);

    // Release views and clean up
    for (Uint32 i = 0; i < windowData->swapchainTextureCount; i += 1) {
        D3D12_INTERNAL_ReleaseStagingDescriptorHandle(
            &windowData->textureContainers[i].activeTexture->srvHandle);
        D3D12_INTERNAL_ReleaseStagingDescriptorHandle(
            &windowData->textureContainers[i].activeTexture->subresources[0].rtvHandles[0]);

        SDL_free(windowData->textureContainers[i].activeTexture->subresources[0].rtvHandles);
        SDL_free(windowData->textureContainers[i].activeTexture->subresources);
        SDL_free(windowData->textureContainers[i].activeTexture);
        SDL_free(windowData->textureContainers[i].textures);
    }

    // Resize the swapchain
    HRESULT res = IDXGISwapChain_ResizeBuffers(
        windowData->swapchain,
        0, // Keep buffer count the same
        0, // use client window width
        0, // use client window height
        DXGI_FORMAT_UNKNOWN, // Keep the old format
        renderer->supportsTearing ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0);
    CHECK_D3D12_ERROR_AND_RETURN("Could not resize swapchain buffers", false);

    // Create texture object for the swapchain
    for (Uint32 i = 0; i < windowData->swapchainTextureCount; i += 1) {
        if (!D3D12_INTERNAL_InitializeSwapchainTexture(
                renderer,
                windowData->swapchain,
                windowData->swapchainComposition,
                i,
                &windowData->textureContainers[i])) {
            return false;
        }
    }

    DXGI_SWAP_CHAIN_DESC1 swapchainDesc;
    IDXGISwapChain3_GetDesc1(windowData->swapchain, &swapchainDesc);
    CHECK_D3D12_ERROR_AND_RETURN("Failed to retrieve swapchain descriptor!", false);

    windowData->width = swapchainDesc.Width;
    windowData->height = swapchainDesc.Height;
    windowData->needsSwapchainRecreate = false;
    return true;
}

static void D3D12_INTERNAL_DestroySwapchain(
    D3D12Renderer *renderer,
    D3D12WindowData *windowData)
{
    // Release views and clean up
    for (Uint32 i = 0; i < windowData->swapchainTextureCount; i += 1) {
        D3D12_INTERNAL_ReleaseStagingDescriptorHandle(
            &windowData->textureContainers[i].activeTexture->srvHandle);
        D3D12_INTERNAL_ReleaseStagingDescriptorHandle(
            &windowData->textureContainers[i].activeTexture->subresources[0].rtvHandles[0]);

        SDL_free(windowData->textureContainers[i].activeTexture->subresources[0].rtvHandles);
        SDL_free(windowData->textureContainers[i].activeTexture->subresources);
        SDL_free(windowData->textureContainers[i].activeTexture);
        SDL_free(windowData->textureContainers[i].textures);
    }

    IDXGISwapChain_Release(windowData->swapchain);
    windowData->swapchain = NULL;
}

static bool D3D12_INTERNAL_CreateSwapchain(
    D3D12Renderer *renderer,
    D3D12WindowData *windowData,
    SDL_GPUSwapchainComposition swapchainComposition,
    SDL_GPUPresentMode presentMode)
{
    HWND dxgiHandle;
    DXGI_SWAP_CHAIN_DESC1 swapchainDesc;
    DXGI_SWAP_CHAIN_FULLSCREEN_DESC fullscreenDesc;
    DXGI_FORMAT swapchainFormat;
    IDXGIFactory1 *pParent;
    IDXGISwapChain1 *swapchain;
    IDXGISwapChain3 *swapchain3;
    HRESULT res;

    // Get the DXGI handle
#ifdef _WIN32
    dxgiHandle = (HWND)SDL_GetPointerProperty(SDL_GetWindowProperties(windowData->window), SDL_PROP_WINDOW_WIN32_HWND_POINTER, NULL);
#else
    dxgiHandle = (HWND)windowData->window;
#endif

    swapchainFormat = SwapchainCompositionToTextureFormat[swapchainComposition];

    // Min swapchain image count is 2
    windowData->swapchainTextureCount = SDL_clamp(renderer->allowedFramesInFlight, 2, 3);

    // Initialize the swapchain buffer descriptor
    swapchainDesc.Width = 0;  // use client window width
    swapchainDesc.Height = 0; // use client window height
    swapchainDesc.Format = swapchainFormat;
    swapchainDesc.SampleDesc.Count = 1;
    swapchainDesc.SampleDesc.Quality = 0;
    swapchainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapchainDesc.BufferCount = windowData->swapchainTextureCount;
    swapchainDesc.Scaling = DXGI_SCALING_NONE;
    swapchainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    swapchainDesc.AlphaMode = DXGI_ALPHA_MODE_UNSPECIFIED;
    swapchainDesc.Flags = 0;
    swapchainDesc.Stereo = 0;

    // Initialize the fullscreen descriptor (if needed)
    fullscreenDesc.RefreshRate.Numerator = 0;
    fullscreenDesc.RefreshRate.Denominator = 0;
    fullscreenDesc.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED;
    fullscreenDesc.Scaling = DXGI_MODE_SCALING_UNSPECIFIED;
    fullscreenDesc.Windowed = true;

    if (renderer->supportsTearing) {
        swapchainDesc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;
    } else {
        swapchainDesc.Flags = 0;
    }

    if (!IsWindow(dxgiHandle)) {
        return false;
    }

    // Create the swapchain!
    res = IDXGIFactory4_CreateSwapChainForHwnd(
        renderer->factory,
        (IUnknown *)renderer->commandQueue,
        dxgiHandle,
        &swapchainDesc,
        &fullscreenDesc,
        NULL,
        &swapchain);
    CHECK_D3D12_ERROR_AND_RETURN("Could not create swapchain", false);

    res = IDXGISwapChain1_QueryInterface(
        swapchain,
        D3D_GUID(D3D_IID_IDXGISwapChain3),
        (void **)&swapchain3);
    IDXGISwapChain1_Release(swapchain);
    CHECK_D3D12_ERROR_AND_RETURN("Could not create IDXGISwapChain3", false);

    if (swapchainComposition != SDL_GPU_SWAPCHAINCOMPOSITION_SDR) {
        // Support already verified if we hit this block
        IDXGISwapChain3_SetColorSpace1(
            swapchain3,
            SwapchainCompositionToColorSpace[swapchainComposition]);
    }

    /*
     * The swapchain's parent is a separate factory from the factory that
     * we used to create the swapchain, and only that parent can be used to
     * set the window association. Trying to set an association on our factory
     * will silently fail and doesn't even verify arguments or return errors.
     * See https://gamedev.net/forums/topic/634235-dxgidisabling-altenter/4999955/
     */
    res = IDXGISwapChain3_GetParent(
        swapchain3,
        D3D_GUID(D3D_IID_IDXGIFactory1),
        (void **)&pParent);
    if (FAILED(res)) {
        SDL_LogWarn(
            SDL_LOG_CATEGORY_GPU,
            "Could not get swapchain parent! Error Code: " HRESULT_FMT,
            res);
    } else {
        // Disable DXGI window crap
        res = IDXGIFactory1_MakeWindowAssociation(
            pParent,
            dxgiHandle,
            DXGI_MWA_NO_WINDOW_CHANGES);
        if (FAILED(res)) {
            SDL_LogWarn(
                SDL_LOG_CATEGORY_GPU,
                "MakeWindowAssociation failed! Error Code: " HRESULT_FMT,
                res);
        }

        // We're done with the parent now
        IDXGIFactory1_Release(pParent);
    }

    IDXGISwapChain3_GetDesc1(swapchain3, &swapchainDesc);
    CHECK_D3D12_ERROR_AND_RETURN("Failed to retrieve swapchain descriptor!", false);

    // Initialize the swapchain data
    windowData->swapchain = swapchain3;
    windowData->present_mode = presentMode;
    windowData->swapchainComposition = swapchainComposition;
    windowData->swapchainColorSpace = SwapchainCompositionToColorSpace[swapchainComposition];
    windowData->frameCounter = 0;
    windowData->width = swapchainDesc.Width;
    windowData->height = swapchainDesc.Height;

    // Precache blit pipelines for the swapchain format
    for (Uint32 i = 0; i < 5; i += 1) {
        SDL_GPU_FetchBlitPipeline(
            renderer->sdlGPUDevice,
            (SDL_GPUTextureType)i,
            SwapchainCompositionToSDLTextureFormat[swapchainComposition],
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

    /* If a you are using a FLIP model format you can't create the swapchain as DXGI_FORMAT_B8G8R8A8_UNORM_SRGB.
     * You have to create the swapchain as DXGI_FORMAT_B8G8R8A8_UNORM and then set the render target view's format to DXGI_FORMAT_B8G8R8A8_UNORM_SRGB
     */
    for (Uint32 i = 0; i < windowData->swapchainTextureCount; i += 1) {
        if (!D3D12_INTERNAL_InitializeSwapchainTexture(
                renderer,
                swapchain3,
                swapchainComposition,
                i,
                &windowData->textureContainers[i])) {
            IDXGISwapChain3_Release(swapchain3);
            return false;
        }
    }

    return true;
}
#endif

static bool D3D12_ClaimWindow(
    SDL_GPURenderer *driverData,
    SDL_Window *window)
{
    D3D12Renderer *renderer = (D3D12Renderer *)driverData;
    D3D12WindowData *windowData = D3D12_INTERNAL_FetchWindowData(window);

    if (windowData == NULL) {
        windowData = (D3D12WindowData *)SDL_calloc(1, sizeof(D3D12WindowData));
        if (!windowData) {
            return false;
        }
        windowData->window = window;

        if (D3D12_INTERNAL_CreateSwapchain(renderer, windowData, SDL_GPU_SWAPCHAINCOMPOSITION_SDR, SDL_GPU_PRESENTMODE_VSYNC)) {
            SDL_SetPointerProperty(SDL_GetWindowProperties(window), WINDOW_PROPERTY_DATA, windowData);

            SDL_LockMutex(renderer->windowLock);
            if (renderer->claimedWindowCount >= renderer->claimedWindowCapacity) {
                renderer->claimedWindowCapacity *= 2;
                renderer->claimedWindows = (D3D12WindowData **)SDL_realloc(
                    renderer->claimedWindows,
                    renderer->claimedWindowCapacity * sizeof(D3D12WindowData *));
            }
            renderer->claimedWindows[renderer->claimedWindowCount] = windowData;
            renderer->claimedWindowCount += 1;
            SDL_UnlockMutex(renderer->windowLock);

            SDL_AddEventWatch(D3D12_INTERNAL_OnWindowResize, window);

            return true;
        } else {
            SDL_free(windowData);
            SET_STRING_ERROR_AND_RETURN("Could not create swapchain, failed to claim window!", false);
        }
    } else {
        SET_STRING_ERROR_AND_RETURN("Window already claimed", false);
    }
}

static void D3D12_ReleaseWindow(
    SDL_GPURenderer *driverData,
    SDL_Window *window)
{
    D3D12Renderer *renderer = (D3D12Renderer *)driverData;
    D3D12WindowData *windowData = D3D12_INTERNAL_FetchWindowData(window);

    if (windowData == NULL) {
        SET_STRING_ERROR_AND_RETURN("Window already unclaimed!", );
    }

    D3D12_Wait(driverData);

    for (Uint32 i = 0; i < MAX_FRAMES_IN_FLIGHT; i += 1) {
        if (windowData->inFlightFences[i] != NULL) {
            D3D12_ReleaseFence(
                driverData,
                windowData->inFlightFences[i]);
            windowData->inFlightFences[i] = NULL;
        }
    }

    D3D12_INTERNAL_DestroySwapchain(renderer, windowData);

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
    SDL_RemoveEventWatch(D3D12_INTERNAL_OnWindowResize, window);
}

static bool D3D12_SetSwapchainParameters(
    SDL_GPURenderer *driverData,
    SDL_Window *window,
    SDL_GPUSwapchainComposition swapchainComposition,
    SDL_GPUPresentMode presentMode)
{
    D3D12Renderer *renderer = (D3D12Renderer *)driverData;
    D3D12WindowData *windowData = D3D12_INTERNAL_FetchWindowData(window);

    if (windowData == NULL) {
        SET_STRING_ERROR_AND_RETURN("Cannot set swapchain parameters on unclaimed window!", false);
    }

    if (!D3D12_SupportsSwapchainComposition(driverData, window, swapchainComposition)) {
        SET_STRING_ERROR_AND_RETURN("Swapchain composition not supported!", false);
    }

    if (!D3D12_SupportsPresentMode(driverData, window, presentMode)) {
        SET_STRING_ERROR_AND_RETURN("Present mode not supported!", false);
    }

    if (
        swapchainComposition != windowData->swapchainComposition ||
        presentMode != windowData->present_mode) {
        D3D12_Wait(driverData);

        // Recreate the swapchain
        D3D12_INTERNAL_DestroySwapchain(
            renderer,
            windowData);

        return D3D12_INTERNAL_CreateSwapchain(
            renderer,
            windowData,
            swapchainComposition,
            presentMode);
    }

    return true;
}

static bool D3D12_SetAllowedFramesInFlight(
    SDL_GPURenderer *driverData,
    Uint32 allowedFramesInFlight)
{
    D3D12Renderer *renderer = (D3D12Renderer *)driverData;

    if (!D3D12_Wait(driverData)) {
        return false;
    }

    // Destroy all swapchains
    for (Uint32 i = 0; i < renderer->claimedWindowCount; i += 1) {
        D3D12WindowData *windowData = renderer->claimedWindows[i];

        D3D12_INTERNAL_DestroySwapchain(renderer, windowData);
    }

    // Set the frames in flight value
    renderer->allowedFramesInFlight = allowedFramesInFlight;

    // Recreate all swapchains
    for (Uint32 i = 0; i < renderer->claimedWindowCount; i += 1) {
        D3D12WindowData *windowData = renderer->claimedWindows[i];

        if (!D3D12_INTERNAL_CreateSwapchain(
            renderer,
            windowData,
            windowData->swapchainComposition,
            windowData->present_mode)) {
            return false;
        }
    }

    return true;
}

static SDL_GPUTextureFormat D3D12_GetSwapchainTextureFormat(
    SDL_GPURenderer *driverData,
    SDL_Window *window)
{
    D3D12Renderer *renderer = (D3D12Renderer *)driverData;
    D3D12WindowData *windowData = D3D12_INTERNAL_FetchWindowData(window);

    if (windowData == NULL) {
        SET_STRING_ERROR_AND_RETURN("Cannot get swapchain format, window has not been claimed!", SDL_GPU_TEXTUREFORMAT_INVALID);
    }

    return windowData->textureContainers[windowData->frameCounter].header.info.format;
}

static D3D12Fence *D3D12_INTERNAL_AcquireFence(
    D3D12Renderer *renderer)
{
    D3D12Fence *fence;
    ID3D12Fence *handle;
    HRESULT res;

    SDL_LockMutex(renderer->fenceLock);

    if (renderer->availableFenceCount == 0) {
        res = ID3D12Device_CreateFence(
            renderer->device,
            D3D12_FENCE_UNSIGNALED_VALUE,
            D3D12_FENCE_FLAG_NONE,
            D3D_GUID(D3D_IID_ID3D12Fence),
            (void **)&handle);
        if (FAILED(res)) {
            D3D12_INTERNAL_SetError(renderer, "Failed to create fence!", res);
            SDL_UnlockMutex(renderer->fenceLock);
            return NULL;
        }

        fence = (D3D12Fence *)SDL_calloc(1, sizeof(D3D12Fence));
        if (!fence) {
            ID3D12Fence_Release(handle);
            SDL_UnlockMutex(renderer->fenceLock);
            return NULL;
        }
        fence->handle = handle;
        fence->event = CreateEvent(NULL, FALSE, FALSE, NULL);
        SDL_SetAtomicInt(&fence->referenceCount, 0);
    } else {
        fence = renderer->availableFences[renderer->availableFenceCount - 1];
        renderer->availableFenceCount -= 1;
        ID3D12Fence_Signal(fence->handle, D3D12_FENCE_UNSIGNALED_VALUE);
    }

    SDL_UnlockMutex(renderer->fenceLock);

    (void)SDL_AtomicIncRef(&fence->referenceCount);
    return fence;
}

static bool D3D12_INTERNAL_AllocateCommandBuffer(
    D3D12Renderer *renderer)
{
    D3D12CommandBuffer *commandBuffer;
    HRESULT res;
    ID3D12CommandAllocator *commandAllocator;
    ID3D12GraphicsCommandList *commandList;

    commandBuffer = (D3D12CommandBuffer *)SDL_calloc(1, sizeof(D3D12CommandBuffer));
    if (!commandBuffer) {
        SET_STRING_ERROR_AND_RETURN("Failed to create ID3D12CommandList. Out of Memory", false);
    }

    res = ID3D12Device_CreateCommandAllocator(
        renderer->device,
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        D3D_GUID(D3D_IID_ID3D12CommandAllocator),
        (void **)&commandAllocator);
    if (FAILED(res)) {
        D3D12_INTERNAL_SetError(renderer, "Failed to create ID3D12CommandAllocator", res);
        D3D12_INTERNAL_DestroyCommandBuffer(commandBuffer);
        return false;
    }
    commandBuffer->commandAllocator = commandAllocator;

    res = ID3D12Device_CreateCommandList(
        renderer->device,
        0,
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        commandAllocator,
        NULL,
        D3D_GUID(D3D_IID_ID3D12GraphicsCommandList),
        (void **)&commandList);

    if (FAILED(res)) {
        D3D12_INTERNAL_SetError(renderer, "Failed to create ID3D12CommandList", res);
        D3D12_INTERNAL_DestroyCommandBuffer(commandBuffer);
        return false;
    }
    commandBuffer->graphicsCommandList = commandList;

    commandBuffer->renderer = renderer;
    commandBuffer->inFlightFence = NULL;

    // Window handling
    commandBuffer->presentDataCapacity = 1;
    commandBuffer->presentDataCount = 0;
    commandBuffer->presentDatas = (D3D12PresentData *)SDL_calloc(
        commandBuffer->presentDataCapacity, sizeof(D3D12PresentData));

    // Resource tracking
    commandBuffer->usedTextureCapacity = 4;
    commandBuffer->usedTextureCount = 0;
    commandBuffer->usedTextures = (D3D12Texture **)SDL_calloc(
        commandBuffer->usedTextureCapacity, sizeof(D3D12Texture *));

    commandBuffer->usedBufferCapacity = 4;
    commandBuffer->usedBufferCount = 0;
    commandBuffer->usedBuffers = (D3D12Buffer **)SDL_calloc(
        commandBuffer->usedBufferCapacity, sizeof(D3D12Buffer *));

    commandBuffer->usedSamplerCapacity = 4;
    commandBuffer->usedSamplerCount = 0;
    commandBuffer->usedSamplers = (D3D12Sampler **)SDL_calloc(
        commandBuffer->usedSamplerCapacity, sizeof(D3D12Sampler *));

    commandBuffer->usedGraphicsPipelineCapacity = 4;
    commandBuffer->usedGraphicsPipelineCount = 0;
    commandBuffer->usedGraphicsPipelines = (D3D12GraphicsPipeline **)SDL_calloc(
        commandBuffer->usedGraphicsPipelineCapacity, sizeof(D3D12GraphicsPipeline *));

    commandBuffer->usedComputePipelineCapacity = 4;
    commandBuffer->usedComputePipelineCount = 0;
    commandBuffer->usedComputePipelines = (D3D12ComputePipeline **)SDL_calloc(
        commandBuffer->usedComputePipelineCapacity, sizeof(D3D12ComputePipeline *));

    commandBuffer->usedUniformBufferCapacity = 4;
    commandBuffer->usedUniformBufferCount = 0;
    commandBuffer->usedUniformBuffers = (D3D12UniformBuffer **)SDL_calloc(
        commandBuffer->usedUniformBufferCapacity, sizeof(D3D12UniformBuffer *));

    commandBuffer->textureDownloadCapacity = 4;
    commandBuffer->textureDownloadCount = 0;
    commandBuffer->textureDownloads = (D3D12TextureDownload **)SDL_calloc(
        commandBuffer->textureDownloadCapacity, sizeof(D3D12TextureDownload *));

    if (
        (!commandBuffer->presentDatas) ||
        (!commandBuffer->usedTextures) ||
        (!commandBuffer->usedBuffers) ||
        (!commandBuffer->usedSamplers) ||
        (!commandBuffer->usedGraphicsPipelines) ||
        (!commandBuffer->usedComputePipelines) ||
        (!commandBuffer->usedUniformBuffers) ||
        (!commandBuffer->textureDownloads)) {
        D3D12_INTERNAL_DestroyCommandBuffer(commandBuffer);
        SET_STRING_ERROR_AND_RETURN("Failed to create ID3D12CommandList. Out of Memory", false);
    }

    D3D12CommandBuffer **resizedAvailableCommandBuffers = (D3D12CommandBuffer **)SDL_realloc(
        renderer->availableCommandBuffers,
        sizeof(D3D12CommandBuffer *) * (renderer->availableCommandBufferCapacity + 1));

    if (!resizedAvailableCommandBuffers) {
        D3D12_INTERNAL_DestroyCommandBuffer(commandBuffer);
        SET_STRING_ERROR_AND_RETURN("Failed to create ID3D12CommandList. Out of Memory", false);
    }
    // Add to inactive command buffer array
    renderer->availableCommandBufferCapacity += 1;
    renderer->availableCommandBuffers = resizedAvailableCommandBuffers;

    renderer->availableCommandBuffers[renderer->availableCommandBufferCount] = commandBuffer;
    renderer->availableCommandBufferCount += 1;

    return true;
}

static D3D12CommandBuffer *D3D12_INTERNAL_AcquireCommandBufferFromPool(
    D3D12Renderer *renderer)
{
    D3D12CommandBuffer *commandBuffer;

    if (renderer->availableCommandBufferCount == 0) {
        if (!D3D12_INTERNAL_AllocateCommandBuffer(renderer)) {
            return NULL;
        }
    }

    commandBuffer = renderer->availableCommandBuffers[renderer->availableCommandBufferCount - 1];
    renderer->availableCommandBufferCount -= 1;

    return commandBuffer;
}

static SDL_GPUCommandBuffer *D3D12_AcquireCommandBuffer(
    SDL_GPURenderer *driverData)
{
    D3D12Renderer *renderer = (D3D12Renderer *)driverData;
    D3D12CommandBuffer *commandBuffer;
    ID3D12DescriptorHeap *heaps[2];
    SDL_zeroa(heaps);

    SDL_LockMutex(renderer->acquireCommandBufferLock);
    commandBuffer = D3D12_INTERNAL_AcquireCommandBufferFromPool(renderer);
    SDL_UnlockMutex(renderer->acquireCommandBufferLock);

    if (commandBuffer == NULL) {
        return NULL;
    }

    // Set the bind state
    commandBuffer->currentGraphicsPipeline = NULL;

    SDL_zeroa(commandBuffer->colorTargetSubresources);
    SDL_zeroa(commandBuffer->colorResolveSubresources);
    commandBuffer->depthStencilTextureSubresource = NULL;

    SDL_zeroa(commandBuffer->vertexBuffers);
    SDL_zeroa(commandBuffer->vertexBufferOffsets);
    commandBuffer->vertexBufferCount = 0;

    SDL_zeroa(commandBuffer->vertexSamplerTextures);
    SDL_zeroa(commandBuffer->vertexSamplers);
    SDL_zeroa(commandBuffer->vertexStorageTextures);
    SDL_zeroa(commandBuffer->vertexStorageBuffers);
    SDL_zeroa(commandBuffer->vertexUniformBuffers);

    SDL_zeroa(commandBuffer->fragmentSamplerTextures);
    SDL_zeroa(commandBuffer->fragmentSamplers);
    SDL_zeroa(commandBuffer->fragmentStorageTextures);
    SDL_zeroa(commandBuffer->fragmentStorageBuffers);
    SDL_zeroa(commandBuffer->fragmentUniformBuffers);

    SDL_zeroa(commandBuffer->computeSamplerTextures);
    SDL_zeroa(commandBuffer->computeSamplers);
    SDL_zeroa(commandBuffer->computeReadOnlyStorageTextures);
    SDL_zeroa(commandBuffer->computeReadOnlyStorageBuffers);
    SDL_zeroa(commandBuffer->computeReadWriteStorageTextureSubresources);
    SDL_zeroa(commandBuffer->computeReadWriteStorageBuffers);
    SDL_zeroa(commandBuffer->computeUniformBuffers);

    commandBuffer->autoReleaseFence = true;

    return (SDL_GPUCommandBuffer *)commandBuffer;
}

static bool D3D12_WaitForSwapchain(
    SDL_GPURenderer *driverData,
    SDL_Window *window)
{
    D3D12Renderer *renderer = (D3D12Renderer *)driverData;
    D3D12WindowData *windowData = D3D12_INTERNAL_FetchWindowData(window);

    if (windowData == NULL) {
        SET_STRING_ERROR_AND_RETURN("Cannot wait for a swapchain from an unclaimed window!", false);
    }

    if (windowData->inFlightFences[windowData->frameCounter] != NULL) {
        if (!D3D12_WaitForFences(
            driverData,
            true,
            &windowData->inFlightFences[windowData->frameCounter],
            1)) {
            return false;
        }
    }

    return true;
}

static bool D3D12_INTERNAL_AcquireSwapchainTexture(
    bool block,
    SDL_GPUCommandBuffer *commandBuffer,
    SDL_Window *window,
    SDL_GPUTexture **swapchainTexture,
    Uint32 *swapchainTextureWidth,
    Uint32 *swapchainTextureHeight)
{
    D3D12CommandBuffer *d3d12CommandBuffer = (D3D12CommandBuffer *)commandBuffer;
    D3D12Renderer *renderer = d3d12CommandBuffer->renderer;
    D3D12WindowData *windowData;
    Uint32 swapchainIndex;
    HRESULT res;

    *swapchainTexture = NULL;
    if (swapchainTextureWidth) {
        *swapchainTextureWidth = 0;
    }
    if (swapchainTextureHeight) {
        *swapchainTextureHeight = 0;
    }

    windowData = D3D12_INTERNAL_FetchWindowData(window);
    if (windowData == NULL) {
        SET_STRING_ERROR_AND_RETURN("Cannot acquire swapchain texture from an unclaimed window!", false);
    }

    if (windowData->needsSwapchainRecreate) {
        if (!D3D12_INTERNAL_ResizeSwapchain(renderer, windowData)) {
            return false;
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
            // In VSYNC mode, block until the least recent presented frame is done
            if (!D3D12_WaitForFences(
                (SDL_GPURenderer *)renderer,
                true,
                &windowData->inFlightFences[windowData->frameCounter],
                1)) {
                return false;
            }
        } else {
            // If we are not blocking and the least recent fence is not signaled,
            // return true to indicate that there is no error but rendering should be skipped.
            if (!D3D12_QueryFence(
                    (SDL_GPURenderer *)renderer,
                    windowData->inFlightFences[windowData->frameCounter])) {
                return true;
            }
        }

        D3D12_ReleaseFence(
            (SDL_GPURenderer *)renderer,
            windowData->inFlightFences[windowData->frameCounter]);

        windowData->inFlightFences[windowData->frameCounter] = NULL;
    }

#if (defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES))
    // FIXME: Should this happen before the inFlightFences stuff above?
    windowData->frameToken = D3D12XBOX_FRAME_PIPELINE_TOKEN_NULL;
    renderer->device->WaitFrameEventX(D3D12XBOX_FRAME_EVENT_ORIGIN, INFINITE, NULL, D3D12XBOX_WAIT_FRAME_EVENT_FLAG_NONE, &windowData->frameToken);
    swapchainIndex = windowData->frameCounter;
#else
    swapchainIndex = IDXGISwapChain3_GetCurrentBackBufferIndex(windowData->swapchain);

    // Set the handle on the windowData texture data.
    res = IDXGISwapChain_GetBuffer(
        windowData->swapchain,
        swapchainIndex,
        D3D_GUID(D3D_IID_ID3D12Resource),
        (void **)&windowData->textureContainers[swapchainIndex].activeTexture->resource);
    CHECK_D3D12_ERROR_AND_RETURN("Could not acquire swapchain!", false);
#endif

    // Set up presentation
    if (d3d12CommandBuffer->presentDataCount == d3d12CommandBuffer->presentDataCapacity) {
        d3d12CommandBuffer->presentDataCapacity += 1;
        d3d12CommandBuffer->presentDatas = (D3D12PresentData *)SDL_realloc(
            d3d12CommandBuffer->presentDatas,
            d3d12CommandBuffer->presentDataCapacity * sizeof(D3D12PresentData));
    }
    d3d12CommandBuffer->presentDatas[d3d12CommandBuffer->presentDataCount].windowData = windowData;
    d3d12CommandBuffer->presentDatas[d3d12CommandBuffer->presentDataCount].swapchainImageIndex = swapchainIndex;
    d3d12CommandBuffer->presentDataCount += 1;

    // Set up resource barrier
    D3D12_RESOURCE_BARRIER barrierDesc;
    barrierDesc.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrierDesc.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrierDesc.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
    barrierDesc.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrierDesc.Transition.pResource = windowData->textureContainers[swapchainIndex].activeTexture->resource;
    barrierDesc.Transition.Subresource = 0;

    ID3D12GraphicsCommandList_ResourceBarrier(
        d3d12CommandBuffer->graphicsCommandList,
        1,
        &barrierDesc);

    *swapchainTexture = (SDL_GPUTexture*)&windowData->textureContainers[swapchainIndex];
    return true;
}

static bool D3D12_AcquireSwapchainTexture(
    SDL_GPUCommandBuffer *command_buffer,
    SDL_Window *window,
    SDL_GPUTexture **swapchain_texture,
    Uint32 *swapchain_texture_width,
    Uint32 *swapchain_texture_height
) {
    return D3D12_INTERNAL_AcquireSwapchainTexture(
        false,
        command_buffer,
        window,
        swapchain_texture,
        swapchain_texture_width,
        swapchain_texture_height);
}

static bool D3D12_WaitAndAcquireSwapchainTexture(
    SDL_GPUCommandBuffer *command_buffer,
    SDL_Window *window,
    SDL_GPUTexture **swapchain_texture,
    Uint32 *swapchain_texture_width,
    Uint32 *swapchain_texture_height
) {
    return D3D12_INTERNAL_AcquireSwapchainTexture(
        true,
        command_buffer,
        window,
        swapchain_texture,
        swapchain_texture_width,
        swapchain_texture_height);
}

static void D3D12_INTERNAL_PerformPendingDestroys(D3D12Renderer *renderer)
{
    SDL_LockMutex(renderer->disposeLock);

    for (Sint32 i = renderer->buffersToDestroyCount - 1; i >= 0; i -= 1) {
        if (SDL_GetAtomicInt(&renderer->buffersToDestroy[i]->referenceCount) == 0) {
            D3D12_INTERNAL_DestroyBuffer(
                renderer->buffersToDestroy[i]);

            renderer->buffersToDestroy[i] = renderer->buffersToDestroy[renderer->buffersToDestroyCount - 1];
            renderer->buffersToDestroyCount -= 1;
        }
    }

    for (Sint32 i = renderer->texturesToDestroyCount - 1; i >= 0; i -= 1) {
        if (SDL_GetAtomicInt(&renderer->texturesToDestroy[i]->referenceCount) == 0) {
            D3D12_INTERNAL_DestroyTexture(
                renderer->texturesToDestroy[i]);

            renderer->texturesToDestroy[i] = renderer->texturesToDestroy[renderer->texturesToDestroyCount - 1];
            renderer->texturesToDestroyCount -= 1;
        }
    }

    for (Sint32 i = renderer->samplersToDestroyCount - 1; i >= 0; i -= 1) {
        if (SDL_GetAtomicInt(&renderer->samplersToDestroy[i]->referenceCount) == 0) {
            D3D12_INTERNAL_DestroySampler(
                renderer->samplersToDestroy[i]);

            renderer->samplersToDestroy[i] = renderer->samplersToDestroy[renderer->samplersToDestroyCount - 1];
            renderer->samplersToDestroyCount -= 1;
        }
    }

    for (Sint32 i = renderer->graphicsPipelinesToDestroyCount - 1; i >= 0; i -= 1) {
        if (SDL_GetAtomicInt(&renderer->graphicsPipelinesToDestroy[i]->referenceCount) == 0) {
            D3D12_INTERNAL_DestroyGraphicsPipeline(
                renderer->graphicsPipelinesToDestroy[i]);

            renderer->graphicsPipelinesToDestroy[i] = renderer->graphicsPipelinesToDestroy[renderer->graphicsPipelinesToDestroyCount - 1];
            renderer->graphicsPipelinesToDestroyCount -= 1;
        }
    }

    for (Sint32 i = renderer->computePipelinesToDestroyCount - 1; i >= 0; i -= 1) {
        if (SDL_GetAtomicInt(&renderer->computePipelinesToDestroy[i]->referenceCount) == 0) {
            D3D12_INTERNAL_DestroyComputePipeline(
                renderer->computePipelinesToDestroy[i]);

            renderer->computePipelinesToDestroy[i] = renderer->computePipelinesToDestroy[renderer->computePipelinesToDestroyCount - 1];
            renderer->computePipelinesToDestroyCount -= 1;
        }
    }

    SDL_UnlockMutex(renderer->disposeLock);
}

static bool D3D12_INTERNAL_CopyTextureDownload(
    D3D12CommandBuffer *commandBuffer,
    D3D12TextureDownload *download)
{
    D3D12Renderer *renderer = commandBuffer->renderer;
    Uint8 *sourcePtr;
    Uint8 *destPtr;
    HRESULT res;

    res = ID3D12Resource_Map(
        download->temporaryBuffer->handle,
        0,
        NULL,
        (void **)&sourcePtr);

    CHECK_D3D12_ERROR_AND_RETURN("Failed to map temporary buffer", false);

    res = ID3D12Resource_Map(
        download->destinationBuffer->handle,
        0,
        NULL,
        (void **)&destPtr);

    CHECK_D3D12_ERROR_AND_RETURN("Failed to map destination buffer", false);

    for (Uint32 sliceIndex = 0; sliceIndex < download->depth; sliceIndex += 1) {
        for (Uint32 rowIndex = 0; rowIndex < download->height; rowIndex += 1) {
            SDL_memcpy(
                destPtr + download->bufferOffset + (sliceIndex * download->bytesPerDepthSlice) + (rowIndex * download->bytesPerRow),
                sourcePtr + (sliceIndex * download->height) + (rowIndex * download->alignedBytesPerRow),
                download->bytesPerRow);
        }
    }

    ID3D12Resource_Unmap(
        download->temporaryBuffer->handle,
        0,
        NULL);

    ID3D12Resource_Unmap(
        download->destinationBuffer->handle,
        0,
        NULL);

    return true;
}

static bool D3D12_INTERNAL_CleanCommandBuffer(
    D3D12Renderer *renderer,
    D3D12CommandBuffer *commandBuffer,
    bool cancel)
{
    Uint32 i;
    HRESULT res;
    bool result = true;

    // Perform deferred texture data copies
    for (i = 0; i < commandBuffer->textureDownloadCount; i += 1) {
        if (!cancel) {
            result &= D3D12_INTERNAL_CopyTextureDownload(
                commandBuffer,
                commandBuffer->textureDownloads[i]);
        }
        SDL_free(commandBuffer->textureDownloads[i]);
    }
    commandBuffer->textureDownloadCount = 0;

    if (!result) {
        return false;
    }

    res = ID3D12CommandAllocator_Reset(commandBuffer->commandAllocator);
    CHECK_D3D12_ERROR_AND_RETURN("Could not reset command allocator", false);

    res = ID3D12GraphicsCommandList_Reset(
        commandBuffer->graphicsCommandList,
        commandBuffer->commandAllocator,
        NULL);
    CHECK_D3D12_ERROR_AND_RETURN("Could not reset command list", false);

    // Return descriptor heaps to pool
    D3D12_INTERNAL_ReturnGPUDescriptorHeapToPool(
        renderer,
        commandBuffer->gpuDescriptorHeaps[D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV]);
    D3D12_INTERNAL_ReturnGPUDescriptorHeapToPool(
        renderer,
        commandBuffer->gpuDescriptorHeaps[D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER]);

    commandBuffer->gpuDescriptorHeaps[D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV] = NULL;
    commandBuffer->gpuDescriptorHeaps[D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER] = NULL;

    // Uniform buffers are now available
    SDL_LockMutex(renderer->acquireUniformBufferLock);

    for (i = 0; i < commandBuffer->usedUniformBufferCount; i += 1) {
        D3D12_INTERNAL_ReturnUniformBufferToPool(
            renderer,
            commandBuffer->usedUniformBuffers[i]);
    }
    commandBuffer->usedUniformBufferCount = 0;

    SDL_UnlockMutex(renderer->acquireUniformBufferLock);

    // TODO: More reference counting

    for (i = 0; i < commandBuffer->usedTextureCount; i += 1) {
        (void)SDL_AtomicDecRef(&commandBuffer->usedTextures[i]->referenceCount);
    }
    commandBuffer->usedTextureCount = 0;

    for (i = 0; i < commandBuffer->usedBufferCount; i += 1) {
        (void)SDL_AtomicDecRef(&commandBuffer->usedBuffers[i]->referenceCount);
    }
    commandBuffer->usedBufferCount = 0;

    for (i = 0; i < commandBuffer->usedSamplerCount; i += 1) {
        (void)SDL_AtomicDecRef(&commandBuffer->usedSamplers[i]->referenceCount);
    }
    commandBuffer->usedSamplerCount = 0;

    for (i = 0; i < commandBuffer->usedGraphicsPipelineCount; i += 1) {
        (void)SDL_AtomicDecRef(&commandBuffer->usedGraphicsPipelines[i]->referenceCount);
    }
    commandBuffer->usedGraphicsPipelineCount = 0;

    for (i = 0; i < commandBuffer->usedComputePipelineCount; i += 1) {
        (void)SDL_AtomicDecRef(&commandBuffer->usedComputePipelines[i]->referenceCount);
    }
    commandBuffer->usedComputePipelineCount = 0;

    // Reset presentation
    commandBuffer->presentDataCount = 0;

    // The fence is now available (unless SubmitAndAcquireFence was called)
    if (commandBuffer->autoReleaseFence) {
        D3D12_ReleaseFence(
            (SDL_GPURenderer *)renderer,
            (SDL_GPUFence *)commandBuffer->inFlightFence);

        commandBuffer->inFlightFence = NULL;
    }

    // Return command buffer to pool
    SDL_LockMutex(renderer->acquireCommandBufferLock);

    if (renderer->availableCommandBufferCount == renderer->availableCommandBufferCapacity) {
        renderer->availableCommandBufferCapacity += 1;
        renderer->availableCommandBuffers = (D3D12CommandBuffer **)SDL_realloc(
            renderer->availableCommandBuffers,
            renderer->availableCommandBufferCapacity * sizeof(D3D12CommandBuffer *));
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

    return true;
}

static bool D3D12_Submit(
    SDL_GPUCommandBuffer *commandBuffer)
{
    D3D12CommandBuffer *d3d12CommandBuffer = (D3D12CommandBuffer *)commandBuffer;
    D3D12Renderer *renderer = d3d12CommandBuffer->renderer;
    ID3D12CommandList *commandLists[1];
    HRESULT res;

    SDL_LockMutex(renderer->submitLock);

    // Unmap uniform buffers
    for (Uint32 i = 0; i < MAX_UNIFORM_BUFFERS_PER_STAGE; i += 1) {
        if (d3d12CommandBuffer->vertexUniformBuffers[i] != NULL) {
            ID3D12Resource_Unmap(
                d3d12CommandBuffer->vertexUniformBuffers[i]->buffer->handle,
                0,
                NULL);
            d3d12CommandBuffer->vertexUniformBuffers[i]->buffer->mapPointer = NULL;
        }

        if (d3d12CommandBuffer->fragmentUniformBuffers[i] != NULL) {
            ID3D12Resource_Unmap(
                d3d12CommandBuffer->fragmentUniformBuffers[i]->buffer->handle,
                0,
                NULL);
            d3d12CommandBuffer->fragmentUniformBuffers[i]->buffer->mapPointer = NULL;
        }

        // TODO: compute uniforms
    }

    // Transition present textures to present mode
    for (Uint32 i = 0; i < d3d12CommandBuffer->presentDataCount; i += 1) {
        Uint32 swapchainIndex = d3d12CommandBuffer->presentDatas[i].swapchainImageIndex;
        D3D12TextureContainer *container = &d3d12CommandBuffer->presentDatas[i].windowData->textureContainers[swapchainIndex];
        D3D12TextureSubresource *subresource = D3D12_INTERNAL_FetchTextureSubresource(container, 0, 0);

        D3D12_RESOURCE_BARRIER barrierDesc;
        barrierDesc.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrierDesc.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
        barrierDesc.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
        barrierDesc.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
        barrierDesc.Transition.pResource = subresource->parent->resource;
        barrierDesc.Transition.Subresource = subresource->index;

        ID3D12GraphicsCommandList_ResourceBarrier(
            d3d12CommandBuffer->graphicsCommandList,
            1,
            &barrierDesc);
    }

    // Notify the command buffer that we have completed recording
    res = ID3D12GraphicsCommandList_Close(d3d12CommandBuffer->graphicsCommandList);
    CHECK_D3D12_ERROR_AND_RETURN("Failed to close command list!", false);

    res = ID3D12GraphicsCommandList_QueryInterface(
        d3d12CommandBuffer->graphicsCommandList,
        D3D_GUID(D3D_IID_ID3D12CommandList),
        (void **)&commandLists[0]);
    if (FAILED(res)) {
        SDL_UnlockMutex(renderer->submitLock);
        CHECK_D3D12_ERROR_AND_RETURN("Failed to convert command list!", false);
    }

    // Submit the command list to the queue
    ID3D12CommandQueue_ExecuteCommandLists(
        renderer->commandQueue,
        1,
        commandLists);

    ID3D12CommandList_Release(commandLists[0]);

    // Acquire a fence and set it to the in-flight fence
    d3d12CommandBuffer->inFlightFence = D3D12_INTERNAL_AcquireFence(renderer);
    if (!d3d12CommandBuffer->inFlightFence) {
        SDL_UnlockMutex(renderer->submitLock);
        return false;
    }

    // Mark that a fence should be signaled after command list execution
    res = ID3D12CommandQueue_Signal(
        renderer->commandQueue,
        d3d12CommandBuffer->inFlightFence->handle,
        D3D12_FENCE_SIGNAL_VALUE);
    if (FAILED(res)) {
        SDL_UnlockMutex(renderer->submitLock);
        CHECK_D3D12_ERROR_AND_RETURN("Failed to enqueue fence signal!", false);
    }

    // Mark the command buffer as submitted
    if (renderer->submittedCommandBufferCount + 1 >= renderer->submittedCommandBufferCapacity) {
        renderer->submittedCommandBufferCapacity = renderer->submittedCommandBufferCount + 1;

        renderer->submittedCommandBuffers = (D3D12CommandBuffer **)SDL_realloc(
            renderer->submittedCommandBuffers,
            sizeof(D3D12CommandBuffer *) * renderer->submittedCommandBufferCapacity);
    }

    renderer->submittedCommandBuffers[renderer->submittedCommandBufferCount] = d3d12CommandBuffer;
    renderer->submittedCommandBufferCount += 1;

    bool result = true;

    // Present, if applicable
    for (Uint32 i = 0; i < d3d12CommandBuffer->presentDataCount; i += 1) {
        D3D12PresentData *presentData = &d3d12CommandBuffer->presentDatas[i];
        D3D12WindowData *windowData = presentData->windowData;

#if defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES)
        D3D12XBOX_PRESENT_PLANE_PARAMETERS planeParams;
        SDL_zero(planeParams);
        planeParams.Token = windowData->frameToken;
        planeParams.ResourceCount = 1;
        planeParams.ppResources = &windowData->textureContainers[windowData->frameCounter].activeTexture->resource;
        planeParams.ColorSpace = DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709; // FIXME

        D3D12XBOX_PRESENT_PARAMETERS presentParams;
        SDL_zero(presentParams);
        presentParams.Flags = (windowData->present_mode == SDL_GPU_PRESENTMODE_IMMEDIATE) ? D3D12XBOX_PRESENT_FLAG_IMMEDIATE : D3D12XBOX_PRESENT_FLAG_NONE;

        renderer->commandQueue->PresentX(1, &planeParams, &presentParams);
        if (FAILED(res)) {
            result = false;
        }
#else
        // NOTE: flip discard always supported since DXGI 1.4 is required
        Uint32 syncInterval = 1;
        if (windowData->present_mode == SDL_GPU_PRESENTMODE_IMMEDIATE ||
            windowData->present_mode == SDL_GPU_PRESENTMODE_MAILBOX) {
            syncInterval = 0;
        }

        Uint32 presentFlags = 0;
        if (renderer->supportsTearing &&
            windowData->present_mode == SDL_GPU_PRESENTMODE_IMMEDIATE) {
            presentFlags = DXGI_PRESENT_ALLOW_TEARING;
        }

        res = IDXGISwapChain_Present(
            windowData->swapchain,
            syncInterval,
            presentFlags);
        if (FAILED(res)) {
            result = false;
        }

        ID3D12Resource_Release(windowData->textureContainers[presentData->swapchainImageIndex].activeTexture->resource);
#endif

        windowData->inFlightFences[windowData->frameCounter] = (SDL_GPUFence*)d3d12CommandBuffer->inFlightFence;
        (void)SDL_AtomicIncRef(&d3d12CommandBuffer->inFlightFence->referenceCount);
        windowData->frameCounter = (windowData->frameCounter + 1) % renderer->allowedFramesInFlight;
    }

    // Check for cleanups
    for (Sint32 i = renderer->submittedCommandBufferCount - 1; i >= 0; i -= 1) {
        Uint64 fenceValue = ID3D12Fence_GetCompletedValue(
            renderer->submittedCommandBuffers[i]->inFlightFence->handle);

        if (fenceValue == D3D12_FENCE_SIGNAL_VALUE) {
            result &= D3D12_INTERNAL_CleanCommandBuffer(
                renderer,
                renderer->submittedCommandBuffers[i],
                false);
        }
    }

    D3D12_INTERNAL_PerformPendingDestroys(renderer);

    SDL_UnlockMutex(renderer->submitLock);

    return result;
}

static SDL_GPUFence *D3D12_SubmitAndAcquireFence(
    SDL_GPUCommandBuffer *commandBuffer)
{
    D3D12CommandBuffer *d3d12CommandBuffer = (D3D12CommandBuffer *)commandBuffer;
    d3d12CommandBuffer->autoReleaseFence = false;
    if (!D3D12_Submit(commandBuffer)) {
        return NULL;
    }
    return (SDL_GPUFence *)d3d12CommandBuffer->inFlightFence;
}

static bool D3D12_Cancel(
    SDL_GPUCommandBuffer *commandBuffer)
{
    D3D12CommandBuffer *d3d12CommandBuffer = (D3D12CommandBuffer *)commandBuffer;
    D3D12Renderer *renderer = d3d12CommandBuffer->renderer;
    bool result;
    HRESULT res;

    // Notify the command buffer that we have completed recording
    res = ID3D12GraphicsCommandList_Close(d3d12CommandBuffer->graphicsCommandList);
    CHECK_D3D12_ERROR_AND_RETURN("Failed to close command list!", false);

    d3d12CommandBuffer->autoReleaseFence = false;
    SDL_LockMutex(renderer->submitLock);
    result = D3D12_INTERNAL_CleanCommandBuffer(renderer, d3d12CommandBuffer, true);
    SDL_UnlockMutex(renderer->submitLock);

    return result;
}

static bool D3D12_Wait(
    SDL_GPURenderer *driverData)
{
    D3D12Renderer *renderer = (D3D12Renderer *)driverData;
    D3D12Fence *fence = D3D12_INTERNAL_AcquireFence(renderer);
    if (!fence) {
        return false;
    }
    HRESULT res;

    SDL_LockMutex(renderer->submitLock);

    if (renderer->commandQueue) {
        // Insert a signal into the end of the command queue...
        ID3D12CommandQueue_Signal(
            renderer->commandQueue,
            fence->handle,
            D3D12_FENCE_SIGNAL_VALUE);

        // ...and then block on it.
        if (ID3D12Fence_GetCompletedValue(fence->handle) != D3D12_FENCE_SIGNAL_VALUE) {
            res = ID3D12Fence_SetEventOnCompletion(
                fence->handle,
                D3D12_FENCE_SIGNAL_VALUE,
                fence->event);
            CHECK_D3D12_ERROR_AND_RETURN("Setting fence event failed", false);

            DWORD waitResult = WaitForSingleObject(fence->event, INFINITE);
            if (waitResult == WAIT_FAILED) {
                SDL_UnlockMutex(renderer->submitLock);
                SET_STRING_ERROR_AND_RETURN("Wait failed", false); // TODO: is there a better way to report this?
            }
        }
    }

    D3D12_ReleaseFence(
        (SDL_GPURenderer *)renderer,
        (SDL_GPUFence *)fence);

    bool result = true;

    // Clean up
    for (Sint32 i = renderer->submittedCommandBufferCount - 1; i >= 0; i -= 1) {
        result &= D3D12_INTERNAL_CleanCommandBuffer(renderer, renderer->submittedCommandBuffers[i], false);
    }

    D3D12_INTERNAL_PerformPendingDestroys(renderer);

    SDL_UnlockMutex(renderer->submitLock);

    return result;
}

static bool D3D12_WaitForFences(
    SDL_GPURenderer *driverData,
    bool waitAll,
    SDL_GPUFence *const *fences,
    Uint32 numFences)
{
    D3D12Renderer *renderer = (D3D12Renderer *)driverData;
    D3D12Fence *fence;
    HANDLE *events = SDL_stack_alloc(HANDLE, numFences);
    HRESULT res;

    SDL_LockMutex(renderer->submitLock);

    for (Uint32 i = 0; i < numFences; i += 1) {
        fence = (D3D12Fence *)fences[i];

        res = ID3D12Fence_SetEventOnCompletion(
            fence->handle,
            D3D12_FENCE_SIGNAL_VALUE,
            fence->event);
        CHECK_D3D12_ERROR_AND_RETURN("Setting fence event failed", false);

        events[i] = fence->event;
    }

    DWORD waitResult = WaitForMultipleObjects(
        numFences,
        events,
        waitAll,
        INFINITE);

    if (waitResult == WAIT_FAILED) {
        SDL_UnlockMutex(renderer->submitLock);
        SET_STRING_ERROR_AND_RETURN("Wait failed", false); // TODO: is there a better way to report this?
    }

    bool result = true;

    // Check for cleanups
    for (Sint32 i = renderer->submittedCommandBufferCount - 1; i >= 0; i -= 1) {
        Uint64 fenceValue = ID3D12Fence_GetCompletedValue(
            renderer->submittedCommandBuffers[i]->inFlightFence->handle);

        if (fenceValue == D3D12_FENCE_SIGNAL_VALUE) {
            result &= D3D12_INTERNAL_CleanCommandBuffer(
                renderer,
                renderer->submittedCommandBuffers[i],
                false);
        }
    }

    D3D12_INTERNAL_PerformPendingDestroys(renderer);

    SDL_stack_free(events);

    SDL_UnlockMutex(renderer->submitLock);

    return result;
}

// Feature Queries

static bool D3D12_SupportsTextureFormat(
    SDL_GPURenderer *driverData,
    SDL_GPUTextureFormat format,
    SDL_GPUTextureType type,
    SDL_GPUTextureUsageFlags usage)
{
    D3D12Renderer *renderer = (D3D12Renderer *)driverData;
    DXGI_FORMAT dxgiFormat = SDLToD3D12_TextureFormat[format];
    D3D12_FEATURE_DATA_FORMAT_SUPPORT formatSupport = { dxgiFormat, D3D12_FORMAT_SUPPORT1_NONE, D3D12_FORMAT_SUPPORT2_NONE };
    HRESULT res;

    res = ID3D12Device_CheckFeatureSupport(
        renderer->device,
        D3D12_FEATURE_FORMAT_SUPPORT,
        &formatSupport,
        sizeof(formatSupport));
    if (FAILED(res)) {
        // Format is apparently unknown
        return false;
    }

    // Is the texture type supported?
    if (type == SDL_GPU_TEXTURETYPE_2D && !(formatSupport.Support1 & D3D12_FORMAT_SUPPORT1_TEXTURE2D)) {
        return false;
    }
    if (type == SDL_GPU_TEXTURETYPE_2D_ARRAY && !(formatSupport.Support1 & D3D12_FORMAT_SUPPORT1_TEXTURE2D)) {
        return false;
    }
    if (type == SDL_GPU_TEXTURETYPE_3D && !(formatSupport.Support1 & D3D12_FORMAT_SUPPORT1_TEXTURE3D)) {
        return false;
    }
    if (type == SDL_GPU_TEXTURETYPE_CUBE && !(formatSupport.Support1 & D3D12_FORMAT_SUPPORT1_TEXTURECUBE)) {
        return false;
    }
    if (type == SDL_GPU_TEXTURETYPE_CUBE_ARRAY && !(formatSupport.Support1 & D3D12_FORMAT_SUPPORT1_TEXTURECUBE)) {
        return false;
    }

    // Are the usage flags supported?
    if ((usage & SDL_GPU_TEXTUREUSAGE_SAMPLER) && !(formatSupport.Support1 & D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE)) {
        return false;
    }
    if ((usage & (SDL_GPU_TEXTUREUSAGE_GRAPHICS_STORAGE_READ | SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_READ)) && !(formatSupport.Support1 & D3D12_FORMAT_SUPPORT1_SHADER_LOAD)) {
        return false;
    }
    if ((usage & SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE) && !(formatSupport.Support2 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE)) {
        return false;
    }
    if ((usage & SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_SIMULTANEOUS_READ_WRITE) && !(formatSupport.Support2 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_LOAD)) {
        return false;
    }
    if ((usage & SDL_GPU_TEXTUREUSAGE_COLOR_TARGET) && !(formatSupport.Support1 & D3D12_FORMAT_SUPPORT1_RENDER_TARGET)) {
        return false;
    }

    // Special case check for depth, because D3D12 is great.
    formatSupport.Format = SDLToD3D12_DepthFormat[format];
    formatSupport.Support1 = D3D12_FORMAT_SUPPORT1_NONE;
    formatSupport.Support2 = D3D12_FORMAT_SUPPORT2_NONE;

    res = ID3D12Device_CheckFeatureSupport(
        renderer->device,
        D3D12_FEATURE_FORMAT_SUPPORT,
        &formatSupport,
        sizeof(formatSupport));
    if (FAILED(res)) {
        // Format is apparently unknown
        return false;
    }

    if ((usage & SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET) && !(formatSupport.Support1 & D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL)) {
        return false;
    }

    return true;
}

static bool D3D12_SupportsSampleCount(
    SDL_GPURenderer *driverData,
    SDL_GPUTextureFormat format,
    SDL_GPUSampleCount sampleCount)
{
    D3D12Renderer *renderer = (D3D12Renderer *)driverData;
    D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS featureData;
    HRESULT res;

#if defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES)
    featureData.Flags = (D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG)0;
#else
    featureData.Flags = (D3D12_MULTISAMPLE_QUALITY_LEVEL_FLAGS)0;
#endif
    featureData.Format = SDLToD3D12_TextureFormat[format];
    featureData.SampleCount = SDLToD3D12_SampleCount[sampleCount];
    res = ID3D12Device_CheckFeatureSupport(
        renderer->device,
        D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS,
        &featureData,
        sizeof(featureData));

    return SUCCEEDED(res) && featureData.NumQualityLevels > 0;
}

static void D3D12_INTERNAL_InitBlitResources(
    D3D12Renderer *renderer)
{
    SDL_GPUShaderCreateInfo shaderCreateInfo;
    SDL_GPUSamplerCreateInfo samplerCreateInfo;

    renderer->blitPipelineCapacity = 2;
    renderer->blitPipelineCount = 0;
    renderer->blitPipelines = (BlitPipelineCacheEntry *)SDL_malloc(
        renderer->blitPipelineCapacity * sizeof(BlitPipelineCacheEntry));

    // Fullscreen vertex shader
    SDL_zero(shaderCreateInfo);
    shaderCreateInfo.code = (Uint8 *)D3D12_FullscreenVert;
    shaderCreateInfo.code_size = sizeof(D3D12_FullscreenVert);
    shaderCreateInfo.stage = SDL_GPU_SHADERSTAGE_VERTEX;
    shaderCreateInfo.format = SDL_GPU_SHADERFORMAT_DXIL;

    renderer->blitVertexShader = D3D12_CreateShader(
        (SDL_GPURenderer *)renderer,
        &shaderCreateInfo);

    if (renderer->blitVertexShader == NULL) {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Failed to compile vertex shader for blit!");
    }

    // BlitFrom2D pixel shader
    shaderCreateInfo.code = (Uint8 *)D3D12_BlitFrom2D;
    shaderCreateInfo.code_size = sizeof(D3D12_BlitFrom2D);
    shaderCreateInfo.stage = SDL_GPU_SHADERSTAGE_FRAGMENT;
    shaderCreateInfo.num_samplers = 1;
    shaderCreateInfo.num_uniform_buffers = 1;

    renderer->blitFrom2DShader = D3D12_CreateShader(
        (SDL_GPURenderer *)renderer,
        &shaderCreateInfo);

    if (renderer->blitFrom2DShader == NULL) {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Failed to compile BlitFrom2D pixel shader!");
    }

    // BlitFrom2DArray pixel shader
    shaderCreateInfo.code = (Uint8 *)D3D12_BlitFrom2DArray;
    shaderCreateInfo.code_size = sizeof(D3D12_BlitFrom2DArray);

    renderer->blitFrom2DArrayShader = D3D12_CreateShader(
        (SDL_GPURenderer *)renderer,
        &shaderCreateInfo);

    if (renderer->blitFrom2DArrayShader == NULL) {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Failed to compile BlitFrom2DArray pixel shader!");
    }

    // BlitFrom3D pixel shader
    shaderCreateInfo.code = (Uint8 *)D3D12_BlitFrom3D;
    shaderCreateInfo.code_size = sizeof(D3D12_BlitFrom3D);

    renderer->blitFrom3DShader = D3D12_CreateShader(
        (SDL_GPURenderer *)renderer,
        &shaderCreateInfo);

    if (renderer->blitFrom3DShader == NULL) {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Failed to compile BlitFrom3D pixel shader!");
    }

    // BlitFromCube pixel shader
    shaderCreateInfo.code = (Uint8 *)D3D12_BlitFromCube;
    shaderCreateInfo.code_size = sizeof(D3D12_BlitFromCube);

    renderer->blitFromCubeShader = D3D12_CreateShader(
        (SDL_GPURenderer *)renderer,
        &shaderCreateInfo);

    if (renderer->blitFromCubeShader == NULL) {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Failed to compile BlitFromCube pixel shader!");
    }

    // BlitFromCubeArray pixel shader
    shaderCreateInfo.code = (Uint8 *)D3D12_BlitFromCubeArray;
    shaderCreateInfo.code_size = sizeof(D3D12_BlitFromCubeArray);

    renderer->blitFromCubeArrayShader = D3D12_CreateShader(
        (SDL_GPURenderer *)renderer,
        &shaderCreateInfo);

    if (renderer->blitFromCubeArrayShader == NULL) {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Failed to compile BlitFromCubeArray pixel shader!");
    }

    // Create samplers
    samplerCreateInfo.address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    samplerCreateInfo.address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    samplerCreateInfo.address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    samplerCreateInfo.enable_anisotropy = 0;
    samplerCreateInfo.enable_compare = 0;
    samplerCreateInfo.mag_filter = SDL_GPU_FILTER_NEAREST;
    samplerCreateInfo.min_filter = SDL_GPU_FILTER_NEAREST;
    samplerCreateInfo.mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_NEAREST;
    samplerCreateInfo.mip_lod_bias = 0.0f;
    samplerCreateInfo.min_lod = 0;
    samplerCreateInfo.max_lod = 1000;
    samplerCreateInfo.max_anisotropy = 1.0f;
    samplerCreateInfo.compare_op = SDL_GPU_COMPAREOP_NEVER;

    renderer->blitNearestSampler = D3D12_CreateSampler(
        (SDL_GPURenderer *)renderer,
        &samplerCreateInfo);

    if (renderer->blitNearestSampler == NULL) {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Failed to create blit nearest sampler!");
    }

    samplerCreateInfo.mag_filter = SDL_GPU_FILTER_LINEAR;
    samplerCreateInfo.min_filter = SDL_GPU_FILTER_LINEAR;
    samplerCreateInfo.mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_LINEAR;

    renderer->blitLinearSampler = D3D12_CreateSampler(
        (SDL_GPURenderer *)renderer,
        &samplerCreateInfo);

    if (renderer->blitLinearSampler == NULL) {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Failed to create blit linear sampler!");
    }
}

static bool D3D12_PrepareDriver(SDL_VideoDevice *_this)
{
#if defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES)
    return true;
#else
    SDL_SharedObject *d3d12Dll;
    SDL_SharedObject *dxgiDll;
    PFN_D3D12_CREATE_DEVICE D3D12CreateDeviceFunc;
    PFN_CREATE_DXGI_FACTORY1 CreateDXGIFactoryFunc;
    HRESULT res;
    ID3D12Device *device;
    IDXGIFactory1 *factory;
    IDXGIFactory4 *factory4;
    IDXGIFactory6 *factory6;
    IDXGIAdapter1 *adapter;

    // Can we load D3D12?

    d3d12Dll = SDL_LoadObject(D3D12_DLL);
    if (d3d12Dll == NULL) {
        SDL_LogWarn(SDL_LOG_CATEGORY_GPU, "D3D12: Could not find " D3D12_DLL);
        return false;
    }

    D3D12CreateDeviceFunc = (PFN_D3D12_CREATE_DEVICE)SDL_LoadFunction(
        d3d12Dll,
        D3D12_CREATE_DEVICE_FUNC);
    if (D3D12CreateDeviceFunc == NULL) {
        SDL_LogWarn(SDL_LOG_CATEGORY_GPU, "D3D12: Could not find function " D3D12_CREATE_DEVICE_FUNC " in " D3D12_DLL);
        SDL_UnloadObject(d3d12Dll);
        return false;
    }

    // Can we load DXGI?

    dxgiDll = SDL_LoadObject(DXGI_DLL);
    if (dxgiDll == NULL) {
        SDL_LogWarn(SDL_LOG_CATEGORY_GPU, "D3D12: Could not find " DXGI_DLL);
        return false;
    }

    CreateDXGIFactoryFunc = (PFN_CREATE_DXGI_FACTORY1)SDL_LoadFunction(
        dxgiDll,
        CREATE_DXGI_FACTORY1_FUNC);
    if (CreateDXGIFactoryFunc == NULL) {
        SDL_LogWarn(SDL_LOG_CATEGORY_GPU, "D3D12: Could not find function " CREATE_DXGI_FACTORY1_FUNC " in " DXGI_DLL);
        SDL_UnloadObject(dxgiDll);
        return false;
    }

    // Can we create a device?

    // Create the DXGI factory
    res = CreateDXGIFactoryFunc(
        &D3D_IID_IDXGIFactory1,
        (void **)&factory);
    if (FAILED(res)) {
        SDL_LogWarn(SDL_LOG_CATEGORY_GPU, "D3D12: Could not create DXGIFactory");
        SDL_UnloadObject(d3d12Dll);
        SDL_UnloadObject(dxgiDll);
        return false;
    }

    // Check for DXGI 1.4 support
    res = IDXGIFactory1_QueryInterface(
        factory,
        D3D_GUID(D3D_IID_IDXGIFactory4),
        (void **)&factory4);
    if (FAILED(res)) {
        IDXGIFactory1_Release(factory);
        SDL_LogWarn(SDL_LOG_CATEGORY_GPU, "D3D12: Failed to find DXGI1.4 support, required for DX12");
        SDL_UnloadObject(d3d12Dll);
        SDL_UnloadObject(dxgiDll);
        return false;
    }
    IDXGIFactory4_Release(factory4);

    res = IDXGIFactory1_QueryInterface(
        factory,
        D3D_GUID(D3D_IID_IDXGIFactory6),
        (void **)&factory6);
    if (SUCCEEDED(res)) {
        res = IDXGIFactory6_EnumAdapterByGpuPreference(
            factory6,
            0,
            DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
            D3D_GUID(D3D_IID_IDXGIAdapter1),
            (void **)&adapter);
        IDXGIFactory6_Release(factory6);
    } else {
        res = IDXGIFactory1_EnumAdapters1(
            factory,
            0,
            &adapter);
    }
    if (FAILED(res)) {
        SDL_LogWarn(SDL_LOG_CATEGORY_GPU, "D3D12: Failed to find adapter for D3D12Device");
        IDXGIFactory1_Release(factory);
        SDL_UnloadObject(d3d12Dll);
        SDL_UnloadObject(dxgiDll);
        return false;
    }

    res = D3D12CreateDeviceFunc(
        (IUnknown *)adapter,
        D3D_FEATURE_LEVEL_CHOICE,
        D3D_GUID(D3D_IID_ID3D12Device),
        (void **)&device);

    if (SUCCEEDED(res)) {
        ID3D12Device_Release(device);
    }
    IDXGIAdapter1_Release(adapter);
    IDXGIFactory1_Release(factory);

    SDL_UnloadObject(d3d12Dll);
    SDL_UnloadObject(dxgiDll);

    if (FAILED(res)) {
        SDL_LogWarn(SDL_LOG_CATEGORY_GPU, "D3D12: Could not create D3D12Device with feature level " D3D_FEATURE_LEVEL_CHOICE_STR);
        return false;
    }

    return true;
#endif
}

#if !(defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES)) && defined(HAVE_IDXGIINFOQUEUE)
static bool D3D12_INTERNAL_TryInitializeDXGIDebug(D3D12Renderer *renderer)
{
    PFN_DXGI_GET_DEBUG_INTERFACE DXGIGetDebugInterfaceFunc;
    HRESULT res;

    renderer->dxgidebug_dll = SDL_LoadObject(DXGIDEBUG_DLL);
    if (renderer->dxgidebug_dll == NULL) {
        return false;
    }

    DXGIGetDebugInterfaceFunc = (PFN_DXGI_GET_DEBUG_INTERFACE)SDL_LoadFunction(
        renderer->dxgidebug_dll,
        DXGI_GET_DEBUG_INTERFACE_FUNC);
    if (DXGIGetDebugInterfaceFunc == NULL) {
        return false;
    }

    res = DXGIGetDebugInterfaceFunc(&D3D_IID_IDXGIDebug, (void **)&renderer->dxgiDebug);
    if (FAILED(res)) {
        return false;
    }

    res = DXGIGetDebugInterfaceFunc(&D3D_IID_IDXGIInfoQueue, (void **)&renderer->dxgiInfoQueue);
    if (FAILED(res)) {
        return false;
    }

    return true;
}
#endif

static bool D3D12_INTERNAL_TryInitializeD3D12Debug(D3D12Renderer *renderer)
{
    PFN_D3D12_GET_DEBUG_INTERFACE D3D12GetDebugInterfaceFunc;
    HRESULT res;

    D3D12GetDebugInterfaceFunc = (PFN_D3D12_GET_DEBUG_INTERFACE)SDL_LoadFunction(
        renderer->d3d12_dll,
        D3D12_GET_DEBUG_INTERFACE_FUNC);
    if (D3D12GetDebugInterfaceFunc == NULL) {
        return false;
    }

    res = D3D12GetDebugInterfaceFunc(D3D_GUID(D3D_IID_ID3D12Debug), (void **)&renderer->d3d12Debug);
    if (FAILED(res)) {
        return false;
    }

    ID3D12Debug_EnableDebugLayer(renderer->d3d12Debug);
    return true;
}

#if !(defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES))
static void D3D12_INTERNAL_TryInitializeD3D12DebugInfoQueue(D3D12Renderer *renderer)
{
    ID3D12InfoQueue *infoQueue = NULL;
    D3D12_MESSAGE_SEVERITY severities[] = { D3D12_MESSAGE_SEVERITY_INFO };
    D3D12_INFO_QUEUE_FILTER filter;
    HRESULT res;

    res = ID3D12Device_QueryInterface(
        renderer->device,
        D3D_GUID(D3D_IID_ID3D12InfoQueue),
        (void **)&infoQueue);
    if (FAILED(res)) {
        return;
    }

    SDL_zero(filter);
    filter.DenyList.NumSeverities = 1;
    filter.DenyList.pSeverityList = severities;
    ID3D12InfoQueue_PushStorageFilter(
        infoQueue,
        &filter);

    ID3D12InfoQueue_SetBreakOnSeverity(
        infoQueue,
        D3D12_MESSAGE_SEVERITY_CORRUPTION,
        true);

    ID3D12InfoQueue_Release(infoQueue);
}

static void WINAPI D3D12_INTERNAL_OnD3D12DebugInfoMsg(
    D3D12_MESSAGE_CATEGORY category,
    D3D12_MESSAGE_SEVERITY severity,
    D3D12_MESSAGE_ID id,
    LPCSTR description,
    void *context)
{
    char *catStr;
    char *sevStr;

    switch (category) {
    case D3D12_MESSAGE_CATEGORY_APPLICATION_DEFINED:
        catStr = "APPLICATION_DEFINED";
        break;
    case D3D12_MESSAGE_CATEGORY_MISCELLANEOUS:
        catStr = "MISCELLANEOUS";
        break;
    case D3D12_MESSAGE_CATEGORY_INITIALIZATION:
        catStr = "INITIALIZATION";
        break;
    case D3D12_MESSAGE_CATEGORY_CLEANUP:
        catStr = "CLEANUP";
        break;
    case D3D12_MESSAGE_CATEGORY_COMPILATION:
        catStr = "COMPILATION";
        break;
    case D3D12_MESSAGE_CATEGORY_STATE_CREATION:
        catStr = "STATE_CREATION";
        break;
    case D3D12_MESSAGE_CATEGORY_STATE_SETTING:
        catStr = "STATE_SETTING";
        break;
    case D3D12_MESSAGE_CATEGORY_STATE_GETTING:
        catStr = "STATE_GETTING";
        break;
    case D3D12_MESSAGE_CATEGORY_RESOURCE_MANIPULATION:
        catStr = "RESOURCE_MANIPULATION";
        break;
    case D3D12_MESSAGE_CATEGORY_EXECUTION:
        catStr = "EXECUTION";
        break;
    case D3D12_MESSAGE_CATEGORY_SHADER:
        catStr = "SHADER";
        break;
    default:
        catStr = "UNKNOWN";
        break;
    }

    switch (severity) {
    case D3D12_MESSAGE_SEVERITY_CORRUPTION:
        sevStr = "CORRUPTION";
        break;
    case D3D12_MESSAGE_SEVERITY_ERROR:
        sevStr = "ERROR";
        break;
    case D3D12_MESSAGE_SEVERITY_WARNING:
        sevStr = "WARNING";
        break;
    case D3D12_MESSAGE_SEVERITY_INFO:
        sevStr = "INFO";
        break;
    case D3D12_MESSAGE_SEVERITY_MESSAGE:
        sevStr = "MESSAGE";
        break;
    default:
        sevStr = "UNKNOWN";
        break;
    }

    if (severity <= D3D12_MESSAGE_SEVERITY_ERROR) {
        SDL_LogError(
            SDL_LOG_CATEGORY_GPU,
            "D3D12 ERROR: %s [%s %s #%d]",
            description,
            catStr,
            sevStr,
            id);
    } else {
        SDL_LogWarn(
            SDL_LOG_CATEGORY_GPU,
            "D3D12 WARNING: %s [%s %s #%d]",
            description,
            catStr,
            sevStr,
            id);
    }
}

static void D3D12_INTERNAL_TryInitializeD3D12DebugInfoLogger(D3D12Renderer *renderer)
{
    ID3D12InfoQueue1 *infoQueue = NULL;
    HRESULT res;

    res = ID3D12Device_QueryInterface(
        renderer->device,
        D3D_GUID(D3D_IID_ID3D12InfoQueue1),
        (void **)&infoQueue);
    if (FAILED(res)) {
        return;
    }

    ID3D12InfoQueue1_RegisterMessageCallback(
        infoQueue,
        D3D12_INTERNAL_OnD3D12DebugInfoMsg,
        D3D12_MESSAGE_CALLBACK_FLAG_NONE,
        NULL,
        NULL);

    ID3D12InfoQueue1_Release(infoQueue);
}
#endif

static SDL_GPUDevice *D3D12_CreateDevice(bool debugMode, bool preferLowPower, SDL_PropertiesID props)
{
    SDL_GPUDevice *result;
    D3D12Renderer *renderer;
    HRESULT res;

#if (defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES))
    PFN_D3D12_XBOX_CREATE_DEVICE D3D12XboxCreateDeviceFunc;
    D3D12XBOX_CREATE_DEVICE_PARAMETERS createDeviceParams;
#else
    PFN_CREATE_DXGI_FACTORY1 CreateDXGIFactoryFunc;
    IDXGIFactory1 *factory1;
    IDXGIFactory5 *factory5;
    IDXGIFactory6 *factory6;
    DXGI_ADAPTER_DESC1 adapterDesc;
    LARGE_INTEGER umdVersion;
    PFN_D3D12_CREATE_DEVICE D3D12CreateDeviceFunc;
#endif
    D3D12_FEATURE_DATA_ARCHITECTURE architecture;
    D3D12_COMMAND_QUEUE_DESC queueDesc;

    bool verboseLogs = SDL_GetBooleanProperty(
        props,
        SDL_PROP_GPU_DEVICE_CREATE_VERBOSE_BOOLEAN,
        true);

    renderer = (D3D12Renderer *)SDL_calloc(1, sizeof(D3D12Renderer));

    bool hasDxgiDebug = false;
#if !(defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES))
    // Load the DXGI library
    renderer->dxgi_dll = SDL_LoadObject(DXGI_DLL);
    if (renderer->dxgi_dll == NULL) {
        D3D12_INTERNAL_DestroyRenderer(renderer);
        SET_STRING_ERROR_AND_RETURN("Could not find " DXGI_DLL, NULL);
    }

#ifdef HAVE_IDXGIINFOQUEUE
    // Initialize the DXGI debug layer, if applicable
    if (debugMode) {
        hasDxgiDebug = D3D12_INTERNAL_TryInitializeDXGIDebug(renderer);
    }
#else
    hasDxgiDebug = true;
#endif

    // Load the CreateDXGIFactory1 function
    CreateDXGIFactoryFunc = (PFN_CREATE_DXGI_FACTORY1)SDL_LoadFunction(
        renderer->dxgi_dll,
        CREATE_DXGI_FACTORY1_FUNC);
    if (CreateDXGIFactoryFunc == NULL) {
        D3D12_INTERNAL_DestroyRenderer(renderer);
        SET_STRING_ERROR_AND_RETURN("Could not load function: " CREATE_DXGI_FACTORY1_FUNC, NULL);
    }

    // Create the DXGI factory
    res = CreateDXGIFactoryFunc(
        &D3D_IID_IDXGIFactory1,
        (void **)&factory1);
    if (FAILED(res)) {
        D3D12_INTERNAL_DestroyRenderer(renderer);
        CHECK_D3D12_ERROR_AND_RETURN("Could not create DXGIFactory", NULL);
    }

    // Check for DXGI 1.4 support
    res = IDXGIFactory1_QueryInterface(
        factory1,
        D3D_GUID(D3D_IID_IDXGIFactory4),
        (void **)&renderer->factory);
    if (FAILED(res)) {
        D3D12_INTERNAL_DestroyRenderer(renderer);
        CHECK_D3D12_ERROR_AND_RETURN("DXGI1.4 support not found, required for DX12", NULL);
    }
    IDXGIFactory1_Release(factory1);

    // Check for explicit tearing support
    res = IDXGIFactory4_QueryInterface(
        renderer->factory,
        D3D_GUID(D3D_IID_IDXGIFactory5),
        (void **)&factory5);
    if (SUCCEEDED(res)) {
        res = IDXGIFactory5_CheckFeatureSupport(
            factory5,
            DXGI_FEATURE_PRESENT_ALLOW_TEARING,
            &renderer->supportsTearing,
            sizeof(renderer->supportsTearing));
        if (FAILED(res)) {
            renderer->supportsTearing = false;
        }
        IDXGIFactory5_Release(factory5);
    }

    // Select the appropriate device for rendering
    res = IDXGIFactory4_QueryInterface(
        renderer->factory,
        D3D_GUID(D3D_IID_IDXGIFactory6),
        (void **)&factory6);
    if (SUCCEEDED(res)) {
        res = IDXGIFactory6_EnumAdapterByGpuPreference(
            factory6,
            0,
            preferLowPower ? DXGI_GPU_PREFERENCE_MINIMUM_POWER : DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
            D3D_GUID(D3D_IID_IDXGIAdapter1),
            (void **)&renderer->adapter);
        IDXGIFactory6_Release(factory6);
    } else {
        res = IDXGIFactory4_EnumAdapters1(
            renderer->factory,
            0,
            &renderer->adapter);
    }

    if (FAILED(res)) {
        D3D12_INTERNAL_DestroyRenderer(renderer);
        CHECK_D3D12_ERROR_AND_RETURN("Could not find adapter for D3D12Device", NULL);
    }

    // Get information about the selected adapter. Used for logging info.
    res = IDXGIAdapter1_GetDesc1(renderer->adapter, &adapterDesc);
    if (FAILED(res)) {
        D3D12_INTERNAL_DestroyRenderer(renderer);
        CHECK_D3D12_ERROR_AND_RETURN("Could not get adapter description", NULL);
    }
    res = IDXGIAdapter1_CheckInterfaceSupport(renderer->adapter, D3D_GUID(D3D_IID_IDXGIDevice), &umdVersion);
    if (FAILED(res)) {
        D3D12_INTERNAL_DestroyRenderer(renderer);
        CHECK_D3D12_ERROR_AND_RETURN("Could not get adapter driver version", NULL);
    }

    renderer->props = SDL_CreateProperties();
    if (verboseLogs) {
        SDL_LogInfo(SDL_LOG_CATEGORY_GPU, "SDL_GPU Driver: D3D12");
    }

    // Record device name
    char *deviceName = SDL_iconv_wchar_utf8(&adapterDesc.Description[0]);
    SDL_SetStringProperty(
        renderer->props,
        SDL_PROP_GPU_DEVICE_NAME_STRING,
        deviceName);
    if (verboseLogs) {
        SDL_LogInfo(SDL_LOG_CATEGORY_GPU, "D3D12 Adapter: %s", deviceName);
    }
    SDL_free(deviceName);

    // Record driver version
    char driverVer[64];
    (void)SDL_snprintf(
        driverVer,
        SDL_arraysize(driverVer),
        "%d.%d.%d.%d",
        HIWORD(umdVersion.HighPart),
        LOWORD(umdVersion.HighPart),
        HIWORD(umdVersion.LowPart),
        LOWORD(umdVersion.LowPart));
    SDL_SetStringProperty(
        renderer->props,
        SDL_PROP_GPU_DEVICE_DRIVER_VERSION_STRING,
        driverVer);
    if (verboseLogs) {
        SDL_LogInfo(SDL_LOG_CATEGORY_GPU, "D3D12 Driver: %s", driverVer);
    }
#endif

    // Load the D3D library
    renderer->d3d12_dll = SDL_LoadObject(D3D12_DLL);
    if (renderer->d3d12_dll == NULL) {
        D3D12_INTERNAL_DestroyRenderer(renderer);
        SET_STRING_ERROR_AND_RETURN("Could not find " D3D12_DLL, NULL);
    }

    // Load the CreateDevice function
#if (defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES))
    D3D12XboxCreateDeviceFunc = (PFN_D3D12_XBOX_CREATE_DEVICE)SDL_LoadFunction(
        renderer->d3d12_dll,
        "D3D12XboxCreateDevice");
    if (D3D12XboxCreateDeviceFunc == NULL) {
        D3D12_INTERNAL_DestroyRenderer(renderer);
        SET_STRING_ERROR_AND_RETURN("Could not load function: D3D12XboxCreateDevice", NULL);
    }
#else
    D3D12CreateDeviceFunc = (PFN_D3D12_CREATE_DEVICE)SDL_LoadFunction(
        renderer->d3d12_dll,
        D3D12_CREATE_DEVICE_FUNC);
    if (D3D12CreateDeviceFunc == NULL) {
        D3D12_INTERNAL_DestroyRenderer(renderer);
        SET_STRING_ERROR_AND_RETURN("Could not load function: " D3D12_CREATE_DEVICE_FUNC, NULL);
    }
#endif

    renderer->D3D12SerializeRootSignature_func = (PFN_D3D12_SERIALIZE_ROOT_SIGNATURE)SDL_LoadFunction(
        renderer->d3d12_dll,
        D3D12_SERIALIZE_ROOT_SIGNATURE_FUNC);
    if (renderer->D3D12SerializeRootSignature_func == NULL) {
        D3D12_INTERNAL_DestroyRenderer(renderer);
        SET_STRING_ERROR_AND_RETURN("Could not load function: " D3D12_SERIALIZE_ROOT_SIGNATURE_FUNC, NULL);
    }

    // Initialize the D3D12 debug layer, if applicable
    if (debugMode) {
        bool hasD3d12Debug = D3D12_INTERNAL_TryInitializeD3D12Debug(renderer);
        if (hasDxgiDebug && hasD3d12Debug) {
            SDL_LogInfo(
                SDL_LOG_CATEGORY_GPU,
                "Validation layers enabled, expect debug level performance!");
        } else if (hasDxgiDebug || hasD3d12Debug) {
            SDL_LogWarn(
                SDL_LOG_CATEGORY_GPU,
                "Validation layers partially enabled, some warnings may not be available");
        } else {
            SDL_LogWarn(
                SDL_LOG_CATEGORY_GPU,
                "Validation layers not found, continuing without validation");
        }
    }

    // Create the D3D12Device
#if (defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES))
    if (s_Device != NULL) {
        renderer->device = s_Device;
    } else {
        SDL_zero(createDeviceParams);
        createDeviceParams.Version = D3D12_SDK_VERSION;
        createDeviceParams.GraphicsCommandQueueRingSizeBytes = D3D12XBOX_DEFAULT_SIZE_BYTES;
        createDeviceParams.GraphicsScratchMemorySizeBytes = D3D12XBOX_DEFAULT_SIZE_BYTES;
        createDeviceParams.ComputeScratchMemorySizeBytes = D3D12XBOX_DEFAULT_SIZE_BYTES;
        createDeviceParams.DisableGeometryShaderAllocations = TRUE;
        createDeviceParams.DisableTessellationShaderAllocations = TRUE;
#if defined(SDL_PLATFORM_XBOXSERIES)
        createDeviceParams.DisableDXR = TRUE;
#endif
        if (debugMode) {
            createDeviceParams.ProcessDebugFlags = D3D12XBOX_PROCESS_DEBUG_FLAG_DEBUG;
        }

        res = D3D12XboxCreateDeviceFunc(
            NULL,
            &createDeviceParams,
            IID_GRAPHICS_PPV_ARGS(&renderer->device));
        if (FAILED(res)) {
            D3D12_INTERNAL_DestroyRenderer(renderer);
            CHECK_D3D12_ERROR_AND_RETURN("Could not create D3D12Device", NULL);
        }

        s_Device = renderer->device;
    }
#else
    res = D3D12CreateDeviceFunc(
        (IUnknown *)renderer->adapter,
        D3D_FEATURE_LEVEL_CHOICE,
        D3D_GUID(D3D_IID_ID3D12Device),
        (void **)&renderer->device);

    if (FAILED(res)) {
        D3D12_INTERNAL_DestroyRenderer(renderer);
        CHECK_D3D12_ERROR_AND_RETURN("Could not create D3D12Device", NULL);
    }

    // Initialize the D3D12 debug info queue, if applicable
    if (debugMode) {
        D3D12_INTERNAL_TryInitializeD3D12DebugInfoQueue(renderer);
        D3D12_INTERNAL_TryInitializeD3D12DebugInfoLogger(renderer);
    }
#endif

    // Check UMA
    architecture.NodeIndex = 0;
    res = ID3D12Device_CheckFeatureSupport(
        renderer->device,
        D3D12_FEATURE_ARCHITECTURE,
        &architecture,
        sizeof(D3D12_FEATURE_DATA_ARCHITECTURE));
    if (FAILED(res)) {
        D3D12_INTERNAL_DestroyRenderer(renderer);
        CHECK_D3D12_ERROR_AND_RETURN("Could not get device architecture", NULL);
    }

    renderer->UMA = (bool)architecture.UMA;
    renderer->UMACacheCoherent = (bool)architecture.CacheCoherentUMA;

#if (defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES))
    renderer->GPUUploadHeapSupported = false;
#else
    // Check "GPU Upload Heap" support (for fast uniform buffers)
    D3D12_FEATURE_DATA_D3D12_OPTIONS16 options16; // 15 wasn't enough, huh?
    renderer->GPUUploadHeapSupported = false;
    res = ID3D12Device_CheckFeatureSupport(
        renderer->device,
        D3D12_FEATURE_D3D12_OPTIONS16,
        &options16,
        sizeof(options16));

    if (SUCCEEDED(res)) {
        renderer->GPUUploadHeapSupported = options16.GPUUploadHeapSupported;
    }
#endif

    // Create command queue
#if (defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES))
    if (s_CommandQueue != NULL) {
        renderer->commandQueue = s_CommandQueue;
    } else {
#endif
        queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
        queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        queueDesc.NodeMask = 0;
        queueDesc.Priority = 0;

        res = ID3D12Device_CreateCommandQueue(
            renderer->device,
            &queueDesc,
            D3D_GUID(D3D_IID_ID3D12CommandQueue),
            (void **)&renderer->commandQueue);

        if (FAILED(res)) {
            D3D12_INTERNAL_DestroyRenderer(renderer);
            CHECK_D3D12_ERROR_AND_RETURN("Could not create D3D12CommandQueue", NULL);
        }
#if (defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES))
        s_CommandQueue = renderer->commandQueue;
    }
#endif

    // Create indirect command signatures

    D3D12_COMMAND_SIGNATURE_DESC commandSignatureDesc;
    D3D12_INDIRECT_ARGUMENT_DESC indirectArgumentDesc;
    SDL_zero(indirectArgumentDesc);

    indirectArgumentDesc.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DRAW;
    commandSignatureDesc.NodeMask = 0;
    commandSignatureDesc.ByteStride = sizeof(SDL_GPUIndirectDrawCommand);
    commandSignatureDesc.NumArgumentDescs = 1;
    commandSignatureDesc.pArgumentDescs = &indirectArgumentDesc;

    res = ID3D12Device_CreateCommandSignature(
        renderer->device,
        &commandSignatureDesc,
        NULL,
        D3D_GUID(D3D_IID_ID3D12CommandSignature),
        (void **)&renderer->indirectDrawCommandSignature);
    if (FAILED(res)) {
        D3D12_INTERNAL_DestroyRenderer(renderer);
        CHECK_D3D12_ERROR_AND_RETURN("Could not create indirect draw command signature", NULL);
    }

    indirectArgumentDesc.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED;
    commandSignatureDesc.ByteStride = sizeof(SDL_GPUIndexedIndirectDrawCommand);
    commandSignatureDesc.pArgumentDescs = &indirectArgumentDesc;

    res = ID3D12Device_CreateCommandSignature(
        renderer->device,
        &commandSignatureDesc,
        NULL,
        D3D_GUID(D3D_IID_ID3D12CommandSignature),
        (void **)&renderer->indirectIndexedDrawCommandSignature);
    if (FAILED(res)) {
        D3D12_INTERNAL_DestroyRenderer(renderer);
        CHECK_D3D12_ERROR_AND_RETURN("Could not create indirect indexed draw command signature", NULL);
    }

    indirectArgumentDesc.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;
    commandSignatureDesc.ByteStride = sizeof(SDL_GPUIndirectDispatchCommand);
    commandSignatureDesc.pArgumentDescs = &indirectArgumentDesc;

    res = ID3D12Device_CreateCommandSignature(
        renderer->device,
        &commandSignatureDesc,
        NULL,
        D3D_GUID(D3D_IID_ID3D12CommandSignature),
        (void **)&renderer->indirectDispatchCommandSignature);
    if (FAILED(res)) {
        D3D12_INTERNAL_DestroyRenderer(renderer);
        CHECK_D3D12_ERROR_AND_RETURN("Could not create indirect dispatch command signature", NULL);
    }

    // Initialize pools

    renderer->submittedCommandBufferCapacity = 4;
    renderer->submittedCommandBufferCount = 0;
    renderer->submittedCommandBuffers = (D3D12CommandBuffer **)SDL_calloc(
        renderer->submittedCommandBufferCapacity, sizeof(D3D12CommandBuffer *));
    if (!renderer->submittedCommandBuffers) {
        D3D12_INTERNAL_DestroyRenderer(renderer);
        return NULL;
    }

    renderer->uniformBufferPoolCapacity = 4;
    renderer->uniformBufferPoolCount = 0;
    renderer->uniformBufferPool = (D3D12UniformBuffer **)SDL_calloc(
        renderer->uniformBufferPoolCapacity, sizeof(D3D12UniformBuffer *));
    if (!renderer->uniformBufferPool) {
        D3D12_INTERNAL_DestroyRenderer(renderer);
        return NULL;
    }

    renderer->claimedWindowCapacity = 4;
    renderer->claimedWindowCount = 0;
    renderer->claimedWindows = (D3D12WindowData **)SDL_calloc(
        renderer->claimedWindowCapacity, sizeof(D3D12WindowData *));
    if (!renderer->claimedWindows) {
        D3D12_INTERNAL_DestroyRenderer(renderer);
        return NULL;
    }

    renderer->availableFenceCapacity = 4;
    renderer->availableFenceCount = 0;
    renderer->availableFences = (D3D12Fence **)SDL_calloc(
        renderer->availableFenceCapacity, sizeof(D3D12Fence *));
    if (!renderer->availableFences) {
        D3D12_INTERNAL_DestroyRenderer(renderer);
        return NULL;
    }

    // Initialize staging descriptor pools
    for (Uint32 i = 0; i < D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES; i += 1) {
        renderer->stagingDescriptorPools[i] = D3D12_INTERNAL_CreateStagingDescriptorPool(
            renderer,
            (D3D12_DESCRIPTOR_HEAP_TYPE)i);

        if (renderer->stagingDescriptorPools[i] == NULL) {
            D3D12_INTERNAL_DestroyRenderer(renderer);
            return NULL;
        }
    }

    // Initialize GPU descriptor heaps
    for (Uint32 i = 0; i < 2; i += 1) {
        renderer->gpuDescriptorHeapPools[i].lock = SDL_CreateMutex();
        renderer->gpuDescriptorHeapPools[i].capacity = 4;
        renderer->gpuDescriptorHeapPools[i].count = 4;
        renderer->gpuDescriptorHeapPools[i].heaps = (D3D12DescriptorHeap **)SDL_calloc(
            renderer->gpuDescriptorHeapPools[i].capacity, sizeof(D3D12DescriptorHeap *));

        for (Uint32 j = 0; j < renderer->gpuDescriptorHeapPools[i].capacity; j += 1) {
            renderer->gpuDescriptorHeapPools[i].heaps[j] = D3D12_INTERNAL_CreateDescriptorHeap(
                renderer,
                (D3D12_DESCRIPTOR_HEAP_TYPE)i,
                i == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV ? VIEW_GPU_DESCRIPTOR_COUNT : SAMPLER_GPU_DESCRIPTOR_COUNT,
                false);

            if (renderer->gpuDescriptorHeapPools[i].heaps[j] == NULL) {
                D3D12_INTERNAL_DestroyRenderer(renderer);
                return NULL;
            }
        }
    }

    // Deferred resource releasing

    renderer->buffersToDestroyCapacity = 4;
    renderer->buffersToDestroyCount = 0;
    renderer->buffersToDestroy = (D3D12Buffer **)SDL_calloc(
        renderer->buffersToDestroyCapacity, sizeof(D3D12Buffer *));
    if (!renderer->buffersToDestroy) {
        D3D12_INTERNAL_DestroyRenderer(renderer);
        return NULL;
    }

    renderer->texturesToDestroyCapacity = 4;
    renderer->texturesToDestroyCount = 0;
    renderer->texturesToDestroy = (D3D12Texture **)SDL_calloc(
        renderer->texturesToDestroyCapacity, sizeof(D3D12Texture *));
    if (!renderer->texturesToDestroy) {
        D3D12_INTERNAL_DestroyRenderer(renderer);
        return NULL;
    }

    renderer->samplersToDestroyCapacity = 4;
    renderer->samplersToDestroyCount = 0;
    renderer->samplersToDestroy = (D3D12Sampler **)SDL_calloc(
        renderer->samplersToDestroyCapacity, sizeof(D3D12Sampler *));
    if (!renderer->samplersToDestroy) {
        D3D12_INTERNAL_DestroyRenderer(renderer);
        return NULL;
    }

    renderer->graphicsPipelinesToDestroyCapacity = 4;
    renderer->graphicsPipelinesToDestroyCount = 0;
    renderer->graphicsPipelinesToDestroy = (D3D12GraphicsPipeline **)SDL_calloc(
        renderer->graphicsPipelinesToDestroyCapacity, sizeof(D3D12GraphicsPipeline *));
    if (!renderer->graphicsPipelinesToDestroy) {
        D3D12_INTERNAL_DestroyRenderer(renderer);
        return NULL;
    }

    renderer->computePipelinesToDestroyCapacity = 4;
    renderer->computePipelinesToDestroyCount = 0;
    renderer->computePipelinesToDestroy = (D3D12ComputePipeline **)SDL_calloc(
        renderer->computePipelinesToDestroyCapacity, sizeof(D3D12ComputePipeline *));
    if (!renderer->computePipelinesToDestroy) {
        D3D12_INTERNAL_DestroyRenderer(renderer);
        return NULL;
    }

    // Locks
    renderer->acquireCommandBufferLock = SDL_CreateMutex();
    renderer->acquireUniformBufferLock = SDL_CreateMutex();
    renderer->submitLock = SDL_CreateMutex();
    renderer->windowLock = SDL_CreateMutex();
    renderer->fenceLock = SDL_CreateMutex();
    renderer->disposeLock = SDL_CreateMutex();

    renderer->debug_mode = debugMode;
    renderer->allowedFramesInFlight = 2;

    renderer->semantic = SDL_GetStringProperty(props, SDL_PROP_GPU_DEVICE_CREATE_D3D12_SEMANTIC_NAME_STRING, "TEXCOORD");

    // Blit resources
    D3D12_INTERNAL_InitBlitResources(renderer);

#if (defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES))
    res = renderer->device->SetFrameIntervalX(
        NULL,
        D3D12XBOX_FRAME_INTERVAL_60_HZ,
        renderer->allowedFramesInFlight - 1,
        D3D12XBOX_FRAME_INTERVAL_FLAG_NONE);
    if (FAILED(res)) {
        D3D12_INTERNAL_DestroyRenderer(renderer);
        CHECK_D3D12_ERROR_AND_RETURN("Could not get set frame interval", NULL);
    }

    res = renderer->device->ScheduleFrameEventX(
        D3D12XBOX_FRAME_EVENT_ORIGIN,
        0,
        NULL,
        D3D12XBOX_SCHEDULE_FRAME_EVENT_FLAG_NONE);
    if (FAILED(res)) {
        D3D12_INTERNAL_DestroyRenderer(renderer);
        CHECK_D3D12_ERROR_AND_RETURN("Could not schedule frame events", NULL);
    }
#endif

    // Create the SDL_GPU Device
    result = (SDL_GPUDevice *)SDL_calloc(1, sizeof(SDL_GPUDevice));

    if (!result) {
        D3D12_INTERNAL_DestroyRenderer(renderer);
        return NULL;
    }

    ASSIGN_DRIVER(D3D12)
    result->driverData = (SDL_GPURenderer *)renderer;
    result->debug_mode = debugMode;
    renderer->sdlGPUDevice = result;

    return result;
}

SDL_GPUBootstrap D3D12Driver = {
    "direct3d12",
    SDL_GPU_SHADERFORMAT_DXIL | SDL_GPU_SHADERFORMAT_DXBC,
    D3D12_PrepareDriver,
    D3D12_CreateDevice
};

#endif // SDL_GPU_D3D12

// GDK-specific APIs

#ifdef SDL_PLATFORM_GDK

void SDL_GDKSuspendGPU(SDL_GPUDevice *device)
{
#if defined(SDL_GPU_D3D12) && (defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES))
    D3D12Renderer *renderer = (D3D12Renderer *)device->driverData;
    HRESULT res;
    if (device == NULL) {
        SET_STRING_ERROR_AND_RETURN("Invalid GPU device", );
    }

    SDL_LockMutex(renderer->submitLock);
    res = renderer->commandQueue->SuspendX(0);
    if (FAILED(res)) {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "SuspendX failed: %X", res);
    }
    SDL_UnlockMutex(renderer->submitLock);
#endif
}

void SDL_GDKResumeGPU(SDL_GPUDevice *device)
{
#if defined(SDL_GPU_D3D12) && (defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES))
    D3D12Renderer *renderer = (D3D12Renderer *)device->driverData;
    HRESULT res;
    if (device == NULL) {
        SET_STRING_ERROR_AND_RETURN("Invalid GPU device", );
    }

    SDL_LockMutex(renderer->submitLock);
    res = renderer->commandQueue->ResumeX();
    if (FAILED(res)) {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "ResumeX failed: %X", res);
    }
    SDL_UnlockMutex(renderer->submitLock);

    res = renderer->device->SetFrameIntervalX(
        NULL,
        D3D12XBOX_FRAME_INTERVAL_60_HZ,
        renderer->allowedFramesInFlight - 1,
        D3D12XBOX_FRAME_INTERVAL_FLAG_NONE);
    if (FAILED(res)) {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Could not set frame interval: %X", res);
    }

    res = renderer->device->ScheduleFrameEventX(
        D3D12XBOX_FRAME_EVENT_ORIGIN,
        0,
        NULL,
        D3D12XBOX_SCHEDULE_FRAME_EVENT_FLAG_NONE);
    if (FAILED(res)) {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "Could not schedule frame events: %X", res);
    }
#endif
}

#endif // SDL_PLATFORM_GDK
