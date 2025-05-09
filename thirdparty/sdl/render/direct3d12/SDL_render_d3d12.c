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

#ifdef SDL_VIDEO_RENDER_D3D12

#define SDL_D3D12_NUM_BUFFERS        2
#define SDL_D3D12_NUM_VERTEX_BUFFERS 256
#define SDL_D3D12_MAX_NUM_TEXTURES   16384
#define SDL_D3D12_NUM_UPLOAD_BUFFERS 32

#include "../../core/windows/SDL_windows.h"
#include "../../video/windows/SDL_windowswindow.h"
#include "../SDL_sysrender.h"
#include "../SDL_d3dmath.h"
#include "../../video/directx/SDL_d3d12.h"

#if defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES)
#include "SDL_render_d3d12_xbox.h"
#endif

#include "SDL_shaders_d3d12.h"

#if defined(_MSC_VER) && !defined(__clang__)
#define SDL_COMPOSE_ERROR(str) __FUNCTION__ ", " str
#else
#define SDL_COMPOSE_ERROR(str) SDL_STRINGIFY_ARG(__FUNCTION__) ", " str
#endif

// Set up for C function definitions, even when using C++
#ifdef __cplusplus
extern "C" {
#endif

// This must be included here as the function definitions in SDL_pixels.c/_c.h are C, not C++
#include "../../video/SDL_pixels_c.h"

/* !!! FIXME: vertex buffer bandwidth could be lower; only use UV coords when
   !!! FIXME:  textures are needed. */

// Vertex shader, common values
typedef struct
{
    Float4X4 model;
    Float4X4 projectionAndView;
} D3D12_VertexShaderConstants;

// These should mirror the definitions in D3D12_PixelShader_Common.hlsli
//static const float TONEMAP_NONE = 0;
//static const float TONEMAP_LINEAR = 1;
static const float TONEMAP_CHROME = 2;

//static const float TEXTURETYPE_NONE = 0;
static const float TEXTURETYPE_RGB = 1;
static const float TEXTURETYPE_RGB_PIXELART = 2;
static const float TEXTURETYPE_NV12 = 3;
static const float TEXTURETYPE_NV21 = 4;
static const float TEXTURETYPE_YUV = 5;

static const float INPUTTYPE_UNSPECIFIED = 0;
static const float INPUTTYPE_SRGB = 1;
static const float INPUTTYPE_SCRGB = 2;
static const float INPUTTYPE_HDR10 = 3;

typedef struct
{
    float scRGB_output;
    float texture_type;
    float input_type;
    float color_scale;

    float texel_width;
    float texel_height;
    float texture_width;
    float texture_height;

    float tonemap_method;
    float tonemap_factor1;
    float tonemap_factor2;
    float sdr_white_point;

    float YCbCr_matrix[16];
} D3D12_PixelShaderConstants;

// Per-vertex data
typedef struct
{
    Float2 pos;
    Float2 tex;
    SDL_FColor color;
} D3D12_VertexPositionColor;

// Per-texture data
typedef struct
{
    int w, h;
    ID3D12Resource *mainTexture;
    D3D12_CPU_DESCRIPTOR_HANDLE mainTextureResourceView;
    D3D12_RESOURCE_STATES mainResourceState;
    SIZE_T mainSRVIndex;
    D3D12_CPU_DESCRIPTOR_HANDLE mainTextureRenderTargetView;
    DXGI_FORMAT mainTextureFormat;
    ID3D12Resource *stagingBuffer;
    D3D12_RESOURCE_STATES stagingResourceState;
    D3D12_Shader shader;
    const float *YCbCr_matrix;
#ifdef SDL_HAVE_YUV
    // YV12 texture support
    bool yuv;
    ID3D12Resource *mainTextureU;
    D3D12_CPU_DESCRIPTOR_HANDLE mainTextureResourceViewU;
    D3D12_RESOURCE_STATES mainResourceStateU;
    SIZE_T mainSRVIndexU;
    ID3D12Resource *mainTextureV;
    D3D12_CPU_DESCRIPTOR_HANDLE mainTextureResourceViewV;
    D3D12_RESOURCE_STATES mainResourceStateV;
    SIZE_T mainSRVIndexV;

    // NV12 texture support
    bool nv12;
    D3D12_CPU_DESCRIPTOR_HANDLE mainTextureResourceViewNV;
    SIZE_T mainSRVIndexNV;

    Uint8 *pixels;
    int pitch;
#endif
    SDL_Rect lockedRect;
} D3D12_TextureData;

// Pipeline State Object data
typedef struct
{
    D3D12_Shader shader;
    D3D12_PixelShaderConstants shader_constants;
    SDL_BlendMode blendMode;
    D3D12_PRIMITIVE_TOPOLOGY_TYPE topology;
    DXGI_FORMAT rtvFormat;
    ID3D12PipelineState *pipelineState;
} D3D12_PipelineState;

// Vertex Buffer
typedef struct
{
    ID3D12Resource *resource;
    D3D12_VERTEX_BUFFER_VIEW view;
    size_t size;
} D3D12_VertexBuffer;

// For SRV pool allocator
typedef struct
{
    SIZE_T index;
    void *next;
} D3D12_SRVPoolNode;

// Private renderer data
typedef struct
{
    SDL_SharedObject *hDXGIMod;
    SDL_SharedObject *hD3D12Mod;
#if defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES)
    UINT64 frameToken;
#else
    IDXGIFactory6 *dxgiFactory;
    IDXGIAdapter4 *dxgiAdapter;
    IDXGIDebug *dxgiDebug;
    IDXGISwapChain4 *swapChain;
#endif
    ID3D12Device1 *d3dDevice;
    ID3D12Debug *debugInterface;
    ID3D12CommandQueue *commandQueue;
    ID3D12GraphicsCommandList2 *commandList;
    DXGI_SWAP_EFFECT swapEffect;
    UINT swapFlags;
    UINT syncInterval;
    UINT presentFlags;
    DXGI_FORMAT renderTargetFormat;
    bool pixelSizeChanged;

    // Descriptor heaps
    ID3D12DescriptorHeap *rtvDescriptorHeap;
    UINT rtvDescriptorSize;
    ID3D12DescriptorHeap *textureRTVDescriptorHeap;
    ID3D12DescriptorHeap *srvDescriptorHeap;
    UINT srvDescriptorSize;
    ID3D12DescriptorHeap *samplerDescriptorHeap;
    UINT samplerDescriptorSize;

    // Data needed per backbuffer
    ID3D12CommandAllocator *commandAllocators[SDL_D3D12_NUM_BUFFERS];
    ID3D12Resource *renderTargets[SDL_D3D12_NUM_BUFFERS];
    UINT64 fenceValue;
    int currentBackBufferIndex;

    // Fences
    ID3D12Fence *fence;
    HANDLE fenceEvent;

    // Root signature and pipeline state data
    ID3D12RootSignature *rootSignatures[NUM_ROOTSIGS];
    int pipelineStateCount;
    D3D12_PipelineState *pipelineStates;
    D3D12_PipelineState *currentPipelineState;

    D3D12_VertexBuffer vertexBuffers[SDL_D3D12_NUM_VERTEX_BUFFERS];
    D3D12_CPU_DESCRIPTOR_HANDLE samplers[RENDER_SAMPLER_COUNT];
    bool samplers_created[RENDER_SAMPLER_COUNT];

    // Data for staging/allocating textures
    ID3D12Resource *uploadBuffers[SDL_D3D12_NUM_UPLOAD_BUFFERS];
    int currentUploadBuffer;

    // Pool allocator to handle reusing SRV heap indices
    D3D12_SRVPoolNode *srvPoolHead;
    D3D12_SRVPoolNode srvPoolNodes[SDL_D3D12_MAX_NUM_TEXTURES];

    // Vertex buffer constants
    D3D12_VertexShaderConstants vertexShaderConstantsData;

    // Cached renderer properties
    DXGI_MODE_ROTATION rotation;
    D3D12_TextureData *textureRenderTarget;
    D3D12_CPU_DESCRIPTOR_HANDLE currentRenderTargetView;
    D3D12_CPU_DESCRIPTOR_HANDLE currentShaderResource;
    D3D12_CPU_DESCRIPTOR_HANDLE currentSampler;
    bool cliprectDirty;
    bool currentCliprectEnabled;
    SDL_Rect currentCliprect;
    SDL_Rect currentViewport;
    int currentViewportRotation;
    bool viewportDirty;
    Float4X4 identity;
    int currentVertexBuffer;
    bool issueBatch;
} D3D12_RenderData;

// Define D3D GUIDs here so we don't have to include uuid.lib.

#ifdef HAVE_GCC_DIAGNOSTIC_PRAGMA
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-const-variable"
#endif

static const GUID SDL_IID_IDXGIFactory6 = { 0xc1b6694f, 0xff09, 0x44a9, { 0xb0, 0x3c, 0x77, 0x90, 0x0a, 0x0a, 0x1d, 0x17 } };
static const GUID SDL_IID_IDXGIAdapter4 = { 0x3c8d99d1, 0x4fbf, 0x4181, { 0xa8, 0x2c, 0xaf, 0x66, 0xbf, 0x7b, 0xd2, 0x4e } };
static const GUID SDL_IID_IDXGIDevice1 = { 0x77db970f, 0x6276, 0x48ba, { 0xba, 0x28, 0x07, 0x01, 0x43, 0xb4, 0x39, 0x2c } };
static const GUID SDL_IID_ID3D12Device1 = { 0x77acce80, 0x638e, 0x4e65, { 0x88, 0x95, 0xc1, 0xf2, 0x33, 0x86, 0x86, 0x3e } };
static const GUID SDL_IID_IDXGISwapChain4 = { 0x3D585D5A, 0xBD4A, 0x489E, { 0xB1, 0xF4, 0x3D, 0xBC, 0xB6, 0x45, 0x2F, 0xFB } };
static const GUID SDL_IID_IDXGIDebug1 = { 0xc5a05f0c, 0x16f2, 0x4adf, { 0x9f, 0x4d, 0xa8, 0xc4, 0xd5, 0x8a, 0xc5, 0x50 } };
static const GUID SDL_IID_IDXGIInfoQueue = { 0xD67441C7, 0x672A, 0x476f, { 0x9E, 0x82, 0xCD, 0x55, 0xB4, 0x49, 0x49, 0xCE } };
static const GUID SDL_IID_ID3D12Debug = { 0x344488b7, 0x6846, 0x474b, { 0xb9, 0x89, 0xf0, 0x27, 0x44, 0x82, 0x45, 0xe0 } };
static const GUID SDL_DXGI_DEBUG_ALL = { 0xe48ae283, 0xda80, 0x490b, { 0x87, 0xe6, 0x43, 0xe9, 0xa9, 0xcf, 0xda, 0x8 } };
static const GUID SDL_IID_ID3D12CommandQueue = { 0x0ec870a6, 0x5d7e, 0x4c22, { 0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed } };
static const GUID SDL_IID_ID3D12DescriptorHeap = { 0x8efb471d, 0x616c, 0x4f49, { 0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51 } };
static const GUID SDL_IID_ID3D12CommandAllocator = { 0x6102dee4, 0xaf59, 0x4b09, { 0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24 } };
static const GUID SDL_IID_ID3D12GraphicsCommandList2 = { 0x38C3E585, 0xFF17, 0x412C, { 0x91, 0x50, 0x4F, 0xC6, 0xF9, 0xD7, 0x2A, 0x28 } };
static const GUID SDL_IID_ID3D12Fence = { 0x0a753dcf, 0xc4d8, 0x4b91, { 0xad, 0xf6, 0xbe, 0x5a, 0x60, 0xd9, 0x5a, 0x76 } };
static const GUID SDL_IID_ID3D12Resource = { 0x696442be, 0xa72e, 0x4059, { 0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad } };
static const GUID SDL_IID_ID3D12RootSignature = { 0xc54a6b66, 0x72df, 0x4ee8, { 0x8b, 0xe5, 0xa9, 0x46, 0xa1, 0x42, 0x92, 0x14 } };
static const GUID SDL_IID_ID3D12PipelineState = { 0x765a30f3, 0xf624, 0x4c6f, { 0xa8, 0x28, 0xac, 0xe9, 0x48, 0x62, 0x24, 0x45 } };
static const GUID SDL_IID_ID3D12Heap = { 0x6b3b2502, 0x6e51, 0x45b3, { 0x90, 0xee, 0x98, 0x84, 0x26, 0x5e, 0x8d, 0xf3 } };
static const GUID SDL_IID_ID3D12InfoQueue = { 0x0742a90b, 0xc387, 0x483f, { 0xb9, 0x46, 0x30, 0xa7, 0xe4, 0xe6, 0x14, 0x58 } };

#ifdef HAVE_GCC_DIAGNOSTIC_PRAGMA
#pragma GCC diagnostic pop
#endif

static UINT D3D12_Align(UINT location, UINT alignment)
{
    return (location + (alignment - 1)) & ~(alignment - 1);
}

static SDL_PixelFormat D3D12_DXGIFormatToSDLPixelFormat(DXGI_FORMAT dxgiFormat)
{
    switch (dxgiFormat) {
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
        return SDL_PIXELFORMAT_ARGB8888;
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
        return SDL_PIXELFORMAT_ABGR8888;
    case DXGI_FORMAT_B8G8R8X8_UNORM:
    case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB:
        return SDL_PIXELFORMAT_XRGB8888;
    case DXGI_FORMAT_R10G10B10A2_UNORM:
        return SDL_PIXELFORMAT_ABGR2101010;
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
        return SDL_PIXELFORMAT_RGBA64_FLOAT;
    default:
        return SDL_PIXELFORMAT_UNKNOWN;
    }
}

static DXGI_FORMAT SDLPixelFormatToDXGITextureFormat(Uint32 format, Uint32 output_colorspace)
{
    switch (format) {
    case SDL_PIXELFORMAT_RGBA64_FLOAT:
        return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case SDL_PIXELFORMAT_ABGR2101010:
        return DXGI_FORMAT_R10G10B10A2_UNORM;
    case SDL_PIXELFORMAT_ARGB8888:
        if (output_colorspace == SDL_COLORSPACE_SRGB_LINEAR) {
            return DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
        }
        return DXGI_FORMAT_B8G8R8A8_UNORM;
    case SDL_PIXELFORMAT_ABGR8888:
        if (output_colorspace == SDL_COLORSPACE_SRGB_LINEAR) {
            return DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
        }
        return DXGI_FORMAT_R8G8B8A8_UNORM;
    case SDL_PIXELFORMAT_XRGB8888:
        if (output_colorspace == SDL_COLORSPACE_SRGB_LINEAR) {
            return DXGI_FORMAT_B8G8R8X8_UNORM_SRGB;
        }
        return DXGI_FORMAT_B8G8R8X8_UNORM;
    case SDL_PIXELFORMAT_YV12:
    case SDL_PIXELFORMAT_IYUV:
        return DXGI_FORMAT_R8_UNORM;
    case SDL_PIXELFORMAT_NV12:
    case SDL_PIXELFORMAT_NV21:
        return DXGI_FORMAT_NV12;
    case SDL_PIXELFORMAT_P010:
        return DXGI_FORMAT_P010;
    default:
        return DXGI_FORMAT_UNKNOWN;
    }
}

static DXGI_FORMAT SDLPixelFormatToDXGIMainResourceViewFormat(Uint32 format, Uint32 colorspace)
{
    switch (format) {
    case SDL_PIXELFORMAT_RGBA64_FLOAT:
        return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case SDL_PIXELFORMAT_ABGR2101010:
        return DXGI_FORMAT_R10G10B10A2_UNORM;
    case SDL_PIXELFORMAT_ARGB8888:
        if (colorspace == SDL_COLORSPACE_SRGB_LINEAR) {
            return DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
        }
        return DXGI_FORMAT_B8G8R8A8_UNORM;
    case SDL_PIXELFORMAT_ABGR8888:
        if (colorspace == SDL_COLORSPACE_SRGB_LINEAR) {
            return DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
        }
        return DXGI_FORMAT_R8G8B8A8_UNORM;
    case SDL_PIXELFORMAT_XRGB8888:
        if (colorspace == SDL_COLORSPACE_SRGB_LINEAR) {
            return DXGI_FORMAT_B8G8R8X8_UNORM_SRGB;
        }
        return DXGI_FORMAT_B8G8R8X8_UNORM;
    case SDL_PIXELFORMAT_YV12:
    case SDL_PIXELFORMAT_IYUV:
    case SDL_PIXELFORMAT_NV12: // For the Y texture
    case SDL_PIXELFORMAT_NV21: // For the Y texture
        return DXGI_FORMAT_R8_UNORM;
    case SDL_PIXELFORMAT_P010:  // For the Y texture
        return DXGI_FORMAT_R16_UNORM;
    default:
        return DXGI_FORMAT_UNKNOWN;
    }
}

static void D3D12_DestroyTexture(SDL_Renderer *renderer, SDL_Texture *texture);

static void D3D12_ReleaseAll(SDL_Renderer *renderer)
{
    D3D12_RenderData *data = (D3D12_RenderData *)renderer->internal;

    SDL_PropertiesID props = SDL_GetRendererProperties(renderer);
    SDL_SetPointerProperty(props, SDL_PROP_RENDERER_D3D12_DEVICE_POINTER, NULL);
    SDL_SetPointerProperty(props, SDL_PROP_RENDERER_D3D12_COMMAND_QUEUE_POINTER, NULL);

    // Release all textures
    for (SDL_Texture *texture = renderer->textures; texture; texture = texture->next) {
        D3D12_DestroyTexture(renderer, texture);
    }

    // Release/reset everything else
    if (data) {
        int i;

#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
        D3D_SAFE_RELEASE(data->dxgiFactory);
        D3D_SAFE_RELEASE(data->dxgiAdapter);
        D3D_SAFE_RELEASE(data->swapChain);
#endif
        D3D_SAFE_RELEASE(data->d3dDevice);
        D3D_SAFE_RELEASE(data->debugInterface);
        D3D_SAFE_RELEASE(data->commandQueue);
        D3D_SAFE_RELEASE(data->commandList);
        D3D_SAFE_RELEASE(data->rtvDescriptorHeap);
        D3D_SAFE_RELEASE(data->textureRTVDescriptorHeap);
        D3D_SAFE_RELEASE(data->srvDescriptorHeap);
        D3D_SAFE_RELEASE(data->samplerDescriptorHeap);
        SDL_zeroa(data->samplers_created);
        D3D_SAFE_RELEASE(data->fence);

        for (i = 0; i < SDL_D3D12_NUM_BUFFERS; ++i) {
            D3D_SAFE_RELEASE(data->commandAllocators[i]);
            D3D_SAFE_RELEASE(data->renderTargets[i]);
        }

        if (data->pipelineStateCount > 0) {
            for (i = 0; i < data->pipelineStateCount; ++i) {
                D3D_SAFE_RELEASE(data->pipelineStates[i].pipelineState);
            }
            SDL_free(data->pipelineStates);
            data->pipelineStates = NULL;
            data->pipelineStateCount = 0;
        }

        for (i = 0; i < NUM_ROOTSIGS; ++i) {
            D3D_SAFE_RELEASE(data->rootSignatures[i]);
        }

        for (i = 0; i < SDL_D3D12_NUM_VERTEX_BUFFERS; ++i) {
            D3D_SAFE_RELEASE(data->vertexBuffers[i].resource);
            data->vertexBuffers[i].size = 0;
        }

        data->swapEffect = (DXGI_SWAP_EFFECT)0;
        data->swapFlags = 0;
        data->currentRenderTargetView.ptr = 0;
        data->currentSampler.ptr = 0;

#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
        // Check for any leaks if in debug mode
        if (data->dxgiDebug) {
            DXGI_DEBUG_RLO_FLAGS rloFlags = (DXGI_DEBUG_RLO_FLAGS)(DXGI_DEBUG_RLO_DETAIL | DXGI_DEBUG_RLO_IGNORE_INTERNAL);
            IDXGIDebug_ReportLiveObjects(data->dxgiDebug, SDL_DXGI_DEBUG_ALL, rloFlags);
            D3D_SAFE_RELEASE(data->dxgiDebug);
        }
#endif

        /* Unload the D3D libraries.  This should be done last, in order
         * to prevent IUnknown::Release() calls from crashing.
         */
        if (data->hD3D12Mod) {
            SDL_UnloadObject(data->hD3D12Mod);
            data->hD3D12Mod = NULL;
        }
        if (data->hDXGIMod) {
            SDL_UnloadObject(data->hDXGIMod);
            data->hDXGIMod = NULL;
        }
    }
}

static D3D12_GPU_DESCRIPTOR_HANDLE D3D12_CPUtoGPUHandle(ID3D12DescriptorHeap *heap, D3D12_CPU_DESCRIPTOR_HANDLE CPUHandle)
{
    D3D12_CPU_DESCRIPTOR_HANDLE CPUHeapStart;
    D3D12_GPU_DESCRIPTOR_HANDLE GPUHandle;
    SIZE_T offset;

    // Calculate the correct offset into the heap
    D3D_CALL_RET(heap, GetCPUDescriptorHandleForHeapStart, &CPUHeapStart);
    offset = CPUHandle.ptr - CPUHeapStart.ptr;

    D3D_CALL_RET(heap, GetGPUDescriptorHandleForHeapStart, &GPUHandle);
    GPUHandle.ptr += offset;

    return GPUHandle;
}

static void D3D12_WaitForGPU(D3D12_RenderData *data)
{
    if (data->commandQueue && data->fence && data->fenceEvent) {
        ID3D12CommandQueue_Signal(data->commandQueue, data->fence, data->fenceValue);
        if (ID3D12Fence_GetCompletedValue(data->fence) < data->fenceValue) {
            ID3D12Fence_SetEventOnCompletion(data->fence,
                     data->fenceValue,
                     data->fenceEvent);
            WaitForSingleObjectEx(data->fenceEvent, INFINITE, FALSE);
        }

        data->fenceValue++;
    }
}

static D3D12_CPU_DESCRIPTOR_HANDLE D3D12_GetCurrentRenderTargetView(SDL_Renderer *renderer)
{
    D3D12_RenderData *data = (D3D12_RenderData *)renderer->internal;
    D3D12_CPU_DESCRIPTOR_HANDLE rtvDescriptor;

    if (data->textureRenderTarget) {
        return data->textureRenderTarget->mainTextureRenderTargetView;
    }

    SDL_zero(rtvDescriptor);
    D3D_CALL_RET(data->rtvDescriptorHeap, GetCPUDescriptorHandleForHeapStart, &rtvDescriptor);
    rtvDescriptor.ptr += data->currentBackBufferIndex * data->rtvDescriptorSize;
    return rtvDescriptor;
}

static void D3D12_TransitionResource(D3D12_RenderData *data,
                                     ID3D12Resource *resource,
                                     D3D12_RESOURCE_STATES beforeState,
                                     D3D12_RESOURCE_STATES afterState)
{
    D3D12_RESOURCE_BARRIER barrier;

    if (beforeState != afterState) {
        SDL_zero(barrier);
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = resource;
        barrier.Transition.StateBefore = beforeState;
        barrier.Transition.StateAfter = afterState;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

        ID3D12GraphicsCommandList2_ResourceBarrier(data->commandList, 1, &barrier);
    }
}

static void D3D12_ResetCommandList(D3D12_RenderData *data)
{
    int i;
    ID3D12DescriptorHeap *rootDescriptorHeaps[] = { data->srvDescriptorHeap, data->samplerDescriptorHeap };
    ID3D12CommandAllocator *commandAllocator = data->commandAllocators[data->currentBackBufferIndex];

    ID3D12CommandAllocator_Reset(commandAllocator);
    ID3D12GraphicsCommandList2_Reset(data->commandList, commandAllocator, NULL);
    data->currentPipelineState = NULL;
    data->currentVertexBuffer = 0;
    data->issueBatch = false;
    data->cliprectDirty = true;
    data->viewportDirty = true;
    data->currentRenderTargetView.ptr = 0;
    // FIXME should we also clear currentSampler.ptr and currentRenderTargetView.ptr ? (and use D3D12_InvalidateCachedState() instead)

    // Release any upload buffers that were inflight
    for (i = 0; i < data->currentUploadBuffer; ++i) {
        D3D_SAFE_RELEASE(data->uploadBuffers[i]);
    }
    data->currentUploadBuffer = 0;

    ID3D12GraphicsCommandList2_SetDescriptorHeaps(data->commandList, 2, rootDescriptorHeaps);
}

static HRESULT D3D12_IssueBatch(D3D12_RenderData *data)
{
    HRESULT result = S_OK;

    // Issue the command list
    result = ID3D12GraphicsCommandList2_Close(data->commandList);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("D3D12_IssueBatch"), result);
        return result;
    }
    ID3D12CommandQueue_ExecuteCommandLists(data->commandQueue, 1, (ID3D12CommandList *const *)&data->commandList);

    D3D12_WaitForGPU(data);

    D3D12_ResetCommandList(data);

    return result;
}

static void D3D12_DestroyRenderer(SDL_Renderer *renderer)
{
    D3D12_RenderData *data = (D3D12_RenderData *)renderer->internal;
    if (data) {
        D3D12_WaitForGPU(data);
        D3D12_ReleaseAll(renderer);
        SDL_free(data);
    }
}

static D3D12_BLEND GetBlendFunc(SDL_BlendFactor factor)
{
    switch (factor) {
    case SDL_BLENDFACTOR_ZERO:
        return D3D12_BLEND_ZERO;
    case SDL_BLENDFACTOR_ONE:
        return D3D12_BLEND_ONE;
    case SDL_BLENDFACTOR_SRC_COLOR:
        return D3D12_BLEND_SRC_COLOR;
    case SDL_BLENDFACTOR_ONE_MINUS_SRC_COLOR:
        return D3D12_BLEND_INV_SRC_COLOR;
    case SDL_BLENDFACTOR_SRC_ALPHA:
        return D3D12_BLEND_SRC_ALPHA;
    case SDL_BLENDFACTOR_ONE_MINUS_SRC_ALPHA:
        return D3D12_BLEND_INV_SRC_ALPHA;
    case SDL_BLENDFACTOR_DST_COLOR:
        return D3D12_BLEND_DEST_COLOR;
    case SDL_BLENDFACTOR_ONE_MINUS_DST_COLOR:
        return D3D12_BLEND_INV_DEST_COLOR;
    case SDL_BLENDFACTOR_DST_ALPHA:
        return D3D12_BLEND_DEST_ALPHA;
    case SDL_BLENDFACTOR_ONE_MINUS_DST_ALPHA:
        return D3D12_BLEND_INV_DEST_ALPHA;
    default:
        return (D3D12_BLEND)0;
    }
}

static D3D12_BLEND_OP GetBlendEquation(SDL_BlendOperation operation)
{
    switch (operation) {
    case SDL_BLENDOPERATION_ADD:
        return D3D12_BLEND_OP_ADD;
    case SDL_BLENDOPERATION_SUBTRACT:
        return D3D12_BLEND_OP_SUBTRACT;
    case SDL_BLENDOPERATION_REV_SUBTRACT:
        return D3D12_BLEND_OP_REV_SUBTRACT;
    case SDL_BLENDOPERATION_MINIMUM:
        return D3D12_BLEND_OP_MIN;
    case SDL_BLENDOPERATION_MAXIMUM:
        return D3D12_BLEND_OP_MAX;
    default:
        return (D3D12_BLEND_OP)0;
    }
}

static void D3D12_CreateBlendState(SDL_Renderer *renderer, SDL_BlendMode blendMode, D3D12_BLEND_DESC *outBlendDesc)
{
    SDL_BlendFactor srcColorFactor = SDL_GetBlendModeSrcColorFactor(blendMode);
    SDL_BlendFactor srcAlphaFactor = SDL_GetBlendModeSrcAlphaFactor(blendMode);
    SDL_BlendOperation colorOperation = SDL_GetBlendModeColorOperation(blendMode);
    SDL_BlendFactor dstColorFactor = SDL_GetBlendModeDstColorFactor(blendMode);
    SDL_BlendFactor dstAlphaFactor = SDL_GetBlendModeDstAlphaFactor(blendMode);
    SDL_BlendOperation alphaOperation = SDL_GetBlendModeAlphaOperation(blendMode);

    SDL_zerop(outBlendDesc);
    outBlendDesc->AlphaToCoverageEnable = FALSE;
    outBlendDesc->IndependentBlendEnable = FALSE;
    outBlendDesc->RenderTarget[0].BlendEnable = TRUE;
    outBlendDesc->RenderTarget[0].SrcBlend = GetBlendFunc(srcColorFactor);
    outBlendDesc->RenderTarget[0].DestBlend = GetBlendFunc(dstColorFactor);
    outBlendDesc->RenderTarget[0].BlendOp = GetBlendEquation(colorOperation);
    outBlendDesc->RenderTarget[0].SrcBlendAlpha = GetBlendFunc(srcAlphaFactor);
    outBlendDesc->RenderTarget[0].DestBlendAlpha = GetBlendFunc(dstAlphaFactor);
    outBlendDesc->RenderTarget[0].BlendOpAlpha = GetBlendEquation(alphaOperation);
    outBlendDesc->RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
}

static D3D12_PipelineState *D3D12_CreatePipelineState(SDL_Renderer *renderer,
                                                      D3D12_Shader shader,
                                                      SDL_BlendMode blendMode,
                                                      D3D12_PRIMITIVE_TOPOLOGY_TYPE topology,
                                                      DXGI_FORMAT rtvFormat)
{
    const D3D12_INPUT_ELEMENT_DESC vertexDesc[] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 8, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 16, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
    };
    D3D12_RenderData *data = (D3D12_RenderData *)renderer->internal;
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pipelineDesc;
    ID3D12PipelineState *pipelineState = NULL;
    D3D12_PipelineState *pipelineStates;
    HRESULT result = S_OK;

    SDL_zero(pipelineDesc);
    pipelineDesc.pRootSignature = data->rootSignatures[D3D12_GetRootSignatureType(shader)];
    D3D12_GetVertexShader(shader, &pipelineDesc.VS);
    D3D12_GetPixelShader(shader, &pipelineDesc.PS);
    D3D12_CreateBlendState(renderer, blendMode, &pipelineDesc.BlendState);
    pipelineDesc.SampleMask = 0xffffffff;

    pipelineDesc.RasterizerState.AntialiasedLineEnable = FALSE;
    pipelineDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pipelineDesc.RasterizerState.DepthBias = 0;
    pipelineDesc.RasterizerState.DepthBiasClamp = 0.0f;
    pipelineDesc.RasterizerState.DepthClipEnable = TRUE;
    pipelineDesc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pipelineDesc.RasterizerState.FrontCounterClockwise = FALSE;
    pipelineDesc.RasterizerState.MultisampleEnable = FALSE;
    pipelineDesc.RasterizerState.SlopeScaledDepthBias = 0.0f;

    pipelineDesc.InputLayout.pInputElementDescs = vertexDesc;
    pipelineDesc.InputLayout.NumElements = 3;

    pipelineDesc.PrimitiveTopologyType = topology;

    pipelineDesc.NumRenderTargets = 1;
    pipelineDesc.RTVFormats[0] = rtvFormat;
    pipelineDesc.SampleDesc.Count = 1;
    pipelineDesc.SampleDesc.Quality = 0;

    result = ID3D12Device1_CreateGraphicsPipelineState(data->d3dDevice,
                      &pipelineDesc,
                      D3D_GUID(SDL_IID_ID3D12PipelineState),
                      (void **)&pipelineState);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D12Device::CreateGraphicsPipelineState"), result);
        return NULL;
    }

    pipelineStates = (D3D12_PipelineState *)SDL_realloc(data->pipelineStates, (data->pipelineStateCount + 1) * sizeof(*pipelineStates));
    if (!pipelineStates) {
        D3D_SAFE_RELEASE(pipelineState);
        return NULL;
    }

    pipelineStates[data->pipelineStateCount].shader = shader;
    pipelineStates[data->pipelineStateCount].blendMode = blendMode;
    pipelineStates[data->pipelineStateCount].topology = topology;
    pipelineStates[data->pipelineStateCount].rtvFormat = rtvFormat;
    pipelineStates[data->pipelineStateCount].pipelineState = pipelineState;
    data->pipelineStates = pipelineStates;
    ++data->pipelineStateCount;

    return &pipelineStates[data->pipelineStateCount - 1];
}

static HRESULT D3D12_CreateVertexBuffer(D3D12_RenderData *data, size_t vbidx, size_t size)
{
    D3D12_HEAP_PROPERTIES vbufferHeapProps;
    D3D12_RESOURCE_DESC vbufferDesc;
    HRESULT result;

    D3D_SAFE_RELEASE(data->vertexBuffers[vbidx].resource);

    SDL_zero(vbufferHeapProps);
    vbufferHeapProps.Type = D3D12_HEAP_TYPE_UPLOAD;
    vbufferHeapProps.CreationNodeMask = 1;
    vbufferHeapProps.VisibleNodeMask = 1;

    SDL_zero(vbufferDesc);
    vbufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    vbufferDesc.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
    vbufferDesc.Width = size;
    vbufferDesc.Height = 1;
    vbufferDesc.DepthOrArraySize = 1;
    vbufferDesc.MipLevels = 1;
    vbufferDesc.Format = DXGI_FORMAT_UNKNOWN;
    vbufferDesc.SampleDesc.Count = 1;
    vbufferDesc.SampleDesc.Quality = 0;
    vbufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    vbufferDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    result = ID3D12Device1_CreateCommittedResource(data->d3dDevice,
                      &vbufferHeapProps,
                      D3D12_HEAP_FLAG_NONE,
                      &vbufferDesc,
                      D3D12_RESOURCE_STATE_GENERIC_READ,
                      NULL,
                      D3D_GUID(SDL_IID_ID3D12Resource),
                      (void **)&data->vertexBuffers[vbidx].resource);

    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D12Device::CreatePlacedResource [vertex buffer]"), result);
        return result;
    }

    data->vertexBuffers[vbidx].view.BufferLocation = ID3D12Resource_GetGPUVirtualAddress(data->vertexBuffers[vbidx].resource);
    data->vertexBuffers[vbidx].view.StrideInBytes = sizeof(D3D12_VertexPositionColor);
    data->vertexBuffers[vbidx].size = size;

    return result;
}

// Create resources that depend on the device.
static HRESULT D3D12_CreateDeviceResources(SDL_Renderer *renderer)
{
#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
    typedef HRESULT(WINAPI * PFN_CREATE_DXGI_FACTORY)(UINT flags, REFIID riid, void **ppFactory);
    PFN_CREATE_DXGI_FACTORY CreateDXGIFactoryFunc;
    PFN_D3D12_CREATE_DEVICE D3D12CreateDeviceFunc;
#endif
    typedef HANDLE(WINAPI * PFN_CREATE_EVENT_EX)(LPSECURITY_ATTRIBUTES lpEventAttributes, LPCWSTR lpName, DWORD dwFlags, DWORD dwDesiredAccess);
    PFN_CREATE_EVENT_EX CreateEventExFunc;

    D3D12_RenderData *data = (D3D12_RenderData *)renderer->internal;
    ID3D12Device *d3dDevice = NULL;
    HRESULT result = S_OK;
    UINT creationFlags = 0;
    int i;
    bool createDebug;

    D3D12_COMMAND_QUEUE_DESC queueDesc;
    D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDesc;
    ID3D12DescriptorHeap *rootDescriptorHeaps[2];

    // See if we need debug interfaces
    createDebug = SDL_GetHintBoolean(SDL_HINT_RENDER_DIRECT3D11_DEBUG, false);

#ifdef SDL_PLATFORM_GDK
    CreateEventExFunc = CreateEventExW;
#else
    // CreateEventEx() arrived in Vista, so we need to load it with GetProcAddress for XP.
    {
        HMODULE kernel32 = GetModuleHandle(TEXT("kernel32.dll"));
        CreateEventExFunc = NULL;
        if (kernel32) {
            CreateEventExFunc = (PFN_CREATE_EVENT_EX)GetProcAddress(kernel32, "CreateEventExW");
        }
    }
#endif
    if (!CreateEventExFunc) {
        result = E_FAIL;
        goto done;
    }

#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
    data->hDXGIMod = SDL_LoadObject("dxgi.dll");
    if (!data->hDXGIMod) {
        result = E_FAIL;
        goto done;
    }

    CreateDXGIFactoryFunc = (PFN_CREATE_DXGI_FACTORY)SDL_LoadFunction(data->hDXGIMod, "CreateDXGIFactory2");
    if (!CreateDXGIFactoryFunc) {
        result = E_FAIL;
        goto done;
    }

    data->hD3D12Mod = SDL_LoadObject("D3D12.dll");
    if (!data->hD3D12Mod) {
        result = E_FAIL;
        goto done;
    }

    D3D12CreateDeviceFunc = (PFN_D3D12_CREATE_DEVICE)SDL_LoadFunction(data->hD3D12Mod, "D3D12CreateDevice");
    if (!D3D12CreateDeviceFunc) {
        result = E_FAIL;
        goto done;
    }

    if (createDebug) {
        PFN_D3D12_GET_DEBUG_INTERFACE D3D12GetDebugInterfaceFunc;

        D3D12GetDebugInterfaceFunc = (PFN_D3D12_GET_DEBUG_INTERFACE)SDL_LoadFunction(data->hD3D12Mod, "D3D12GetDebugInterface");
        if (!D3D12GetDebugInterfaceFunc) {
            result = E_FAIL;
            goto done;
        }
        if (SUCCEEDED(D3D12GetDebugInterfaceFunc(D3D_GUID(SDL_IID_ID3D12Debug), (void **)&data->debugInterface))) {
            ID3D12Debug_EnableDebugLayer(data->debugInterface);
        }
    }
#endif // !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)

#if defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES)
    result = D3D12_XBOX_CreateDevice(&d3dDevice, createDebug);
    if (FAILED(result)) {
        // SDL Error is set by D3D12_XBOX_CreateDevice
        goto done;
    }
#else
    if (createDebug) {
#ifdef __IDXGIInfoQueue_INTERFACE_DEFINED__
        IDXGIInfoQueue *dxgiInfoQueue = NULL;
        PFN_CREATE_DXGI_FACTORY DXGIGetDebugInterfaceFunc;

        // If the debug hint is set, also create the DXGI factory in debug mode
        DXGIGetDebugInterfaceFunc = (PFN_CREATE_DXGI_FACTORY)SDL_LoadFunction(data->hDXGIMod, "DXGIGetDebugInterface1");
        if (!DXGIGetDebugInterfaceFunc) {
            result = E_FAIL;
            goto done;
        }

        result = DXGIGetDebugInterfaceFunc(0, D3D_GUID(SDL_IID_IDXGIDebug1), (void **)&data->dxgiDebug);
        if (FAILED(result)) {
            WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("DXGIGetDebugInterface1"), result);
            goto done;
        }

        result = DXGIGetDebugInterfaceFunc(0, D3D_GUID(SDL_IID_IDXGIInfoQueue), (void **)&dxgiInfoQueue);
        if (FAILED(result)) {
            WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("DXGIGetDebugInterface1"), result);
            goto done;
        }

        IDXGIInfoQueue_SetBreakOnSeverity(dxgiInfoQueue, SDL_DXGI_DEBUG_ALL, DXGI_INFO_QUEUE_MESSAGE_SEVERITY_ERROR, TRUE);
        IDXGIInfoQueue_SetBreakOnSeverity(dxgiInfoQueue, SDL_DXGI_DEBUG_ALL, DXGI_INFO_QUEUE_MESSAGE_SEVERITY_CORRUPTION, TRUE);
        D3D_SAFE_RELEASE(dxgiInfoQueue);
#endif // __IDXGIInfoQueue_INTERFACE_DEFINED__
        creationFlags = DXGI_CREATE_FACTORY_DEBUG;
    }

    result = CreateDXGIFactoryFunc(creationFlags, D3D_GUID(SDL_IID_IDXGIFactory6), (void **)&data->dxgiFactory);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("CreateDXGIFactory"), result);
        goto done;
    }

    // Prefer a high performance adapter if there are multiple choices
    result = IDXGIFactory6_EnumAdapterByGpuPreference(data->dxgiFactory,
                      0,
                      DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                      D3D_GUID(SDL_IID_IDXGIAdapter4),
                      (void **)&data->dxgiAdapter);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("IDXGIFactory6::EnumAdapterByGPUPreference"), result);
        goto done;
    }

    result = D3D12CreateDeviceFunc((IUnknown *)data->dxgiAdapter,
                                   D3D_FEATURE_LEVEL_11_0, // Request minimum feature level 11.0 for maximum compatibility
                                   D3D_GUID(SDL_IID_ID3D12Device1),
                                   (void **)&d3dDevice);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("D3D12CreateDevice"), result);
        goto done;
    }

    // Setup the info queue if in debug mode
    if (createDebug) {
        ID3D12InfoQueue *infoQueue = NULL;
        D3D12_MESSAGE_SEVERITY severities[] = { D3D12_MESSAGE_SEVERITY_INFO };
        D3D12_INFO_QUEUE_FILTER filter;

        result = ID3D12Device1_QueryInterface(d3dDevice, D3D_GUID(SDL_IID_ID3D12InfoQueue), (void **)&infoQueue);
        if (FAILED(result)) {
            WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D12Device to ID3D12InfoQueue"), result);
            goto done;
        }

        SDL_zero(filter);
        filter.DenyList.NumSeverities = 1;
        filter.DenyList.pSeverityList = severities;
        ID3D12InfoQueue_PushStorageFilter(infoQueue, &filter);

        ID3D12InfoQueue_SetBreakOnSeverity(infoQueue, D3D12_MESSAGE_SEVERITY_ERROR, TRUE);
        ID3D12InfoQueue_SetBreakOnSeverity(infoQueue, D3D12_MESSAGE_SEVERITY_CORRUPTION, TRUE);

        D3D_SAFE_RELEASE(infoQueue);
    }
#endif // !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)

    result = ID3D12Device_QueryInterface(d3dDevice, D3D_GUID(SDL_IID_ID3D12Device1), (void **)&data->d3dDevice);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D12Device to ID3D12Device1"), result);
        goto done;
    }

    // Create a command queue
    SDL_zero(queueDesc);
    queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

    result = ID3D12Device1_CreateCommandQueue(data->d3dDevice,
                      &queueDesc,
                      D3D_GUID(SDL_IID_ID3D12CommandQueue),
                      (void **)&data->commandQueue);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D12Device::CreateCommandQueue"), result);
        goto done;
    }

    // Create the descriptor heaps for the render target view, texture SRVs, and samplers
    SDL_zero(descriptorHeapDesc);
    descriptorHeapDesc.NumDescriptors = SDL_D3D12_NUM_BUFFERS;
    descriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    result = ID3D12Device1_CreateDescriptorHeap(data->d3dDevice,
                      &descriptorHeapDesc,
                      D3D_GUID(SDL_IID_ID3D12DescriptorHeap),
                      (void **)&data->rtvDescriptorHeap);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D12Device::CreateDescriptorHeap [rtv]"), result);
        goto done;
    }
    data->rtvDescriptorSize = ID3D12Device1_GetDescriptorHandleIncrementSize(d3dDevice, D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    descriptorHeapDesc.NumDescriptors = SDL_D3D12_MAX_NUM_TEXTURES;
    result = ID3D12Device1_CreateDescriptorHeap(data->d3dDevice,
                      &descriptorHeapDesc,
                      D3D_GUID(SDL_IID_ID3D12DescriptorHeap),
                      (void **)&data->textureRTVDescriptorHeap);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D12Device::CreateDescriptorHeap [texture rtv]"), result);
        goto done;
    }

    SDL_zero(descriptorHeapDesc);
    descriptorHeapDesc.NumDescriptors = SDL_D3D12_MAX_NUM_TEXTURES;
    descriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    descriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    result = ID3D12Device1_CreateDescriptorHeap(data->d3dDevice,
                      &descriptorHeapDesc,
                      D3D_GUID(SDL_IID_ID3D12DescriptorHeap),
                      (void **)&data->srvDescriptorHeap);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D12Device::CreateDescriptorHeap  [srv]"), result);
        goto done;
    }
    rootDescriptorHeaps[0] = data->srvDescriptorHeap;
    data->srvDescriptorSize = ID3D12Device1_GetDescriptorHandleIncrementSize(d3dDevice, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    SDL_zero(descriptorHeapDesc);
    descriptorHeapDesc.NumDescriptors = 4;
    descriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
    descriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    result = ID3D12Device1_CreateDescriptorHeap(data->d3dDevice,
                      &descriptorHeapDesc,
                      D3D_GUID(SDL_IID_ID3D12DescriptorHeap),
                      (void **)&data->samplerDescriptorHeap);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D12Device::CreateDescriptorHeap  [sampler]"), result);
        goto done;
    }
    rootDescriptorHeaps[1] = data->samplerDescriptorHeap;
    data->samplerDescriptorSize = ID3D12Device1_GetDescriptorHandleIncrementSize(d3dDevice, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);

    // Create a command allocator for each back buffer
    for (i = 0; i < SDL_D3D12_NUM_BUFFERS; ++i) {
        result = ID3D12Device1_CreateCommandAllocator(data->d3dDevice,
                          D3D12_COMMAND_LIST_TYPE_DIRECT,
                          D3D_GUID(SDL_IID_ID3D12CommandAllocator),
                          (void **)&data->commandAllocators[i]);
        if (FAILED(result)) {
            WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D12Device::CreateCommandAllocator"), result);
            goto done;
        }
    }

    // Create the command list
    result = ID3D12Device1_CreateCommandList(data->d3dDevice,
                      0,
                      D3D12_COMMAND_LIST_TYPE_DIRECT,
                      data->commandAllocators[0],
                      NULL,
                      D3D_GUID(SDL_IID_ID3D12GraphicsCommandList2),
                      (void **)&data->commandList);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D12Device::CreateCommandList"), result);
        goto done;
    }

    // Set the descriptor heaps to the correct initial value
    ID3D12GraphicsCommandList2_SetDescriptorHeaps(data->commandList, 2, rootDescriptorHeaps);

    // Create the fence and fence event
    result = ID3D12Device_CreateFence(data->d3dDevice,
                      data->fenceValue,
                      D3D12_FENCE_FLAG_NONE,
                      D3D_GUID(SDL_IID_ID3D12Fence),
                      (void **)&data->fence);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D12Device::CreateFence"), result);
        goto done;
    }

    data->fenceValue++;

    data->fenceEvent = CreateEventExFunc(NULL, NULL, 0, EVENT_MODIFY_STATE | SYNCHRONIZE);
    if (!data->fenceEvent) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("CreateEventEx"), result);
        goto done;
    }

    // Create all the root signatures
    for (i = 0; i < NUM_ROOTSIGS; ++i) {
        D3D12_SHADER_BYTECODE rootSigData;
        D3D12_GetRootSignatureData((D3D12_RootSignature)i, &rootSigData);
        result = ID3D12Device1_CreateRootSignature(data->d3dDevice,
                          0,
                          rootSigData.pShaderBytecode,
                          rootSigData.BytecodeLength,
                          D3D_GUID(SDL_IID_ID3D12RootSignature),
                          (void **)&data->rootSignatures[i]);
        if (FAILED(result)) {
            WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D12Device::CreateRootSignature"), result);
            goto done;
        }
    }

    {
        const SDL_BlendMode defaultBlendModes[] = {
            SDL_BLENDMODE_BLEND,
        };
        const DXGI_FORMAT defaultRTVFormats[] = {
            DXGI_FORMAT_B8G8R8A8_UNORM,
        };
        int j, k, l;

        // Create a few default pipeline state objects, to verify that this renderer will work
        for (i = 0; i < NUM_SHADERS; ++i) {
            for (j = 0; j < SDL_arraysize(defaultBlendModes); ++j) {
                for (k = D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT; k < D3D12_PRIMITIVE_TOPOLOGY_TYPE_PATCH; ++k) {
                    for (l = 0; l < SDL_arraysize(defaultRTVFormats); ++l) {
                        if (!D3D12_CreatePipelineState(renderer, (D3D12_Shader)i, defaultBlendModes[j], (D3D12_PRIMITIVE_TOPOLOGY_TYPE)k, defaultRTVFormats[l])) {
                            // D3D12_CreatePipelineState will set the SDL error, if it fails
                            result = E_FAIL;
                            goto done;
                        }
                    }
                }
            }
        }
    }

    // Create default vertex buffers
    for (i = 0; i < SDL_D3D12_NUM_VERTEX_BUFFERS; ++i) {
        D3D12_CreateVertexBuffer(data, i, D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT);
    }

    // Create samplers to use when drawing textures:
    D3D_CALL_RET(data->samplerDescriptorHeap, GetCPUDescriptorHandleForHeapStart, &data->samplers[0]);
    for (i = 0; i < SDL_arraysize(data->samplers); ++i) {
        data->samplers[i].ptr = data->samplers[0].ptr + i * data->samplerDescriptorSize;
    }

    // Initialize the pool allocator for SRVs
    for (i = 0; i < SDL_D3D12_MAX_NUM_TEXTURES; ++i) {
        data->srvPoolNodes[i].index = (SIZE_T)i;
        if (i != SDL_D3D12_MAX_NUM_TEXTURES - 1) {
            data->srvPoolNodes[i].next = &data->srvPoolNodes[i + 1];
        }
    }
    data->srvPoolHead = &data->srvPoolNodes[0];

    SDL_PropertiesID props = SDL_GetRendererProperties(renderer);
    SDL_SetPointerProperty(props, SDL_PROP_RENDERER_D3D12_DEVICE_POINTER, data->d3dDevice);
    SDL_SetPointerProperty(props, SDL_PROP_RENDERER_D3D12_COMMAND_QUEUE_POINTER, data->commandQueue);

done:
    D3D_SAFE_RELEASE(d3dDevice);
    return result;
}

static DXGI_MODE_ROTATION D3D12_GetCurrentRotation(void)
{
    // FIXME
    return DXGI_MODE_ROTATION_IDENTITY;
}

static BOOL D3D12_IsDisplayRotated90Degrees(DXGI_MODE_ROTATION rotation)
{
    switch (rotation) {
    case DXGI_MODE_ROTATION_ROTATE90:
    case DXGI_MODE_ROTATION_ROTATE270:
        return TRUE;
    default:
        return FALSE;
    }
}

static int D3D12_GetRotationForCurrentRenderTarget(SDL_Renderer *renderer)
{
    D3D12_RenderData *data = (D3D12_RenderData *)renderer->internal;
    if (data->textureRenderTarget) {
        return DXGI_MODE_ROTATION_IDENTITY;
    } else {
        return data->rotation;
    }
}

static bool D3D12_GetViewportAlignedD3DRect(SDL_Renderer *renderer, const SDL_Rect *sdlRect, D3D12_RECT *outRect, BOOL includeViewportOffset)
{
    D3D12_RenderData *data = (D3D12_RenderData *)renderer->internal;
    const int rotation = D3D12_GetRotationForCurrentRenderTarget(renderer);
    const SDL_Rect *viewport = &data->currentViewport;

    switch (rotation) {
    case DXGI_MODE_ROTATION_IDENTITY:
        outRect->left = sdlRect->x;
        outRect->right = (LONG)sdlRect->x + sdlRect->w;
        outRect->top = sdlRect->y;
        outRect->bottom = (LONG)sdlRect->y + sdlRect->h;
        if (includeViewportOffset) {
            outRect->left += viewport->x;
            outRect->right += viewport->x;
            outRect->top += viewport->y;
            outRect->bottom += viewport->y;
        }
        break;
    case DXGI_MODE_ROTATION_ROTATE270:
        outRect->left = sdlRect->y;
        outRect->right = (LONG)sdlRect->y + sdlRect->h;
        outRect->top = viewport->w - sdlRect->x - sdlRect->w;
        outRect->bottom = viewport->w - sdlRect->x;
        break;
    case DXGI_MODE_ROTATION_ROTATE180:
        outRect->left = viewport->w - sdlRect->x - sdlRect->w;
        outRect->right = viewport->w - sdlRect->x;
        outRect->top = viewport->h - sdlRect->y - sdlRect->h;
        outRect->bottom = viewport->h - sdlRect->y;
        break;
    case DXGI_MODE_ROTATION_ROTATE90:
        outRect->left = viewport->h - sdlRect->y - sdlRect->h;
        outRect->right = viewport->h - sdlRect->y;
        outRect->top = sdlRect->x;
        outRect->bottom = (LONG)sdlRect->x + sdlRect->h;
        break;
    default:
        return SDL_SetError("The physical display is in an unknown or unsupported rotation");
    }
    return true;
}

#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
static HRESULT D3D12_CreateSwapChain(SDL_Renderer *renderer, int w, int h)
{
    D3D12_RenderData *data = (D3D12_RenderData *)renderer->internal;
    IDXGISwapChain1 *swapChain = NULL;
    HRESULT result = S_OK;

    // Create a swap chain using the same adapter as the existing Direct3D device.
    DXGI_SWAP_CHAIN_DESC1 swapChainDesc;
    SDL_zero(swapChainDesc);
    swapChainDesc.Width = w;
    swapChainDesc.Height = h;
    switch (renderer->output_colorspace) {
    case SDL_COLORSPACE_SRGB_LINEAR:
        swapChainDesc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        data->renderTargetFormat = DXGI_FORMAT_R16G16B16A16_FLOAT;
        break;
    case SDL_COLORSPACE_HDR10:
        swapChainDesc.Format = DXGI_FORMAT_R10G10B10A2_UNORM;
        data->renderTargetFormat = DXGI_FORMAT_R10G10B10A2_UNORM;
        break;
    default:
        swapChainDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM; // This is the most common swap chain format.
        data->renderTargetFormat = DXGI_FORMAT_B8G8R8A8_UNORM;
        break;
    }
    swapChainDesc.Stereo = FALSE;
    swapChainDesc.SampleDesc.Count = 1; // Don't use multi-sampling.
    swapChainDesc.SampleDesc.Quality = 0;
    swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapChainDesc.BufferCount = 2; // Use double-buffering to minimize latency.
    if (WIN_IsWindows8OrGreater()) {
        swapChainDesc.Scaling = DXGI_SCALING_NONE;
    } else {
        swapChainDesc.Scaling = DXGI_SCALING_STRETCH;
    }
    swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;               // All Windows Store apps must use this SwapEffect.
    swapChainDesc.Flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT | // To support SetMaximumFrameLatency
                          DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;                  // To support presenting with allow tearing on

    HWND hwnd = (HWND)SDL_GetPointerProperty(SDL_GetWindowProperties(renderer->window), SDL_PROP_WINDOW_WIN32_HWND_POINTER, NULL);
    if (!hwnd) {
        SDL_SetError("Couldn't get window handle");
        result = E_FAIL;
        goto done;
    }

    result = IDXGIFactory2_CreateSwapChainForHwnd(data->dxgiFactory,
                      (IUnknown *)data->commandQueue,
                      hwnd,
                      &swapChainDesc,
                      NULL,
                      NULL, // Allow on all displays.
                      &swapChain);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("IDXGIFactory2::CreateSwapChainForHwnd"), result);
        goto done;
    }

    IDXGIFactory6_MakeWindowAssociation(data->dxgiFactory, hwnd, DXGI_MWA_NO_WINDOW_CHANGES);

    result = IDXGISwapChain1_QueryInterface(swapChain, D3D_GUID(SDL_IID_IDXGISwapChain4), (void **)&data->swapChain);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("IDXGISwapChain1::QueryInterface"), result);
        goto done;
    }

    /* Ensure that the swapchain does not queue more than one frame at a time. This both reduces latency
     * and ensures that the application will only render after each VSync, minimizing power consumption.
     */
    result = IDXGISwapChain4_SetMaximumFrameLatency(data->swapChain, 1);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("IDXGISwapChain4::SetMaximumFrameLatency"), result);
        goto done;
    }

    data->swapEffect = swapChainDesc.SwapEffect;
    data->swapFlags = swapChainDesc.Flags;

    UINT colorspace_support = 0;
    DXGI_COLOR_SPACE_TYPE colorspace;
    switch (renderer->output_colorspace) {
    case SDL_COLORSPACE_SRGB_LINEAR:
        colorspace = DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709;
        break;
    case SDL_COLORSPACE_HDR10:
        colorspace = DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020;
        break;
    default:
        // sRGB
        colorspace = DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;
        break;
    }
    if (SUCCEEDED(IDXGISwapChain3_CheckColorSpaceSupport(data->swapChain, colorspace, &colorspace_support)) &&
        (colorspace_support & DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT)) {
        result = IDXGISwapChain3_SetColorSpace1(data->swapChain, colorspace);
        if (FAILED(result)) {
            WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("IDXGISwapChain3::SetColorSpace1"), result);
            goto done;
        }
    } else {
        // Not the default, we're not going to be able to present in this colorspace
        SDL_SetError("Unsupported output colorspace");
        result = DXGI_ERROR_UNSUPPORTED;
    }

    SDL_SetPointerProperty(SDL_GetRendererProperties(renderer), SDL_PROP_RENDERER_D3D12_SWAPCHAIN_POINTER, data->swapChain);

done:
    D3D_SAFE_RELEASE(swapChain);
    return result;
}
#endif

// Initialize all resources that change when the window's size changes.
static HRESULT D3D12_CreateWindowSizeDependentResources(SDL_Renderer *renderer)
{
    D3D12_RenderData *data = (D3D12_RenderData *)renderer->internal;
    HRESULT result = S_OK;
    int i, w, h;

    D3D12_RENDER_TARGET_VIEW_DESC rtvDesc;
    D3D12_CPU_DESCRIPTOR_HANDLE rtvDescriptor;

    // Release resources in the current command list
    D3D12_IssueBatch(data);
    ID3D12GraphicsCommandList2_OMSetRenderTargets(data->commandList, 0, NULL, FALSE, NULL);

    // Release render targets
    for (i = 0; i < SDL_D3D12_NUM_BUFFERS; ++i) {
        D3D_SAFE_RELEASE(data->renderTargets[i]);
    }

    /* The width and height of the swap chain must be based on the display's
     * non-rotated size.
     */
    SDL_GetWindowSizeInPixels(renderer->window, &w, &h);
    data->rotation = D3D12_GetCurrentRotation();
    if (D3D12_IsDisplayRotated90Degrees(data->rotation)) {
        int tmp = w;
        w = h;
        h = tmp;
    }

#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
    if (data->swapChain) {
        // If the swap chain already exists, resize it.
        result = IDXGISwapChain_ResizeBuffers(data->swapChain,
                          0,
                          w, h,
                          DXGI_FORMAT_UNKNOWN,
                          data->swapFlags);
        if (FAILED(result)) {
            WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("IDXGISwapChain::ResizeBuffers"), result);
            goto done;
        }
    } else {
        result = D3D12_CreateSwapChain(renderer, w, h);
        if (FAILED(result) || !data->swapChain) {
            goto done;
        }
    }

    // Set the proper rotation for the swap chain.
    if (WIN_IsWindows8OrGreater()) {
        if (data->swapEffect == DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL) {
            result = IDXGISwapChain4_SetRotation(data->swapChain, data->rotation); // NOLINT(clang-analyzer-core.NullDereference)
            if (FAILED(result)) {
                WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("IDXGISwapChain4::SetRotation"), result);
                goto done;
            }
        }
    }
#endif // !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)

    // Get each back buffer render target and create render target views
    for (i = 0; i < SDL_D3D12_NUM_BUFFERS; ++i) {
#if defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES)
        result = D3D12_XBOX_CreateBackBufferTarget(data->d3dDevice, renderer->window->w, renderer->window->h, (void **)&data->renderTargets[i]);
        if (FAILED(result)) {
            WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("D3D12_XBOX_CreateBackBufferTarget"), result);
            goto done;
        }
#else
        result = IDXGISwapChain4_GetBuffer(data->swapChain, // NOLINT(clang-analyzer-core.NullDereference)
                          i,
                          D3D_GUID(SDL_IID_ID3D12Resource),
                          (void **)&data->renderTargets[i]);
        if (FAILED(result)) {
            WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("IDXGISwapChain4::GetBuffer"), result);
            goto done;
        }
#endif

        SDL_zero(rtvDesc);
        rtvDesc.Format = data->renderTargetFormat;
        rtvDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;

        SDL_zero(rtvDescriptor);
        D3D_CALL_RET(data->rtvDescriptorHeap, GetCPUDescriptorHandleForHeapStart, &rtvDescriptor);
        rtvDescriptor.ptr += i * data->rtvDescriptorSize;
        ID3D12Device1_CreateRenderTargetView(data->d3dDevice, data->renderTargets[i], &rtvDesc, rtvDescriptor);
    }

    // Set back buffer index to current buffer
#if defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES)
    data->currentBackBufferIndex = 0;
#else
    data->currentBackBufferIndex = IDXGISwapChain4_GetCurrentBackBufferIndex(data->swapChain);
#endif

    /* Set the swap chain target immediately, so that a target is always set
     * even before we get to SetDrawState. Without this it's possible to hit
     * null references in places like ReadPixels!
     */
    data->currentRenderTargetView = D3D12_GetCurrentRenderTargetView(renderer);
    ID3D12GraphicsCommandList2_OMSetRenderTargets(data->commandList, 1, &data->currentRenderTargetView, FALSE, NULL);
    D3D12_TransitionResource(data,
                             data->renderTargets[data->currentBackBufferIndex],
                             D3D12_RESOURCE_STATE_PRESENT,
                             D3D12_RESOURCE_STATE_RENDER_TARGET);

    data->viewportDirty = true;

#if defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES)
    D3D12_XBOX_StartFrame(data->d3dDevice, &data->frameToken);
#endif

done:
    return result;
}

static bool D3D12_HandleDeviceLost(SDL_Renderer *renderer)
{
    bool recovered = false;

    D3D12_ReleaseAll(renderer);

    if (SUCCEEDED(D3D12_CreateDeviceResources(renderer)) &&
        SUCCEEDED(D3D12_CreateWindowSizeDependentResources(renderer))) {
        recovered = true;
    } else {
        SDL_LogError(SDL_LOG_CATEGORY_RENDER, "Renderer couldn't recover from device lost: %s", SDL_GetError());
        D3D12_ReleaseAll(renderer);
    }

    // Let the application know that the device has been reset or lost
    SDL_Event event;
    SDL_zero(event);
    event.type = recovered ? SDL_EVENT_RENDER_DEVICE_RESET : SDL_EVENT_RENDER_DEVICE_LOST;
    event.render.windowID = SDL_GetWindowID(SDL_GetRenderWindow(renderer));
    SDL_PushEvent(&event);

    return recovered;
}

// This method is called when the window's size changes.
static HRESULT D3D12_UpdateForWindowSizeChange(SDL_Renderer *renderer)
{
    D3D12_RenderData *data = (D3D12_RenderData *)renderer->internal;
    // If the GPU has previous work, wait for it to be done first
    D3D12_WaitForGPU(data);
    return D3D12_CreateWindowSizeDependentResources(renderer);
}

static void D3D12_WindowEvent(SDL_Renderer *renderer, const SDL_WindowEvent *event)
{
    D3D12_RenderData *data = (D3D12_RenderData *)renderer->internal;

    if (event->type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED) {
        data->pixelSizeChanged = true;
    }
}

static bool D3D12_SupportsBlendMode(SDL_Renderer *renderer, SDL_BlendMode blendMode)
{
    SDL_BlendFactor srcColorFactor = SDL_GetBlendModeSrcColorFactor(blendMode);
    SDL_BlendFactor srcAlphaFactor = SDL_GetBlendModeSrcAlphaFactor(blendMode);
    SDL_BlendOperation colorOperation = SDL_GetBlendModeColorOperation(blendMode);
    SDL_BlendFactor dstColorFactor = SDL_GetBlendModeDstColorFactor(blendMode);
    SDL_BlendFactor dstAlphaFactor = SDL_GetBlendModeDstAlphaFactor(blendMode);
    SDL_BlendOperation alphaOperation = SDL_GetBlendModeAlphaOperation(blendMode);

    if (!GetBlendFunc(srcColorFactor) || !GetBlendFunc(srcAlphaFactor) ||
        !GetBlendEquation(colorOperation) ||
        !GetBlendFunc(dstColorFactor) || !GetBlendFunc(dstAlphaFactor) ||
        !GetBlendEquation(alphaOperation)) {
        return false;
    }
    return true;
}

static SIZE_T D3D12_GetAvailableSRVIndex(SDL_Renderer *renderer)
{
    D3D12_RenderData *rendererData = (D3D12_RenderData *)renderer->internal;
    if (rendererData->srvPoolHead) {
        SIZE_T index = rendererData->srvPoolHead->index;
        rendererData->srvPoolHead = (D3D12_SRVPoolNode *)(rendererData->srvPoolHead->next);
        return index;
    } else {
        SDL_SetError("[d3d12] Cannot allocate more than %d textures!", SDL_D3D12_MAX_NUM_TEXTURES);
        return SDL_D3D12_MAX_NUM_TEXTURES + 1;
    }
}

static void D3D12_FreeSRVIndex(SDL_Renderer *renderer, SIZE_T index)
{
    D3D12_RenderData *rendererData = (D3D12_RenderData *)renderer->internal;
    rendererData->srvPoolNodes[index].next = rendererData->srvPoolHead;
    rendererData->srvPoolHead = &rendererData->srvPoolNodes[index];
}

static bool GetTextureProperty(SDL_PropertiesID props, const char *name, ID3D12Resource **texture)
{
    IUnknown *unknown = (IUnknown*)SDL_GetPointerProperty(props, name, NULL);
    if (unknown) {
#if defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES)
        HRESULT result = unknown->QueryInterface(D3D_GUID(SDL_IID_ID3D12Resource), (void **)texture);
#else
        HRESULT result = IUnknown_QueryInterface(unknown, D3D_GUID(SDL_IID_ID3D12Resource), (void **)texture);
#endif
        if (FAILED(result)) {
            return WIN_SetErrorFromHRESULT(name, result);
        }
    }
    return true;
}

static bool D3D12_CreateTexture(SDL_Renderer *renderer, SDL_Texture *texture, SDL_PropertiesID create_props)
{
    D3D12_RenderData *rendererData = (D3D12_RenderData *)renderer->internal;
    D3D12_TextureData *textureData;
    HRESULT result;
    DXGI_FORMAT textureFormat = SDLPixelFormatToDXGITextureFormat(texture->format, renderer->output_colorspace);
    D3D12_RESOURCE_DESC textureDesc;
    D3D12_HEAP_PROPERTIES heapProps;
    D3D12_SHADER_RESOURCE_VIEW_DESC resourceViewDesc;

    if (!rendererData->d3dDevice) {
        return SDL_SetError("Device lost and couldn't be recovered");
    }

    if (textureFormat == DXGI_FORMAT_UNKNOWN) {
        return SDL_SetError("%s, An unsupported SDL pixel format (0x%x) was specified", __FUNCTION__, texture->format);
    }

    textureData = (D3D12_TextureData *)SDL_calloc(1, sizeof(*textureData));
    if (!textureData) {
        return false;
    }

    texture->internal = textureData;
    textureData->mainTextureFormat = textureFormat;

    SDL_zero(textureDesc);
    textureDesc.Width = texture->w;
    textureDesc.Height = texture->h;
    textureDesc.MipLevels = 1;
    textureDesc.DepthOrArraySize = 1;
    textureDesc.Format = textureFormat;
    textureDesc.SampleDesc.Count = 1;
    textureDesc.SampleDesc.Quality = 0;
    textureDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    textureDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    // NV12 textures must have even width and height
    if (texture->format == SDL_PIXELFORMAT_NV12 ||
        texture->format == SDL_PIXELFORMAT_NV21 ||
        texture->format == SDL_PIXELFORMAT_P010) {
        textureDesc.Width = (textureDesc.Width + 1) & ~1;
        textureDesc.Height = (textureDesc.Height + 1) & ~1;
    }
    textureData->w = (int)textureDesc.Width;
    textureData->h = (int)textureDesc.Height;
    if (SDL_COLORSPACETRANSFER(texture->colorspace) == SDL_TRANSFER_CHARACTERISTICS_SRGB) {
        textureData->shader = SHADER_RGB;
    } else {
        textureData->shader = SHADER_ADVANCED;
    }

    if (texture->access == SDL_TEXTUREACCESS_TARGET) {
        textureDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    }

    SDL_zero(heapProps);
    heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;
    heapProps.CreationNodeMask = 1;
    heapProps.VisibleNodeMask = 1;

    if (!GetTextureProperty(create_props, SDL_PROP_TEXTURE_CREATE_D3D12_TEXTURE_POINTER, &textureData->mainTexture)) {
        return false;
    }
    if (!textureData->mainTexture) {
        result = ID3D12Device1_CreateCommittedResource(rendererData->d3dDevice,
                          &heapProps,
                          D3D12_HEAP_FLAG_NONE,
                          &textureDesc,
                          D3D12_RESOURCE_STATE_COPY_DEST,
                          NULL,
                          D3D_GUID(SDL_IID_ID3D12Resource),
                          (void **)&textureData->mainTexture);
        if (FAILED(result)) {
            return WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D12Device::CreateCommittedResource [texture]"), result);
        }
    }
    textureData->mainResourceState = D3D12_RESOURCE_STATE_COPY_DEST;
    SDL_SetPointerProperty(SDL_GetTextureProperties(texture), SDL_PROP_TEXTURE_D3D12_TEXTURE_POINTER, textureData->mainTexture);
#ifdef SDL_HAVE_YUV
    if (texture->format == SDL_PIXELFORMAT_YV12 ||
        texture->format == SDL_PIXELFORMAT_IYUV) {
        textureData->yuv = true;

        textureDesc.Width = (textureDesc.Width + 1) / 2;
        textureDesc.Height = (textureDesc.Height + 1) / 2;

        if (!GetTextureProperty(create_props, SDL_PROP_TEXTURE_CREATE_D3D12_TEXTURE_U_POINTER, &textureData->mainTextureU)) {
            return false;
        }
        if (!textureData->mainTextureU) {
            result = ID3D12Device1_CreateCommittedResource(rendererData->d3dDevice,
                              &heapProps,
                              D3D12_HEAP_FLAG_NONE,
                              &textureDesc,
                              D3D12_RESOURCE_STATE_COPY_DEST,
                              NULL,
                              D3D_GUID(SDL_IID_ID3D12Resource),
                              (void **)&textureData->mainTextureU);
            if (FAILED(result)) {
                return WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D12Device::CreateCommittedResource [texture]"), result);
            }
        }
        textureData->mainResourceStateU = D3D12_RESOURCE_STATE_COPY_DEST;
        SDL_SetPointerProperty(SDL_GetTextureProperties(texture), SDL_PROP_TEXTURE_D3D12_TEXTURE_U_POINTER, textureData->mainTextureU);

        if (!GetTextureProperty(create_props, SDL_PROP_TEXTURE_CREATE_D3D12_TEXTURE_V_POINTER, &textureData->mainTextureV)) {
            return false;
        }
        if (!textureData->mainTextureV) {
            result = ID3D12Device1_CreateCommittedResource(rendererData->d3dDevice,
                              &heapProps,
                              D3D12_HEAP_FLAG_NONE,
                              &textureDesc,
                              D3D12_RESOURCE_STATE_COPY_DEST,
                              NULL,
                              D3D_GUID(SDL_IID_ID3D12Resource),
                              (void **)&textureData->mainTextureV);
            if (FAILED(result)) {
                return WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D12Device::CreateCommittedResource [texture]"), result);
            }
        }
        textureData->mainResourceStateV = D3D12_RESOURCE_STATE_COPY_DEST;
        SDL_SetPointerProperty(SDL_GetTextureProperties(texture), SDL_PROP_TEXTURE_D3D12_TEXTURE_V_POINTER, textureData->mainTextureV);

        textureData->YCbCr_matrix = SDL_GetYCbCRtoRGBConversionMatrix(texture->colorspace, texture->w, texture->h, 8);
        if (!textureData->YCbCr_matrix) {
            return SDL_SetError("Unsupported YUV colorspace");
        }
    }

    if (texture->format == SDL_PIXELFORMAT_NV12 ||
        texture->format == SDL_PIXELFORMAT_NV21 ||
        texture->format == SDL_PIXELFORMAT_P010) {
        int bits_per_pixel;

        textureData->nv12 = true;

        switch (texture->format) {
        case SDL_PIXELFORMAT_P010:
            bits_per_pixel = 10;
            break;
        default:
            bits_per_pixel = 8;
            break;
        }
        textureData->YCbCr_matrix = SDL_GetYCbCRtoRGBConversionMatrix(texture->colorspace, texture->w, texture->h, bits_per_pixel);
        if (!textureData->YCbCr_matrix) {
            return SDL_SetError("Unsupported YUV colorspace");
        }
    }
#endif // SDL_HAVE_YUV
    SDL_zero(resourceViewDesc);
    resourceViewDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    resourceViewDesc.Format = SDLPixelFormatToDXGIMainResourceViewFormat(texture->format, renderer->output_colorspace);
    resourceViewDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    resourceViewDesc.Texture2D.MipLevels = textureDesc.MipLevels;

    textureData->mainSRVIndex = D3D12_GetAvailableSRVIndex(renderer);
    D3D_CALL_RET(rendererData->srvDescriptorHeap, GetCPUDescriptorHandleForHeapStart, &textureData->mainTextureResourceView);
    textureData->mainTextureResourceView.ptr += textureData->mainSRVIndex * rendererData->srvDescriptorSize;

    ID3D12Device1_CreateShaderResourceView(rendererData->d3dDevice,
             textureData->mainTexture,
             &resourceViewDesc,
             textureData->mainTextureResourceView);
#ifdef SDL_HAVE_YUV
    if (textureData->yuv) {
        D3D_CALL_RET(rendererData->srvDescriptorHeap, GetCPUDescriptorHandleForHeapStart, &textureData->mainTextureResourceViewU);
        textureData->mainSRVIndexU = D3D12_GetAvailableSRVIndex(renderer);
        textureData->mainTextureResourceViewU.ptr += textureData->mainSRVIndexU * rendererData->srvDescriptorSize;
        ID3D12Device1_CreateShaderResourceView(rendererData->d3dDevice,
                 textureData->mainTextureU,
                 &resourceViewDesc,
                 textureData->mainTextureResourceViewU);

        D3D_CALL_RET(rendererData->srvDescriptorHeap, GetCPUDescriptorHandleForHeapStart, &textureData->mainTextureResourceViewV);
        textureData->mainSRVIndexV = D3D12_GetAvailableSRVIndex(renderer);
        textureData->mainTextureResourceViewV.ptr += textureData->mainSRVIndexV * rendererData->srvDescriptorSize;
        ID3D12Device1_CreateShaderResourceView(rendererData->d3dDevice,
                 textureData->mainTextureV,
                 &resourceViewDesc,
                 textureData->mainTextureResourceViewV);
    }

    if (textureData->nv12) {
        D3D12_SHADER_RESOURCE_VIEW_DESC nvResourceViewDesc = resourceViewDesc;

        if (texture->format == SDL_PIXELFORMAT_NV12 || texture->format == SDL_PIXELFORMAT_NV21) {
            nvResourceViewDesc.Format = DXGI_FORMAT_R8G8_UNORM;
        } else if (texture->format == SDL_PIXELFORMAT_P010) {
            nvResourceViewDesc.Format = DXGI_FORMAT_R16G16_UNORM;
        }
        nvResourceViewDesc.Texture2D.PlaneSlice = 1;

        D3D_CALL_RET(rendererData->srvDescriptorHeap, GetCPUDescriptorHandleForHeapStart, &textureData->mainTextureResourceViewNV);
        textureData->mainSRVIndexNV = D3D12_GetAvailableSRVIndex(renderer);
        textureData->mainTextureResourceViewNV.ptr += textureData->mainSRVIndexNV * rendererData->srvDescriptorSize;
        ID3D12Device1_CreateShaderResourceView(rendererData->d3dDevice,
                 textureData->mainTexture,
                 &nvResourceViewDesc,
                 textureData->mainTextureResourceViewNV);
    }
#endif // SDL_HAVE_YUV

    if (texture->access & SDL_TEXTUREACCESS_TARGET) {
        D3D12_RENDER_TARGET_VIEW_DESC renderTargetViewDesc;
        SDL_zero(renderTargetViewDesc);
        renderTargetViewDesc.Format = textureDesc.Format;
        renderTargetViewDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
        renderTargetViewDesc.Texture2D.MipSlice = 0;

        D3D_CALL_RET(rendererData->textureRTVDescriptorHeap, GetCPUDescriptorHandleForHeapStart, &textureData->mainTextureRenderTargetView);
        textureData->mainTextureRenderTargetView.ptr += textureData->mainSRVIndex * rendererData->rtvDescriptorSize;

        ID3D12Device1_CreateRenderTargetView(rendererData->d3dDevice,
                 (ID3D12Resource *)textureData->mainTexture,
                 &renderTargetViewDesc,
                 textureData->mainTextureRenderTargetView);
    }

    return true;
}

static void D3D12_DestroyTexture(SDL_Renderer *renderer,
                                 SDL_Texture *texture)
{
    D3D12_RenderData *rendererData = (D3D12_RenderData *)renderer->internal;
    D3D12_TextureData *textureData = (D3D12_TextureData *)texture->internal;

    if (!textureData) {
        return;
    }

    /* Because SDL_DestroyTexture might be called while the data is in-flight, we need to issue the batch first
       Unfortunately, this means that deleting a lot of textures mid-frame will have poor performance. */
    D3D12_IssueBatch(rendererData);

    D3D_SAFE_RELEASE(textureData->mainTexture);
    D3D_SAFE_RELEASE(textureData->stagingBuffer);
    D3D12_FreeSRVIndex(renderer, textureData->mainSRVIndex);
#ifdef SDL_HAVE_YUV
    D3D_SAFE_RELEASE(textureData->mainTextureU);
    D3D_SAFE_RELEASE(textureData->mainTextureV);
    if (textureData->yuv) {
        D3D12_FreeSRVIndex(renderer, textureData->mainSRVIndexU);
        D3D12_FreeSRVIndex(renderer, textureData->mainSRVIndexV);
    }
    if (textureData->nv12) {
        D3D12_FreeSRVIndex(renderer, textureData->mainSRVIndexNV);
    }
    SDL_free(textureData->pixels);
#endif
    SDL_free(textureData);
    texture->internal = NULL;
}

static bool D3D12_UpdateTextureInternal(D3D12_RenderData *rendererData, ID3D12Resource *texture, int plane, int x, int y, int w, int h, const void *pixels, int pitch, D3D12_RESOURCE_STATES *resourceState)
{
    const Uint8 *src;
    Uint8 *dst;
    UINT length;
    HRESULT result;
    D3D12_RESOURCE_DESC textureDesc;
    D3D12_RESOURCE_DESC uploadDesc;
    D3D12_HEAP_PROPERTIES heapProps;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT placedTextureDesc;
    D3D12_TEXTURE_COPY_LOCATION srcLocation;
    D3D12_TEXTURE_COPY_LOCATION dstLocation;
    BYTE *textureMemory;
    ID3D12Resource *uploadBuffer;
    UINT row, NumRows, RowPitch;
    UINT64 RowLength;

    // Create an upload buffer, which will be used to write to the main texture.
    SDL_zero(textureDesc);
    D3D_CALL_RET(texture, GetDesc, &textureDesc);
    textureDesc.Width = w;
    textureDesc.Height = h;
    if (textureDesc.Format == DXGI_FORMAT_NV12 ||
        textureDesc.Format == DXGI_FORMAT_P010) {
        textureDesc.Width = (textureDesc.Width + 1) & ~1;
        textureDesc.Height = (textureDesc.Height + 1) & ~1;
    }

    SDL_zero(uploadDesc);
    uploadDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    uploadDesc.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
    uploadDesc.Height = 1;
    uploadDesc.DepthOrArraySize = 1;
    uploadDesc.MipLevels = 1;
    uploadDesc.Format = DXGI_FORMAT_UNKNOWN;
    uploadDesc.SampleDesc.Count = 1;
    uploadDesc.SampleDesc.Quality = 0;
    uploadDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    uploadDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    // Figure out how much we need to allocate for the upload buffer
    ID3D12Device1_GetCopyableFootprints(rendererData->d3dDevice,
             &textureDesc,
             plane,
             1,
             0,
             &placedTextureDesc,
             &NumRows,
             &RowLength,
             &uploadDesc.Width);
    RowPitch = placedTextureDesc.Footprint.RowPitch;

    SDL_zero(heapProps);
    heapProps.Type = D3D12_HEAP_TYPE_UPLOAD;
    heapProps.CreationNodeMask = 1;
    heapProps.VisibleNodeMask = 1;

    // Create the upload buffer
    result = ID3D12Device1_CreateCommittedResource(rendererData->d3dDevice,
                      &heapProps,
                      D3D12_HEAP_FLAG_NONE,
                      &uploadDesc,
                      D3D12_RESOURCE_STATE_GENERIC_READ,
                      NULL,
                      D3D_GUID(SDL_IID_ID3D12Resource),
                      (void **)&rendererData->uploadBuffers[rendererData->currentUploadBuffer]);
    if (FAILED(result)) {
        return WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D12Device::CreateCommittedResource [create upload buffer]"), result);
    }

    // Get a write-only pointer to data in the upload buffer:
    uploadBuffer = rendererData->uploadBuffers[rendererData->currentUploadBuffer];
    result = ID3D12Resource_Map(uploadBuffer,
                      0,
                      NULL,
                      (void **)&textureMemory);
    if (FAILED(result)) {
        D3D_SAFE_RELEASE(rendererData->uploadBuffers[rendererData->currentUploadBuffer]);
        return WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D12Resource::Map [map staging texture]"), result);
    }

    src = (const Uint8 *)pixels;
    dst = textureMemory;
    length = (UINT)RowLength;
    if (length == (UINT)pitch && length == RowPitch) {
        SDL_memcpy(dst, src, (size_t)length * NumRows);
    } else {
        if (length > (UINT)pitch) {
            length = pitch;
        }
        if (length > RowPitch) {
            length = RowPitch;
        }
        for (row = NumRows; row--; ) {
            SDL_memcpy(dst, src, length);
            src += pitch;
            dst += RowPitch;
        }
    }

    // Commit the changes back to the upload buffer:
    ID3D12Resource_Unmap(uploadBuffer, 0, NULL);

    // Make sure the destination is in the correct resource state
    D3D12_TransitionResource(rendererData, texture, *resourceState, D3D12_RESOURCE_STATE_COPY_DEST);
    *resourceState = D3D12_RESOURCE_STATE_COPY_DEST;

    SDL_zero(dstLocation);
    dstLocation.pResource = texture;
    dstLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dstLocation.SubresourceIndex = plane;

    SDL_zero(srcLocation);
    srcLocation.pResource = rendererData->uploadBuffers[rendererData->currentUploadBuffer];
    srcLocation.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    srcLocation.PlacedFootprint = placedTextureDesc;

    ID3D12GraphicsCommandList2_CopyTextureRegion(rendererData->commandList,
             &dstLocation,
             x,
             y,
             0,
             &srcLocation,
             NULL);

    // Transition the texture to be shader accessible
    D3D12_TransitionResource(rendererData, texture, *resourceState, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    *resourceState = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;

    rendererData->currentUploadBuffer++;
    // If we've used up all the upload buffers, we need to issue the batch
    if (rendererData->currentUploadBuffer == SDL_D3D12_NUM_UPLOAD_BUFFERS) {
        D3D12_IssueBatch(rendererData);
    }

    return true;
}

static bool D3D12_UpdateTexture(SDL_Renderer *renderer, SDL_Texture *texture,
                               const SDL_Rect *rect, const void *srcPixels,
                               int srcPitch)
{
    D3D12_RenderData *rendererData = (D3D12_RenderData *)renderer->internal;
    D3D12_TextureData *textureData = (D3D12_TextureData *)texture->internal;

    if (!textureData) {
        return SDL_SetError("Texture is not currently available");
    }

    if (!D3D12_UpdateTextureInternal(rendererData, textureData->mainTexture, 0, rect->x, rect->y, rect->w, rect->h, srcPixels, srcPitch, &textureData->mainResourceState)) {
        return false;
    }
#ifdef SDL_HAVE_YUV
    if (textureData->yuv) {
        // Skip to the correct offset into the next texture
        srcPixels = (const void *)((const Uint8 *)srcPixels + rect->h * srcPitch);

        if (!D3D12_UpdateTextureInternal(rendererData, texture->format == SDL_PIXELFORMAT_YV12 ? textureData->mainTextureV : textureData->mainTextureU, 0, rect->x / 2, rect->y / 2, (rect->w + 1) / 2, (rect->h + 1) / 2, srcPixels, (srcPitch + 1) / 2, texture->format == SDL_PIXELFORMAT_YV12 ? &textureData->mainResourceStateV : &textureData->mainResourceStateU)) {
            return false;
        }

        // Skip to the correct offset into the next texture
        srcPixels = (const void *)((const Uint8 *)srcPixels + ((rect->h + 1) / 2) * ((srcPitch + 1) / 2));
        if (!D3D12_UpdateTextureInternal(rendererData, texture->format == SDL_PIXELFORMAT_YV12 ? textureData->mainTextureU : textureData->mainTextureV, 0, rect->x / 2, rect->y / 2, (rect->w + 1) / 2, (rect->h + 1) / 2, srcPixels, (srcPitch + 1) / 2, texture->format == SDL_PIXELFORMAT_YV12 ? &textureData->mainResourceStateU : &textureData->mainResourceStateV)) {
            return false;
        }
    }

    if (textureData->nv12) {
        // Skip to the correct offset into the next texture
        srcPixels = (const void *)((const Uint8 *)srcPixels + rect->h * srcPitch);

        if (texture->format == SDL_PIXELFORMAT_P010) {
            srcPitch = (srcPitch + 3) & ~3;
        } else {
            srcPitch = (srcPitch + 1) & ~1;
        }
        if (!D3D12_UpdateTextureInternal(rendererData, textureData->mainTexture, 1, rect->x, rect->y, (rect->w + 1) & ~1, (rect->h + 1) & ~1, srcPixels, srcPitch, &textureData->mainResourceState)) {
            return false;
        }
    }
#endif // SDL_HAVE_YUV
    if (textureData->mainTextureResourceView.ptr == rendererData->currentShaderResource.ptr) {
        // We'll need to rebind this resource after updating it
        rendererData->currentShaderResource.ptr = 0;
    }
    return true;
}

#ifdef SDL_HAVE_YUV
static bool D3D12_UpdateTextureYUV(SDL_Renderer *renderer, SDL_Texture *texture,
                                  const SDL_Rect *rect,
                                  const Uint8 *Yplane, int Ypitch,
                                  const Uint8 *Uplane, int Upitch,
                                  const Uint8 *Vplane, int Vpitch)
{
    D3D12_RenderData *rendererData = (D3D12_RenderData *)renderer->internal;
    D3D12_TextureData *textureData = (D3D12_TextureData *)texture->internal;

    if (!textureData) {
        return SDL_SetError("Texture is not currently available");
    }

    if (!D3D12_UpdateTextureInternal(rendererData, textureData->mainTexture, 0, rect->x, rect->y, rect->w, rect->h, Yplane, Ypitch, &textureData->mainResourceState)) {
        return false;
    }
    if (!D3D12_UpdateTextureInternal(rendererData, textureData->mainTextureU, 0, rect->x / 2, rect->y / 2, rect->w / 2, rect->h / 2, Uplane, Upitch, &textureData->mainResourceStateU)) {
        return false;
    }
    if (!D3D12_UpdateTextureInternal(rendererData, textureData->mainTextureV, 0, rect->x / 2, rect->y / 2, rect->w / 2, rect->h / 2, Vplane, Vpitch, &textureData->mainResourceStateV)) {
        return false;
    }
    if (textureData->mainTextureResourceView.ptr == rendererData->currentShaderResource.ptr) {
        // We'll need to rebind this resource after updating it
        rendererData->currentShaderResource.ptr = 0;
    }
    return true;
}

static bool D3D12_UpdateTextureNV(SDL_Renderer *renderer, SDL_Texture *texture,
                                 const SDL_Rect *rect,
                                 const Uint8 *Yplane, int Ypitch,
                                 const Uint8 *UVplane, int UVpitch)
{
    D3D12_RenderData *rendererData = (D3D12_RenderData *)renderer->internal;
    D3D12_TextureData *textureData = (D3D12_TextureData *)texture->internal;

    if (!textureData) {
        return SDL_SetError("Texture is not currently available");
    }

    if (!D3D12_UpdateTextureInternal(rendererData, textureData->mainTexture, 0, rect->x, rect->y, rect->w, rect->h, Yplane, Ypitch, &textureData->mainResourceState)) {
        return false;
    }
    if (!D3D12_UpdateTextureInternal(rendererData, textureData->mainTexture, 1, rect->x, rect->y, rect->w, rect->h, UVplane, UVpitch, &textureData->mainResourceState)) {
        return false;
    }
    if (textureData->mainTextureResourceView.ptr == rendererData->currentShaderResource.ptr) {
        // We'll need to rebind this resource after updating it
        rendererData->currentShaderResource.ptr = 0;
    }
    return true;
}
#endif

static bool D3D12_LockTexture(SDL_Renderer *renderer, SDL_Texture *texture,
                             const SDL_Rect *rect, void **pixels, int *pitch)
{
    D3D12_RenderData *rendererData = (D3D12_RenderData *)renderer->internal;
    D3D12_TextureData *textureData = (D3D12_TextureData *)texture->internal;
    HRESULT result = S_OK;

    D3D12_RESOURCE_DESC textureDesc;
    D3D12_RESOURCE_DESC uploadDesc;
    D3D12_HEAP_PROPERTIES heapProps;
    D3D12_SUBRESOURCE_FOOTPRINT pitchedDesc;
    BYTE *textureMemory;
    int bpp;

    if (!textureData) {
        return SDL_SetError("Texture is not currently available");
    }
#ifdef SDL_HAVE_YUV
    if (textureData->yuv || textureData->nv12) {
        // It's more efficient to upload directly...
        if (!textureData->pixels) {
            textureData->pitch = texture->w;
            textureData->pixels = (Uint8 *)SDL_malloc((texture->h * textureData->pitch * 3) / 2);
            if (!textureData->pixels) {
                return false;
            }
        }
        textureData->lockedRect = *rect;
        *pixels =
            (void *)(textureData->pixels + rect->y * textureData->pitch +
                     rect->x * SDL_BYTESPERPIXEL(texture->format));
        *pitch = textureData->pitch;
        return true;
    }
#endif
    if (textureData->stagingBuffer) {
        return SDL_SetError("texture is already locked");
    }

    // Create an upload buffer, which will be used to write to the main texture.
    SDL_zero(textureDesc);
    D3D_CALL_RET(textureData->mainTexture, GetDesc, &textureDesc);
    textureDesc.Width = rect->w;
    textureDesc.Height = rect->h;

    SDL_zero(uploadDesc);
    uploadDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    uploadDesc.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
    uploadDesc.Height = 1;
    uploadDesc.DepthOrArraySize = 1;
    uploadDesc.MipLevels = 1;
    uploadDesc.Format = DXGI_FORMAT_UNKNOWN;
    uploadDesc.SampleDesc.Count = 1;
    uploadDesc.SampleDesc.Quality = 0;
    uploadDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    uploadDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    // Figure out how much we need to allocate for the upload buffer
    ID3D12Device1_GetCopyableFootprints(rendererData->d3dDevice,
             &textureDesc,
             0,
             1,
             0,
             NULL,
             NULL,
             NULL,
             &uploadDesc.Width);

    SDL_zero(heapProps);
    heapProps.Type = D3D12_HEAP_TYPE_UPLOAD;
    heapProps.CreationNodeMask = 1;
    heapProps.VisibleNodeMask = 1;

    // Create the upload buffer
    result = ID3D12Device1_CreateCommittedResource(rendererData->d3dDevice,
                      &heapProps,
                      D3D12_HEAP_FLAG_NONE,
                      &uploadDesc,
                      D3D12_RESOURCE_STATE_GENERIC_READ,
                      NULL,
                      D3D_GUID(SDL_IID_ID3D12Resource),
                      (void **)&textureData->stagingBuffer);
    if (FAILED(result)) {
        return WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D12Device::CreateCommittedResource [create upload buffer]"), result);
    }

    // Get a write-only pointer to data in the upload buffer:
    result = ID3D12Resource_Map(textureData->stagingBuffer,
                      0,
                      NULL,
                      (void **)&textureMemory);
    if (FAILED(result)) {
        D3D_SAFE_RELEASE(rendererData->uploadBuffers[rendererData->currentUploadBuffer]);
        return WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D12Resource::Map [map staging texture]"), result);
    }

    SDL_zero(pitchedDesc);
    pitchedDesc.Format = textureDesc.Format;
    pitchedDesc.Width = rect->w;
    pitchedDesc.Height = rect->h;
    pitchedDesc.Depth = 1;
    if (pitchedDesc.Format == DXGI_FORMAT_R8_UNORM) {
        bpp = 1;
    } else {
        bpp = 4;
    }
    pitchedDesc.RowPitch = D3D12_Align(rect->w * bpp, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);

    /* Make note of where the staging texture will be written to
     * (on a call to SDL_UnlockTexture):
     */
    textureData->lockedRect = *rect;

    /* Make sure the caller has information on the texture's pixel buffer,
     * then return:
     */
    *pixels = textureMemory;
    *pitch = pitchedDesc.RowPitch;
    return true;
}

static void D3D12_UnlockTexture(SDL_Renderer *renderer, SDL_Texture *texture)
{
    D3D12_RenderData *rendererData = (D3D12_RenderData *)renderer->internal;
    D3D12_TextureData *textureData = (D3D12_TextureData *)texture->internal;

    D3D12_RESOURCE_DESC textureDesc;
    D3D12_SUBRESOURCE_FOOTPRINT pitchedDesc;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT placedTextureDesc;
    D3D12_TEXTURE_COPY_LOCATION srcLocation;
    D3D12_TEXTURE_COPY_LOCATION dstLocation;
    int bpp;

    if (!textureData) {
        return;
    }
#ifdef SDL_HAVE_YUV
    if (textureData->yuv || textureData->nv12) {
        const SDL_Rect *rect = &textureData->lockedRect;
        void *pixels =
            (void *)(textureData->pixels + rect->y * textureData->pitch +
                     rect->x * SDL_BYTESPERPIXEL(texture->format));
        D3D12_UpdateTexture(renderer, texture, rect, pixels, textureData->pitch);
        return;
    }
#endif
    // Commit the pixel buffer's changes back to the staging texture:
    ID3D12Resource_Unmap(textureData->stagingBuffer, 0, NULL);

    SDL_zero(textureDesc);
    D3D_CALL_RET(textureData->mainTexture, GetDesc, &textureDesc);
    textureDesc.Width = textureData->lockedRect.w;
    textureDesc.Height = textureData->lockedRect.h;

    SDL_zero(pitchedDesc);
    pitchedDesc.Format = textureDesc.Format;
    pitchedDesc.Width = (UINT)textureDesc.Width;
    pitchedDesc.Height = textureDesc.Height;
    pitchedDesc.Depth = 1;
    if (pitchedDesc.Format == DXGI_FORMAT_R8_UNORM) {
        bpp = 1;
    } else {
        bpp = 4;
    }
    pitchedDesc.RowPitch = D3D12_Align(textureData->lockedRect.w * bpp, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);

    SDL_zero(placedTextureDesc);
    placedTextureDesc.Offset = 0;
    placedTextureDesc.Footprint = pitchedDesc;

    D3D12_TransitionResource(rendererData, textureData->mainTexture, textureData->mainResourceState, D3D12_RESOURCE_STATE_COPY_DEST);
    textureData->mainResourceState = D3D12_RESOURCE_STATE_COPY_DEST;

    SDL_zero(dstLocation);
    dstLocation.pResource = textureData->mainTexture;
    dstLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dstLocation.SubresourceIndex = 0;

    SDL_zero(srcLocation);
    srcLocation.pResource = textureData->stagingBuffer;
    srcLocation.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    srcLocation.PlacedFootprint = placedTextureDesc;

    ID3D12GraphicsCommandList2_CopyTextureRegion(rendererData->commandList,
             &dstLocation,
             textureData->lockedRect.x,
             textureData->lockedRect.y,
             0,
             &srcLocation,
             NULL);

    // Transition the texture to be shader accessible
    D3D12_TransitionResource(rendererData, textureData->mainTexture, textureData->mainResourceState, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    textureData->mainResourceState = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;

    // Execute the command list before releasing the staging buffer
    D3D12_IssueBatch(rendererData);
    D3D_SAFE_RELEASE(textureData->stagingBuffer);
}

static bool D3D12_SetRenderTarget(SDL_Renderer *renderer, SDL_Texture *texture)
{
    D3D12_RenderData *rendererData = (D3D12_RenderData *)renderer->internal;
    D3D12_TextureData *textureData = NULL;

    if (!texture) {
        if (rendererData->textureRenderTarget) {
            D3D12_TransitionResource(rendererData,
                                     rendererData->textureRenderTarget->mainTexture,
                                     rendererData->textureRenderTarget->mainResourceState,
                                     D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            rendererData->textureRenderTarget->mainResourceState = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        }
        rendererData->textureRenderTarget = NULL;
        return true;
    }

    textureData = (D3D12_TextureData *)texture->internal;

    if (!textureData->mainTextureRenderTargetView.ptr) {
        return SDL_SetError("specified texture is not a render target");
    }

    rendererData->textureRenderTarget = textureData;
    D3D12_TransitionResource(rendererData,
                             rendererData->textureRenderTarget->mainTexture,
                             rendererData->textureRenderTarget->mainResourceState,
                             D3D12_RESOURCE_STATE_RENDER_TARGET);
    rendererData->textureRenderTarget->mainResourceState = D3D12_RESOURCE_STATE_RENDER_TARGET;

    return true;
}

static bool D3D12_QueueNoOp(SDL_Renderer *renderer, SDL_RenderCommand *cmd)
{
    return true; // nothing to do in this backend.
}

static bool D3D12_QueueDrawPoints(SDL_Renderer *renderer, SDL_RenderCommand *cmd, const SDL_FPoint *points, int count)
{
    D3D12_VertexPositionColor *verts = (D3D12_VertexPositionColor *)SDL_AllocateRenderVertices(renderer, count * sizeof(D3D12_VertexPositionColor), 0, &cmd->data.draw.first);
    int i;
    SDL_FColor color = cmd->data.draw.color;
    bool convert_color = SDL_RenderingLinearSpace(renderer);

    if (!verts) {
        return false;
    }

    cmd->data.draw.count = count;

    if (convert_color) {
        SDL_ConvertToLinear(&color);
    }

    for (i = 0; i < count; i++) {
        verts->pos.x = points[i].x + 0.5f;
        verts->pos.y = points[i].y + 0.5f;
        verts->tex.x = 0.0f;
        verts->tex.y = 0.0f;
        verts->color = color;
        verts++;
    }

    return true;
}

static bool D3D12_QueueGeometry(SDL_Renderer *renderer, SDL_RenderCommand *cmd, SDL_Texture *texture,
                               const float *xy, int xy_stride, const SDL_FColor *color, int color_stride, const float *uv, int uv_stride,
                               int num_vertices, const void *indices, int num_indices, int size_indices,
                               float scale_x, float scale_y)
{
    int i;
    int count = indices ? num_indices : num_vertices;
    D3D12_VertexPositionColor *verts = (D3D12_VertexPositionColor *)SDL_AllocateRenderVertices(renderer, count * sizeof(D3D12_VertexPositionColor), 0, &cmd->data.draw.first);
    bool convert_color = SDL_RenderingLinearSpace(renderer);
    D3D12_TextureData *textureData = texture ? (D3D12_TextureData *)texture->internal : NULL;
    float u_scale = textureData ? (float)texture->w / textureData->w : 0.0f;
    float v_scale = textureData ? (float)texture->h / textureData->h : 0.0f;

    if (!verts) {
        return false;
    }

    cmd->data.draw.count = count;
    size_indices = indices ? size_indices : 0;

    for (i = 0; i < count; i++) {
        int j;
        float *xy_;
        if (size_indices == 4) {
            j = ((const Uint32 *)indices)[i];
        } else if (size_indices == 2) {
            j = ((const Uint16 *)indices)[i];
        } else if (size_indices == 1) {
            j = ((const Uint8 *)indices)[i];
        } else {
            j = i;
        }

        xy_ = (float *)((char *)xy + j * xy_stride);

        verts->pos.x = xy_[0] * scale_x;
        verts->pos.y = xy_[1] * scale_y;
        verts->color = *(SDL_FColor *)((char *)color + j * color_stride);
        if (convert_color) {
            SDL_ConvertToLinear(&verts->color);
        }

        if (texture) {
            float *uv_ = (float *)((char *)uv + j * uv_stride);
            verts->tex.x = uv_[0] * u_scale;
            verts->tex.y = uv_[1] * v_scale;
        } else {
            verts->tex.x = 0.0f;
            verts->tex.y = 0.0f;
        }

        verts += 1;
    }
    return true;
}

static bool D3D12_UpdateVertexBuffer(SDL_Renderer *renderer,
                                    const void *vertexData, size_t dataSizeInBytes)
{
    D3D12_RenderData *rendererData = (D3D12_RenderData *)renderer->internal;
    HRESULT result = S_OK;
    const int vbidx = rendererData->currentVertexBuffer;
    UINT8 *vertexBufferData = NULL;
    D3D12_RANGE range;
    ID3D12Resource *vertexBuffer;

    range.Begin = 0;
    range.End = 0;

    if (dataSizeInBytes == 0) {
        return true; // nothing to do.
    }

    if (rendererData->issueBatch) {
        if (FAILED(D3D12_IssueBatch(rendererData))) {
            return SDL_SetError("Failed to issue intermediate batch");
        }
    }

    // If the existing vertex buffer isn't big enough, we need to recreate a big enough one
    if (dataSizeInBytes > rendererData->vertexBuffers[vbidx].size) {
        D3D12_CreateVertexBuffer(rendererData, vbidx, dataSizeInBytes);
    }

    vertexBuffer = rendererData->vertexBuffers[vbidx].resource;
    result = ID3D12Resource_Map(vertexBuffer, 0, &range, (void **)&vertexBufferData);
    if (FAILED(result)) {
        return WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D12Resource::Map [vertex buffer]"), result);
    }
    SDL_memcpy(vertexBufferData, vertexData, dataSizeInBytes);
    ID3D12Resource_Unmap(vertexBuffer, 0, NULL);

    rendererData->vertexBuffers[vbidx].view.SizeInBytes = (UINT)dataSizeInBytes;

    ID3D12GraphicsCommandList2_IASetVertexBuffers(rendererData->commandList, 0, 1, &rendererData->vertexBuffers[vbidx].view);

    rendererData->currentVertexBuffer++;
    if (rendererData->currentVertexBuffer >= SDL_D3D12_NUM_VERTEX_BUFFERS) {
        rendererData->currentVertexBuffer = 0;
        rendererData->issueBatch = true;
    }

    return true;
}

static bool D3D12_UpdateViewport(SDL_Renderer *renderer)
{
    D3D12_RenderData *data = (D3D12_RenderData *)renderer->internal;
    const SDL_Rect *viewport = &data->currentViewport;
    Float4X4 projection;
    Float4X4 view;
    SDL_FRect orientationAlignedViewport;
    BOOL swapDimensions;
    D3D12_VIEWPORT d3dviewport;
    const int rotation = D3D12_GetRotationForCurrentRenderTarget(renderer);

    if (viewport->w == 0 || viewport->h == 0) {
        /* If the viewport is empty, assume that it is because
         * SDL_CreateRenderer is calling it, and will call it again later
         * with a non-empty viewport.
         */
        // SDL_Log("%s, no viewport was set!", __FUNCTION__);
        return false;
    }

    /* Make sure the SDL viewport gets rotated to that of the physical display's rotation.
     * Keep in mind here that the Y-axis will be been inverted (from Direct3D's
     * default coordinate system) so rotations will be done in the opposite
     * direction of the DXGI_MODE_ROTATION enumeration.
     */
    switch (rotation) {
    case DXGI_MODE_ROTATION_IDENTITY:
        projection = MatrixIdentity();
        break;
    case DXGI_MODE_ROTATION_ROTATE270:
        projection = MatrixRotationZ(SDL_PI_F * 0.5f);
        break;
    case DXGI_MODE_ROTATION_ROTATE180:
        projection = MatrixRotationZ(SDL_PI_F);
        break;
    case DXGI_MODE_ROTATION_ROTATE90:
        projection = MatrixRotationZ(-SDL_PI_F * 0.5f);
        break;
    default:
        return SDL_SetError("An unknown DisplayOrientation is being used");
    }

    // Update the view matrix
    SDL_zero(view);
    view.m[0][0] = 2.0f / viewport->w;
    view.m[1][1] = -2.0f / viewport->h;
    view.m[2][2] = 1.0f;
    view.m[3][0] = -1.0f;
    view.m[3][1] = 1.0f;
    view.m[3][3] = 1.0f;

    /* Combine the projection + view matrix together now, as both only get
     * set here (as of this writing, on Dec 26, 2013).  When done, store it
     * for eventual transfer to the GPU.
     */
    data->vertexShaderConstantsData.projectionAndView = MatrixMultiply(
        view,
        projection);

    /* Update the Direct3D viewport, which seems to be aligned to the
     * swap buffer's coordinate space, which is always in either
     * a landscape mode, for all Windows 8/RT devices, or a portrait mode,
     * for Windows Phone devices.
     */
    swapDimensions = D3D12_IsDisplayRotated90Degrees((DXGI_MODE_ROTATION)rotation);
    if (swapDimensions) {
        orientationAlignedViewport.x = (float)viewport->y;
        orientationAlignedViewport.y = (float)viewport->x;
        orientationAlignedViewport.w = (float)viewport->h;
        orientationAlignedViewport.h = (float)viewport->w;
    } else {
        orientationAlignedViewport.x = (float)viewport->x;
        orientationAlignedViewport.y = (float)viewport->y;
        orientationAlignedViewport.w = (float)viewport->w;
        orientationAlignedViewport.h = (float)viewport->h;
    }

    d3dviewport.TopLeftX = orientationAlignedViewport.x;
    d3dviewport.TopLeftY = orientationAlignedViewport.y;
    d3dviewport.Width = orientationAlignedViewport.w;
    d3dviewport.Height = orientationAlignedViewport.h;
    d3dviewport.MinDepth = 0.0f;
    d3dviewport.MaxDepth = 1.0f;
    // SDL_Log("%s: D3D viewport = {%f,%f,%f,%f}", __FUNCTION__, d3dviewport.TopLeftX, d3dviewport.TopLeftY, d3dviewport.Width, d3dviewport.Height);
    ID3D12GraphicsCommandList_RSSetViewports(data->commandList, 1, &d3dviewport);

    data->viewportDirty = false;

    return true;
}

static void D3D12_SetupShaderConstants(SDL_Renderer *renderer, const SDL_RenderCommand *cmd, const SDL_Texture *texture, D3D12_PixelShaderConstants *constants)
{
    float output_headroom;

    SDL_zerop(constants);

    constants->scRGB_output = (float)SDL_RenderingLinearSpace(renderer);
    constants->color_scale = cmd->data.draw.color_scale;

    if (texture) {
        D3D12_TextureData *textureData = (D3D12_TextureData *)texture->internal;

        switch (texture->format) {
        case SDL_PIXELFORMAT_YV12:
        case SDL_PIXELFORMAT_IYUV:
            constants->texture_type = TEXTURETYPE_YUV;
            constants->input_type = INPUTTYPE_SRGB;
            break;
        case SDL_PIXELFORMAT_NV12:
            constants->texture_type = TEXTURETYPE_NV12;
            constants->input_type = INPUTTYPE_SRGB;
            break;
        case SDL_PIXELFORMAT_NV21:
            constants->texture_type = TEXTURETYPE_NV21;
            constants->input_type = INPUTTYPE_SRGB;
            break;
        case SDL_PIXELFORMAT_P010:
            constants->texture_type = TEXTURETYPE_NV12;
            constants->input_type = INPUTTYPE_HDR10;
            break;
        default:
            if (cmd->data.draw.texture_scale_mode == SDL_SCALEMODE_PIXELART) {
                constants->texture_type = TEXTURETYPE_RGB_PIXELART;
                constants->texture_width = texture->w;
                constants->texture_height = texture->h;
                constants->texel_width = 1.0f / constants->texture_width;
                constants->texel_height = 1.0f / constants->texture_height;
            } else {
                constants->texture_type = TEXTURETYPE_RGB;
            }
            if (texture->colorspace == SDL_COLORSPACE_SRGB_LINEAR) {
                constants->input_type = INPUTTYPE_SCRGB;
            } else if (texture->colorspace == SDL_COLORSPACE_HDR10) {
                constants->input_type = INPUTTYPE_HDR10;
            } else {
                // The sampler will convert from sRGB to linear on load if working in linear colorspace
                constants->input_type = INPUTTYPE_UNSPECIFIED;
            }
            break;
        }

        constants->sdr_white_point = texture->SDR_white_point;

        if (renderer->target) {
            output_headroom = renderer->target->HDR_headroom;
        } else {
            output_headroom = renderer->HDR_headroom;
        }

        if (texture->HDR_headroom > output_headroom) {
            constants->tonemap_method = TONEMAP_CHROME;
            constants->tonemap_factor1 = (output_headroom / (texture->HDR_headroom * texture->HDR_headroom));
            constants->tonemap_factor2 = (1.0f / output_headroom);
        }

        if (textureData->YCbCr_matrix) {
            SDL_memcpy(constants->YCbCr_matrix, textureData->YCbCr_matrix, sizeof(constants->YCbCr_matrix));
        }
    }
}

static bool D3D12_SetDrawState(SDL_Renderer *renderer, const SDL_RenderCommand *cmd, D3D12_Shader shader, const D3D12_PixelShaderConstants *shader_constants,
                              D3D12_PRIMITIVE_TOPOLOGY_TYPE topology,
                              const int numShaderResources, D3D12_CPU_DESCRIPTOR_HANDLE *shaderResources,
                              D3D12_CPU_DESCRIPTOR_HANDLE *sampler, const Float4X4 *matrix)

{
    D3D12_RenderData *rendererData = (D3D12_RenderData *)renderer->internal;
    const Float4X4 *newmatrix = matrix ? matrix : &rendererData->identity;
    D3D12_CPU_DESCRIPTOR_HANDLE renderTargetView = D3D12_GetCurrentRenderTargetView(renderer);
    const SDL_BlendMode blendMode = cmd->data.draw.blend;
    bool updateSubresource = false;
    int i;
    D3D12_CPU_DESCRIPTOR_HANDLE firstShaderResource;
    DXGI_FORMAT rtvFormat = rendererData->renderTargetFormat;
    D3D12_PipelineState *currentPipelineState = rendererData->currentPipelineState;;
    D3D12_PixelShaderConstants solid_constants;

    if (rendererData->textureRenderTarget) {
        rtvFormat = rendererData->textureRenderTarget->mainTextureFormat;
    }

    // See if we need to change the pipeline state
    if (!currentPipelineState ||
        currentPipelineState->shader != shader ||
        currentPipelineState->blendMode != blendMode ||
        currentPipelineState->topology != topology ||
        currentPipelineState->rtvFormat != rtvFormat) {

        /* Find the matching pipeline.
           NOTE: Although it may seem inefficient to linearly search through ~450 pipelines
           to find the correct one, in profiling this doesn't come up at all.
           It's unlikely that using a hash table would affect performance a measurable amount unless
           it's a degenerate case that's changing the pipeline state dozens of times per frame.
        */
        currentPipelineState = NULL;
        for (i = 0; i < rendererData->pipelineStateCount; ++i) {
            D3D12_PipelineState *candidatePiplineState = &rendererData->pipelineStates[i];
            if (candidatePiplineState->shader == shader &&
                candidatePiplineState->blendMode == blendMode &&
                candidatePiplineState->topology == topology &&
                candidatePiplineState->rtvFormat == rtvFormat) {
                currentPipelineState = candidatePiplineState;
                break;
            }
        }

        // If we didn't find a match, create a new one -- it must mean the blend mode is non-standard
        if (!currentPipelineState) {
            currentPipelineState = D3D12_CreatePipelineState(renderer, shader, blendMode, topology, rtvFormat);
        }

        if (!currentPipelineState) {
            // The error has been set inside D3D12_CreatePipelineState()
            return false;
        }

        ID3D12GraphicsCommandList2_SetPipelineState(rendererData->commandList, currentPipelineState->pipelineState);
        ID3D12GraphicsCommandList2_SetGraphicsRootSignature(rendererData->commandList,
                 rendererData->rootSignatures[D3D12_GetRootSignatureType(currentPipelineState->shader)]);
        // When we change these we will need to re-upload the constant buffer and reset any descriptors
        updateSubresource = true;
        rendererData->currentSampler.ptr = 0;
        rendererData->currentShaderResource.ptr = 0;
        rendererData->currentPipelineState = currentPipelineState;
    }

    if (renderTargetView.ptr != rendererData->currentRenderTargetView.ptr) {
        ID3D12GraphicsCommandList2_OMSetRenderTargets(rendererData->commandList, 1, &renderTargetView, FALSE, NULL);
        rendererData->currentRenderTargetView = renderTargetView;
    }

    if (rendererData->viewportDirty) {
        if (D3D12_UpdateViewport(renderer)) {
            // vertexShaderConstantsData.projectionAndView has changed
            updateSubresource = true;
        }
    }

    if (rendererData->cliprectDirty) {
        D3D12_RECT scissorRect;
        if (!D3D12_GetViewportAlignedD3DRect(renderer, &rendererData->currentCliprect, &scissorRect, TRUE)) {
            // D3D12_GetViewportAlignedD3DRect will have set the SDL error
            return false;
        }
        ID3D12GraphicsCommandList2_RSSetScissorRects(rendererData->commandList, 1, &scissorRect);
        rendererData->cliprectDirty = false;
    }

    if (numShaderResources > 0) {
        firstShaderResource = shaderResources[0];
    } else {
        firstShaderResource.ptr = 0;
    }
    if (firstShaderResource.ptr != rendererData->currentShaderResource.ptr) {
        for (i = 0; i < numShaderResources; ++i) {
            D3D12_GPU_DESCRIPTOR_HANDLE GPUHandle = D3D12_CPUtoGPUHandle(rendererData->srvDescriptorHeap, shaderResources[i]);
            ID3D12GraphicsCommandList2_SetGraphicsRootDescriptorTable(rendererData->commandList, i + 2, GPUHandle);
        }
        rendererData->currentShaderResource.ptr = firstShaderResource.ptr;
    }

    if (sampler && sampler->ptr != rendererData->currentSampler.ptr) {
        D3D12_GPU_DESCRIPTOR_HANDLE GPUHandle = D3D12_CPUtoGPUHandle(rendererData->samplerDescriptorHeap, *sampler);
        UINT tableIndex = 0;

        // Figure out the correct sampler descriptor table index based on the type of shader
        switch (shader) {
        case SHADER_RGB:
            tableIndex = 3;
            break;
        case SHADER_ADVANCED:
            tableIndex = 5;
            break;
        default:
            return SDL_SetError("[direct3d12] Trying to set a sampler for a shader which doesn't have one");
            break;
        }

        ID3D12GraphicsCommandList2_SetGraphicsRootDescriptorTable(rendererData->commandList, tableIndex, GPUHandle);
        rendererData->currentSampler = *sampler;
    }

    if (updateSubresource == true || SDL_memcmp(&rendererData->vertexShaderConstantsData.model, newmatrix, sizeof(*newmatrix)) != 0) {
        SDL_memcpy(&rendererData->vertexShaderConstantsData.model, newmatrix, sizeof(*newmatrix));
        ID3D12GraphicsCommandList2_SetGraphicsRoot32BitConstants(rendererData->commandList,
                 0,
                 32,
                 &rendererData->vertexShaderConstantsData,
                 0);
    }

    if (!shader_constants) {
        D3D12_SetupShaderConstants(renderer, cmd, NULL, &solid_constants);
        shader_constants = &solid_constants;
    }

    if (updateSubresource == true ||
        SDL_memcmp(shader_constants, &currentPipelineState->shader_constants, sizeof(*shader_constants)) != 0) {
        ID3D12GraphicsCommandList2_SetGraphicsRoot32BitConstants(rendererData->commandList,
                 1,
                 sizeof(*shader_constants) / sizeof(float),
                 shader_constants,
                 0);

        SDL_memcpy(&currentPipelineState->shader_constants, shader_constants, sizeof(*shader_constants));
    }

    return true;
}

static D3D12_CPU_DESCRIPTOR_HANDLE *D3D12_GetSamplerState(D3D12_RenderData *data, SDL_ScaleMode scale_mode, SDL_TextureAddressMode address_u, SDL_TextureAddressMode address_v)
{
    Uint32 key = RENDER_SAMPLER_HASHKEY(scale_mode, address_u, address_v);
    SDL_assert(key < SDL_arraysize(data->samplers));
    if (!data->samplers_created[key]) {
        D3D12_SAMPLER_DESC samplerDesc;
        SDL_zero(samplerDesc);
        samplerDesc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        samplerDesc.MipLODBias = 0.0f;
        samplerDesc.MaxAnisotropy = 1;
        samplerDesc.ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS;
        samplerDesc.MinLOD = 0.0f;
        samplerDesc.MaxLOD = D3D12_FLOAT32_MAX;
        switch (scale_mode) {
        case SDL_SCALEMODE_NEAREST:
            samplerDesc.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
            break;
        case SDL_SCALEMODE_PIXELART:    // Uses linear sampling
        case SDL_SCALEMODE_LINEAR:
            samplerDesc.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
            break;
        default:
            SDL_SetError("Unknown scale mode: %d", scale_mode);
            return NULL;
        }
        switch (address_u) {
        case SDL_TEXTURE_ADDRESS_CLAMP:
            samplerDesc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
            break;
        case SDL_TEXTURE_ADDRESS_WRAP:
            samplerDesc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
            break;
        default:
            SDL_SetError("Unknown texture address mode: %d", address_u);
            return NULL;
        }
        switch (address_v) {
        case SDL_TEXTURE_ADDRESS_CLAMP:
            samplerDesc.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
            break;
        case SDL_TEXTURE_ADDRESS_WRAP:
            samplerDesc.AddressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
            break;
        default:
            SDL_SetError("Unknown texture address mode: %d", address_v);
            return NULL;
        }
        ID3D12Device1_CreateSampler(data->d3dDevice, &samplerDesc, data->samplers[key]);
        data->samplers_created[key] = true;
    }
    return &data->samplers[key];
}

static bool D3D12_SetCopyState(SDL_Renderer *renderer, const SDL_RenderCommand *cmd, const Float4X4 *matrix)
{
    SDL_Texture *texture = cmd->data.draw.texture;
    D3D12_RenderData *rendererData = (D3D12_RenderData *)renderer->internal;
    D3D12_TextureData *textureData = (D3D12_TextureData *)texture->internal;
    D3D12_CPU_DESCRIPTOR_HANDLE *textureSampler;
    D3D12_PixelShaderConstants constants;

    if (!textureData) {
        return SDL_SetError("Texture is not currently available");
    }

    D3D12_SetupShaderConstants(renderer, cmd, texture, &constants);

    textureSampler = D3D12_GetSamplerState(rendererData, cmd->data.draw.texture_scale_mode, cmd->data.draw.texture_address_mode_u, cmd->data.draw.texture_address_mode_v);
    if (!textureSampler) {
        return false;
    }

#ifdef SDL_HAVE_YUV
    if (textureData->yuv) {
        D3D12_CPU_DESCRIPTOR_HANDLE shaderResources[3];

        shaderResources[0] = textureData->mainTextureResourceView;
        shaderResources[1] = textureData->mainTextureResourceViewU;
        shaderResources[2] = textureData->mainTextureResourceViewV;

        // Make sure each texture is in the correct state to be accessed by the pixel shader.
        D3D12_TransitionResource(rendererData, textureData->mainTexture, textureData->mainResourceState, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        textureData->mainResourceState = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        D3D12_TransitionResource(rendererData, textureData->mainTextureU, textureData->mainResourceStateU, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        textureData->mainResourceStateU = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        D3D12_TransitionResource(rendererData, textureData->mainTextureV, textureData->mainResourceStateV, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        textureData->mainResourceStateV = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;

        return D3D12_SetDrawState(renderer, cmd, textureData->shader, &constants, D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE, SDL_arraysize(shaderResources), shaderResources, textureSampler, matrix);
    } else if (textureData->nv12) {
        D3D12_CPU_DESCRIPTOR_HANDLE shaderResources[2];

        shaderResources[0] = textureData->mainTextureResourceView;
        shaderResources[1] = textureData->mainTextureResourceViewNV;

        // Make sure each texture is in the correct state to be accessed by the pixel shader.
        D3D12_TransitionResource(rendererData, textureData->mainTexture, textureData->mainResourceState, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        textureData->mainResourceState = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;

        return D3D12_SetDrawState(renderer, cmd, textureData->shader, &constants, D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE, SDL_arraysize(shaderResources), shaderResources, textureSampler, matrix);
    }
#endif // SDL_HAVE_YUV
    D3D12_TransitionResource(rendererData, textureData->mainTexture, textureData->mainResourceState, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    textureData->mainResourceState = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    return D3D12_SetDrawState(renderer, cmd, textureData->shader, &constants, D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE, 1, &textureData->mainTextureResourceView, textureSampler, matrix);
}

static void D3D12_DrawPrimitives(SDL_Renderer *renderer, D3D12_PRIMITIVE_TOPOLOGY primitiveTopology, const size_t vertexStart, const size_t vertexCount)
{
    D3D12_RenderData *rendererData = (D3D12_RenderData *)renderer->internal;
    ID3D12GraphicsCommandList2_IASetPrimitiveTopology(rendererData->commandList, primitiveTopology);
    ID3D12GraphicsCommandList2_DrawInstanced(rendererData->commandList, (UINT)vertexCount, 1, (UINT)vertexStart, 0);
}

static void D3D12_InvalidateCachedState(SDL_Renderer *renderer)
{
    D3D12_RenderData *data = (D3D12_RenderData *)renderer->internal;
    data->currentRenderTargetView.ptr = 0;
    data->currentShaderResource.ptr = 0;
    data->currentSampler.ptr = 0;
    data->cliprectDirty = true;
    data->viewportDirty = true;
}

static bool D3D12_RunCommandQueue(SDL_Renderer *renderer, SDL_RenderCommand *cmd, void *vertices, size_t vertsize)
{
    D3D12_RenderData *rendererData = (D3D12_RenderData *)renderer->internal;
    const int viewportRotation = D3D12_GetRotationForCurrentRenderTarget(renderer);

    if (!rendererData->d3dDevice) {
        return SDL_SetError("Device lost and couldn't be recovered");
    }

    if (rendererData->pixelSizeChanged) {
        D3D12_UpdateForWindowSizeChange(renderer);
        rendererData->pixelSizeChanged = false;
    }

    if (rendererData->currentViewportRotation != viewportRotation) {
        rendererData->currentViewportRotation = viewportRotation;
        rendererData->viewportDirty = true;
    }

    if (!D3D12_UpdateVertexBuffer(renderer, vertices, vertsize)) {
        return false;
    }

    while (cmd) {
        switch (cmd->command) {
        case SDL_RENDERCMD_SETDRAWCOLOR:
        {
            break; // this isn't currently used in this render backend.
        }

        case SDL_RENDERCMD_SETVIEWPORT:
        {
            SDL_Rect *viewport = &rendererData->currentViewport;
            if (SDL_memcmp(viewport, &cmd->data.viewport.rect, sizeof(cmd->data.viewport.rect)) != 0) {
                SDL_copyp(viewport, &cmd->data.viewport.rect);
                rendererData->viewportDirty = true;
                rendererData->cliprectDirty = true;
            }
            break;
        }

        case SDL_RENDERCMD_SETCLIPRECT:
        {
            const SDL_Rect *rect = &cmd->data.cliprect.rect;
            SDL_Rect viewport_cliprect;
            if (rendererData->currentCliprectEnabled != cmd->data.cliprect.enabled) {
                rendererData->currentCliprectEnabled = cmd->data.cliprect.enabled;
                rendererData->cliprectDirty = true;
            }
            if (!rendererData->currentCliprectEnabled) {
                /* If the clip rect is disabled, then the scissor rect should be the whole viewport,
                   since direct3d12 doesn't allow disabling the scissor rectangle */
                viewport_cliprect.x = 0;
                viewport_cliprect.y = 0;
                viewport_cliprect.w = rendererData->currentViewport.w;
                viewport_cliprect.h = rendererData->currentViewport.h;
                rect = &viewport_cliprect;
            }
            if (SDL_memcmp(&rendererData->currentCliprect, rect, sizeof(*rect)) != 0) {
                SDL_copyp(&rendererData->currentCliprect, rect);
                rendererData->cliprectDirty = true;
            }
            break;
        }

        case SDL_RENDERCMD_CLEAR:
        {
            D3D12_CPU_DESCRIPTOR_HANDLE rtvDescriptor = D3D12_GetCurrentRenderTargetView(renderer);
            bool convert_color = SDL_RenderingLinearSpace(renderer);
            SDL_FColor color = cmd->data.color.color;
            if (convert_color) {
                SDL_ConvertToLinear(&color);
            }
            color.r *= cmd->data.color.color_scale;
            color.g *= cmd->data.color.color_scale;
            color.b *= cmd->data.color.color_scale;
            ID3D12GraphicsCommandList2_ClearRenderTargetView(rendererData->commandList, rtvDescriptor, &color.r, 0, NULL);
            break;
        }

        case SDL_RENDERCMD_DRAW_POINTS:
        {
            const size_t count = cmd->data.draw.count;
            const size_t first = cmd->data.draw.first;
            const size_t start = first / sizeof(D3D12_VertexPositionColor);
            D3D12_SetDrawState(renderer, cmd, SHADER_SOLID, NULL, D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT, 0, NULL, NULL, NULL);
            D3D12_DrawPrimitives(renderer, D3D_PRIMITIVE_TOPOLOGY_POINTLIST, start, count);
            break;
        }

        case SDL_RENDERCMD_DRAW_LINES:
        {
            const size_t count = cmd->data.draw.count;
            const size_t first = cmd->data.draw.first;
            const size_t start = first / sizeof(D3D12_VertexPositionColor);
            const D3D12_VertexPositionColor *verts = (D3D12_VertexPositionColor *)(((Uint8 *)vertices) + first);
            D3D12_SetDrawState(renderer, cmd, SHADER_SOLID, NULL, D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE, 0, NULL, NULL, NULL);
            D3D12_DrawPrimitives(renderer, D3D_PRIMITIVE_TOPOLOGY_LINESTRIP, start, count);
            if (verts[0].pos.x != verts[count - 1].pos.x || verts[0].pos.y != verts[count - 1].pos.y) {
                D3D12_DrawPrimitives(renderer, D3D_PRIMITIVE_TOPOLOGY_POINTLIST, start + (count - 1), 1);
            }
            break;
        }

        case SDL_RENDERCMD_FILL_RECTS: // unused
            break;

        case SDL_RENDERCMD_COPY: // unused
            break;

        case SDL_RENDERCMD_COPY_EX: // unused
            break;

        case SDL_RENDERCMD_GEOMETRY:
        {
            SDL_Texture *texture = cmd->data.draw.texture;
            const size_t count = cmd->data.draw.count;
            const size_t first = cmd->data.draw.first;
            const size_t start = first / sizeof(D3D12_VertexPositionColor);

            if (texture) {
                D3D12_SetCopyState(renderer, cmd, NULL);
            } else {
                D3D12_SetDrawState(renderer, cmd, SHADER_SOLID, NULL, D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE, 0, NULL, NULL, NULL);
            }

            D3D12_DrawPrimitives(renderer, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, start, count);
            break;
        }

        case SDL_RENDERCMD_NO_OP:
            break;
        }

        cmd = cmd->next;
    }

    return true;
}

static SDL_Surface *D3D12_RenderReadPixels(SDL_Renderer *renderer, const SDL_Rect *rect)
{
    D3D12_RenderData *data = (D3D12_RenderData *)renderer->internal;
    ID3D12Resource *backBuffer = NULL;
    ID3D12Resource *readbackBuffer = NULL;
    HRESULT result;
    D3D12_RESOURCE_DESC textureDesc;
    D3D12_RESOURCE_DESC readbackDesc;
    D3D12_HEAP_PROPERTIES heapProps;
    D3D12_RECT srcRect = { 0, 0, 0, 0 };
    D3D12_BOX srcBox;
    D3D12_TEXTURE_COPY_LOCATION dstLocation;
    D3D12_TEXTURE_COPY_LOCATION srcLocation;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT placedTextureDesc;
    D3D12_SUBRESOURCE_FOOTPRINT pitchedDesc;
    BYTE *textureMemory;
    int bpp;
    SDL_Surface *output = NULL;

    if (data->textureRenderTarget) {
        backBuffer = data->textureRenderTarget->mainTexture;
    } else {
        backBuffer = data->renderTargets[data->currentBackBufferIndex];
    }

    // Create a staging texture to copy the screen's data to:
    SDL_zero(textureDesc);
    D3D_CALL_RET(backBuffer, GetDesc, &textureDesc);
    textureDesc.Width = rect->w;
    textureDesc.Height = rect->h;

    SDL_zero(readbackDesc);
    readbackDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    readbackDesc.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
    readbackDesc.Height = 1;
    readbackDesc.DepthOrArraySize = 1;
    readbackDesc.MipLevels = 1;
    readbackDesc.Format = DXGI_FORMAT_UNKNOWN;
    readbackDesc.SampleDesc.Count = 1;
    readbackDesc.SampleDesc.Quality = 0;
    readbackDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    readbackDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    // Figure out how much we need to allocate for the upload buffer
    ID3D12Device1_GetCopyableFootprints(data->d3dDevice,
             &textureDesc,
             0,
             1,
             0,
             NULL,
             NULL,
             NULL,
             &readbackDesc.Width);

    SDL_zero(heapProps);
    heapProps.Type = D3D12_HEAP_TYPE_READBACK;
    heapProps.CreationNodeMask = 1;
    heapProps.VisibleNodeMask = 1;

    result = ID3D12Device1_CreateCommittedResource(data->d3dDevice,
                      &heapProps,
                      D3D12_HEAP_FLAG_NONE,
                      &readbackDesc,
                      D3D12_RESOURCE_STATE_COPY_DEST,
                      NULL,
                      D3D_GUID(SDL_IID_ID3D12Resource),
                      (void **)&readbackBuffer);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D12Device::CreateTexture2D [create staging texture]"), result);
        goto done;
    }

    // Transition the render target to be copyable from
    D3D12_TransitionResource(data, backBuffer, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);

    // Copy the desired portion of the back buffer to the staging texture:
    if (!D3D12_GetViewportAlignedD3DRect(renderer, rect, &srcRect, FALSE)) {
        // D3D12_GetViewportAlignedD3DRect will have set the SDL error
        goto done;
    }
    srcBox.left = srcRect.left;
    srcBox.right = srcRect.right;
    srcBox.top = srcRect.top;
    srcBox.bottom = srcRect.bottom;
    srcBox.front = 0;
    srcBox.back = 1;

    // Issue the copy texture region
    SDL_zero(pitchedDesc);
    pitchedDesc.Format = textureDesc.Format;
    pitchedDesc.Width = (UINT)textureDesc.Width;
    pitchedDesc.Height = textureDesc.Height;
    pitchedDesc.Depth = 1;
    bpp = SDL_BYTESPERPIXEL(D3D12_DXGIFormatToSDLPixelFormat(pitchedDesc.Format));
    pitchedDesc.RowPitch = D3D12_Align(pitchedDesc.Width * bpp, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);

    SDL_zero(placedTextureDesc);
    placedTextureDesc.Offset = 0;
    placedTextureDesc.Footprint = pitchedDesc;

    SDL_zero(dstLocation);
    dstLocation.pResource = readbackBuffer;
    dstLocation.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dstLocation.PlacedFootprint = placedTextureDesc;

    SDL_zero(srcLocation);
    srcLocation.pResource = backBuffer;
    srcLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    srcLocation.SubresourceIndex = 0;

    ID3D12GraphicsCommandList2_CopyTextureRegion(data->commandList,
             &dstLocation,
             0, 0, 0,
             &srcLocation,
             &srcBox);

    // We need to issue the command list for the copy to finish
    D3D12_IssueBatch(data);

    // Transition the render target back to a render target
    D3D12_TransitionResource(data, backBuffer, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);

    // Map the staging texture's data to CPU-accessible memory:
    result = ID3D12Resource_Map(readbackBuffer,
                      0,
                      NULL,
                      (void **)&textureMemory);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D12Resource::Map [map staging texture]"), result);
        goto done;
    }

    output = SDL_DuplicatePixels(
        rect->w, rect->h,
        D3D12_DXGIFormatToSDLPixelFormat(textureDesc.Format),
        renderer->target ? renderer->target->colorspace : renderer->output_colorspace,
        textureMemory,
        pitchedDesc.RowPitch);

    // Unmap the texture:
    ID3D12Resource_Unmap(readbackBuffer, 0, NULL);

done:
    D3D_SAFE_RELEASE(readbackBuffer);
    return output;
}

static bool D3D12_RenderPresent(SDL_Renderer *renderer)
{
    D3D12_RenderData *data = (D3D12_RenderData *)renderer->internal;
    HRESULT result;

    if (!data->d3dDevice) {
        return SDL_SetError("Device lost and couldn't be recovered");
    }

    // Transition the render target to present state
    D3D12_TransitionResource(data,
                             data->renderTargets[data->currentBackBufferIndex],
                             D3D12_RESOURCE_STATE_RENDER_TARGET,
                             D3D12_RESOURCE_STATE_PRESENT);

    // Issue the command list
    result = ID3D12GraphicsCommandList2_Close(data->commandList);
    ID3D12CommandQueue_ExecuteCommandLists(data->commandQueue, 1, (ID3D12CommandList *const *)&data->commandList);

#if defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES)
    result = D3D12_XBOX_PresentFrame(data->commandQueue, data->frameToken, data->renderTargets[data->currentBackBufferIndex]);
#else
    /* The application may optionally specify "dirty" or "scroll"
     * rects to improve efficiency in certain scenarios.
     */
    result = IDXGISwapChain_Present(data->swapChain, data->syncInterval, data->presentFlags);
#endif

    if (FAILED(result) && result != DXGI_ERROR_WAS_STILL_DRAWING) {
        /* If the device was removed either by a disconnect or a driver upgrade, we
         * must recreate all device resources.
         */
        if (result == DXGI_ERROR_DEVICE_REMOVED) {
            if (D3D12_HandleDeviceLost(renderer)) {
                SDL_SetError("Present failed, device lost");
            } else {
                // Recovering from device lost failed, error is already set
            }
        } else if (result == DXGI_ERROR_INVALID_CALL) {
            // We probably went through a fullscreen <-> windowed transition
            D3D12_CreateWindowSizeDependentResources(renderer);
            WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("IDXGISwapChain::Present"), result);
        } else {
            WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("IDXGISwapChain::Present"), result);
        }
        return false;
    } else {
        // Wait for the GPU and move to the next frame
        result = ID3D12CommandQueue_Signal(data->commandQueue, data->fence, data->fenceValue);

        if (ID3D12Fence_GetCompletedValue(data->fence) < data->fenceValue) {
            result = ID3D12Fence_SetEventOnCompletion(data->fence,
                              data->fenceValue,
                              data->fenceEvent);
            WaitForSingleObjectEx(data->fenceEvent, INFINITE, FALSE);
        }

        data->fenceValue++;
#if defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES)
        data->currentBackBufferIndex++;
        data->currentBackBufferIndex %= SDL_D3D12_NUM_BUFFERS;
#else
        data->currentBackBufferIndex = IDXGISwapChain4_GetCurrentBackBufferIndex(data->swapChain);
#endif

        // Reset the command allocator and command list, and transition back to render target
        D3D12_ResetCommandList(data);
        D3D12_TransitionResource(data,
                                 data->renderTargets[data->currentBackBufferIndex],
                                 D3D12_RESOURCE_STATE_PRESENT,
                                 D3D12_RESOURCE_STATE_RENDER_TARGET);

#if defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES)
        D3D12_XBOX_StartFrame(data->d3dDevice, &data->frameToken);
#endif
        return true;
    }
}

static bool D3D12_SetVSync(SDL_Renderer *renderer, const int vsync)
{
    D3D12_RenderData *data = (D3D12_RenderData *)renderer->internal;

    if (vsync < 0) {
        return SDL_Unsupported();
    }

    if (vsync > 0) {
        data->syncInterval = vsync;
        data->presentFlags = 0;
    } else {
        data->syncInterval = 0;
        data->presentFlags = DXGI_PRESENT_ALLOW_TEARING;
    }
    return true;
}

bool D3D12_CreateRenderer(SDL_Renderer *renderer, SDL_Window *window, SDL_PropertiesID create_props)
{
    D3D12_RenderData *data;

    HWND hwnd = (HWND)SDL_GetPointerProperty(SDL_GetWindowProperties(window), SDL_PROP_WINDOW_WIN32_HWND_POINTER, NULL);
    if (!hwnd) {
        return SDL_SetError("Couldn't get window handle");
    }

    if (SDL_GetWindowFlags(window) & SDL_WINDOW_TRANSPARENT) {
		// D3D12 removed the swap effect needed to support transparent windows, use D3D11 instead
		return SDL_SetError("The direct3d12 renderer doesn't work with transparent windows");
	}

    SDL_SetupRendererColorspace(renderer, create_props);

    if (renderer->output_colorspace != SDL_COLORSPACE_SRGB &&
        renderer->output_colorspace != SDL_COLORSPACE_SRGB_LINEAR
        /*&& renderer->output_colorspace != SDL_COLORSPACE_HDR10*/) {
        return SDL_SetError("Unsupported output colorspace");
    }

    data = (D3D12_RenderData *)SDL_calloc(1, sizeof(*data));
    if (!data) {
        return false;
    }

    data->identity = MatrixIdentity();

    renderer->WindowEvent = D3D12_WindowEvent;
    renderer->SupportsBlendMode = D3D12_SupportsBlendMode;
    renderer->CreateTexture = D3D12_CreateTexture;
    renderer->UpdateTexture = D3D12_UpdateTexture;
#ifdef SDL_HAVE_YUV
    renderer->UpdateTextureYUV = D3D12_UpdateTextureYUV;
    renderer->UpdateTextureNV = D3D12_UpdateTextureNV;
#endif
    renderer->LockTexture = D3D12_LockTexture;
    renderer->UnlockTexture = D3D12_UnlockTexture;
    renderer->SetRenderTarget = D3D12_SetRenderTarget;
    renderer->QueueSetViewport = D3D12_QueueNoOp;
    renderer->QueueSetDrawColor = D3D12_QueueNoOp;
    renderer->QueueDrawPoints = D3D12_QueueDrawPoints;
    renderer->QueueDrawLines = D3D12_QueueDrawPoints; // lines and points queue vertices the same way.
    renderer->QueueGeometry = D3D12_QueueGeometry;
    renderer->InvalidateCachedState = D3D12_InvalidateCachedState;
    renderer->RunCommandQueue = D3D12_RunCommandQueue;
    renderer->RenderReadPixels = D3D12_RenderReadPixels;
    renderer->RenderPresent = D3D12_RenderPresent;
    renderer->DestroyTexture = D3D12_DestroyTexture;
    renderer->DestroyRenderer = D3D12_DestroyRenderer;
    renderer->SetVSync = D3D12_SetVSync;
    renderer->internal = data;
    D3D12_InvalidateCachedState(renderer);

    renderer->name = D3D12_RenderDriver.name;
    SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_ARGB8888);
    SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_ABGR8888);
    SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_XRGB8888);
    SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_ABGR2101010);
    SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_RGBA64_FLOAT);
    SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_YV12);
    SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_IYUV);
    SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_NV12);
    SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_NV21);
    SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_P010);
    SDL_SetNumberProperty(SDL_GetRendererProperties(renderer), SDL_PROP_RENDERER_MAX_TEXTURE_SIZE_NUMBER, 16384);

    data->syncInterval = 0;
    data->presentFlags = DXGI_PRESENT_ALLOW_TEARING;

    /* HACK: make sure the SDL_Renderer references the SDL_Window data now, in
     * order to give init functions access to the underlying window handle:
     */
    renderer->window = window;

    // Initialize Direct3D resources
    if (FAILED(D3D12_CreateDeviceResources(renderer))) {
        return false;
    }
    if (FAILED(D3D12_CreateWindowSizeDependentResources(renderer))) {
        return false;
    }

    return true;
}

SDL_RenderDriver D3D12_RenderDriver = {
    D3D12_CreateRenderer, "direct3d12"
};

// Ends C function definitions when using C++
#ifdef __cplusplus
}
#endif

#endif // SDL_VIDEO_RENDER_D3D12
