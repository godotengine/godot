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

#ifdef SDL_VIDEO_RENDER_D3D11

#define COBJMACROS
#include "../../core/windows/SDL_windows.h"
#include "../../video/windows/SDL_windowswindow.h"
#include "../SDL_sysrender.h"
#include "../SDL_d3dmath.h"
#include "../../video/SDL_pixels_c.h"

#include <d3d11_1.h>
#include <dxgi1_4.h>
#include <dxgidebug.h>

#include "SDL_shaders_d3d11.h"

#if defined(_MSC_VER) && !defined(__clang__)
#define SDL_COMPOSE_ERROR(str) __FUNCTION__ ", " str
#else
#define SDL_COMPOSE_ERROR(str) SDL_STRINGIFY_ARG(__FUNCTION__) ", " str
#endif

#define SAFE_RELEASE(X)                                   \
    if ((X)) {                                            \
        IUnknown_Release(SDL_static_cast(IUnknown *, X)); \
        X = NULL;                                         \
    }

/* !!! FIXME: vertex buffer bandwidth could be lower; only use UV coords when
   !!! FIXME:  textures are needed. */

// Vertex shader, common values
typedef struct
{
    Float4X4 model;
    Float4X4 projectionAndView;
} D3D11_VertexShaderConstants;

// These should mirror the definitions in D3D11_PixelShader_Common.hlsli
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
} D3D11_PixelShaderConstants;

typedef struct
{
    ID3D11Buffer *constants;
    D3D11_PixelShaderConstants shader_constants;
} D3D11_PixelShaderState;

// Per-vertex data
typedef struct
{
    Float2 pos;
    Float2 tex;
    SDL_FColor color;
} D3D11_VertexPositionColor;

// Per-texture data
typedef struct
{
    int w, h;
    ID3D11Texture2D *mainTexture;
    ID3D11ShaderResourceView *mainTextureResourceView;
    ID3D11RenderTargetView *mainTextureRenderTargetView;
    ID3D11Texture2D *stagingTexture;
    int lockedTexturePositionX;
    int lockedTexturePositionY;
    D3D11_Shader shader;
    const float *YCbCr_matrix;
#ifdef SDL_HAVE_YUV
    // YV12 texture support
    bool yuv;
    ID3D11Texture2D *mainTextureU;
    ID3D11ShaderResourceView *mainTextureResourceViewU;
    ID3D11Texture2D *mainTextureV;
    ID3D11ShaderResourceView *mainTextureResourceViewV;

    // NV12 texture support
    bool nv12;
    ID3D11ShaderResourceView *mainTextureResourceViewNV;

    Uint8 *pixels;
    int pitch;
    SDL_Rect locked_rect;
#endif
} D3D11_TextureData;

// Blend mode data
typedef struct
{
    SDL_BlendMode blendMode;
    ID3D11BlendState *blendState;
} D3D11_BlendMode;

// Private renderer data
typedef struct
{
    SDL_SharedObject *hDXGIMod;
    SDL_SharedObject *hD3D11Mod;
    IDXGIFactory2 *dxgiFactory;
    IDXGIAdapter *dxgiAdapter;
    IDXGIDebug *dxgiDebug;
    ID3D11Device1 *d3dDevice;
    ID3D11DeviceContext1 *d3dContext;
    IDXGISwapChain1 *swapChain;
    DXGI_SWAP_EFFECT swapEffect;
    UINT syncInterval;
    UINT presentFlags;
    ID3D11RenderTargetView *mainRenderTargetView;
    ID3D11RenderTargetView *currentOffscreenRenderTargetView;
    ID3D11InputLayout *inputLayout;
    ID3D11Buffer *vertexBuffers[8];
    size_t vertexBufferSizes[8];
    ID3D11VertexShader *vertexShader;
    ID3D11PixelShader *pixelShaders[NUM_SHADERS];
    int blendModesCount;
    D3D11_BlendMode *blendModes;
    ID3D11SamplerState *samplers[RENDER_SAMPLER_COUNT];
    D3D_FEATURE_LEVEL featureLevel;
    bool pixelSizeChanged;

    // Rasterizers
    ID3D11RasterizerState *mainRasterizer;
    ID3D11RasterizerState *clippedRasterizer;

    // Vertex buffer constants
    D3D11_VertexShaderConstants vertexShaderConstantsData;
    ID3D11Buffer *vertexShaderConstants;

    // Cached renderer properties
    DXGI_MODE_ROTATION rotation;
    ID3D11RenderTargetView *currentRenderTargetView;
    ID3D11RasterizerState *currentRasterizerState;
    ID3D11BlendState *currentBlendState;
    D3D11_Shader currentShader;
    D3D11_PixelShaderState currentShaderState[NUM_SHADERS];
    ID3D11ShaderResourceView *currentShaderResource;
    ID3D11SamplerState *currentSampler;
    bool cliprectDirty;
    bool currentCliprectEnabled;
    SDL_Rect currentCliprect;
    SDL_Rect currentViewport;
    int currentViewportRotation;
    bool viewportDirty;
    Float4X4 identity;
    int currentVertexBuffer;
} D3D11_RenderData;

// Define D3D GUIDs here so we don't have to include uuid.lib.

#ifdef HAVE_GCC_DIAGNOSTIC_PRAGMA
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-const-variable"
#endif

static const GUID SDL_IID_IDXGIFactory2 = { 0x50c83a1c, 0xe072, 0x4c48, { 0x87, 0xb0, 0x36, 0x30, 0xfa, 0x36, 0xa6, 0xd0 } };
static const GUID SDL_IID_IDXGIDevice1 = { 0x77db970f, 0x6276, 0x48ba, { 0xba, 0x28, 0x07, 0x01, 0x43, 0xb4, 0x39, 0x2c } };
static const GUID SDL_IID_ID3D11Texture2D = { 0x6f15aaf2, 0xd208, 0x4e89, { 0x9a, 0xb4, 0x48, 0x95, 0x35, 0xd3, 0x4f, 0x9c } };
static const GUID SDL_IID_ID3D11Device1 = { 0xa04bfb29, 0x08ef, 0x43d6, { 0xa4, 0x9c, 0xa9, 0xbd, 0xbd, 0xcb, 0xe6, 0x86 } };
static const GUID SDL_IID_ID3D11DeviceContext1 = { 0xbb2c6faa, 0xb5fb, 0x4082, { 0x8e, 0x6b, 0x38, 0x8b, 0x8c, 0xfa, 0x90, 0xe1 } };
static const GUID SDL_IID_IDXGISwapChain2 = { 0x94d99bdb, 0xf1f8, 0x4ab0, { 0xb2, 0x36, 0x7d, 0xa0, 0x17, 0x0e, 0xda, 0xb1 } };
static const GUID SDL_IID_IDXGIDebug1 = { 0xc5a05f0c, 0x16f2, 0x4adf, { 0x9f, 0x4d, 0xa8, 0xc4, 0xd5, 0x8a, 0xc5, 0x50 } };
static const GUID SDL_IID_IDXGIInfoQueue = { 0xD67441C7, 0x672A, 0x476f, { 0x9E, 0x82, 0xCD, 0x55, 0xB4, 0x49, 0x49, 0xCE } };
static const GUID SDL_DXGI_DEBUG_ALL = { 0xe48ae283, 0xda80, 0x490b, { 0x87, 0xe6, 0x43, 0xe9, 0xa9, 0xcf, 0xda, 0x8 } };

#ifdef HAVE_GCC_DIAGNOSTIC_PRAGMA
#pragma GCC diagnostic pop
#endif

SDL_PixelFormat D3D11_DXGIFormatToSDLPixelFormat(DXGI_FORMAT dxgiFormat)
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
    case SDL_PIXELFORMAT_NV12:  // For the Y texture
    case SDL_PIXELFORMAT_NV21:  // For the Y texture
        return DXGI_FORMAT_R8_UNORM;
    case SDL_PIXELFORMAT_P010:  // For the Y texture
        return DXGI_FORMAT_R16_UNORM;
    default:
        return DXGI_FORMAT_UNKNOWN;
    }
}

static void D3D11_DestroyTexture(SDL_Renderer *renderer, SDL_Texture *texture);

static void D3D11_ReleaseAll(SDL_Renderer *renderer)
{
    D3D11_RenderData *data = (D3D11_RenderData *)renderer->internal;

    // Release all textures
    for (SDL_Texture *texture = renderer->textures; texture; texture = texture->next) {
        D3D11_DestroyTexture(renderer, texture);
    }

    // Release/reset everything else
    if (data) {
        int i;

        // Make sure the swap chain is fully released
        if (data->d3dContext) {
            ID3D11DeviceContext_ClearState(data->d3dContext);
            ID3D11DeviceContext_Flush(data->d3dContext);
        }

        SAFE_RELEASE(data->vertexShaderConstants);
        SAFE_RELEASE(data->clippedRasterizer);
        SAFE_RELEASE(data->mainRasterizer);
        for (i = 0; i < SDL_arraysize(data->samplers); ++i) {
            SAFE_RELEASE(data->samplers[i]);
        }

        if (data->blendModesCount > 0) {
            for (i = 0; i < data->blendModesCount; ++i) {
                SAFE_RELEASE(data->blendModes[i].blendState);
            }
            SDL_free(data->blendModes);
            data->blendModes = NULL;
            data->blendModesCount = 0;
        }
        for (i = 0; i < SDL_arraysize(data->pixelShaders); ++i) {
            SAFE_RELEASE(data->pixelShaders[i]);
        }
        for (i = 0; i < SDL_arraysize(data->currentShaderState); ++i) {
            SAFE_RELEASE(data->currentShaderState[i].constants);
        }
        SAFE_RELEASE(data->vertexShader);
        for (i = 0; i < SDL_arraysize(data->vertexBuffers); ++i) {
            SAFE_RELEASE(data->vertexBuffers[i]);
        }
        SAFE_RELEASE(data->inputLayout);
        SAFE_RELEASE(data->mainRenderTargetView);
        SAFE_RELEASE(data->swapChain);

        SAFE_RELEASE(data->d3dContext);
        SAFE_RELEASE(data->d3dDevice);
        SAFE_RELEASE(data->dxgiAdapter);
        SAFE_RELEASE(data->dxgiFactory);

        data->swapEffect = (DXGI_SWAP_EFFECT)0;
        data->rotation = DXGI_MODE_ROTATION_UNSPECIFIED;
        data->currentOffscreenRenderTargetView = NULL;
        data->currentRenderTargetView = NULL;
        data->currentRasterizerState = NULL;
        data->currentBlendState = NULL;
        data->currentShader = SHADER_NONE;
        SDL_zero(data->currentShaderState);
        data->currentShaderResource = NULL;
        data->currentSampler = NULL;

        // Check for any leaks if in debug mode
        if (data->dxgiDebug) {
            DXGI_DEBUG_RLO_FLAGS rloFlags = (DXGI_DEBUG_RLO_FLAGS)(DXGI_DEBUG_RLO_DETAIL | DXGI_DEBUG_RLO_IGNORE_INTERNAL);
            IDXGIDebug_ReportLiveObjects(data->dxgiDebug, SDL_DXGI_DEBUG_ALL, rloFlags);
            SAFE_RELEASE(data->dxgiDebug);
        }

        /* Unload the D3D libraries.  This should be done last, in order
         * to prevent IUnknown::Release() calls from crashing.
         */
        if (data->hD3D11Mod) {
            SDL_UnloadObject(data->hD3D11Mod);
            data->hD3D11Mod = NULL;
        }
        if (data->hDXGIMod) {
            SDL_UnloadObject(data->hDXGIMod);
            data->hDXGIMod = NULL;
        }
    }
}

static void D3D11_DestroyRenderer(SDL_Renderer *renderer)
{
    D3D11_RenderData *data = (D3D11_RenderData *)renderer->internal;
    if (data) {
        D3D11_ReleaseAll(renderer);
        SDL_free(data);
    }
}

static D3D11_BLEND GetBlendFunc(SDL_BlendFactor factor)
{
    switch (factor) {
    case SDL_BLENDFACTOR_ZERO:
        return D3D11_BLEND_ZERO;
    case SDL_BLENDFACTOR_ONE:
        return D3D11_BLEND_ONE;
    case SDL_BLENDFACTOR_SRC_COLOR:
        return D3D11_BLEND_SRC_COLOR;
    case SDL_BLENDFACTOR_ONE_MINUS_SRC_COLOR:
        return D3D11_BLEND_INV_SRC_COLOR;
    case SDL_BLENDFACTOR_SRC_ALPHA:
        return D3D11_BLEND_SRC_ALPHA;
    case SDL_BLENDFACTOR_ONE_MINUS_SRC_ALPHA:
        return D3D11_BLEND_INV_SRC_ALPHA;
    case SDL_BLENDFACTOR_DST_COLOR:
        return D3D11_BLEND_DEST_COLOR;
    case SDL_BLENDFACTOR_ONE_MINUS_DST_COLOR:
        return D3D11_BLEND_INV_DEST_COLOR;
    case SDL_BLENDFACTOR_DST_ALPHA:
        return D3D11_BLEND_DEST_ALPHA;
    case SDL_BLENDFACTOR_ONE_MINUS_DST_ALPHA:
        return D3D11_BLEND_INV_DEST_ALPHA;
    default:
        return (D3D11_BLEND)0;
    }
}

static D3D11_BLEND_OP GetBlendEquation(SDL_BlendOperation operation)
{
    switch (operation) {
    case SDL_BLENDOPERATION_ADD:
        return D3D11_BLEND_OP_ADD;
    case SDL_BLENDOPERATION_SUBTRACT:
        return D3D11_BLEND_OP_SUBTRACT;
    case SDL_BLENDOPERATION_REV_SUBTRACT:
        return D3D11_BLEND_OP_REV_SUBTRACT;
    case SDL_BLENDOPERATION_MINIMUM:
        return D3D11_BLEND_OP_MIN;
    case SDL_BLENDOPERATION_MAXIMUM:
        return D3D11_BLEND_OP_MAX;
    default:
        return (D3D11_BLEND_OP)0;
    }
}

static ID3D11BlendState *D3D11_CreateBlendState(SDL_Renderer *renderer, SDL_BlendMode blendMode)
{
    D3D11_RenderData *data = (D3D11_RenderData *)renderer->internal;
    SDL_BlendFactor srcColorFactor = SDL_GetBlendModeSrcColorFactor(blendMode);
    SDL_BlendFactor srcAlphaFactor = SDL_GetBlendModeSrcAlphaFactor(blendMode);
    SDL_BlendOperation colorOperation = SDL_GetBlendModeColorOperation(blendMode);
    SDL_BlendFactor dstColorFactor = SDL_GetBlendModeDstColorFactor(blendMode);
    SDL_BlendFactor dstAlphaFactor = SDL_GetBlendModeDstAlphaFactor(blendMode);
    SDL_BlendOperation alphaOperation = SDL_GetBlendModeAlphaOperation(blendMode);
    ID3D11BlendState *blendState = NULL;
    D3D11_BlendMode *blendModes;
    HRESULT result = S_OK;

    D3D11_BLEND_DESC blendDesc;
    SDL_zero(blendDesc);
    blendDesc.AlphaToCoverageEnable = FALSE;
    blendDesc.IndependentBlendEnable = FALSE;
    blendDesc.RenderTarget[0].BlendEnable = TRUE;
    blendDesc.RenderTarget[0].SrcBlend = GetBlendFunc(srcColorFactor);
    blendDesc.RenderTarget[0].DestBlend = GetBlendFunc(dstColorFactor);
    blendDesc.RenderTarget[0].BlendOp = GetBlendEquation(colorOperation);
    blendDesc.RenderTarget[0].SrcBlendAlpha = GetBlendFunc(srcAlphaFactor);
    blendDesc.RenderTarget[0].DestBlendAlpha = GetBlendFunc(dstAlphaFactor);
    blendDesc.RenderTarget[0].BlendOpAlpha = GetBlendEquation(alphaOperation);
    blendDesc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    result = ID3D11Device_CreateBlendState(data->d3dDevice, &blendDesc, &blendState);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D11Device1::CreateBlendState"), result);
        return NULL;
    }

    blendModes = (D3D11_BlendMode *)SDL_realloc(data->blendModes, (data->blendModesCount + 1) * sizeof(*blendModes));
    if (!blendModes) {
        SAFE_RELEASE(blendState);
        return NULL;
    }
    blendModes[data->blendModesCount].blendMode = blendMode;
    blendModes[data->blendModesCount].blendState = blendState;
    data->blendModes = blendModes;
    ++data->blendModesCount;

    return blendState;
}

// Create resources that depend on the device.
static HRESULT D3D11_CreateDeviceResources(SDL_Renderer *renderer)
{
    typedef HRESULT(WINAPI * PFN_CREATE_DXGI_FACTORY)(REFIID riid, void **ppFactory);
    typedef HRESULT(WINAPI * PFN_CREATE_DXGI_FACTORY2)(UINT flags, REFIID riid, void **ppFactory);
    PFN_CREATE_DXGI_FACTORY CreateDXGIFactoryFunc = NULL;
    PFN_CREATE_DXGI_FACTORY2 CreateDXGIFactory2Func = NULL;
    D3D11_RenderData *data = (D3D11_RenderData *)renderer->internal;
    PFN_D3D11_CREATE_DEVICE D3D11CreateDeviceFunc;
    ID3D11Device *d3dDevice = NULL;
    ID3D11DeviceContext *d3dContext = NULL;
    IDXGIDevice1 *dxgiDevice = NULL;
    HRESULT result = S_OK;
    UINT creationFlags = 0;
    bool createDebug;

    /* This array defines the set of DirectX hardware feature levels this app will support.
     * Note the ordering should be preserved.
     * Don't forget to declare your application's minimum required feature level in its
     * description.  All applications are assumed to support 9.1 unless otherwise stated.
     */
    D3D_FEATURE_LEVEL featureLevels[] = {
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0,
        D3D_FEATURE_LEVEL_9_3,
        D3D_FEATURE_LEVEL_9_2,
        D3D_FEATURE_LEVEL_9_1
    };

    D3D11_BUFFER_DESC constantBufferDesc;
    D3D11_RASTERIZER_DESC rasterDesc;

    // See if we need debug interfaces
    createDebug = SDL_GetHintBoolean(SDL_HINT_RENDER_DIRECT3D11_DEBUG, false);

    data->hDXGIMod = SDL_LoadObject("dxgi.dll");
    if (!data->hDXGIMod) {
        result = E_FAIL;
        goto done;
    }

    CreateDXGIFactory2Func = (PFN_CREATE_DXGI_FACTORY2)SDL_LoadFunction(data->hDXGIMod, "CreateDXGIFactory2");
    if (!CreateDXGIFactory2Func) {
        CreateDXGIFactoryFunc = (PFN_CREATE_DXGI_FACTORY)SDL_LoadFunction(data->hDXGIMod, "CreateDXGIFactory");
        if (!CreateDXGIFactoryFunc) {
            result = E_FAIL;
            goto done;
        }
    }

    data->hD3D11Mod = SDL_LoadObject("d3d11.dll");
    if (!data->hD3D11Mod) {
        result = E_FAIL;
        goto done;
    }

    D3D11CreateDeviceFunc = (PFN_D3D11_CREATE_DEVICE)SDL_LoadFunction(data->hD3D11Mod, "D3D11CreateDevice");
    if (!D3D11CreateDeviceFunc) {
        result = E_FAIL;
        goto done;
    }

    if (createDebug) {
#ifdef __IDXGIInfoQueue_INTERFACE_DEFINED__
        IDXGIInfoQueue *dxgiInfoQueue = NULL;
        PFN_CREATE_DXGI_FACTORY2 DXGIGetDebugInterfaceFunc;

        // If the debug hint is set, also create the DXGI factory in debug mode
        DXGIGetDebugInterfaceFunc = (PFN_CREATE_DXGI_FACTORY2)SDL_LoadFunction(data->hDXGIMod, "DXGIGetDebugInterface1");
        if (!DXGIGetDebugInterfaceFunc) {
            result = E_FAIL;
            goto done;
        }

        result = DXGIGetDebugInterfaceFunc(0, &SDL_IID_IDXGIDebug1, (void **)&data->dxgiDebug);
        if (FAILED(result)) {
            WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("DXGIGetDebugInterface1"), result);
            goto done;
        }

        result = DXGIGetDebugInterfaceFunc(0, &SDL_IID_IDXGIInfoQueue, (void **)&dxgiInfoQueue);
        if (FAILED(result)) {
            WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("DXGIGetDebugInterface1"), result);
            goto done;
        }

        IDXGIInfoQueue_SetBreakOnSeverity(dxgiInfoQueue, SDL_DXGI_DEBUG_ALL, DXGI_INFO_QUEUE_MESSAGE_SEVERITY_ERROR, TRUE);
        IDXGIInfoQueue_SetBreakOnSeverity(dxgiInfoQueue, SDL_DXGI_DEBUG_ALL, DXGI_INFO_QUEUE_MESSAGE_SEVERITY_CORRUPTION, TRUE);
        SAFE_RELEASE(dxgiInfoQueue);
#endif // __IDXGIInfoQueue_INTERFACE_DEFINED__
        creationFlags = DXGI_CREATE_FACTORY_DEBUG;
    }

    if (CreateDXGIFactory2Func) {
        result = CreateDXGIFactory2Func(creationFlags, &SDL_IID_IDXGIFactory2, (void **)&data->dxgiFactory);
    } else {
        result = CreateDXGIFactoryFunc(&SDL_IID_IDXGIFactory2, (void **)&data->dxgiFactory);
    }
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("CreateDXGIFactory"), result);
        goto done;
    }

    // FIXME: Should we use the default adapter?
    result = IDXGIFactory2_EnumAdapters(data->dxgiFactory, 0, &data->dxgiAdapter);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("D3D11CreateDevice"), result);
        goto done;
    }

    /* This flag adds support for surfaces with a different color channel ordering
     * than the API default. It is required for compatibility with Direct2D.
     */
    creationFlags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;

    // Make sure Direct3D's debugging feature gets used, if the app requests it.
    if (createDebug) {
        creationFlags |= D3D11_CREATE_DEVICE_DEBUG;
    }

    // Create a single-threaded device unless the app requests otherwise.
    if (!SDL_GetHintBoolean(SDL_HINT_RENDER_DIRECT3D_THREADSAFE, false)) {
        creationFlags |= D3D11_CREATE_DEVICE_SINGLETHREADED;
    }

    // Create the Direct3D 11 API device object and a corresponding context.
    result = D3D11CreateDeviceFunc(
        data->dxgiAdapter,
        D3D_DRIVER_TYPE_UNKNOWN,
        NULL,
        creationFlags, // Set set debug and Direct2D compatibility flags.
        featureLevels, // List of feature levels this app can support.
        SDL_arraysize(featureLevels),
        D3D11_SDK_VERSION,   // Always set this to D3D11_SDK_VERSION for Windows Store apps.
        &d3dDevice,          // Returns the Direct3D device created.
        &data->featureLevel, // Returns feature level of device created.
        &d3dContext          // Returns the device immediate context.
    );
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("D3D11CreateDevice"), result);
        goto done;
    }

    result = ID3D11Device_QueryInterface(d3dDevice, &SDL_IID_ID3D11Device1, (void **)&data->d3dDevice);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D11Device to ID3D11Device1"), result);
        goto done;
    }

    result = ID3D11DeviceContext_QueryInterface(d3dContext, &SDL_IID_ID3D11DeviceContext1, (void **)&data->d3dContext);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D11DeviceContext to ID3D11DeviceContext1"), result);
        goto done;
    }

    result = ID3D11Device_QueryInterface(d3dDevice, &SDL_IID_IDXGIDevice1, (void **)&dxgiDevice);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D11Device to IDXGIDevice1"), result);
        goto done;
    }

    /* Ensure that DXGI does not queue more than one frame at a time. This both reduces latency and
     * ensures that the application will only render after each VSync, minimizing power consumption.
     */
    result = IDXGIDevice1_SetMaximumFrameLatency(dxgiDevice, 1);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("IDXGIDevice1::SetMaximumFrameLatency"), result);
        goto done;
    }

    /* Make note of the maximum texture size
     * Max texture sizes are documented on MSDN, at:
     * http://msdn.microsoft.com/en-us/library/windows/apps/ff476876.aspx
     */
    switch (data->featureLevel) {
    case D3D_FEATURE_LEVEL_11_1:
    case D3D_FEATURE_LEVEL_11_0:
        SDL_SetNumberProperty(SDL_GetRendererProperties(renderer), SDL_PROP_RENDERER_MAX_TEXTURE_SIZE_NUMBER, 16384);
        break;

    case D3D_FEATURE_LEVEL_10_1:
    case D3D_FEATURE_LEVEL_10_0:
        SDL_SetNumberProperty(SDL_GetRendererProperties(renderer), SDL_PROP_RENDERER_MAX_TEXTURE_SIZE_NUMBER, 8192);
        break;

    case D3D_FEATURE_LEVEL_9_3:
        SDL_SetNumberProperty(SDL_GetRendererProperties(renderer), SDL_PROP_RENDERER_MAX_TEXTURE_SIZE_NUMBER, 4096);
        break;

    case D3D_FEATURE_LEVEL_9_2:
    case D3D_FEATURE_LEVEL_9_1:
        SDL_SetNumberProperty(SDL_GetRendererProperties(renderer), SDL_PROP_RENDERER_MAX_TEXTURE_SIZE_NUMBER, 2048);
        break;

    default:
        SDL_SetError("%s, Unexpected feature level: %d", __FUNCTION__, data->featureLevel);
        result = E_FAIL;
        goto done;
    }

    if (!D3D11_CreateVertexShader(data->d3dDevice, &data->vertexShader, &data->inputLayout)) {
        goto done;
    }

    // Setup space to hold vertex shader constants:
    SDL_zero(constantBufferDesc);
    constantBufferDesc.ByteWidth = sizeof(D3D11_VertexShaderConstants);
    constantBufferDesc.Usage = D3D11_USAGE_DEFAULT;
    constantBufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    result = ID3D11Device_CreateBuffer(data->d3dDevice,
                                       &constantBufferDesc,
                                       NULL,
                                       &data->vertexShaderConstants);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D11Device1::CreateBuffer [vertex shader constants]"), result);
        goto done;
    }

    // Setup Direct3D rasterizer states
    SDL_zero(rasterDesc);
    rasterDesc.AntialiasedLineEnable = FALSE;
    rasterDesc.CullMode = D3D11_CULL_NONE;
    rasterDesc.DepthBias = 0;
    rasterDesc.DepthBiasClamp = 0.0f;
    rasterDesc.DepthClipEnable = TRUE;
    rasterDesc.FillMode = D3D11_FILL_SOLID;
    rasterDesc.FrontCounterClockwise = FALSE;
    rasterDesc.MultisampleEnable = FALSE;
    rasterDesc.ScissorEnable = FALSE;
    rasterDesc.SlopeScaledDepthBias = 0.0f;
    result = ID3D11Device_CreateRasterizerState(data->d3dDevice, &rasterDesc, &data->mainRasterizer);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D11Device1::CreateRasterizerState [main rasterizer]"), result);
        goto done;
    }

    rasterDesc.ScissorEnable = TRUE;
    result = ID3D11Device_CreateRasterizerState(data->d3dDevice, &rasterDesc, &data->clippedRasterizer);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D11Device1::CreateRasterizerState [clipped rasterizer]"), result);
        goto done;
    }

    // Create blending states:
    if (!D3D11_CreateBlendState(renderer, SDL_BLENDMODE_BLEND)) {
        // D3D11_CreateBlendState will set the SDL error, if it fails
        goto done;
    }

    // Setup render state that doesn't change
    ID3D11DeviceContext_IASetInputLayout(data->d3dContext, data->inputLayout);
    ID3D11DeviceContext_VSSetShader(data->d3dContext, data->vertexShader, NULL, 0);
    ID3D11DeviceContext_VSSetConstantBuffers(data->d3dContext, 0, 1, &data->vertexShaderConstants);

    SDL_SetPointerProperty(SDL_GetRendererProperties(renderer), SDL_PROP_RENDERER_D3D11_DEVICE_POINTER, data->d3dDevice);

done:
    SAFE_RELEASE(d3dDevice);
    SAFE_RELEASE(d3dContext);
    SAFE_RELEASE(dxgiDevice);
    return result;
}

static DXGI_MODE_ROTATION D3D11_GetCurrentRotation(void)
{
    // FIXME
    return DXGI_MODE_ROTATION_IDENTITY;
}

static BOOL D3D11_IsDisplayRotated90Degrees(DXGI_MODE_ROTATION rotation)
{
    switch (rotation) {
    case DXGI_MODE_ROTATION_ROTATE90:
    case DXGI_MODE_ROTATION_ROTATE270:
        return TRUE;
    default:
        return FALSE;
    }
}

static int D3D11_GetRotationForCurrentRenderTarget(SDL_Renderer *renderer)
{
    D3D11_RenderData *data = (D3D11_RenderData *)renderer->internal;
    if (data->currentOffscreenRenderTargetView) {
        return DXGI_MODE_ROTATION_IDENTITY;
    } else {
        return data->rotation;
    }
}

static bool D3D11_GetViewportAlignedD3DRect(SDL_Renderer *renderer, const SDL_Rect *sdlRect, D3D11_RECT *outRect, BOOL includeViewportOffset)
{
    D3D11_RenderData *data = (D3D11_RenderData *)renderer->internal;
    const int rotation = D3D11_GetRotationForCurrentRenderTarget(renderer);
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

static HRESULT D3D11_CreateSwapChain(SDL_Renderer *renderer, int w, int h)
{
    D3D11_RenderData *data = (D3D11_RenderData *)renderer->internal;
    IUnknown *coreWindow = NULL;
    IDXGISwapChain3 *swapChain3 = NULL;
    HRESULT result = S_OK;

    // Create a swap chain using the same adapter as the existing Direct3D device.
    DXGI_SWAP_CHAIN_DESC1 swapChainDesc;
    SDL_zero(swapChainDesc);
    swapChainDesc.Width = w;
    swapChainDesc.Height = h;
    switch (renderer->output_colorspace) {
    case SDL_COLORSPACE_SRGB_LINEAR:
        swapChainDesc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        break;
    case SDL_COLORSPACE_HDR10:
        swapChainDesc.Format = DXGI_FORMAT_R10G10B10A2_UNORM;
        break;
    default:
        swapChainDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM; // This is the most common swap chain format.
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
    if (SDL_GetWindowFlags(renderer->window) & SDL_WINDOW_TRANSPARENT) {
        swapChainDesc.Scaling = DXGI_SCALING_STRETCH;
        swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
    } else {
        swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL; // All Windows Store apps must use this SwapEffect.
    }
    swapChainDesc.Flags = 0;

    if (coreWindow) {
        result = IDXGIFactory2_CreateSwapChainForCoreWindow(data->dxgiFactory,
                                                            (IUnknown *)data->d3dDevice,
                                                            coreWindow,
                                                            &swapChainDesc,
                                                            NULL, // Allow on all displays.
                                                            &data->swapChain);
        if (FAILED(result)) {
            WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("IDXGIFactory2::CreateSwapChainForCoreWindow"), result);
            goto done;
        }
    } else {
#if defined(SDL_PLATFORM_WIN32) || defined(SDL_PLATFORM_WINGDK)
        HWND hwnd = (HWND)SDL_GetPointerProperty(SDL_GetWindowProperties(renderer->window), SDL_PROP_WINDOW_WIN32_HWND_POINTER, NULL);
        if (!hwnd) {
            SDL_SetError("Couldn't get window handle");
            result = E_FAIL;
            goto done;
        }

        result = IDXGIFactory2_CreateSwapChainForHwnd(data->dxgiFactory,
                                                      (IUnknown *)data->d3dDevice,
                                                      hwnd,
                                                      &swapChainDesc,
                                                      NULL,
                                                      NULL, // Allow on all displays.
                                                      &data->swapChain);
        if (FAILED(result)) {
            WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("IDXGIFactory2::CreateSwapChainForHwnd"), result);
            goto done;
        }

        IDXGIFactory_MakeWindowAssociation(data->dxgiFactory, hwnd, DXGI_MWA_NO_WINDOW_CHANGES);
#else
        SDL_SetError(__FUNCTION__ ", Unable to find something to attach a swap chain to");
        goto done;
#endif // defined(SDL_PLATFORM_WIN32) || defined(SDL_PLATFORM_WINGDK) / else
    }
    data->swapEffect = swapChainDesc.SwapEffect;

    if (SUCCEEDED(IDXGISwapChain1_QueryInterface(data->swapChain, &SDL_IID_IDXGISwapChain2, (void **)&swapChain3))) {
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
        if (SUCCEEDED(IDXGISwapChain3_CheckColorSpaceSupport(swapChain3, colorspace, &colorspace_support)) &&
            (colorspace_support & DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT)) {
            result = IDXGISwapChain3_SetColorSpace1(swapChain3, colorspace);
            if (FAILED(result)) {
                WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("IDXGISwapChain3::SetColorSpace1"), result);
                goto done;
            }
        } else if (colorspace != DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709) {
            // Not the default, we're not going to be able to present in this colorspace
            SDL_SetError("Unsupported output colorspace");
            result = DXGI_ERROR_UNSUPPORTED;
        }
    }

    SDL_SetPointerProperty(SDL_GetRendererProperties(renderer), SDL_PROP_RENDERER_D3D11_SWAPCHAIN_POINTER, data->swapChain);

done:
    SAFE_RELEASE(swapChain3);
    SAFE_RELEASE(coreWindow);
    return result;
}

static void D3D11_ReleaseMainRenderTargetView(SDL_Renderer *renderer)
{
    D3D11_RenderData *data = (D3D11_RenderData *)renderer->internal;
    ID3D11DeviceContext_OMSetRenderTargets(data->d3dContext, 0, NULL, NULL);
    SAFE_RELEASE(data->mainRenderTargetView);
}

// Initialize all resources that change when the window's size changes.
static HRESULT D3D11_CreateWindowSizeDependentResources(SDL_Renderer *renderer)
{
    D3D11_RenderData *data = (D3D11_RenderData *)renderer->internal;
    ID3D11Texture2D *backBuffer = NULL;
    HRESULT result = S_OK;
    int w, h;

    // Release the previous render target view
    D3D11_ReleaseMainRenderTargetView(renderer);

    /* The width and height of the swap chain must be based on the display's
     * non-rotated size.
     */
    SDL_GetWindowSizeInPixels(renderer->window, &w, &h);
    data->rotation = D3D11_GetCurrentRotation();
    // SDL_Log("%s: windowSize={%d,%d}, orientation=%d", __FUNCTION__, w, h, (int)data->rotation);
    if (D3D11_IsDisplayRotated90Degrees(data->rotation)) {
        int tmp = w;
        w = h;
        h = tmp;
    }

    if (data->swapChain) {
        // If the swap chain already exists, resize it.
        result = IDXGISwapChain_ResizeBuffers(data->swapChain,
                                              0,
                                              w, h,
                                              DXGI_FORMAT_UNKNOWN,
                                              0);
        if (FAILED(result)) {
            WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("IDXGISwapChain::ResizeBuffers"), result);
            goto done;
        }
    } else {
        result = D3D11_CreateSwapChain(renderer, w, h);
        if (FAILED(result) || !data->swapChain) {
            goto done;
        }
    }

    // Set the proper rotation for the swap chain.
    if (WIN_IsWindows8OrGreater()) {
        if (data->swapEffect == DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL) {
            result = IDXGISwapChain1_SetRotation(data->swapChain, data->rotation);
            if (FAILED(result)) {
                WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("IDXGISwapChain1::SetRotation"), result);
                goto done;
            }
        }
    }

    result = IDXGISwapChain_GetBuffer(data->swapChain,
                                      0,
                                      &SDL_IID_ID3D11Texture2D,
                                      (void **)&backBuffer);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("IDXGISwapChain::GetBuffer [back-buffer]"), result);
        goto done;
    }

    // Create a render target view of the swap chain back buffer.
    result = ID3D11Device_CreateRenderTargetView(data->d3dDevice,
                                                 (ID3D11Resource *)backBuffer,
                                                 NULL,
                                                 &data->mainRenderTargetView);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D11Device::CreateRenderTargetView"), result);
        goto done;
    }

    /* Set the swap chain target immediately, so that a target is always set
     * even before we get to SetDrawState. Without this it's possible to hit
     * null references in places like ReadPixels!
     */
    ID3D11DeviceContext_OMSetRenderTargets(data->d3dContext,
                                           1,
                                           &data->mainRenderTargetView,
                                           NULL);

    data->viewportDirty = true;

done:
    SAFE_RELEASE(backBuffer);
    return result;
}

static bool D3D11_HandleDeviceLost(SDL_Renderer *renderer)
{
    bool recovered = false;

    D3D11_ReleaseAll(renderer);

    if (SUCCEEDED(D3D11_CreateDeviceResources(renderer)) &&
        SUCCEEDED(D3D11_CreateWindowSizeDependentResources(renderer))) {
        recovered = true;
    } else {
        SDL_LogError(SDL_LOG_CATEGORY_RENDER, "Renderer couldn't recover from device lost: %s", SDL_GetError());
        D3D11_ReleaseAll(renderer);
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
static HRESULT D3D11_UpdateForWindowSizeChange(SDL_Renderer *renderer)
{
    return D3D11_CreateWindowSizeDependentResources(renderer);
}

static void D3D11_WindowEvent(SDL_Renderer *renderer, const SDL_WindowEvent *event)
{
    D3D11_RenderData *data = (D3D11_RenderData *)renderer->internal;

    if (event->type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED) {
        data->pixelSizeChanged = true;
    }
}

static bool D3D11_SupportsBlendMode(SDL_Renderer *renderer, SDL_BlendMode blendMode)
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

static bool GetTextureProperty(SDL_PropertiesID props, const char *name, ID3D11Texture2D **texture)
{
    IUnknown *unknown = SDL_GetPointerProperty(props, name, NULL);
    if (unknown) {
        HRESULT result = IUnknown_QueryInterface(unknown, &SDL_IID_ID3D11Texture2D, (void **)texture);
        if (FAILED(result)) {
            return WIN_SetErrorFromHRESULT(name, result);
        }
    }
    return true;
}

static bool D3D11_CreateTexture(SDL_Renderer *renderer, SDL_Texture *texture, SDL_PropertiesID create_props)
{
    D3D11_RenderData *rendererData = (D3D11_RenderData *)renderer->internal;
    D3D11_TextureData *textureData;
    HRESULT result;
    DXGI_FORMAT textureFormat = SDLPixelFormatToDXGITextureFormat(texture->format, renderer->output_colorspace);
    D3D11_TEXTURE2D_DESC textureDesc;
    D3D11_SHADER_RESOURCE_VIEW_DESC resourceViewDesc;

    if (!rendererData->d3dDevice) {
        return SDL_SetError("Device lost and couldn't be recovered");
    }

    if (textureFormat == DXGI_FORMAT_UNKNOWN) {
        return SDL_SetError("%s, An unsupported SDL pixel format (0x%x) was specified",
                            __FUNCTION__, texture->format);
    }

    textureData = (D3D11_TextureData *)SDL_calloc(1, sizeof(*textureData));
    if (!textureData) {
        return false;
    }

    texture->internal = textureData;

    SDL_zero(textureDesc);
    textureDesc.Width = texture->w;
    textureDesc.Height = texture->h;
    textureDesc.MipLevels = 1;
    textureDesc.ArraySize = 1;
    textureDesc.Format = textureFormat;
    textureDesc.SampleDesc.Count = 1;
    textureDesc.SampleDesc.Quality = 0;
    textureDesc.MiscFlags = 0;

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

    if (texture->access == SDL_TEXTUREACCESS_STREAMING) {
        textureDesc.Usage = D3D11_USAGE_DYNAMIC;
        textureDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    } else {
        textureDesc.Usage = D3D11_USAGE_DEFAULT;
        textureDesc.CPUAccessFlags = 0;
    }

    if (texture->access == SDL_TEXTUREACCESS_TARGET) {
        textureDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
    } else {
        textureDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    }

    if (!GetTextureProperty(create_props, SDL_PROP_TEXTURE_CREATE_D3D11_TEXTURE_POINTER, &textureData->mainTexture)) {
        return false;
    }
    if (!textureData->mainTexture) {
        result = ID3D11Device_CreateTexture2D(rendererData->d3dDevice,
                                              &textureDesc,
                                              NULL,
                                              &textureData->mainTexture);
        if (FAILED(result)) {
            return WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D11Device1::CreateTexture2D"), result);
        }
    }
    SDL_SetPointerProperty(SDL_GetTextureProperties(texture), SDL_PROP_TEXTURE_D3D11_TEXTURE_POINTER, textureData->mainTexture);
#ifdef SDL_HAVE_YUV
    if (texture->format == SDL_PIXELFORMAT_YV12 ||
        texture->format == SDL_PIXELFORMAT_IYUV) {
        textureData->yuv = true;

        textureDesc.Width = (textureDesc.Width + 1) / 2;
        textureDesc.Height = (textureDesc.Height + 1) / 2;

        if (!GetTextureProperty(create_props, SDL_PROP_TEXTURE_CREATE_D3D11_TEXTURE_U_POINTER, &textureData->mainTextureU)) {
            return false;
        }
        if (!textureData->mainTextureU) {
            result = ID3D11Device_CreateTexture2D(rendererData->d3dDevice,
                                                  &textureDesc,
                                                  NULL,
                                                  &textureData->mainTextureU);
            if (FAILED(result)) {
                return WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D11Device1::CreateTexture2D"), result);
            }
        }
        SDL_SetPointerProperty(SDL_GetTextureProperties(texture), SDL_PROP_TEXTURE_D3D11_TEXTURE_U_POINTER, textureData->mainTextureU);

        if (!GetTextureProperty(create_props, SDL_PROP_TEXTURE_CREATE_D3D11_TEXTURE_V_POINTER, &textureData->mainTextureV)) {
            return false;
        }
        if (!textureData->mainTextureV) {
            result = ID3D11Device_CreateTexture2D(rendererData->d3dDevice,
                                                  &textureDesc,
                                                  NULL,
                                                  &textureData->mainTextureV);
            if (FAILED(result)) {
                return WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D11Device1::CreateTexture2D"), result);
            }
        }
        SDL_SetPointerProperty(SDL_GetTextureProperties(texture), SDL_PROP_TEXTURE_D3D11_TEXTURE_V_POINTER, textureData->mainTextureV);

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
    resourceViewDesc.Format = SDLPixelFormatToDXGIMainResourceViewFormat(texture->format, renderer->output_colorspace);
    resourceViewDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    resourceViewDesc.Texture2D.MostDetailedMip = 0;
    resourceViewDesc.Texture2D.MipLevels = textureDesc.MipLevels;
    result = ID3D11Device_CreateShaderResourceView(rendererData->d3dDevice,
                                                   (ID3D11Resource *)textureData->mainTexture,
                                                   &resourceViewDesc,
                                                   &textureData->mainTextureResourceView);
    if (FAILED(result)) {
        return WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D11Device1::CreateShaderResourceView"), result);
    }
#ifdef SDL_HAVE_YUV
    if (textureData->yuv) {
        result = ID3D11Device_CreateShaderResourceView(rendererData->d3dDevice,
                                                       (ID3D11Resource *)textureData->mainTextureU,
                                                       &resourceViewDesc,
                                                       &textureData->mainTextureResourceViewU);
        if (FAILED(result)) {
            return WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D11Device1::CreateShaderResourceView"), result);
        }
        result = ID3D11Device_CreateShaderResourceView(rendererData->d3dDevice,
                                                       (ID3D11Resource *)textureData->mainTextureV,
                                                       &resourceViewDesc,
                                                       &textureData->mainTextureResourceViewV);
        if (FAILED(result)) {
            return WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D11Device1::CreateShaderResourceView"), result);
        }
    }

    if (textureData->nv12) {
        D3D11_SHADER_RESOURCE_VIEW_DESC nvResourceViewDesc = resourceViewDesc;

        if (texture->format == SDL_PIXELFORMAT_NV12 || texture->format == SDL_PIXELFORMAT_NV21) {
            nvResourceViewDesc.Format = DXGI_FORMAT_R8G8_UNORM;
        } else if (texture->format == SDL_PIXELFORMAT_P010) {
            nvResourceViewDesc.Format = DXGI_FORMAT_R16G16_UNORM;
        }

        result = ID3D11Device_CreateShaderResourceView(rendererData->d3dDevice,
                                                       (ID3D11Resource *)textureData->mainTexture,
                                                       &nvResourceViewDesc,
                                                       &textureData->mainTextureResourceViewNV);
        if (FAILED(result)) {
            return WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D11Device1::CreateShaderResourceView"), result);
        }
    }
#endif // SDL_HAVE_YUV

    if (texture->access & SDL_TEXTUREACCESS_TARGET) {
        D3D11_RENDER_TARGET_VIEW_DESC renderTargetViewDesc;
        SDL_zero(renderTargetViewDesc);
        renderTargetViewDesc.Format = textureDesc.Format;
        renderTargetViewDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
        renderTargetViewDesc.Texture2D.MipSlice = 0;

        result = ID3D11Device_CreateRenderTargetView(rendererData->d3dDevice,
                                                     (ID3D11Resource *)textureData->mainTexture,
                                                     &renderTargetViewDesc,
                                                     &textureData->mainTextureRenderTargetView);
        if (FAILED(result)) {
            return WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D11Device1::CreateRenderTargetView"), result);
        }
    }

    return true;
}

static void D3D11_DestroyTexture(SDL_Renderer *renderer,
                                 SDL_Texture *texture)
{
    D3D11_TextureData *data = (D3D11_TextureData *)texture->internal;

    if (!data) {
        return;
    }

    SAFE_RELEASE(data->mainTexture);
    SAFE_RELEASE(data->mainTextureResourceView);
    SAFE_RELEASE(data->mainTextureRenderTargetView);
    SAFE_RELEASE(data->stagingTexture);
#ifdef SDL_HAVE_YUV
    SAFE_RELEASE(data->mainTextureU);
    SAFE_RELEASE(data->mainTextureResourceViewU);
    SAFE_RELEASE(data->mainTextureV);
    SAFE_RELEASE(data->mainTextureResourceViewV);
    SAFE_RELEASE(data->mainTextureResourceViewNV);
    SDL_free(data->pixels);
#endif
    SDL_free(data);
    texture->internal = NULL;
}

static bool D3D11_UpdateTextureInternal(D3D11_RenderData *rendererData, ID3D11Texture2D *texture, int bpp, int x, int y, int w, int h, const void *pixels, int pitch)
{
    ID3D11Texture2D *stagingTexture;
    const Uint8 *src;
    Uint8 *dst;
    int row;
    UINT length;
    HRESULT result;
    D3D11_TEXTURE2D_DESC stagingTextureDesc;
    D3D11_MAPPED_SUBRESOURCE textureMemory;

    // Create a 'staging' texture, which will be used to write to a portion of the main texture.
    ID3D11Texture2D_GetDesc(texture, &stagingTextureDesc);
    stagingTextureDesc.Width = w;
    stagingTextureDesc.Height = h;
    stagingTextureDesc.BindFlags = 0;
    stagingTextureDesc.MiscFlags = 0;
    stagingTextureDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    stagingTextureDesc.Usage = D3D11_USAGE_STAGING;
    if (stagingTextureDesc.Format == DXGI_FORMAT_NV12 ||
        stagingTextureDesc.Format == DXGI_FORMAT_P010) {
        stagingTextureDesc.Width = (stagingTextureDesc.Width + 1) & ~1;
        stagingTextureDesc.Height = (stagingTextureDesc.Height + 1) & ~1;
    }
    result = ID3D11Device_CreateTexture2D(rendererData->d3dDevice,
                                          &stagingTextureDesc,
                                          NULL,
                                          &stagingTexture);
    if (FAILED(result)) {
        return WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D11Device1::CreateTexture2D [create staging texture]"), result);
    }

    // Get a write-only pointer to data in the staging texture:
    result = ID3D11DeviceContext_Map(rendererData->d3dContext,
                                     (ID3D11Resource *)stagingTexture,
                                     0,
                                     D3D11_MAP_WRITE,
                                     0,
                                     &textureMemory);
    if (FAILED(result)) {
        SAFE_RELEASE(stagingTexture);
        return WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D11DeviceContext1::Map [map staging texture]"), result);
    }

    src = (const Uint8 *)pixels;
    dst = (Uint8 *)textureMemory.pData;
    length = w * bpp;
    if (length == (UINT)pitch && length == textureMemory.RowPitch) {
        SDL_memcpy(dst, src, (size_t)length * h);
    } else {
        if (length > (UINT)pitch) {
            length = pitch;
        }
        if (length > textureMemory.RowPitch) {
            length = textureMemory.RowPitch;
        }
        for (row = 0; row < h; ++row) {
            SDL_memcpy(dst, src, length);
            src += pitch;
            dst += textureMemory.RowPitch;
        }
    }

    if (stagingTextureDesc.Format == DXGI_FORMAT_NV12 ||
        stagingTextureDesc.Format == DXGI_FORMAT_P010) {
        // Copy the UV plane as well
        h = (h + 1) / 2;
        if (stagingTextureDesc.Format == DXGI_FORMAT_P010) {
            length = (length + 3) & ~3;
            pitch = (pitch + 3) & ~3;
        } else {
            length = (length + 1) & ~1;
            pitch = (pitch + 1) & ~1;
        }
        dst = (Uint8 *)textureMemory.pData + stagingTextureDesc.Height * textureMemory.RowPitch;
        for (row = 0; row < h; ++row) {
            SDL_memcpy(dst, src, length);
            src += pitch;
            dst += textureMemory.RowPitch;
        }
    }

    // Commit the pixel buffer's changes back to the staging texture:
    ID3D11DeviceContext_Unmap(rendererData->d3dContext,
                              (ID3D11Resource *)stagingTexture,
                              0);

    // Copy the staging texture's contents back to the texture:
    ID3D11DeviceContext_CopySubresourceRegion(rendererData->d3dContext,
                                              (ID3D11Resource *)texture,
                                              0,
                                              x,
                                              y,
                                              0,
                                              (ID3D11Resource *)stagingTexture,
                                              0,
                                              NULL);

    SAFE_RELEASE(stagingTexture);

    return true;
}

#ifdef SDL_HAVE_YUV
static bool D3D11_UpdateTextureNV(SDL_Renderer *renderer, SDL_Texture *texture,
                                 const SDL_Rect *rect,
                                 const Uint8 *Yplane, int Ypitch,
                                 const Uint8 *UVplane, int UVpitch);

static bool D3D11_UpdateTextureYUV(SDL_Renderer *renderer, SDL_Texture *texture,
                                  const SDL_Rect *rect,
                                  const Uint8 *Yplane, int Ypitch,
                                  const Uint8 *Uplane, int Upitch,
                                  const Uint8 *Vplane, int Vpitch);
#endif

static bool D3D11_UpdateTexture(SDL_Renderer *renderer, SDL_Texture *texture,
                               const SDL_Rect *rect, const void *srcPixels,
                               int srcPitch)
{
    D3D11_RenderData *rendererData = (D3D11_RenderData *)renderer->internal;
    D3D11_TextureData *textureData = (D3D11_TextureData *)texture->internal;

    if (!textureData) {
        return SDL_SetError("Texture is not currently available");
    }

#ifdef SDL_HAVE_YUV
    if (textureData->nv12) {
        const Uint8 *Yplane = (const Uint8 *)srcPixels;
        const Uint8 *UVplane = Yplane + rect->h * srcPitch;

        return D3D11_UpdateTextureNV(renderer, texture, rect, Yplane, srcPitch, UVplane, srcPitch);

    } else if (textureData->yuv) {
        int Ypitch = srcPitch;
        int UVpitch = ((Ypitch + 1) / 2);
        const Uint8 *Yplane = (const Uint8 *)srcPixels;
        const Uint8 *Uplane = Yplane + rect->h * Ypitch;
        const Uint8 *Vplane = Uplane + ((rect->h + 1) / 2) * UVpitch;

        if (texture->format == SDL_PIXELFORMAT_YV12) {
            return D3D11_UpdateTextureYUV(renderer, texture, rect, Yplane, Ypitch, Vplane, UVpitch, Uplane, UVpitch);
        } else {
            return D3D11_UpdateTextureYUV(renderer, texture, rect, Yplane, Ypitch, Uplane, UVpitch, Vplane, UVpitch);
        }
    }
#endif

    if (!D3D11_UpdateTextureInternal(rendererData, textureData->mainTexture, SDL_BYTESPERPIXEL(texture->format), rect->x, rect->y, rect->w, rect->h, srcPixels, srcPitch)) {
        return false;
    }
    return true;
}

#ifdef SDL_HAVE_YUV
static bool D3D11_UpdateTextureYUV(SDL_Renderer *renderer, SDL_Texture *texture,
                                  const SDL_Rect *rect,
                                  const Uint8 *Yplane, int Ypitch,
                                  const Uint8 *Uplane, int Upitch,
                                  const Uint8 *Vplane, int Vpitch)
{
    D3D11_RenderData *rendererData = (D3D11_RenderData *)renderer->internal;
    D3D11_TextureData *textureData = (D3D11_TextureData *)texture->internal;

    if (!textureData) {
        return SDL_SetError("Texture is not currently available");
    }

    if (!D3D11_UpdateTextureInternal(rendererData, textureData->mainTexture, SDL_BYTESPERPIXEL(texture->format), rect->x, rect->y, rect->w, rect->h, Yplane, Ypitch)) {
        return false;
    }
    if (!D3D11_UpdateTextureInternal(rendererData, textureData->mainTextureU, SDL_BYTESPERPIXEL(texture->format), rect->x / 2, rect->y / 2, (rect->w + 1) / 2, (rect->h + 1) / 2, Uplane, Upitch)) {
        return false;
    }
    if (!D3D11_UpdateTextureInternal(rendererData, textureData->mainTextureV, SDL_BYTESPERPIXEL(texture->format), rect->x / 2, rect->y / 2, (rect->w + 1) / 2, (rect->h + 1) / 2, Vplane, Vpitch)) {
        return false;
    }
    return true;
}

static bool D3D11_UpdateTextureNV(SDL_Renderer *renderer, SDL_Texture *texture,
                                 const SDL_Rect *rect,
                                 const Uint8 *Yplane, int Ypitch,
                                 const Uint8 *UVplane, int UVpitch)
{
    D3D11_RenderData *rendererData = (D3D11_RenderData *)renderer->internal;
    D3D11_TextureData *textureData = (D3D11_TextureData *)texture->internal;
    ID3D11Texture2D *stagingTexture;
    const Uint8 *src;
    Uint8 *dst;
    int w, h, row;
    UINT length;
    HRESULT result;
    D3D11_TEXTURE2D_DESC stagingTextureDesc;
    D3D11_MAPPED_SUBRESOURCE textureMemory;

    if (!textureData) {
        return SDL_SetError("Texture is not currently available");
    }

    w = rect->w;
    h = rect->h;

    // Create a 'staging' texture, which will be used to write to a portion of the main texture.
    ID3D11Texture2D_GetDesc(textureData->mainTexture, &stagingTextureDesc);
    stagingTextureDesc.Width = w;
    stagingTextureDesc.Height = h;
    stagingTextureDesc.BindFlags = 0;
    stagingTextureDesc.MiscFlags = 0;
    stagingTextureDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    stagingTextureDesc.Usage = D3D11_USAGE_STAGING;
    if (stagingTextureDesc.Format == DXGI_FORMAT_NV12 ||
        stagingTextureDesc.Format == DXGI_FORMAT_P010) {
        stagingTextureDesc.Width = (stagingTextureDesc.Width + 1) & ~1;
        stagingTextureDesc.Height = (stagingTextureDesc.Height + 1) & ~1;
    }
    result = ID3D11Device_CreateTexture2D(rendererData->d3dDevice,
                                          &stagingTextureDesc,
                                          NULL,
                                          &stagingTexture);
    if (FAILED(result)) {
        return WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D11Device1::CreateTexture2D [create staging texture]"), result);
    }

    // Get a write-only pointer to data in the staging texture:
    result = ID3D11DeviceContext_Map(rendererData->d3dContext,
                                     (ID3D11Resource *)stagingTexture,
                                     0,
                                     D3D11_MAP_WRITE,
                                     0,
                                     &textureMemory);
    if (FAILED(result)) {
        SAFE_RELEASE(stagingTexture);
        return WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D11DeviceContext1::Map [map staging texture]"), result);
    }

    src = Yplane;
    dst = (Uint8 *)textureMemory.pData;
    length = w;
    if (length == (UINT)Ypitch && length == textureMemory.RowPitch) {
        SDL_memcpy(dst, src, (size_t)length * h);
    } else {
        if (length > (UINT)Ypitch) {
            length = Ypitch;
        }
        if (length > textureMemory.RowPitch) {
            length = textureMemory.RowPitch;
        }
        for (row = 0; row < h; ++row) {
            SDL_memcpy(dst, src, length);
            src += Ypitch;
            dst += textureMemory.RowPitch;
        }
    }

    src = UVplane;
    length = w;
    h = (h + 1) / 2;
    if (stagingTextureDesc.Format == DXGI_FORMAT_P010) {
        length = (length + 3) & ~3;
        UVpitch = (UVpitch + 3) & ~3;
    } else {
        length = (length + 1) & ~1;
        UVpitch = (UVpitch + 1) & ~1;
    }
    dst = (Uint8 *)textureMemory.pData + stagingTextureDesc.Height * textureMemory.RowPitch;
    for (row = 0; row < h; ++row) {
        SDL_memcpy(dst, src, length);
        src += UVpitch;
        dst += textureMemory.RowPitch;
    }

    // Commit the pixel buffer's changes back to the staging texture:
    ID3D11DeviceContext_Unmap(rendererData->d3dContext,
                              (ID3D11Resource *)stagingTexture,
                              0);

    // Copy the staging texture's contents back to the texture:
    ID3D11DeviceContext_CopySubresourceRegion(rendererData->d3dContext,
                                              (ID3D11Resource *)textureData->mainTexture,
                                              0,
                                              rect->x,
                                              rect->y,
                                              0,
                                              (ID3D11Resource *)stagingTexture,
                                              0,
                                              NULL);

    SAFE_RELEASE(stagingTexture);

    return true;
}
#endif

static bool D3D11_LockTexture(SDL_Renderer *renderer, SDL_Texture *texture,
                             const SDL_Rect *rect, void **pixels, int *pitch)
{
    D3D11_RenderData *rendererData = (D3D11_RenderData *)renderer->internal;
    D3D11_TextureData *textureData = (D3D11_TextureData *)texture->internal;
    HRESULT result = S_OK;
    D3D11_TEXTURE2D_DESC stagingTextureDesc;
    D3D11_MAPPED_SUBRESOURCE textureMemory;

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
        textureData->locked_rect = *rect;
        *pixels =
            (void *)(textureData->pixels + rect->y * textureData->pitch +
                     rect->x * SDL_BYTESPERPIXEL(texture->format));
        *pitch = textureData->pitch;
        return true;
    }
#endif
    if (textureData->stagingTexture) {
        return SDL_SetError("texture is already locked");
    }

    /* Create a 'staging' texture, which will be used to write to a portion
     * of the main texture.  This is necessary, as Direct3D 11.1 does not
     * have the ability to write a CPU-bound pixel buffer to a rectangular
     * subrect of a texture.  Direct3D 11.1 can, however, write a pixel
     * buffer to an entire texture, hence the use of a staging texture.
     */
    ID3D11Texture2D_GetDesc(textureData->mainTexture, &stagingTextureDesc);
    stagingTextureDesc.Width = rect->w;
    stagingTextureDesc.Height = rect->h;
    stagingTextureDesc.BindFlags = 0;
    stagingTextureDesc.MiscFlags = 0;
    stagingTextureDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    stagingTextureDesc.Usage = D3D11_USAGE_STAGING;
    result = ID3D11Device_CreateTexture2D(rendererData->d3dDevice,
                                          &stagingTextureDesc,
                                          NULL,
                                          &textureData->stagingTexture);
    if (FAILED(result)) {
        return WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D11Device1::CreateTexture2D [create staging texture]"), result);
    }

    // Get a write-only pointer to data in the staging texture:
    result = ID3D11DeviceContext_Map(rendererData->d3dContext,
                                     (ID3D11Resource *)textureData->stagingTexture,
                                     0,
                                     D3D11_MAP_WRITE,
                                     0,
                                     &textureMemory);
    if (FAILED(result)) {
        SAFE_RELEASE(textureData->stagingTexture);
        return WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D11DeviceContext1::Map [map staging texture]"), result);
    }

    /* Make note of where the staging texture will be written to
     * (on a call to SDL_UnlockTexture):
     */
    textureData->lockedTexturePositionX = rect->x;
    textureData->lockedTexturePositionY = rect->y;

    /* Make sure the caller has information on the texture's pixel buffer,
     * then return:
     */
    *pixels = textureMemory.pData;
    *pitch = textureMemory.RowPitch;
    return true;
}

static void D3D11_UnlockTexture(SDL_Renderer *renderer, SDL_Texture *texture)
{
    D3D11_RenderData *rendererData = (D3D11_RenderData *)renderer->internal;
    D3D11_TextureData *textureData = (D3D11_TextureData *)texture->internal;

    if (!textureData) {
        return;
    }
#ifdef SDL_HAVE_YUV
    if (textureData->yuv || textureData->nv12) {
        const SDL_Rect *rect = &textureData->locked_rect;
        void *pixels =
            (void *)(textureData->pixels + rect->y * textureData->pitch +
                     rect->x * SDL_BYTESPERPIXEL(texture->format));
        D3D11_UpdateTexture(renderer, texture, rect, pixels, textureData->pitch);
        return;
    }
#endif
    // Commit the pixel buffer's changes back to the staging texture:
    ID3D11DeviceContext_Unmap(rendererData->d3dContext,
                              (ID3D11Resource *)textureData->stagingTexture,
                              0);

    // Copy the staging texture's contents back to the main texture:
    ID3D11DeviceContext_CopySubresourceRegion(rendererData->d3dContext,
                                              (ID3D11Resource *)textureData->mainTexture,
                                              0,
                                              textureData->lockedTexturePositionX,
                                              textureData->lockedTexturePositionY,
                                              0,
                                              (ID3D11Resource *)textureData->stagingTexture,
                                              0,
                                              NULL);

    SAFE_RELEASE(textureData->stagingTexture);
}

static bool D3D11_SetRenderTarget(SDL_Renderer *renderer, SDL_Texture *texture)
{
    D3D11_RenderData *rendererData = (D3D11_RenderData *)renderer->internal;
    D3D11_TextureData *textureData = NULL;

    if (!texture) {
        rendererData->currentOffscreenRenderTargetView = NULL;
        return true;
    }

    textureData = (D3D11_TextureData *)texture->internal;

    if (!textureData->mainTextureRenderTargetView) {
        return SDL_SetError("specified texture is not a render target");
    }

    rendererData->currentOffscreenRenderTargetView = textureData->mainTextureRenderTargetView;

    return true;
}

static bool D3D11_QueueNoOp(SDL_Renderer *renderer, SDL_RenderCommand *cmd)
{
    return true; // nothing to do in this backend.
}

static bool D3D11_QueueDrawPoints(SDL_Renderer *renderer, SDL_RenderCommand *cmd, const SDL_FPoint *points, int count)
{
    D3D11_VertexPositionColor *verts = (D3D11_VertexPositionColor *)SDL_AllocateRenderVertices(renderer, count * sizeof(D3D11_VertexPositionColor), 0, &cmd->data.draw.first);
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

static bool D3D11_QueueGeometry(SDL_Renderer *renderer, SDL_RenderCommand *cmd, SDL_Texture *texture,
                               const float *xy, int xy_stride, const SDL_FColor *color, int color_stride, const float *uv, int uv_stride,
                               int num_vertices, const void *indices, int num_indices, int size_indices,
                               float scale_x, float scale_y)
{
    int i;
    int count = indices ? num_indices : num_vertices;
    D3D11_VertexPositionColor *verts = (D3D11_VertexPositionColor *)SDL_AllocateRenderVertices(renderer, count * sizeof(D3D11_VertexPositionColor), 0, &cmd->data.draw.first);
    bool convert_color = SDL_RenderingLinearSpace(renderer);
    D3D11_TextureData *textureData = texture ? (D3D11_TextureData *)texture->internal : NULL;
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

static bool D3D11_UpdateVertexBuffer(SDL_Renderer *renderer,
                                    const void *vertexData, size_t dataSizeInBytes)
{
    D3D11_RenderData *rendererData = (D3D11_RenderData *)renderer->internal;
    HRESULT result = S_OK;
    const int vbidx = rendererData->currentVertexBuffer;
    const UINT stride = sizeof(D3D11_VertexPositionColor);
    const UINT offset = 0;

    if (dataSizeInBytes == 0) {
        return true; // nothing to do.
    }

    if (rendererData->vertexBuffers[vbidx] && rendererData->vertexBufferSizes[vbidx] >= dataSizeInBytes) {
        D3D11_MAPPED_SUBRESOURCE mappedResource;
        result = ID3D11DeviceContext_Map(rendererData->d3dContext,
                                         (ID3D11Resource *)rendererData->vertexBuffers[vbidx],
                                         0,
                                         D3D11_MAP_WRITE_DISCARD,
                                         0,
                                         &mappedResource);
        if (FAILED(result)) {
            return WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D11DeviceContext1::Map [vertex buffer]"), result);
        }
        SDL_memcpy(mappedResource.pData, vertexData, dataSizeInBytes);
        ID3D11DeviceContext_Unmap(rendererData->d3dContext, (ID3D11Resource *)rendererData->vertexBuffers[vbidx], 0);
    } else {
        D3D11_BUFFER_DESC vertexBufferDesc;
        D3D11_SUBRESOURCE_DATA vertexBufferData;

        SAFE_RELEASE(rendererData->vertexBuffers[vbidx]);

        SDL_zero(vertexBufferDesc);
        vertexBufferDesc.ByteWidth = (UINT)dataSizeInBytes;
        vertexBufferDesc.Usage = D3D11_USAGE_DYNAMIC;
        vertexBufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
        vertexBufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

        SDL_zero(vertexBufferData);
        vertexBufferData.pSysMem = vertexData;
        vertexBufferData.SysMemPitch = 0;
        vertexBufferData.SysMemSlicePitch = 0;

        result = ID3D11Device_CreateBuffer(rendererData->d3dDevice,
                                           &vertexBufferDesc,
                                           &vertexBufferData,
                                           &rendererData->vertexBuffers[vbidx]);
        if (FAILED(result)) {
            return WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D11Device1::CreateBuffer [vertex buffer]"), result);
        }

        rendererData->vertexBufferSizes[vbidx] = dataSizeInBytes;
    }

    ID3D11DeviceContext_IASetVertexBuffers(rendererData->d3dContext,
                                           0,
                                           1,
                                           &rendererData->vertexBuffers[vbidx],
                                           &stride,
                                           &offset);

    rendererData->currentVertexBuffer++;
    if (rendererData->currentVertexBuffer >= SDL_arraysize(rendererData->vertexBuffers)) {
        rendererData->currentVertexBuffer = 0;
    }

    return true;
}

static bool D3D11_UpdateViewport(SDL_Renderer *renderer)
{
    D3D11_RenderData *data = (D3D11_RenderData *)renderer->internal;
    const SDL_Rect *viewport = &data->currentViewport;
    Float4X4 projection;
    Float4X4 view;
    SDL_FRect orientationAlignedViewport;
    BOOL swapDimensions;
    D3D11_VIEWPORT d3dviewport;
    const int rotation = D3D11_GetRotationForCurrentRenderTarget(renderer);

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
    swapDimensions = D3D11_IsDisplayRotated90Degrees(rotation);
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
    ID3D11DeviceContext_RSSetViewports(data->d3dContext, 1, &d3dviewport);

    data->viewportDirty = false;

    return true;
}

static ID3D11RenderTargetView *D3D11_GetCurrentRenderTargetView(SDL_Renderer *renderer)
{
    D3D11_RenderData *data = (D3D11_RenderData *)renderer->internal;
    if (data->currentOffscreenRenderTargetView) {
        return data->currentOffscreenRenderTargetView;
    } else {
        return data->mainRenderTargetView;
    }
}

static void D3D11_SetupShaderConstants(SDL_Renderer *renderer, const SDL_RenderCommand *cmd, const SDL_Texture *texture, D3D11_PixelShaderConstants *constants)
{
    float output_headroom;

    SDL_zerop(constants);

    constants->scRGB_output = (float)SDL_RenderingLinearSpace(renderer);
    constants->color_scale = cmd->data.draw.color_scale;

    if (texture) {
        D3D11_TextureData *textureData = (D3D11_TextureData *)texture->internal;

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

static bool D3D11_SetDrawState(SDL_Renderer *renderer, const SDL_RenderCommand *cmd,
                              D3D11_Shader shader, const D3D11_PixelShaderConstants *shader_constants,
                              const int numShaderResources, ID3D11ShaderResourceView **shaderResources,
                              ID3D11SamplerState *sampler, const Float4X4 *matrix)

{
    D3D11_RenderData *rendererData = (D3D11_RenderData *)renderer->internal;
    const Float4X4 *newmatrix = matrix ? matrix : &rendererData->identity;
    ID3D11RasterizerState *rasterizerState;
    ID3D11RenderTargetView *renderTargetView = D3D11_GetCurrentRenderTargetView(renderer);
    ID3D11ShaderResourceView *shaderResource;
    const SDL_BlendMode blendMode = cmd->data.draw.blend;
    ID3D11BlendState *blendState = NULL;
    bool updateSubresource = false;
    D3D11_PixelShaderState *shader_state = &rendererData->currentShaderState[shader];
    D3D11_PixelShaderConstants solid_constants;

    if (numShaderResources > 0) {
        shaderResource = shaderResources[0];
    } else {
        shaderResource = NULL;
    }

    // Make sure the render target isn't bound to a shader
    if (shaderResource != rendererData->currentShaderResource) {
        ID3D11ShaderResourceView *pNullResource = NULL;
        ID3D11DeviceContext_PSSetShaderResources(rendererData->d3dContext, 0, 1, &pNullResource);
        rendererData->currentShaderResource = NULL;
    }

    if (renderTargetView != rendererData->currentRenderTargetView) {
        ID3D11DeviceContext_OMSetRenderTargets(rendererData->d3dContext,
                                               1,
                                               &renderTargetView,
                                               NULL);
        rendererData->currentRenderTargetView = renderTargetView;
    }

    if (rendererData->viewportDirty) {
        if (D3D11_UpdateViewport(renderer)) {
            // vertexShaderConstantsData.projectionAndView has changed
            updateSubresource = true;
        }
    }

    if (rendererData->cliprectDirty) {
        if (!rendererData->currentCliprectEnabled) {
            ID3D11DeviceContext_RSSetScissorRects(rendererData->d3dContext, 0, NULL);
        } else {
            D3D11_RECT scissorRect;
            if (!D3D11_GetViewportAlignedD3DRect(renderer, &rendererData->currentCliprect, &scissorRect, TRUE)) {
                // D3D11_GetViewportAlignedD3DRect will have set the SDL error
                return false;
            }
            ID3D11DeviceContext_RSSetScissorRects(rendererData->d3dContext, 1, &scissorRect);
        }
        rendererData->cliprectDirty = false;
    }

    if (!rendererData->currentCliprectEnabled) {
        rasterizerState = rendererData->mainRasterizer;
    } else {
        rasterizerState = rendererData->clippedRasterizer;
    }
    if (rasterizerState != rendererData->currentRasterizerState) {
        ID3D11DeviceContext_RSSetState(rendererData->d3dContext, rasterizerState);
        rendererData->currentRasterizerState = rasterizerState;
    }

    if (blendMode != SDL_BLENDMODE_NONE) {
        int i;
        for (i = 0; i < rendererData->blendModesCount; ++i) {
            if (blendMode == rendererData->blendModes[i].blendMode) {
                blendState = rendererData->blendModes[i].blendState;
                break;
            }
        }
        if (!blendState) {
            blendState = D3D11_CreateBlendState(renderer, blendMode);
            if (!blendState) {
                return false;
            }
        }
    }
    if (blendState != rendererData->currentBlendState) {
        ID3D11DeviceContext_OMSetBlendState(rendererData->d3dContext, blendState, 0, 0xFFFFFFFF);
        rendererData->currentBlendState = blendState;
    }

    if (!shader_constants) {
        D3D11_SetupShaderConstants(renderer, cmd, NULL, &solid_constants);
        shader_constants = &solid_constants;
    }

    if (!shader_state->constants ||
        SDL_memcmp(shader_constants, &shader_state->shader_constants, sizeof(*shader_constants)) != 0) {
        SAFE_RELEASE(shader_state->constants);

        D3D11_BUFFER_DESC desc;
        SDL_zero(desc);
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.ByteWidth = sizeof(*shader_constants);
        desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;

        D3D11_SUBRESOURCE_DATA data;
        SDL_zero(data);
        data.pSysMem = shader_constants;

        HRESULT result = ID3D11Device_CreateBuffer(rendererData->d3dDevice, &desc, &data, &shader_state->constants);
        if (FAILED(result)) {
            WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D11Device::CreateBuffer [create shader constants]"), result);
            return false;
        }
        SDL_memcpy(&shader_state->shader_constants, shader_constants, sizeof(*shader_constants));

        // Force the shader parameters to be re-set
        rendererData->currentShader = SHADER_NONE;
    }
    if (shader != rendererData->currentShader) {
        if (!rendererData->pixelShaders[shader]) {
            if (!D3D11_CreatePixelShader(rendererData->d3dDevice, shader, &rendererData->pixelShaders[shader])) {
                return false;
            }
        }
        ID3D11DeviceContext_PSSetShader(rendererData->d3dContext, rendererData->pixelShaders[shader], NULL, 0);
        if (shader_state->constants) {
            ID3D11DeviceContext_PSSetConstantBuffers(rendererData->d3dContext, 0, 1, &shader_state->constants);
        }
        rendererData->currentShader = shader;
    }
    if (shaderResource != rendererData->currentShaderResource) {
        ID3D11DeviceContext_PSSetShaderResources(rendererData->d3dContext, 0, numShaderResources, shaderResources);
        rendererData->currentShaderResource = shaderResource;
    }
    if (sampler != rendererData->currentSampler) {
        ID3D11DeviceContext_PSSetSamplers(rendererData->d3dContext, 0, 1, &sampler);
        rendererData->currentSampler = sampler;
    }

    if (updateSubresource == true || SDL_memcmp(&rendererData->vertexShaderConstantsData.model, newmatrix, sizeof(*newmatrix)) != 0) {
        SDL_copyp(&rendererData->vertexShaderConstantsData.model, newmatrix);
        ID3D11DeviceContext_UpdateSubresource(rendererData->d3dContext,
                                              (ID3D11Resource *)rendererData->vertexShaderConstants,
                                              0,
                                              NULL,
                                              &rendererData->vertexShaderConstantsData,
                                              0,
                                              0);
    }

    return true;
}

static ID3D11SamplerState *D3D11_GetSamplerState(D3D11_RenderData *data, SDL_ScaleMode scale_mode, SDL_TextureAddressMode address_u, SDL_TextureAddressMode address_v)
{
    Uint32 key = RENDER_SAMPLER_HASHKEY(scale_mode, address_u, address_v);
    SDL_assert(key < SDL_arraysize(data->samplers));
    if (!data->samplers[key]) {
        D3D11_SAMPLER_DESC samplerDesc;
        SDL_zero(samplerDesc);
        samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
        samplerDesc.MipLODBias = 0.0f;
        samplerDesc.MaxAnisotropy = 1;
        samplerDesc.ComparisonFunc = D3D11_COMPARISON_ALWAYS;
        samplerDesc.MinLOD = 0.0f;
        samplerDesc.MaxLOD = D3D11_FLOAT32_MAX;
        switch (scale_mode) {
        case SDL_SCALEMODE_NEAREST:
            samplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
            break;
        case SDL_SCALEMODE_PIXELART:    // Uses linear sampling
        case SDL_SCALEMODE_LINEAR:
            samplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
            break;
        default:
            SDL_SetError("Unknown scale mode: %d", scale_mode);
            return NULL;
        }
        switch (address_u) {
        case SDL_TEXTURE_ADDRESS_CLAMP:
            samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
            break;
        case SDL_TEXTURE_ADDRESS_WRAP:
            samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
            break;
        default:
            SDL_SetError("Unknown texture address mode: %d", address_u);
            return NULL;
        }
        switch (address_v) {
        case SDL_TEXTURE_ADDRESS_CLAMP:
            samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
            break;
        case SDL_TEXTURE_ADDRESS_WRAP:
            samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
            break;
        default:
            SDL_SetError("Unknown texture address mode: %d", address_v);
            return NULL;
        }
        HRESULT result = ID3D11Device_CreateSamplerState(data->d3dDevice,
                                                 &samplerDesc,
                                                 &data->samplers[key]);
        if (FAILED(result)) {
            WIN_SetErrorFromHRESULT("ID3D11Device::CreateSamplerState", result);
            return NULL;
        }
    }
    return data->samplers[key];
}

static bool D3D11_SetCopyState(SDL_Renderer *renderer, const SDL_RenderCommand *cmd, const Float4X4 *matrix)
{
    SDL_Texture *texture = cmd->data.draw.texture;
    D3D11_RenderData *rendererData = (D3D11_RenderData *)renderer->internal;
    D3D11_TextureData *textureData = (D3D11_TextureData *)texture->internal;
    ID3D11SamplerState *textureSampler;
    D3D11_PixelShaderConstants constants;

    if (!textureData) {
        return SDL_SetError("Texture is not currently available");
    }

    D3D11_SetupShaderConstants(renderer, cmd, texture, &constants);

    textureSampler = D3D11_GetSamplerState(rendererData, cmd->data.draw.texture_scale_mode, cmd->data.draw.texture_address_mode_u, cmd->data.draw.texture_address_mode_v);
    if (!textureSampler) {
        return false;
    }

#ifdef SDL_HAVE_YUV
    if (textureData->yuv) {
        ID3D11ShaderResourceView *shaderResources[3];

        shaderResources[0] = textureData->mainTextureResourceView;
        shaderResources[1] = textureData->mainTextureResourceViewU;
        shaderResources[2] = textureData->mainTextureResourceViewV;

        return D3D11_SetDrawState(renderer, cmd, textureData->shader, &constants,
                                  SDL_arraysize(shaderResources), shaderResources, textureSampler, matrix);

    } else if (textureData->nv12) {
        ID3D11ShaderResourceView *shaderResources[2];

        shaderResources[0] = textureData->mainTextureResourceView;
        shaderResources[1] = textureData->mainTextureResourceViewNV;

        return D3D11_SetDrawState(renderer, cmd, textureData->shader, &constants,
                                  SDL_arraysize(shaderResources), shaderResources, textureSampler, matrix);
    }
#endif // SDL_HAVE_YUV
    return D3D11_SetDrawState(renderer, cmd, textureData->shader, &constants,
                              1, &textureData->mainTextureResourceView, textureSampler, matrix);
}

static void D3D11_DrawPrimitives(SDL_Renderer *renderer, D3D11_PRIMITIVE_TOPOLOGY primitiveTopology, const size_t vertexStart, const size_t vertexCount)
{
    D3D11_RenderData *rendererData = (D3D11_RenderData *)renderer->internal;
    ID3D11DeviceContext_IASetPrimitiveTopology(rendererData->d3dContext, primitiveTopology);
    ID3D11DeviceContext_Draw(rendererData->d3dContext, (UINT)vertexCount, (UINT)vertexStart);
}

static void D3D11_InvalidateCachedState(SDL_Renderer *renderer)
{
    D3D11_RenderData *data = (D3D11_RenderData *)renderer->internal;
    data->currentRenderTargetView = NULL;
    data->currentRasterizerState = NULL;
    data->currentBlendState = NULL;
    data->currentShader = SHADER_NONE;
    data->currentShaderResource = NULL;
    data->currentSampler = NULL;
    data->cliprectDirty = true;
    data->viewportDirty = true;
}

static bool D3D11_RunCommandQueue(SDL_Renderer *renderer, SDL_RenderCommand *cmd, void *vertices, size_t vertsize)
{
    D3D11_RenderData *rendererData = (D3D11_RenderData *)renderer->internal;
    const int viewportRotation = D3D11_GetRotationForCurrentRenderTarget(renderer);

    if (!rendererData->d3dDevice) {
        return SDL_SetError("Device lost and couldn't be recovered");
    }

    if (rendererData->pixelSizeChanged) {
        D3D11_UpdateForWindowSizeChange(renderer);
        rendererData->pixelSizeChanged = false;
    }

    if (rendererData->currentViewportRotation != viewportRotation) {
        rendererData->currentViewportRotation = viewportRotation;
        rendererData->viewportDirty = true;
    }

    if (!D3D11_UpdateVertexBuffer(renderer, vertices, vertsize)) {
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
            if (rendererData->currentCliprectEnabled != cmd->data.cliprect.enabled) {
                rendererData->currentCliprectEnabled = cmd->data.cliprect.enabled;
                rendererData->cliprectDirty = true;
            }
            if (SDL_memcmp(&rendererData->currentCliprect, rect, sizeof(*rect)) != 0) {
                SDL_copyp(&rendererData->currentCliprect, rect);
                rendererData->cliprectDirty = true;
            }
            break;
        }

        case SDL_RENDERCMD_CLEAR:
        {
            bool convert_color = SDL_RenderingLinearSpace(renderer);
            SDL_FColor color = cmd->data.color.color;
            if (convert_color) {
                SDL_ConvertToLinear(&color);
            }
            color.r *= cmd->data.color.color_scale;
            color.g *= cmd->data.color.color_scale;
            color.b *= cmd->data.color.color_scale;
            ID3D11DeviceContext_ClearRenderTargetView(rendererData->d3dContext, D3D11_GetCurrentRenderTargetView(renderer), &color.r);
            break;
        }

        case SDL_RENDERCMD_DRAW_POINTS:
        {
            const size_t count = cmd->data.draw.count;
            const size_t first = cmd->data.draw.first;
            const size_t start = first / sizeof(D3D11_VertexPositionColor);
            D3D11_SetDrawState(renderer, cmd, SHADER_SOLID, NULL, 0, NULL, NULL, NULL);
            D3D11_DrawPrimitives(renderer, D3D11_PRIMITIVE_TOPOLOGY_POINTLIST, start, count);
            break;
        }

        case SDL_RENDERCMD_DRAW_LINES:
        {
            const size_t count = cmd->data.draw.count;
            const size_t first = cmd->data.draw.first;
            const size_t start = first / sizeof(D3D11_VertexPositionColor);
            const D3D11_VertexPositionColor *verts = (D3D11_VertexPositionColor *)(((Uint8 *)vertices) + first);
            D3D11_SetDrawState(renderer, cmd, SHADER_SOLID, NULL, 0, NULL, NULL, NULL);
            D3D11_DrawPrimitives(renderer, D3D11_PRIMITIVE_TOPOLOGY_LINESTRIP, start, count);
            if (verts[0].pos.x != verts[count - 1].pos.x || verts[0].pos.y != verts[count - 1].pos.y) {
                D3D11_DrawPrimitives(renderer, D3D11_PRIMITIVE_TOPOLOGY_POINTLIST, start + (count - 1), 1);
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
            const size_t start = first / sizeof(D3D11_VertexPositionColor);

            if (texture) {
                D3D11_SetCopyState(renderer, cmd, NULL);
            } else {
                D3D11_SetDrawState(renderer, cmd, SHADER_SOLID, NULL, 0, NULL, NULL, NULL);
            }

            D3D11_DrawPrimitives(renderer, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST, start, count);
            break;
        }

        case SDL_RENDERCMD_NO_OP:
            break;
        }

        cmd = cmd->next;
    }

    return true;
}

static SDL_Surface *D3D11_RenderReadPixels(SDL_Renderer *renderer, const SDL_Rect *rect)
{
    D3D11_RenderData *data = (D3D11_RenderData *)renderer->internal;
    ID3D11RenderTargetView *renderTargetView = NULL;
    ID3D11Texture2D *backBuffer = NULL;
    ID3D11Texture2D *stagingTexture = NULL;
    HRESULT result;
    D3D11_TEXTURE2D_DESC stagingTextureDesc;
    D3D11_RECT srcRect = { 0, 0, 0, 0 };
    D3D11_BOX srcBox;
    D3D11_MAPPED_SUBRESOURCE textureMemory;
    SDL_Surface *output = NULL;

    renderTargetView = D3D11_GetCurrentRenderTargetView(renderer);
    if (!renderTargetView) {
        SDL_SetError("%s, ID3D11DeviceContext::OMGetRenderTargets failed", __FUNCTION__);
        goto done;
    }

    ID3D11View_GetResource(renderTargetView, (ID3D11Resource **)&backBuffer);
    if (!backBuffer) {
        SDL_SetError("%s, ID3D11View::GetResource failed", __FUNCTION__);
        goto done;
    }

    // Create a staging texture to copy the screen's data to:
    ID3D11Texture2D_GetDesc(backBuffer, &stagingTextureDesc);
    stagingTextureDesc.Width = rect->w;
    stagingTextureDesc.Height = rect->h;
    stagingTextureDesc.BindFlags = 0;
    stagingTextureDesc.MiscFlags = 0;
    stagingTextureDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    stagingTextureDesc.Usage = D3D11_USAGE_STAGING;
    result = ID3D11Device_CreateTexture2D(data->d3dDevice,
                                          &stagingTextureDesc,
                                          NULL,
                                          &stagingTexture);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D11Device1::CreateTexture2D [create staging texture]"), result);
        goto done;
    }

    // Copy the desired portion of the back buffer to the staging texture:
    if (!D3D11_GetViewportAlignedD3DRect(renderer, rect, &srcRect, FALSE)) {
        // D3D11_GetViewportAlignedD3DRect will have set the SDL error
        goto done;
    }

    srcBox.left = srcRect.left;
    srcBox.right = srcRect.right;
    srcBox.top = srcRect.top;
    srcBox.bottom = srcRect.bottom;
    srcBox.front = 0;
    srcBox.back = 1;
    ID3D11DeviceContext_CopySubresourceRegion(data->d3dContext,
                                              (ID3D11Resource *)stagingTexture,
                                              0,
                                              0, 0, 0,
                                              (ID3D11Resource *)backBuffer,
                                              0,
                                              &srcBox);

    // Map the staging texture's data to CPU-accessible memory:
    result = ID3D11DeviceContext_Map(data->d3dContext,
                                     (ID3D11Resource *)stagingTexture,
                                     0,
                                     D3D11_MAP_READ,
                                     0,
                                     &textureMemory);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("ID3D11DeviceContext1::Map [map staging texture]"), result);
        goto done;
    }

    output = SDL_DuplicatePixels(
        rect->w, rect->h,
        D3D11_DXGIFormatToSDLPixelFormat(stagingTextureDesc.Format),
        renderer->target ? renderer->target->colorspace : renderer->output_colorspace,
        textureMemory.pData,
        textureMemory.RowPitch);

    // Unmap the texture:
    ID3D11DeviceContext_Unmap(data->d3dContext,
                              (ID3D11Resource *)stagingTexture,
                              0);

done:
    SAFE_RELEASE(backBuffer);
    SAFE_RELEASE(stagingTexture);
    return output;
}

static bool D3D11_RenderPresent(SDL_Renderer *renderer)
{
    D3D11_RenderData *data = (D3D11_RenderData *)renderer->internal;
    HRESULT result;
    DXGI_PRESENT_PARAMETERS parameters;

    if (!data->d3dDevice) {
        return SDL_SetError("Device lost and couldn't be recovered");
    }

    SDL_zero(parameters);

    /* The application may optionally specify "dirty" or "scroll"
     * rects to improve efficiency in certain scenarios.
     */
    result = IDXGISwapChain1_Present1(data->swapChain, data->syncInterval, data->presentFlags, &parameters);

    /* Discard the contents of the render target.
     * This is a valid operation only when the existing contents will be entirely
     * overwritten. If dirty or scroll rects are used, this call should be removed.
     */
    ID3D11DeviceContext1_DiscardView(data->d3dContext, (ID3D11View *)data->mainRenderTargetView);

    // When the present flips, it unbinds the current view, so bind it again on the next draw call
    data->currentRenderTargetView = NULL;

    if (FAILED(result) && result != DXGI_ERROR_WAS_STILL_DRAWING) {
        /* If the device was removed either by a disconnect or a driver upgrade, we
         * must recreate all device resources.
         */
        if (result == DXGI_ERROR_DEVICE_REMOVED) {
            if (D3D11_HandleDeviceLost(renderer)) {
                SDL_SetError("Present failed, device lost");
            } else {
                // Recovering from device lost failed, error is already set
            }
        } else if (result == DXGI_ERROR_INVALID_CALL) {
            // We probably went through a fullscreen <-> windowed transition
            D3D11_CreateWindowSizeDependentResources(renderer);
            WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("IDXGISwapChain::Present"), result);
        } else {
            WIN_SetErrorFromHRESULT(SDL_COMPOSE_ERROR("IDXGISwapChain::Present"), result);
        }
        return false;
    }
    return true;
}

static bool D3D11_SetVSync(SDL_Renderer *renderer, const int vsync)
{
    D3D11_RenderData *data = (D3D11_RenderData *)renderer->internal;

    if (vsync < 0) {
        return SDL_Unsupported();
    }

    if (vsync > 0) {
        data->syncInterval = vsync;
        data->presentFlags = 0;
    } else {
        data->syncInterval = 0;
        data->presentFlags = DXGI_PRESENT_DO_NOT_WAIT;
    }
    return true;
}

static bool D3D11_CreateRenderer(SDL_Renderer *renderer, SDL_Window *window, SDL_PropertiesID create_props)
{
    D3D11_RenderData *data;

    HWND hwnd = (HWND)SDL_GetPointerProperty(SDL_GetWindowProperties(window), SDL_PROP_WINDOW_WIN32_HWND_POINTER, NULL);
    if (!hwnd) {
        return SDL_SetError("Couldn't get window handle");
    }

    SDL_SetupRendererColorspace(renderer, create_props);

    if (renderer->output_colorspace != SDL_COLORSPACE_SRGB &&
        renderer->output_colorspace != SDL_COLORSPACE_SRGB_LINEAR
        /*&& renderer->output_colorspace != SDL_COLORSPACE_HDR10*/) {
        return SDL_SetError("Unsupported output colorspace");
    }

    data = (D3D11_RenderData *)SDL_calloc(1, sizeof(*data));
    if (!data) {
        return false;
    }

    data->identity = MatrixIdentity();

    renderer->WindowEvent = D3D11_WindowEvent;
    renderer->SupportsBlendMode = D3D11_SupportsBlendMode;
    renderer->CreateTexture = D3D11_CreateTexture;
    renderer->UpdateTexture = D3D11_UpdateTexture;
#ifdef SDL_HAVE_YUV
    renderer->UpdateTextureYUV = D3D11_UpdateTextureYUV;
    renderer->UpdateTextureNV = D3D11_UpdateTextureNV;
#endif
    renderer->LockTexture = D3D11_LockTexture;
    renderer->UnlockTexture = D3D11_UnlockTexture;
    renderer->SetRenderTarget = D3D11_SetRenderTarget;
    renderer->QueueSetViewport = D3D11_QueueNoOp;
    renderer->QueueSetDrawColor = D3D11_QueueNoOp;
    renderer->QueueDrawPoints = D3D11_QueueDrawPoints;
    renderer->QueueDrawLines = D3D11_QueueDrawPoints; // lines and points queue vertices the same way.
    renderer->QueueGeometry = D3D11_QueueGeometry;
    renderer->InvalidateCachedState = D3D11_InvalidateCachedState;
    renderer->RunCommandQueue = D3D11_RunCommandQueue;
    renderer->RenderReadPixels = D3D11_RenderReadPixels;
    renderer->RenderPresent = D3D11_RenderPresent;
    renderer->DestroyTexture = D3D11_DestroyTexture;
    renderer->DestroyRenderer = D3D11_DestroyRenderer;
    renderer->SetVSync = D3D11_SetVSync;
    renderer->internal = data;
    D3D11_InvalidateCachedState(renderer);

    renderer->name = D3D11_RenderDriver.name;
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

    data->syncInterval = 0;
    data->presentFlags = DXGI_PRESENT_DO_NOT_WAIT;

    /* HACK: make sure the SDL_Renderer references the SDL_Window data now, in
     * order to give init functions access to the underlying window handle:
     */
    renderer->window = window;

    // Initialize Direct3D resources
    if (FAILED(D3D11_CreateDeviceResources(renderer))) {
        return false;
    }
    if (FAILED(D3D11_CreateWindowSizeDependentResources(renderer))) {
        return false;
    }

    return true;
}

SDL_RenderDriver D3D11_RenderDriver = {
    D3D11_CreateRenderer, "direct3d11"
};

#endif // SDL_VIDEO_RENDER_D3D11
