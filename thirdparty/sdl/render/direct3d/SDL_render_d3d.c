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

#ifdef SDL_VIDEO_RENDER_D3D

#include "../../core/windows/SDL_windows.h"

#include "../SDL_sysrender.h"
#include "../SDL_d3dmath.h"
#include "../../video/windows/SDL_windowsvideo.h"
#include "../../video/SDL_pixels_c.h"

#define D3D_DEBUG_INFO
#include <d3d9.h>

#include "SDL_shaders_d3d.h"

typedef struct
{
    SDL_Rect viewport;
    bool viewport_dirty;
    SDL_Texture *texture;
    SDL_BlendMode blend;
    bool cliprect_enabled;
    bool cliprect_enabled_dirty;
    SDL_Rect cliprect;
    bool cliprect_dirty;
    D3D9_Shader shader;
    const float *shader_params;
} D3D_DrawStateCache;

// Direct3D renderer implementation

typedef struct
{
    void *d3dDLL;
    IDirect3D9 *d3d;
    IDirect3DDevice9 *device;
    UINT adapter;
    D3DPRESENT_PARAMETERS pparams;
    bool updateSize;
    bool beginScene;
    bool enableSeparateAlphaBlend;
    SDL_ScaleMode scaleMode[3];
    SDL_TextureAddressMode addressModeU[3];
    SDL_TextureAddressMode addressModeV[3];
    IDirect3DSurface9 *defaultRenderTarget;
    IDirect3DSurface9 *currentRenderTarget;
    void *d3dxDLL;
#ifdef SDL_HAVE_YUV
    LPDIRECT3DPIXELSHADER9 shaders[NUM_SHADERS];
#endif
    LPDIRECT3DVERTEXBUFFER9 vertexBuffers[8];
    size_t vertexBufferSize[8];
    int currentVertexBuffer;
    bool reportedVboProblem;
    D3D_DrawStateCache drawstate;
} D3D_RenderData;

typedef struct
{
    bool dirty;
    int w, h;
    DWORD usage;
    Uint32 format;
    D3DFORMAT d3dfmt;
    IDirect3DTexture9 *texture;
    IDirect3DTexture9 *staging;
} D3D_TextureRep;

typedef struct
{
    D3D_TextureRep texture;
    D3D9_Shader shader;
    const float *shader_params;

#ifdef SDL_HAVE_YUV
    // YV12 texture support
    bool yuv;
    D3D_TextureRep utexture;
    D3D_TextureRep vtexture;
    Uint8 *pixels;
    int pitch;
    SDL_Rect locked_rect;
#endif
} D3D_TextureData;

typedef struct
{
    float x, y, z;
    DWORD color;
    float u, v;
} Vertex;

static bool D3D_SetError(const char *prefix, HRESULT result)
{
    const char *error;

    switch (result) {
    case D3DERR_WRONGTEXTUREFORMAT:
        error = "WRONGTEXTUREFORMAT";
        break;
    case D3DERR_UNSUPPORTEDCOLOROPERATION:
        error = "UNSUPPORTEDCOLOROPERATION";
        break;
    case D3DERR_UNSUPPORTEDCOLORARG:
        error = "UNSUPPORTEDCOLORARG";
        break;
    case D3DERR_UNSUPPORTEDALPHAOPERATION:
        error = "UNSUPPORTEDALPHAOPERATION";
        break;
    case D3DERR_UNSUPPORTEDALPHAARG:
        error = "UNSUPPORTEDALPHAARG";
        break;
    case D3DERR_TOOMANYOPERATIONS:
        error = "TOOMANYOPERATIONS";
        break;
    case D3DERR_CONFLICTINGTEXTUREFILTER:
        error = "CONFLICTINGTEXTUREFILTER";
        break;
    case D3DERR_UNSUPPORTEDFACTORVALUE:
        error = "UNSUPPORTEDFACTORVALUE";
        break;
    case D3DERR_CONFLICTINGRENDERSTATE:
        error = "CONFLICTINGRENDERSTATE";
        break;
    case D3DERR_UNSUPPORTEDTEXTUREFILTER:
        error = "UNSUPPORTEDTEXTUREFILTER";
        break;
    case D3DERR_CONFLICTINGTEXTUREPALETTE:
        error = "CONFLICTINGTEXTUREPALETTE";
        break;
    case D3DERR_DRIVERINTERNALERROR:
        error = "DRIVERINTERNALERROR";
        break;
    case D3DERR_NOTFOUND:
        error = "NOTFOUND";
        break;
    case D3DERR_MOREDATA:
        error = "MOREDATA";
        break;
    case D3DERR_DEVICELOST:
        error = "DEVICELOST";
        break;
    case D3DERR_DEVICENOTRESET:
        error = "DEVICENOTRESET";
        break;
    case D3DERR_NOTAVAILABLE:
        error = "NOTAVAILABLE";
        break;
    case D3DERR_OUTOFVIDEOMEMORY:
        error = "OUTOFVIDEOMEMORY";
        break;
    case D3DERR_INVALIDDEVICE:
        error = "INVALIDDEVICE";
        break;
    case D3DERR_INVALIDCALL:
        error = "INVALIDCALL";
        break;
    case D3DERR_DRIVERINVALIDCALL:
        error = "DRIVERINVALIDCALL";
        break;
    case D3DERR_WASSTILLDRAWING:
        error = "WASSTILLDRAWING";
        break;
    default:
        error = "UNKNOWN";
        break;
    }
    return SDL_SetError("%s: %s", prefix, error);
}

static D3DFORMAT PixelFormatToD3DFMT(Uint32 format)
{
    switch (format) {
    case SDL_PIXELFORMAT_RGB565:
        return D3DFMT_R5G6B5;
    case SDL_PIXELFORMAT_XRGB8888:
        return D3DFMT_X8R8G8B8;
    case SDL_PIXELFORMAT_ARGB8888:
        return D3DFMT_A8R8G8B8;
    case SDL_PIXELFORMAT_YV12:
    case SDL_PIXELFORMAT_IYUV:
    case SDL_PIXELFORMAT_NV12:
    case SDL_PIXELFORMAT_NV21:
        return D3DFMT_L8;
    default:
        return D3DFMT_UNKNOWN;
    }
}

static SDL_PixelFormat D3DFMTToPixelFormat(D3DFORMAT format)
{
    switch (format) {
    case D3DFMT_R5G6B5:
        return SDL_PIXELFORMAT_RGB565;
    case D3DFMT_X8R8G8B8:
        return SDL_PIXELFORMAT_XRGB8888;
    case D3DFMT_A8R8G8B8:
        return SDL_PIXELFORMAT_ARGB8888;
    default:
        return SDL_PIXELFORMAT_UNKNOWN;
    }
}

static void D3D_InitRenderState(D3D_RenderData *data)
{
    D3DMATRIX matrix;

    IDirect3DDevice9 *device = data->device;
    IDirect3DDevice9_SetPixelShader(device, NULL);
    IDirect3DDevice9_SetTexture(device, 0, NULL);
    IDirect3DDevice9_SetTexture(device, 1, NULL);
    IDirect3DDevice9_SetTexture(device, 2, NULL);
    IDirect3DDevice9_SetFVF(device, D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1);
    IDirect3DDevice9_SetVertexShader(device, NULL);
    IDirect3DDevice9_SetRenderState(device, D3DRS_ZENABLE, D3DZB_FALSE);
    IDirect3DDevice9_SetRenderState(device, D3DRS_CULLMODE, D3DCULL_NONE);
    IDirect3DDevice9_SetRenderState(device, D3DRS_LIGHTING, FALSE);

    // Enable color modulation by diffuse color
    IDirect3DDevice9_SetTextureStageState(device, 0, D3DTSS_COLOROP,
                                          D3DTOP_MODULATE);
    IDirect3DDevice9_SetTextureStageState(device, 0, D3DTSS_COLORARG1,
                                          D3DTA_TEXTURE);
    IDirect3DDevice9_SetTextureStageState(device, 0, D3DTSS_COLORARG2,
                                          D3DTA_DIFFUSE);

    // Enable alpha modulation by diffuse alpha
    IDirect3DDevice9_SetTextureStageState(device, 0, D3DTSS_ALPHAOP,
                                          D3DTOP_MODULATE);
    IDirect3DDevice9_SetTextureStageState(device, 0, D3DTSS_ALPHAARG1,
                                          D3DTA_TEXTURE);
    IDirect3DDevice9_SetTextureStageState(device, 0, D3DTSS_ALPHAARG2,
                                          D3DTA_DIFFUSE);

    // Enable separate alpha blend function, if possible
    if (data->enableSeparateAlphaBlend) {
        IDirect3DDevice9_SetRenderState(device, D3DRS_SEPARATEALPHABLENDENABLE, TRUE);
    }

    // Disable second texture stage, since we're done
    IDirect3DDevice9_SetTextureStageState(device, 1, D3DTSS_COLOROP,
                                          D3DTOP_DISABLE);
    IDirect3DDevice9_SetTextureStageState(device, 1, D3DTSS_ALPHAOP,
                                          D3DTOP_DISABLE);

    // Set an identity world and view matrix
    SDL_zero(matrix);
    matrix.m[0][0] = 1.0f;
    matrix.m[1][1] = 1.0f;
    matrix.m[2][2] = 1.0f;
    matrix.m[3][3] = 1.0f;
    IDirect3DDevice9_SetTransform(device, D3DTS_WORLD, &matrix);
    IDirect3DDevice9_SetTransform(device, D3DTS_VIEW, &matrix);

    // Reset our current scale mode
    for (int i = 0; i < SDL_arraysize(data->scaleMode); ++i) {
        data->scaleMode[i] = SDL_SCALEMODE_INVALID;
    }

    // Reset our current address mode
    for (int i = 0; i < SDL_arraysize(data->addressModeU); ++i) {
        data->addressModeU[i] = SDL_TEXTURE_ADDRESS_INVALID;
    }
    for (int i = 0; i < SDL_arraysize(data->addressModeV); ++i) {
        data->addressModeV[i] = SDL_TEXTURE_ADDRESS_INVALID;
    }

    // Start the render with beginScene
    data->beginScene = true;
}

static bool D3D_Reset(SDL_Renderer *renderer);

static bool D3D_ActivateRenderer(SDL_Renderer *renderer)
{
    D3D_RenderData *data = (D3D_RenderData *)renderer->internal;
    HRESULT result;

    if (data->updateSize) {
        SDL_Window *window = renderer->window;
        int w, h;
        const SDL_DisplayMode *fullscreen_mode = NULL;

        SDL_GetWindowSizeInPixels(window, &w, &h);
        data->pparams.BackBufferWidth = w;
        data->pparams.BackBufferHeight = h;
        if (SDL_GetWindowFlags(window) & SDL_WINDOW_FULLSCREEN) {
            fullscreen_mode = SDL_GetWindowFullscreenMode(window);
        }
        if (fullscreen_mode) {
            data->pparams.Windowed = FALSE;
            data->pparams.BackBufferFormat = PixelFormatToD3DFMT(fullscreen_mode->format);
            data->pparams.FullScreen_RefreshRateInHz = (UINT)SDL_ceilf(fullscreen_mode->refresh_rate);
        } else {
            data->pparams.Windowed = TRUE;
            data->pparams.BackBufferFormat = D3DFMT_UNKNOWN;
            data->pparams.FullScreen_RefreshRateInHz = 0;
        }
        if (!D3D_Reset(renderer)) {
            return false;
        }

        data->updateSize = false;
    }
    if (data->beginScene) {
        result = IDirect3DDevice9_BeginScene(data->device);
        if (result == D3DERR_DEVICELOST) {
            if (!D3D_Reset(renderer)) {
                return false;
            }
            result = IDirect3DDevice9_BeginScene(data->device);
        }
        if (FAILED(result)) {
            return D3D_SetError("BeginScene()", result);
        }
        data->beginScene = false;
    }
    return true;
}

static void D3D_WindowEvent(SDL_Renderer *renderer, const SDL_WindowEvent *event)
{
    D3D_RenderData *data = (D3D_RenderData *)renderer->internal;

    if (event->type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED) {
        data->updateSize = true;
    }
}

static D3DBLEND GetBlendFunc(SDL_BlendFactor factor)
{
    switch (factor) {
    case SDL_BLENDFACTOR_ZERO:
        return D3DBLEND_ZERO;
    case SDL_BLENDFACTOR_ONE:
        return D3DBLEND_ONE;
    case SDL_BLENDFACTOR_SRC_COLOR:
        return D3DBLEND_SRCCOLOR;
    case SDL_BLENDFACTOR_ONE_MINUS_SRC_COLOR:
        return D3DBLEND_INVSRCCOLOR;
    case SDL_BLENDFACTOR_SRC_ALPHA:
        return D3DBLEND_SRCALPHA;
    case SDL_BLENDFACTOR_ONE_MINUS_SRC_ALPHA:
        return D3DBLEND_INVSRCALPHA;
    case SDL_BLENDFACTOR_DST_COLOR:
        return D3DBLEND_DESTCOLOR;
    case SDL_BLENDFACTOR_ONE_MINUS_DST_COLOR:
        return D3DBLEND_INVDESTCOLOR;
    case SDL_BLENDFACTOR_DST_ALPHA:
        return D3DBLEND_DESTALPHA;
    case SDL_BLENDFACTOR_ONE_MINUS_DST_ALPHA:
        return D3DBLEND_INVDESTALPHA;
    default:
        break;
    }
    return (D3DBLEND)0;
}

static D3DBLENDOP GetBlendEquation(SDL_BlendOperation operation)
{
    switch (operation) {
    case SDL_BLENDOPERATION_ADD:
        return D3DBLENDOP_ADD;
    case SDL_BLENDOPERATION_SUBTRACT:
        return D3DBLENDOP_SUBTRACT;
    case SDL_BLENDOPERATION_REV_SUBTRACT:
        return D3DBLENDOP_REVSUBTRACT;
    case SDL_BLENDOPERATION_MINIMUM:
        return D3DBLENDOP_MIN;
    case SDL_BLENDOPERATION_MAXIMUM:
        return D3DBLENDOP_MAX;
    default:
        break;
    }
    return (D3DBLENDOP)0;
}

static bool D3D_SupportsBlendMode(SDL_Renderer *renderer, SDL_BlendMode blendMode)
{
    D3D_RenderData *data = (D3D_RenderData *)renderer->internal;
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

    if (!data->enableSeparateAlphaBlend) {
        if ((srcColorFactor != srcAlphaFactor) || (dstColorFactor != dstAlphaFactor) || (colorOperation != alphaOperation)) {
            return false;
        }
    }
    return true;
}

static bool D3D_CreateTextureRep(IDirect3DDevice9 *device, D3D_TextureRep *texture, DWORD usage, Uint32 format, D3DFORMAT d3dfmt, int w, int h)
{
    HRESULT result;

    texture->dirty = false;
    texture->w = w;
    texture->h = h;
    texture->usage = usage;
    texture->format = format;
    texture->d3dfmt = d3dfmt;

    result = IDirect3DDevice9_CreateTexture(device, w, h, 1, usage,
                                            PixelFormatToD3DFMT(format),
                                            D3DPOOL_DEFAULT, &texture->texture, NULL);
    if (FAILED(result)) {
        return D3D_SetError("CreateTexture(D3DPOOL_DEFAULT)", result);
    }
    return true;
}

static bool D3D_CreateStagingTexture(IDirect3DDevice9 *device, D3D_TextureRep *texture)
{
    HRESULT result;

    if (!texture->staging) {
        result = IDirect3DDevice9_CreateTexture(device, texture->w, texture->h, 1, 0,
                                                texture->d3dfmt, D3DPOOL_SYSTEMMEM, &texture->staging, NULL);
        if (FAILED(result)) {
            return D3D_SetError("CreateTexture(D3DPOOL_SYSTEMMEM)", result);
        }
    }
    return true;
}

static bool D3D_RecreateTextureRep(IDirect3DDevice9 *device, D3D_TextureRep *texture)
{
    if (texture->texture) {
        IDirect3DTexture9_Release(texture->texture);
        texture->texture = NULL;
    }
    if (texture->staging) {
        IDirect3DTexture9_AddDirtyRect(texture->staging, NULL);
        texture->dirty = true;
    }
    return true;
}

static bool D3D_UpdateTextureRep(IDirect3DDevice9 *device, D3D_TextureRep *texture, int x, int y, int w, int h, const void *pixels, int pitch)
{
    RECT d3drect;
    D3DLOCKED_RECT locked;
    const Uint8 *src;
    Uint8 *dst;
    int row, length;
    HRESULT result;

    if (!D3D_CreateStagingTexture(device, texture)) {
        return false;
    }

    d3drect.left = x;
    d3drect.right = (LONG)x + w;
    d3drect.top = y;
    d3drect.bottom = (LONG)y + h;

    result = IDirect3DTexture9_LockRect(texture->staging, 0, &locked, &d3drect, 0);
    if (FAILED(result)) {
        return D3D_SetError("LockRect()", result);
    }

    src = (const Uint8 *)pixels;
    dst = (Uint8 *)locked.pBits;
    length = w * SDL_BYTESPERPIXEL(texture->format);
    if (length == pitch && length == locked.Pitch) {
        SDL_memcpy(dst, src, (size_t)length * h);
    } else {
        if (length > pitch) {
            length = pitch;
        }
        if (length > locked.Pitch) {
            length = locked.Pitch;
        }
        for (row = 0; row < h; ++row) {
            SDL_memcpy(dst, src, length);
            src += pitch;
            dst += locked.Pitch;
        }
    }
    result = IDirect3DTexture9_UnlockRect(texture->staging, 0);
    if (FAILED(result)) {
        return D3D_SetError("UnlockRect()", result);
    }
    texture->dirty = true;

    return true;
}

static void D3D_DestroyTextureRep(D3D_TextureRep *texture)
{
    if (texture->texture) {
        IDirect3DTexture9_Release(texture->texture);
        texture->texture = NULL;
    }
    if (texture->staging) {
        IDirect3DTexture9_Release(texture->staging);
        texture->staging = NULL;
    }
}

static bool D3D_CreateTexture(SDL_Renderer *renderer, SDL_Texture *texture, SDL_PropertiesID create_props)
{
    D3D_RenderData *data = (D3D_RenderData *)renderer->internal;
    D3D_TextureData *texturedata;
    DWORD usage;

    texturedata = (D3D_TextureData *)SDL_calloc(1, sizeof(*texturedata));
    if (!texturedata) {
        return false;
    }

    texture->internal = texturedata;

    if (texture->access == SDL_TEXTUREACCESS_TARGET) {
        usage = D3DUSAGE_RENDERTARGET;
    } else {
        usage = 0;
    }

    if (!D3D_CreateTextureRep(data->device, &texturedata->texture, usage, texture->format, PixelFormatToD3DFMT(texture->format), texture->w, texture->h)) {
        return false;
    }
#ifdef SDL_HAVE_YUV
    if (texture->format == SDL_PIXELFORMAT_YV12 ||
        texture->format == SDL_PIXELFORMAT_IYUV) {
        texturedata->yuv = true;

        if (!D3D_CreateTextureRep(data->device, &texturedata->utexture, usage, texture->format, PixelFormatToD3DFMT(texture->format), (texture->w + 1) / 2, (texture->h + 1) / 2)) {
            return false;
        }

        if (!D3D_CreateTextureRep(data->device, &texturedata->vtexture, usage, texture->format, PixelFormatToD3DFMT(texture->format), (texture->w + 1) / 2, (texture->h + 1) / 2)) {
            return false;
        }

        texturedata->shader = SHADER_YUV;
        texturedata->shader_params = SDL_GetYCbCRtoRGBConversionMatrix(texture->colorspace, texture->w, texture->h, 8);
        if (texturedata->shader_params == NULL) {
            return SDL_SetError("Unsupported YUV colorspace");
        }
    }
#endif
    return true;
}

static bool D3D_RecreateTexture(SDL_Renderer *renderer, SDL_Texture *texture)
{
    D3D_RenderData *data = (D3D_RenderData *)renderer->internal;
    D3D_TextureData *texturedata = (D3D_TextureData *)texture->internal;

    if (!texturedata) {
        return true;
    }

    if (!D3D_RecreateTextureRep(data->device, &texturedata->texture)) {
        return false;
    }
#ifdef SDL_HAVE_YUV
    if (texturedata->yuv) {
        if (!D3D_RecreateTextureRep(data->device, &texturedata->utexture)) {
            return false;
        }

        if (!D3D_RecreateTextureRep(data->device, &texturedata->vtexture)) {
            return false;
        }
    }
#endif
    return true;
}

static bool D3D_UpdateTexture(SDL_Renderer *renderer, SDL_Texture *texture,
                             const SDL_Rect *rect, const void *pixels, int pitch)
{
    D3D_RenderData *data = (D3D_RenderData *)renderer->internal;
    D3D_TextureData *texturedata = (D3D_TextureData *)texture->internal;

    if (!texturedata) {
        return SDL_SetError("Texture is not currently available");
    }

    if (!D3D_UpdateTextureRep(data->device, &texturedata->texture, rect->x, rect->y, rect->w, rect->h, pixels, pitch)) {
        return false;
    }
#ifdef SDL_HAVE_YUV
    if (texturedata->yuv) {
        // Skip to the correct offset into the next texture
        pixels = (const void *)((const Uint8 *)pixels + rect->h * pitch);

        if (!D3D_UpdateTextureRep(data->device, texture->format == SDL_PIXELFORMAT_YV12 ? &texturedata->vtexture : &texturedata->utexture, rect->x / 2, rect->y / 2, (rect->w + 1) / 2, (rect->h + 1) / 2, pixels, (pitch + 1) / 2)) {
            return false;
        }

        // Skip to the correct offset into the next texture
        pixels = (const void *)((const Uint8 *)pixels + ((rect->h + 1) / 2) * ((pitch + 1) / 2));
        if (!D3D_UpdateTextureRep(data->device, texture->format == SDL_PIXELFORMAT_YV12 ? &texturedata->utexture : &texturedata->vtexture, rect->x / 2, (rect->y + 1) / 2, (rect->w + 1) / 2, (rect->h + 1) / 2, pixels, (pitch + 1) / 2)) {
            return false;
        }
    }
#endif
    return true;
}

#ifdef SDL_HAVE_YUV
static bool D3D_UpdateTextureYUV(SDL_Renderer *renderer, SDL_Texture *texture,
                                const SDL_Rect *rect,
                                const Uint8 *Yplane, int Ypitch,
                                const Uint8 *Uplane, int Upitch,
                                const Uint8 *Vplane, int Vpitch)
{
    D3D_RenderData *data = (D3D_RenderData *)renderer->internal;
    D3D_TextureData *texturedata = (D3D_TextureData *)texture->internal;

    if (!texturedata) {
        return SDL_SetError("Texture is not currently available");
    }

    if (!D3D_UpdateTextureRep(data->device, &texturedata->texture, rect->x, rect->y, rect->w, rect->h, Yplane, Ypitch)) {
        return false;
    }
    if (!D3D_UpdateTextureRep(data->device, &texturedata->utexture, rect->x / 2, rect->y / 2, (rect->w + 1) / 2, (rect->h + 1) / 2, Uplane, Upitch)) {
        return false;
    }
    if (!D3D_UpdateTextureRep(data->device, &texturedata->vtexture, rect->x / 2, rect->y / 2, (rect->w + 1) / 2, (rect->h + 1) / 2, Vplane, Vpitch)) {
        return false;
    }
    return true;
}
#endif

static bool D3D_LockTexture(SDL_Renderer *renderer, SDL_Texture *texture,
                           const SDL_Rect *rect, void **pixels, int *pitch)
{
    D3D_RenderData *data = (D3D_RenderData *)renderer->internal;
    D3D_TextureData *texturedata = (D3D_TextureData *)texture->internal;
    IDirect3DDevice9 *device = data->device;

    if (!texturedata) {
        return SDL_SetError("Texture is not currently available");
    }
#ifdef SDL_HAVE_YUV
    texturedata->locked_rect = *rect;

    if (texturedata->yuv) {
        // It's more efficient to upload directly...
        if (!texturedata->pixels) {
            texturedata->pitch = texture->w;
            texturedata->pixels = (Uint8 *)SDL_malloc((texture->h * texturedata->pitch * 3) / 2);
            if (!texturedata->pixels) {
                return false;
            }
        }
        *pixels =
            (void *)(texturedata->pixels + rect->y * texturedata->pitch +
                     rect->x * SDL_BYTESPERPIXEL(texture->format));
        *pitch = texturedata->pitch;
    } else
#endif
    {
        RECT d3drect;
        D3DLOCKED_RECT locked;
        HRESULT result;

        if (!D3D_CreateStagingTexture(device, &texturedata->texture)) {
            return false;
        }

        d3drect.left = rect->x;
        d3drect.right = (LONG)rect->x + rect->w;
        d3drect.top = rect->y;
        d3drect.bottom = (LONG)rect->y + rect->h;

        result = IDirect3DTexture9_LockRect(texturedata->texture.staging, 0, &locked, &d3drect, 0);
        if (FAILED(result)) {
            return D3D_SetError("LockRect()", result);
        }
        *pixels = locked.pBits;
        *pitch = locked.Pitch;
    }
    return true;
}

static void D3D_UnlockTexture(SDL_Renderer *renderer, SDL_Texture *texture)
{
    D3D_RenderData *data = (D3D_RenderData *)renderer->internal;
    D3D_TextureData *texturedata = (D3D_TextureData *)texture->internal;

    if (!texturedata) {
        return;
    }
#ifdef SDL_HAVE_YUV
    if (texturedata->yuv) {
        const SDL_Rect *rect = &texturedata->locked_rect;
        void *pixels =
            (void *)(texturedata->pixels + rect->y * texturedata->pitch +
                     rect->x * SDL_BYTESPERPIXEL(texture->format));
        D3D_UpdateTexture(renderer, texture, rect, pixels, texturedata->pitch);
    } else
#endif
    {
        IDirect3DTexture9_UnlockRect(texturedata->texture.staging, 0);
        texturedata->texture.dirty = true;
        if (data->drawstate.texture == texture) {
            data->drawstate.texture = NULL;
            data->drawstate.shader = SHADER_NONE;
            data->drawstate.shader_params = NULL;
            IDirect3DDevice9_SetPixelShader(data->device, NULL);
            IDirect3DDevice9_SetTexture(data->device, 0, NULL);
        }
    }
}

static bool D3D_SetRenderTargetInternal(SDL_Renderer *renderer, SDL_Texture *texture)
{
    D3D_RenderData *data = (D3D_RenderData *)renderer->internal;
    D3D_TextureData *texturedata;
    D3D_TextureRep *texturerep;
    HRESULT result;
    IDirect3DDevice9 *device = data->device;

    // Release the previous render target if it wasn't the default one
    if (data->currentRenderTarget) {
        IDirect3DSurface9_Release(data->currentRenderTarget);
        data->currentRenderTarget = NULL;
    }

    if (!texture) {
        IDirect3DDevice9_SetRenderTarget(data->device, 0, data->defaultRenderTarget);
        return true;
    }

    texturedata = (D3D_TextureData *)texture->internal;
    if (!texturedata) {
        return SDL_SetError("Texture is not currently available");
    }

    // Make sure the render target is updated if it was locked and written to
    texturerep = &texturedata->texture;
    if (texturerep->dirty && texturerep->staging) {
        if (!texturerep->texture) {
            result = IDirect3DDevice9_CreateTexture(device, texturerep->w, texturerep->h, 1, texturerep->usage,
                                                    PixelFormatToD3DFMT(texturerep->format), D3DPOOL_DEFAULT, &texturerep->texture, NULL);
            if (FAILED(result)) {
                return D3D_SetError("CreateTexture(D3DPOOL_DEFAULT)", result);
            }
        }

        result = IDirect3DDevice9_UpdateTexture(device, (IDirect3DBaseTexture9 *)texturerep->staging, (IDirect3DBaseTexture9 *)texturerep->texture);
        if (FAILED(result)) {
            return D3D_SetError("UpdateTexture()", result);
        }
        texturerep->dirty = false;
    }

    result = IDirect3DTexture9_GetSurfaceLevel(texturedata->texture.texture, 0, &data->currentRenderTarget);
    if (FAILED(result)) {
        return D3D_SetError("GetSurfaceLevel()", result);
    }
    result = IDirect3DDevice9_SetRenderTarget(data->device, 0, data->currentRenderTarget);
    if (FAILED(result)) {
        return D3D_SetError("SetRenderTarget()", result);
    }

    return true;
}

static bool D3D_SetRenderTarget(SDL_Renderer *renderer, SDL_Texture *texture)
{
    if (!D3D_ActivateRenderer(renderer)) {
        return false;
    }

    return D3D_SetRenderTargetInternal(renderer, texture);
}

static bool D3D_QueueNoOp(SDL_Renderer *renderer, SDL_RenderCommand *cmd)
{
    return true; // nothing to do in this backend.
}

static bool D3D_QueueDrawPoints(SDL_Renderer *renderer, SDL_RenderCommand *cmd, const SDL_FPoint *points, int count)
{
    const DWORD color = D3DCOLOR_COLORVALUE(cmd->data.draw.color.r * cmd->data.draw.color_scale,
                                            cmd->data.draw.color.g * cmd->data.draw.color_scale,
                                            cmd->data.draw.color.b * cmd->data.draw.color_scale,
                                            cmd->data.draw.color.a);
    const size_t vertslen = count * sizeof(Vertex);
    Vertex *verts = (Vertex *)SDL_AllocateRenderVertices(renderer, vertslen, 0, &cmd->data.draw.first);
    int i;

    if (!verts) {
        return false;
    }

    SDL_memset(verts, '\0', vertslen);
    cmd->data.draw.count = count;

    for (i = 0; i < count; i++, verts++, points++) {
        verts->x = points->x;
        verts->y = points->y;
        verts->color = color;
    }

    return true;
}

static bool D3D_QueueGeometry(SDL_Renderer *renderer, SDL_RenderCommand *cmd, SDL_Texture *texture,
                             const float *xy, int xy_stride, const SDL_FColor *color, int color_stride, const float *uv, int uv_stride,
                             int num_vertices, const void *indices, int num_indices, int size_indices,
                             float scale_x, float scale_y)
{
    int i;
    int count = indices ? num_indices : num_vertices;
    Vertex *verts = (Vertex *)SDL_AllocateRenderVertices(renderer, count * sizeof(Vertex), 0, &cmd->data.draw.first);
    const float color_scale = cmd->data.draw.color_scale;

    if (!verts) {
        return false;
    }

    cmd->data.draw.count = count;
    size_indices = indices ? size_indices : 0;

    for (i = 0; i < count; i++) {
        int j;
        float *xy_;
        SDL_FColor *col_;
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
        col_ = (SDL_FColor *)((char *)color + j * color_stride);

        verts->x = xy_[0] * scale_x - 0.5f;
        verts->y = xy_[1] * scale_y - 0.5f;
        verts->z = 0.0f;
        verts->color = D3DCOLOR_COLORVALUE(col_->r * color_scale, col_->g * color_scale, col_->b * color_scale, col_->a);

        if (texture) {
            float *uv_ = (float *)((char *)uv + j * uv_stride);
            verts->u = uv_[0];
            verts->v = uv_[1];
        } else {
            verts->u = 0.0f;
            verts->v = 0.0f;
        }

        verts += 1;
    }
    return true;
}

static bool UpdateDirtyTexture(IDirect3DDevice9 *device, D3D_TextureRep *texture)
{
    if (texture->dirty && texture->staging) {
        HRESULT result;
        if (!texture->texture) {
            result = IDirect3DDevice9_CreateTexture(device, texture->w, texture->h, 1, texture->usage,
                                                    PixelFormatToD3DFMT(texture->format), D3DPOOL_DEFAULT, &texture->texture, NULL);
            if (FAILED(result)) {
                return D3D_SetError("CreateTexture(D3DPOOL_DEFAULT)", result);
            }
        }

        result = IDirect3DDevice9_UpdateTexture(device, (IDirect3DBaseTexture9 *)texture->staging, (IDirect3DBaseTexture9 *)texture->texture);
        if (FAILED(result)) {
            return D3D_SetError("UpdateTexture()", result);
        }
        texture->dirty = false;
    }
    return true;
}

static bool BindTextureRep(IDirect3DDevice9 *device, D3D_TextureRep *texture, DWORD sampler)
{
    HRESULT result;
    UpdateDirtyTexture(device, texture);
    result = IDirect3DDevice9_SetTexture(device, sampler, (IDirect3DBaseTexture9 *)texture->texture);
    if (FAILED(result)) {
        return D3D_SetError("SetTexture()", result);
    }
    return true;
}

static void UpdateTextureScaleMode(D3D_RenderData *data, SDL_ScaleMode scaleMode, unsigned index)
{
    if (scaleMode != data->scaleMode[index]) {
        switch (scaleMode) {
        case SDL_SCALEMODE_PIXELART:
        case SDL_SCALEMODE_NEAREST:
            IDirect3DDevice9_SetSamplerState(data->device, index, D3DSAMP_MINFILTER, D3DTEXF_POINT);
            IDirect3DDevice9_SetSamplerState(data->device, index, D3DSAMP_MAGFILTER, D3DTEXF_POINT);
            break;
        case SDL_SCALEMODE_LINEAR:
            IDirect3DDevice9_SetSamplerState(data->device, index, D3DSAMP_MINFILTER, D3DTEXF_LINEAR);
            IDirect3DDevice9_SetSamplerState(data->device, index, D3DSAMP_MAGFILTER, D3DTEXF_LINEAR);
            break;
        default:
            break;
        }
        data->scaleMode[index] = scaleMode;
    }
}

static DWORD TranslateAddressMode(SDL_TextureAddressMode addressMode)
{
    switch (addressMode) {
    case SDL_TEXTURE_ADDRESS_CLAMP:
        return D3DTADDRESS_CLAMP;
    case SDL_TEXTURE_ADDRESS_WRAP:
        return D3DTADDRESS_WRAP;
    default:
        SDL_assert(!"Unknown texture address mode");
        return D3DTADDRESS_CLAMP;
    }
}

static void UpdateTextureAddressMode(D3D_RenderData *data, SDL_TextureAddressMode addressModeU, SDL_TextureAddressMode addressModeV, unsigned index)
{
    if (addressModeU != data->addressModeU[index]) {
        IDirect3DDevice9_SetSamplerState(data->device, index, D3DSAMP_ADDRESSU, TranslateAddressMode(addressModeU));
        data->addressModeU[index] = addressModeU;
    }
    if (addressModeV != data->addressModeV[index]) {
        IDirect3DDevice9_SetSamplerState(data->device, index, D3DSAMP_ADDRESSV, TranslateAddressMode(addressModeV));
        data->addressModeV[index] = addressModeV;
    }
}

static bool SetupTextureState(D3D_RenderData *data, SDL_Texture *texture, D3D9_Shader *shader, const float **shader_params)
{
    D3D_TextureData *texturedata = (D3D_TextureData *)texture->internal;

    if (!texturedata) {
        return SDL_SetError("Texture is not currently available");
    }

    *shader = texturedata->shader;
    *shader_params = texturedata->shader_params;

    if (!BindTextureRep(data->device, &texturedata->texture, 0)) {
        return false;
    }
#ifdef SDL_HAVE_YUV
    if (texturedata->yuv) {
        if (!BindTextureRep(data->device, &texturedata->utexture, 1)) {
            return false;
        }
        if (!BindTextureRep(data->device, &texturedata->vtexture, 2)) {
            return false;
        }
    }
#endif
    return true;
}

static bool SetDrawState(D3D_RenderData *data, const SDL_RenderCommand *cmd)
{
    SDL_Texture *texture = cmd->data.draw.texture;
    const SDL_BlendMode blend = cmd->data.draw.blend;

    if (texture != data->drawstate.texture) {
#ifdef SDL_HAVE_YUV
        D3D_TextureData *oldtexturedata = data->drawstate.texture ? (D3D_TextureData *)data->drawstate.texture->internal : NULL;
        D3D_TextureData *newtexturedata = texture ? (D3D_TextureData *)texture->internal : NULL;
#endif
        D3D9_Shader shader = SHADER_NONE;
        const float *shader_params = NULL;

        // disable any enabled textures we aren't going to use, let SetupTextureState() do the rest.
        if (!texture) {
            IDirect3DDevice9_SetTexture(data->device, 0, NULL);
        }
#ifdef SDL_HAVE_YUV
        if ((!newtexturedata || !newtexturedata->yuv) && (oldtexturedata && oldtexturedata->yuv)) {
            IDirect3DDevice9_SetTexture(data->device, 1, NULL);
            IDirect3DDevice9_SetTexture(data->device, 2, NULL);
        }
#endif
        if (texture && !SetupTextureState(data, texture, &shader, &shader_params)) {
            return false;
        }

#ifdef SDL_HAVE_YUV
        if (shader != data->drawstate.shader) {
            const HRESULT result = IDirect3DDevice9_SetPixelShader(data->device, data->shaders[shader]);
            if (FAILED(result)) {
                return D3D_SetError("IDirect3DDevice9_SetPixelShader()", result);
            }
            data->drawstate.shader = shader;
        }

        if (shader_params != data->drawstate.shader_params) {
            if (shader_params) {
                const UINT shader_params_length = 4; // The YUV shader takes 4 float4 parameters
                const HRESULT result = IDirect3DDevice9_SetPixelShaderConstantF(data->device, 0, shader_params, shader_params_length);
                if (FAILED(result)) {
                    return D3D_SetError("IDirect3DDevice9_SetPixelShaderConstantF()", result);
                }
            }
            data->drawstate.shader_params = shader_params;
        }
#endif // SDL_HAVE_YUV

        data->drawstate.texture = texture;
    } else if (texture) {
        D3D_TextureData *texturedata = (D3D_TextureData *)texture->internal;
        if (texturedata) {
            UpdateDirtyTexture(data->device, &texturedata->texture);
#ifdef SDL_HAVE_YUV
            if (texturedata->yuv) {
                UpdateDirtyTexture(data->device, &texturedata->utexture);
                UpdateDirtyTexture(data->device, &texturedata->vtexture);
            }
#endif // SDL_HAVE_YUV
        }
    }

    if (texture) {
        UpdateTextureScaleMode(data, cmd->data.draw.texture_scale_mode, 0);
        UpdateTextureAddressMode(data, cmd->data.draw.texture_address_mode_u, cmd->data.draw.texture_address_mode_v, 0);

#ifdef SDL_HAVE_YUV
        D3D_TextureData *texturedata = (D3D_TextureData *)texture->internal;
        if (texturedata && texturedata->yuv) {
            UpdateTextureScaleMode(data, cmd->data.draw.texture_scale_mode, 1);
            UpdateTextureScaleMode(data, cmd->data.draw.texture_scale_mode, 2);
            UpdateTextureAddressMode(data, cmd->data.draw.texture_address_mode_u, cmd->data.draw.texture_address_mode_v, 1);
            UpdateTextureAddressMode(data, cmd->data.draw.texture_address_mode_u, cmd->data.draw.texture_address_mode_v, 2);
        }
#endif // SDL_HAVE_YUV
    }

    if (blend != data->drawstate.blend) {
        if (blend == SDL_BLENDMODE_NONE) {
            IDirect3DDevice9_SetRenderState(data->device, D3DRS_ALPHABLENDENABLE, FALSE);
        } else {
            IDirect3DDevice9_SetRenderState(data->device, D3DRS_ALPHABLENDENABLE, TRUE);
            IDirect3DDevice9_SetRenderState(data->device, D3DRS_SRCBLEND,
                                            GetBlendFunc(SDL_GetBlendModeSrcColorFactor(blend)));
            IDirect3DDevice9_SetRenderState(data->device, D3DRS_DESTBLEND,
                                            GetBlendFunc(SDL_GetBlendModeDstColorFactor(blend)));
            IDirect3DDevice9_SetRenderState(data->device, D3DRS_BLENDOP,
                                            GetBlendEquation(SDL_GetBlendModeColorOperation(blend)));
            if (data->enableSeparateAlphaBlend) {
                IDirect3DDevice9_SetRenderState(data->device, D3DRS_SRCBLENDALPHA,
                                                GetBlendFunc(SDL_GetBlendModeSrcAlphaFactor(blend)));
                IDirect3DDevice9_SetRenderState(data->device, D3DRS_DESTBLENDALPHA,
                                                GetBlendFunc(SDL_GetBlendModeDstAlphaFactor(blend)));
                IDirect3DDevice9_SetRenderState(data->device, D3DRS_BLENDOPALPHA,
                                                GetBlendEquation(SDL_GetBlendModeAlphaOperation(blend)));
            }
        }

        data->drawstate.blend = blend;
    }

    if (data->drawstate.viewport_dirty) {
        const SDL_Rect *viewport = &data->drawstate.viewport;
        D3DVIEWPORT9 d3dviewport;
        d3dviewport.X = viewport->x;
        d3dviewport.Y = viewport->y;
        d3dviewport.Width = viewport->w;
        d3dviewport.Height = viewport->h;
        d3dviewport.MinZ = 0.0f;
        d3dviewport.MaxZ = 1.0f;
        IDirect3DDevice9_SetViewport(data->device, &d3dviewport);

        // Set an orthographic projection matrix
        if (viewport->w && viewport->h) {
            D3DMATRIX d3dmatrix;
            SDL_zero(d3dmatrix);
            d3dmatrix.m[0][0] = 2.0f / viewport->w;
            d3dmatrix.m[1][1] = -2.0f / viewport->h;
            d3dmatrix.m[2][2] = 1.0f;
            d3dmatrix.m[3][0] = -1.0f;
            d3dmatrix.m[3][1] = 1.0f;
            d3dmatrix.m[3][3] = 1.0f;
            IDirect3DDevice9_SetTransform(data->device, D3DTS_PROJECTION, &d3dmatrix);
        }

        data->drawstate.viewport_dirty = false;
    }

    if (data->drawstate.cliprect_enabled_dirty) {
        IDirect3DDevice9_SetRenderState(data->device, D3DRS_SCISSORTESTENABLE, data->drawstate.cliprect_enabled ? TRUE : FALSE);
        data->drawstate.cliprect_enabled_dirty = false;
    }

    if (data->drawstate.cliprect_dirty) {
        const SDL_Rect *viewport = &data->drawstate.viewport;
        const SDL_Rect *rect = &data->drawstate.cliprect;
        RECT d3drect;
        d3drect.left = (LONG)viewport->x + rect->x;
        d3drect.top = (LONG)viewport->y + rect->y;
        d3drect.right = (LONG)viewport->x + rect->x + rect->w;
        d3drect.bottom = (LONG)viewport->y + rect->y + rect->h;
        IDirect3DDevice9_SetScissorRect(data->device, &d3drect);
        data->drawstate.cliprect_dirty = false;
    }

    return true;
}

static void D3D_InvalidateCachedState(SDL_Renderer *renderer)
{
    D3D_RenderData *data = (D3D_RenderData *)renderer->internal;
    data->drawstate.viewport_dirty = true;
    data->drawstate.cliprect_enabled_dirty = true;
    data->drawstate.cliprect_dirty = true;
    data->drawstate.blend = SDL_BLENDMODE_INVALID;
    data->drawstate.texture = NULL;
    data->drawstate.shader = SHADER_NONE;
    data->drawstate.shader_params = NULL;
}

static bool D3D_RunCommandQueue(SDL_Renderer *renderer, SDL_RenderCommand *cmd, void *vertices, size_t vertsize)
{
    D3D_RenderData *data = (D3D_RenderData *)renderer->internal;
    const int vboidx = data->currentVertexBuffer;
    IDirect3DVertexBuffer9 *vbo = NULL;
    const bool istarget = renderer->target != NULL;

    if (!D3D_ActivateRenderer(renderer)) {
        return false;
    }

    if (vertsize > 0) {
        // upload the new VBO data for this set of commands.
        vbo = data->vertexBuffers[vboidx];
        if (data->vertexBufferSize[vboidx] < vertsize) {
            const DWORD usage = D3DUSAGE_DYNAMIC | D3DUSAGE_WRITEONLY;
            const DWORD fvf = D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1;
            if (vbo) {
                IDirect3DVertexBuffer9_Release(vbo);
            }

            if (FAILED(IDirect3DDevice9_CreateVertexBuffer(data->device, (UINT)vertsize, usage, fvf, D3DPOOL_DEFAULT, &vbo, NULL))) {
                vbo = NULL;
            }
            data->vertexBuffers[vboidx] = vbo;
            data->vertexBufferSize[vboidx] = vbo ? vertsize : 0;
        }

        if (vbo) {
            void *ptr;
            if (FAILED(IDirect3DVertexBuffer9_Lock(vbo, 0, (UINT)vertsize, &ptr, D3DLOCK_DISCARD))) {
                vbo = NULL; // oh well, we'll do immediate mode drawing.  :(
            } else {
                SDL_memcpy(ptr, vertices, vertsize);
                if (FAILED(IDirect3DVertexBuffer9_Unlock(vbo))) {
                    vbo = NULL; // oh well, we'll do immediate mode drawing.  :(
                }
            }
        }

        // cycle through a few VBOs so D3D has some time with the data before we replace it.
        if (vbo) {
            data->currentVertexBuffer++;
            if (data->currentVertexBuffer >= SDL_arraysize(data->vertexBuffers)) {
                data->currentVertexBuffer = 0;
            }
        } else if (!data->reportedVboProblem) {
            SDL_LogError(SDL_LOG_CATEGORY_RENDER, "SDL failed to get a vertex buffer for this Direct3D 9 rendering batch!");
            SDL_LogError(SDL_LOG_CATEGORY_RENDER, "Dropping back to a slower method.");
            SDL_LogError(SDL_LOG_CATEGORY_RENDER, "This might be a brief hiccup, but if performance is bad, this is probably why.");
            SDL_LogError(SDL_LOG_CATEGORY_RENDER, "This error will not be logged again for this renderer.");
            data->reportedVboProblem = true;
        }
    }

    IDirect3DDevice9_SetStreamSource(data->device, 0, vbo, 0, sizeof(Vertex));

    while (cmd) {
        switch (cmd->command) {
        case SDL_RENDERCMD_SETDRAWCOLOR:
        {
            /* currently this is sent with each vertex, but if we move to
               shaders, we can put this in a uniform here and reduce vertex
               buffer bandwidth */
            break;
        }

        case SDL_RENDERCMD_SETVIEWPORT:
        {
            SDL_Rect *viewport = &data->drawstate.viewport;
            if (SDL_memcmp(viewport, &cmd->data.viewport.rect, sizeof(cmd->data.viewport.rect)) != 0) {
                SDL_copyp(viewport, &cmd->data.viewport.rect);
                data->drawstate.viewport_dirty = true;
                data->drawstate.cliprect_dirty = true;
            }
            break;
        }

        case SDL_RENDERCMD_SETCLIPRECT:
        {
            const SDL_Rect *rect = &cmd->data.cliprect.rect;
            if (data->drawstate.cliprect_enabled != cmd->data.cliprect.enabled) {
                data->drawstate.cliprect_enabled = cmd->data.cliprect.enabled;
                data->drawstate.cliprect_enabled_dirty = true;
            }

            if (SDL_memcmp(&data->drawstate.cliprect, rect, sizeof(*rect)) != 0) {
                SDL_copyp(&data->drawstate.cliprect, rect);
                data->drawstate.cliprect_dirty = true;
            }
            break;
        }

        case SDL_RENDERCMD_CLEAR:
        {
            const DWORD color = D3DCOLOR_COLORVALUE(cmd->data.color.color.r * cmd->data.color.color_scale,
                                                    cmd->data.color.color.g * cmd->data.color.color_scale,
                                                    cmd->data.color.color.b * cmd->data.color.color_scale,
                                                    cmd->data.color.color.a);
            const SDL_Rect *viewport = &data->drawstate.viewport;
            const int backw = istarget ? renderer->target->w : data->pparams.BackBufferWidth;
            const int backh = istarget ? renderer->target->h : data->pparams.BackBufferHeight;
            const bool viewport_equal = ((viewport->x == 0) && (viewport->y == 0) && (viewport->w == backw) && (viewport->h == backh));

            if (data->drawstate.cliprect_enabled || data->drawstate.cliprect_enabled_dirty) {
                IDirect3DDevice9_SetRenderState(data->device, D3DRS_SCISSORTESTENABLE, FALSE);
                data->drawstate.cliprect_enabled_dirty = data->drawstate.cliprect_enabled;
            }

            // Don't reset the viewport if we don't have to!
            if (!data->drawstate.viewport_dirty && viewport_equal) {
                IDirect3DDevice9_Clear(data->device, 0, NULL, D3DCLEAR_TARGET, color, 0.0f, 0);
            } else {
                // Clear is defined to clear the entire render target
                D3DVIEWPORT9 wholeviewport = { 0, 0, 0, 0, 0.0f, 1.0f };
                wholeviewport.Width = backw;
                wholeviewport.Height = backh;
                IDirect3DDevice9_SetViewport(data->device, &wholeviewport);
                data->drawstate.viewport_dirty = true; // we still need to (re)set orthographic projection, so always mark it dirty.
                IDirect3DDevice9_Clear(data->device, 0, NULL, D3DCLEAR_TARGET, color, 0.0f, 0);
            }

            break;
        }

        case SDL_RENDERCMD_DRAW_POINTS:
        {
            const size_t count = cmd->data.draw.count;
            const size_t first = cmd->data.draw.first;
            SetDrawState(data, cmd);
            if (vbo) {
                IDirect3DDevice9_DrawPrimitive(data->device, D3DPT_POINTLIST, (UINT)(first / sizeof(Vertex)), (UINT)count);
            } else {
                const Vertex *verts = (Vertex *)(((Uint8 *)vertices) + first);
                IDirect3DDevice9_DrawPrimitiveUP(data->device, D3DPT_POINTLIST, (UINT)count, verts, sizeof(Vertex));
            }
            break;
        }

        case SDL_RENDERCMD_DRAW_LINES:
        {
            const size_t count = cmd->data.draw.count;
            const size_t first = cmd->data.draw.first;
            const Vertex *verts = (Vertex *)(((Uint8 *)vertices) + first);

            /* DirectX 9 has the same line rasterization semantics as GDI,
               so we need to close the endpoint of the line with a second draw call.
               NOLINTNEXTLINE(clang-analyzer-core.NullDereference): FIXME: Can verts truly not be NULL ? */
            const bool close_endpoint = ((count == 2) || (verts[0].x != verts[count - 1].x) || (verts[0].y != verts[count - 1].y));

            SetDrawState(data, cmd);

            if (vbo) {
                IDirect3DDevice9_DrawPrimitive(data->device, D3DPT_LINESTRIP, (UINT)(first / sizeof(Vertex)), (UINT)(count - 1));
                if (close_endpoint) {
                    IDirect3DDevice9_DrawPrimitive(data->device, D3DPT_POINTLIST, (UINT)((first / sizeof(Vertex)) + (count - 1)), 1);
                }
            } else {
                IDirect3DDevice9_DrawPrimitiveUP(data->device, D3DPT_LINESTRIP, (UINT)(count - 1), verts, sizeof(Vertex));
                if (close_endpoint) {
                    IDirect3DDevice9_DrawPrimitiveUP(data->device, D3DPT_POINTLIST, 1, &verts[count - 1], sizeof(Vertex));
                }
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
            const size_t count = cmd->data.draw.count;
            const size_t first = cmd->data.draw.first;
            SetDrawState(data, cmd);
            if (vbo) {
                IDirect3DDevice9_DrawPrimitive(data->device, D3DPT_TRIANGLELIST, (UINT)(first / sizeof(Vertex)), (UINT)count / 3);
            } else {
                const Vertex *verts = (Vertex *)(((Uint8 *)vertices) + first);
                IDirect3DDevice9_DrawPrimitiveUP(data->device, D3DPT_TRIANGLELIST, (UINT)count / 3, verts, sizeof(Vertex));
            }
            break;
        }

        case SDL_RENDERCMD_NO_OP:
            break;
        }

        cmd = cmd->next;
    }

    return true;
}

static SDL_Surface *D3D_RenderReadPixels(SDL_Renderer *renderer, const SDL_Rect *rect)
{
    D3D_RenderData *data = (D3D_RenderData *)renderer->internal;
    D3DSURFACE_DESC desc;
    LPDIRECT3DSURFACE9 backBuffer;
    LPDIRECT3DSURFACE9 surface;
    RECT d3drect;
    D3DLOCKED_RECT locked;
    HRESULT result;
    SDL_Surface *output;

    if (data->currentRenderTarget) {
        backBuffer = data->currentRenderTarget;
    } else {
        backBuffer = data->defaultRenderTarget;
    }

    result = IDirect3DSurface9_GetDesc(backBuffer, &desc);
    if (FAILED(result)) {
        D3D_SetError("GetDesc()", result);
        return NULL;
    }

    result = IDirect3DDevice9_CreateOffscreenPlainSurface(data->device, desc.Width, desc.Height, desc.Format, D3DPOOL_SYSTEMMEM, &surface, NULL);
    if (FAILED(result)) {
        D3D_SetError("CreateOffscreenPlainSurface()", result);
        return NULL;
    }

    result = IDirect3DDevice9_GetRenderTargetData(data->device, backBuffer, surface);
    if (FAILED(result)) {
        IDirect3DSurface9_Release(surface);
        D3D_SetError("GetRenderTargetData()", result);
        return NULL;
    }

    d3drect.left = rect->x;
    d3drect.right = (LONG)rect->x + rect->w;
    d3drect.top = rect->y;
    d3drect.bottom = (LONG)rect->y + rect->h;

    result = IDirect3DSurface9_LockRect(surface, &locked, &d3drect, D3DLOCK_READONLY);
    if (FAILED(result)) {
        IDirect3DSurface9_Release(surface);
        D3D_SetError("LockRect()", result);
        return NULL;
    }

    output = SDL_DuplicatePixels(rect->w, rect->h, D3DFMTToPixelFormat(desc.Format), SDL_COLORSPACE_SRGB, locked.pBits, locked.Pitch);

    IDirect3DSurface9_UnlockRect(surface);

    IDirect3DSurface9_Release(surface);

    return output;
}

static bool D3D_RenderPresent(SDL_Renderer *renderer)
{
    D3D_RenderData *data = (D3D_RenderData *)renderer->internal;
    HRESULT result;

    if (!data->beginScene) {
        IDirect3DDevice9_EndScene(data->device);
        data->beginScene = true;
    }

    result = IDirect3DDevice9_TestCooperativeLevel(data->device);
    if (result == D3DERR_DEVICELOST) {
        // We'll reset later
        return false;
    }
    if (result == D3DERR_DEVICENOTRESET) {
        D3D_Reset(renderer);
    }
    result = IDirect3DDevice9_Present(data->device, NULL, NULL, NULL, NULL);
    if (FAILED(result)) {
        return D3D_SetError("Present()", result);
    }
    return true;
}

static void D3D_DestroyTexture(SDL_Renderer *renderer, SDL_Texture *texture)
{
    D3D_RenderData *renderdata = (D3D_RenderData *)renderer->internal;
    D3D_TextureData *data = (D3D_TextureData *)texture->internal;

    if (renderdata->drawstate.texture == texture) {
        renderdata->drawstate.texture = NULL;
        renderdata->drawstate.shader = SHADER_NONE;
        renderdata->drawstate.shader_params = NULL;
        IDirect3DDevice9_SetPixelShader(renderdata->device, NULL);
        IDirect3DDevice9_SetTexture(renderdata->device, 0, NULL);
#ifdef SDL_HAVE_YUV
        if (data->yuv) {
            IDirect3DDevice9_SetTexture(renderdata->device, 1, NULL);
            IDirect3DDevice9_SetTexture(renderdata->device, 2, NULL);
        }
#endif
    }

    if (!data) {
        return;
    }

    D3D_DestroyTextureRep(&data->texture);
#ifdef SDL_HAVE_YUV
    D3D_DestroyTextureRep(&data->utexture);
    D3D_DestroyTextureRep(&data->vtexture);
    SDL_free(data->pixels);
#endif
    SDL_free(data);
    texture->internal = NULL;
}

static void D3D_DestroyRenderer(SDL_Renderer *renderer)
{
    D3D_RenderData *data = (D3D_RenderData *)renderer->internal;

    if (data) {
        int i;

        // Release the render target
        if (data->defaultRenderTarget) {
            IDirect3DSurface9_Release(data->defaultRenderTarget);
            data->defaultRenderTarget = NULL;
        }
        if (data->currentRenderTarget) {
            IDirect3DSurface9_Release(data->currentRenderTarget);
            data->currentRenderTarget = NULL;
        }
#ifdef SDL_HAVE_YUV
        for (i = 0; i < SDL_arraysize(data->shaders); ++i) {
            if (data->shaders[i]) {
                IDirect3DPixelShader9_Release(data->shaders[i]);
                data->shaders[i] = NULL;
            }
        }
#endif
        // Release all vertex buffers
        for (i = 0; i < SDL_arraysize(data->vertexBuffers); ++i) {
            if (data->vertexBuffers[i]) {
                IDirect3DVertexBuffer9_Release(data->vertexBuffers[i]);
            }
            data->vertexBuffers[i] = NULL;
        }
        if (data->device) {
            IDirect3DDevice9_Release(data->device);
            data->device = NULL;
        }
        if (data->d3d) {
            IDirect3D9_Release(data->d3d);
            SDL_UnloadObject(data->d3dDLL);
        }
        SDL_free(data);
    }
}

static bool D3D_Reset(SDL_Renderer *renderer)
{
    D3D_RenderData *data = (D3D_RenderData *)renderer->internal;
    const Float4X4 d3dmatrix = MatrixIdentity();
    HRESULT result;
    SDL_Texture *texture;
    int i;

    // Cancel any scene that we've started
    if (!data->beginScene) {
        IDirect3DDevice9_EndScene(data->device);
        data->beginScene = true;
    }

    // Release the default render target before reset
    if (data->defaultRenderTarget) {
        IDirect3DSurface9_Release(data->defaultRenderTarget);
        data->defaultRenderTarget = NULL;
    }
    if (data->currentRenderTarget) {
        IDirect3DSurface9_Release(data->currentRenderTarget);
        data->currentRenderTarget = NULL;
    }

    // Release application render targets
    for (texture = renderer->textures; texture; texture = texture->next) {
        if (texture->access == SDL_TEXTUREACCESS_TARGET) {
            D3D_DestroyTexture(renderer, texture);
        } else {
            D3D_RecreateTexture(renderer, texture);
        }
    }

    // Release all vertex buffers
    for (i = 0; i < SDL_arraysize(data->vertexBuffers); ++i) {
        if (data->vertexBuffers[i]) {
            IDirect3DVertexBuffer9_Release(data->vertexBuffers[i]);
        }
        data->vertexBuffers[i] = NULL;
        data->vertexBufferSize[i] = 0;
    }

    result = IDirect3DDevice9_Reset(data->device, &data->pparams);
    if (FAILED(result)) {
        if (result == D3DERR_DEVICELOST) {
            // Don't worry about it, we'll reset later...
            return true;
        } else {
            return D3D_SetError("Reset()", result);
        }
    }

    // Allocate application render targets
    for (texture = renderer->textures; texture; texture = texture->next) {
        if (texture->access == SDL_TEXTUREACCESS_TARGET) {
            D3D_CreateTexture(renderer, texture, 0);
        }
    }

    IDirect3DDevice9_GetRenderTarget(data->device, 0, &data->defaultRenderTarget);
    D3D_InitRenderState(data);
    D3D_SetRenderTargetInternal(renderer, renderer->target);

    D3D_InvalidateCachedState(renderer);

    IDirect3DDevice9_SetTransform(data->device, D3DTS_VIEW, (D3DMATRIX *)&d3dmatrix);

    // Let the application know that render targets were reset
    {
        SDL_Event event;
        SDL_zero(event);
        event.type = SDL_EVENT_RENDER_TARGETS_RESET;
        event.render.windowID = SDL_GetWindowID(SDL_GetRenderWindow(renderer));
        SDL_PushEvent(&event);
    }

    return true;
}

static bool D3D_SetVSync(SDL_Renderer *renderer, const int vsync)
{
    D3D_RenderData *data = (D3D_RenderData *)renderer->internal;

    DWORD PresentationInterval;
    switch (vsync) {
    case 0:
        PresentationInterval = D3DPRESENT_INTERVAL_IMMEDIATE;
        break;
    case 1:
        PresentationInterval = D3DPRESENT_INTERVAL_ONE;
        break;
    case 2:
        PresentationInterval = D3DPRESENT_INTERVAL_TWO;
        break;
    case 3:
        PresentationInterval = D3DPRESENT_INTERVAL_THREE;
        break;
    case 4:
        PresentationInterval = D3DPRESENT_INTERVAL_FOUR;
        break;
    default:
        return SDL_Unsupported();
    }

    D3DCAPS9 caps;
    HRESULT result = IDirect3D9_GetDeviceCaps(data->d3d, data->adapter, D3DDEVTYPE_HAL, &caps);
    if (FAILED(result)) {
        return D3D_SetError("GetDeviceCaps()", result);
    }
    if (!(caps.PresentationIntervals & PresentationInterval)) {
        return SDL_Unsupported();
    }
    data->pparams.PresentationInterval = PresentationInterval;

    if (!D3D_Reset(renderer)) {
        // D3D_Reset will call SDL_SetError()
        return false;
    }
    return true;
}

static bool D3D_CreateRenderer(SDL_Renderer *renderer, SDL_Window *window, SDL_PropertiesID create_props)
{
    D3D_RenderData *data;
    HRESULT result;
    HWND hwnd;
    D3DPRESENT_PARAMETERS pparams;
    IDirect3DSwapChain9 *chain;
    D3DCAPS9 caps;
    DWORD device_flags;
    int w, h;
    SDL_DisplayID displayID;
    const SDL_DisplayMode *fullscreen_mode = NULL;

    hwnd = (HWND)SDL_GetPointerProperty(SDL_GetWindowProperties(window), SDL_PROP_WINDOW_WIN32_HWND_POINTER, NULL);
    if (!hwnd) {
        return SDL_SetError("Couldn't get window handle");
    }

    SDL_SetupRendererColorspace(renderer, create_props);

    if (renderer->output_colorspace != SDL_COLORSPACE_SRGB) {
        return SDL_SetError("Unsupported output colorspace");
    }

    data = (D3D_RenderData *)SDL_calloc(1, sizeof(*data));
    if (!data) {
        return false;
    }

    if (!D3D_LoadDLL(&data->d3dDLL, &data->d3d)) {
        SDL_free(data);
        return SDL_SetError("Unable to create Direct3D interface");
    }

    renderer->WindowEvent = D3D_WindowEvent;
    renderer->SupportsBlendMode = D3D_SupportsBlendMode;
    renderer->CreateTexture = D3D_CreateTexture;
    renderer->UpdateTexture = D3D_UpdateTexture;
#ifdef SDL_HAVE_YUV
    renderer->UpdateTextureYUV = D3D_UpdateTextureYUV;
#endif
    renderer->LockTexture = D3D_LockTexture;
    renderer->UnlockTexture = D3D_UnlockTexture;
    renderer->SetRenderTarget = D3D_SetRenderTarget;
    renderer->QueueSetViewport = D3D_QueueNoOp;
    renderer->QueueSetDrawColor = D3D_QueueNoOp;
    renderer->QueueDrawPoints = D3D_QueueDrawPoints;
    renderer->QueueDrawLines = D3D_QueueDrawPoints; // lines and points queue vertices the same way.
    renderer->QueueGeometry = D3D_QueueGeometry;
    renderer->InvalidateCachedState = D3D_InvalidateCachedState;
    renderer->RunCommandQueue = D3D_RunCommandQueue;
    renderer->RenderReadPixels = D3D_RenderReadPixels;
    renderer->RenderPresent = D3D_RenderPresent;
    renderer->DestroyTexture = D3D_DestroyTexture;
    renderer->DestroyRenderer = D3D_DestroyRenderer;
    renderer->SetVSync = D3D_SetVSync;
    renderer->internal = data;
    D3D_InvalidateCachedState(renderer);

    renderer->name = D3D_RenderDriver.name;
    SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_ARGB8888);

    SDL_GetWindowSizeInPixels(window, &w, &h);
    if (SDL_GetWindowFlags(window) & SDL_WINDOW_FULLSCREEN) {
        fullscreen_mode = SDL_GetWindowFullscreenMode(window);
    }

    SDL_zero(pparams);
    pparams.hDeviceWindow = hwnd;
    pparams.BackBufferWidth = w;
    pparams.BackBufferHeight = h;
    pparams.BackBufferCount = 1;
    pparams.SwapEffect = D3DSWAPEFFECT_DISCARD;

    if (fullscreen_mode) {
        pparams.Windowed = FALSE;
        pparams.BackBufferFormat = PixelFormatToD3DFMT(fullscreen_mode->format);
        pparams.FullScreen_RefreshRateInHz = (UINT)SDL_ceilf(fullscreen_mode->refresh_rate);
    } else {
        pparams.Windowed = TRUE;
        pparams.BackBufferFormat = D3DFMT_UNKNOWN;
        pparams.FullScreen_RefreshRateInHz = 0;
    }
    pparams.PresentationInterval = D3DPRESENT_INTERVAL_IMMEDIATE;

    // Get the adapter for the display that the window is on
    displayID = SDL_GetDisplayForWindow(window);
    data->adapter = SDL_GetDirect3D9AdapterIndex(displayID);

    result = IDirect3D9_GetDeviceCaps(data->d3d, data->adapter, D3DDEVTYPE_HAL, &caps);
    if (FAILED(result)) {
        return D3D_SetError("GetDeviceCaps()", result);
    }

    device_flags = D3DCREATE_FPU_PRESERVE;
    if (caps.DevCaps & D3DDEVCAPS_HWTRANSFORMANDLIGHT) {
        device_flags |= D3DCREATE_HARDWARE_VERTEXPROCESSING;
    } else {
        device_flags |= D3DCREATE_SOFTWARE_VERTEXPROCESSING;
    }

    if (SDL_GetHintBoolean(SDL_HINT_RENDER_DIRECT3D_THREADSAFE, false)) {
        device_flags |= D3DCREATE_MULTITHREADED;
    }

    result = IDirect3D9_CreateDevice(data->d3d, data->adapter,
                                     D3DDEVTYPE_HAL,
                                     pparams.hDeviceWindow,
                                     device_flags,
                                     &pparams, &data->device);
    if (FAILED(result)) {
        return D3D_SetError("CreateDevice()", result);
    }

    // Get presentation parameters to fill info
    result = IDirect3DDevice9_GetSwapChain(data->device, 0, &chain);
    if (FAILED(result)) {
        return D3D_SetError("GetSwapChain()", result);
    }
    result = IDirect3DSwapChain9_GetPresentParameters(chain, &pparams);
    if (FAILED(result)) {
        IDirect3DSwapChain9_Release(chain);
        return D3D_SetError("GetPresentParameters()", result);
    }
    IDirect3DSwapChain9_Release(chain);
    data->pparams = pparams;

    IDirect3DDevice9_GetDeviceCaps(data->device, &caps);
    SDL_SetNumberProperty(SDL_GetRendererProperties(renderer), SDL_PROP_RENDERER_MAX_TEXTURE_SIZE_NUMBER, SDL_min(caps.MaxTextureWidth, caps.MaxTextureHeight));

    if (caps.PrimitiveMiscCaps & D3DPMISCCAPS_SEPARATEALPHABLEND) {
        data->enableSeparateAlphaBlend = true;
    }

    // Store the default render target
    IDirect3DDevice9_GetRenderTarget(data->device, 0, &data->defaultRenderTarget);
    data->currentRenderTarget = NULL;

    // Set up parameters for rendering
    D3D_InitRenderState(data);
#ifdef SDL_HAVE_YUV
    if (caps.MaxSimultaneousTextures >= 3) {
        int i;
        for (i = SHADER_NONE + 1; i < SDL_arraysize(data->shaders); ++i) {
            result = D3D9_CreatePixelShader(data->device, (D3D9_Shader)i, &data->shaders[i]);
            if (FAILED(result)) {
                D3D_SetError("CreatePixelShader()", result);
            }
        }
        if (data->shaders[SHADER_YUV]) {
            SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_YV12);
            SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_IYUV);
        }
    }
#endif

    SDL_SetPointerProperty(SDL_GetRendererProperties(renderer), SDL_PROP_RENDERER_D3D9_DEVICE_POINTER, data->device);

    return true;
}

SDL_RenderDriver D3D_RenderDriver = {
    D3D_CreateRenderer, "direct3d"
};
#endif // SDL_VIDEO_RENDER_D3D
