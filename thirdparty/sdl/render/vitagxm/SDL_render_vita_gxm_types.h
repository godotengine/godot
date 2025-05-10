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

#ifndef SDL_RENDER_VITA_GXM_TYPES_H
#define SDL_RENDER_VITA_GXM_TYPES_H

#include "SDL_internal.h"

#include "../SDL_sysrender.h"

#include <psp2/kernel/processmgr.h>
#include <psp2/appmgr.h>
#include <psp2/display.h>
#include <psp2/gxm.h>
#include <psp2/types.h>
#include <psp2/kernel/sysmem.h>
#include <psp2/kernel/clib.h>

#include <string.h>

#define VITA_GXM_SCREEN_WIDTH  960
#define VITA_GXM_SCREEN_HEIGHT 544
#define VITA_GXM_SCREEN_STRIDE 960

#define VITA_GXM_COLOR_FORMAT SCE_GXM_COLOR_FORMAT_A8B8G8R8
#define VITA_GXM_PIXEL_FORMAT SCE_DISPLAY_PIXELFORMAT_A8B8G8R8

#define VITA_GXM_BUFFERS       3
#define VITA_GXM_PENDING_SWAPS 2
#define VITA_GXM_POOL_SIZE     2 * 1024 * 1024

typedef struct
{
    void *address;
    Uint8 wait_vblank;
} VITA_GXM_DisplayData;

typedef struct clear_vertex
{
    float x;
    float y;
} clear_vertex;

typedef struct color_vertex
{
    float x;
    float y;
    SDL_FColor color;
} color_vertex;

typedef struct texture_vertex
{
    float x;
    float y;
    float u;
    float v;
    SDL_FColor color;
} texture_vertex;

typedef struct gxm_texture
{
    SceGxmTexture gxm_tex;
    SceUID data_UID;
    SceGxmRenderTarget *gxm_rendertarget;
    SceGxmColorSurface gxm_colorsurface;
    SceGxmDepthStencilSurface gxm_depthstencil;
    SceUID depth_UID;
    bool cdram;
} gxm_texture;

typedef struct fragment_programs
{
    SceGxmFragmentProgram *color;
    SceGxmFragmentProgram *texture;
} fragment_programs;

typedef struct blend_fragment_programs
{
    fragment_programs blend_mode_none;
    fragment_programs blend_mode_blend;
    fragment_programs blend_mode_add;
    fragment_programs blend_mode_mod;
    fragment_programs blend_mode_mul;
} blend_fragment_programs;

typedef struct
{
    SDL_Rect viewport;
    bool viewport_dirty;
    SDL_Texture *texture;
    SDL_Texture *target;
    SDL_FColor color;
    SceGxmFragmentProgram *fragment_program;
    SceGxmVertexProgram *vertex_program;
    int last_command;

    bool cliprect_enabled_dirty;
    bool cliprect_enabled;
    bool cliprect_dirty;
    SDL_Rect cliprect;
    bool texturing;
    int drawablew;
    int drawableh;
} gxm_drawstate_cache;

typedef struct
{
    bool initialized;
    bool drawing;

    unsigned int psm;
    unsigned int bpp;

    int currentBlendMode;

    VITA_GXM_DisplayData displayData;

    SceUID vdmRingBufferUid;
    SceUID vertexRingBufferUid;
    SceUID fragmentRingBufferUid;
    SceUID fragmentUsseRingBufferUid;
    SceGxmContextParams contextParams;
    SceGxmContext *gxm_context;
    SceGxmRenderTarget *renderTarget;
    SceUID displayBufferUid[VITA_GXM_BUFFERS];
    void *displayBufferData[VITA_GXM_BUFFERS];
    SceGxmColorSurface displaySurface[VITA_GXM_BUFFERS];
    SceGxmSyncObject *displayBufferSync[VITA_GXM_BUFFERS];

    SceUID depthBufferUid;
    SceUID stencilBufferUid;
    SceGxmDepthStencilSurface depthSurface;
    void *depthBufferData;
    void *stencilBufferData;

    unsigned int backBufferIndex;
    unsigned int frontBufferIndex;

    void *pool_addr[2];
    SceUID poolUid[2];
    unsigned int pool_index;
    unsigned int current_pool;

    float ortho_matrix[4 * 4];

    SceGxmVertexProgram *colorVertexProgram;
    SceGxmFragmentProgram *colorFragmentProgram;
    SceGxmVertexProgram *textureVertexProgram;
    SceGxmFragmentProgram *textureFragmentProgram;
    SceGxmProgramParameter *clearClearColorParam;
    SceGxmProgramParameter *colorWvpParam;
    SceGxmProgramParameter *textureWvpParam;

    SceGxmShaderPatcher *shaderPatcher;
    SceGxmVertexProgram *clearVertexProgram;
    SceGxmFragmentProgram *clearFragmentProgram;

    SceGxmShaderPatcherId clearVertexProgramId;
    SceGxmShaderPatcherId clearFragmentProgramId;
    SceGxmShaderPatcherId colorVertexProgramId;
    SceGxmShaderPatcherId colorFragmentProgramId;
    SceGxmShaderPatcherId textureVertexProgramId;
    SceGxmShaderPatcherId textureFragmentProgramId;

    SceUID patcherBufferUid;
    SceUID patcherVertexUsseUid;
    SceUID patcherFragmentUsseUid;

    SceUID clearVerticesUid;
    SceUID linearIndicesUid;
    clear_vertex *clearVertices;
    uint16_t *linearIndices;

    blend_fragment_programs blendFragmentPrograms;

    gxm_drawstate_cache drawstate;
    SceClibMspace texturePool;
    SceUID texturePoolUID;
} VITA_GXM_RenderData;

typedef struct
{
    gxm_texture *tex;
    unsigned int pitch;
    unsigned int w;
    unsigned int h;
    float wscale;
    bool yuv;
    bool nv12;
    SDL_ScaleMode scale_mode;
    SDL_TextureAddressMode address_mode_u;
    SDL_TextureAddressMode address_mode_v;
} VITA_GXM_TextureData;

#endif // SDL_RENDER_VITA_GXM_TYPES_H
