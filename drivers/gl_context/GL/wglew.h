/*
** The OpenGL Extension Wrangler Library
** Copyright (C) 2002-2008, Milan Ikits <milan ikits[]ieee org>
** Copyright (C) 2002-2008, Marcelo E. Magallon <mmagallo[]debian org>
** Copyright (C) 2002, Lev Povalahev
** All rights reserved.
** 
** Redistribution and use in source and binary forms, with or without 
** modification, are permitted provided that the following conditions are met:
** 
** * Redistributions of source code must retain the above copyright notice, 
**   this list of conditions and the following disclaimer.
** * Redistributions in binary form must reproduce the above copyright notice, 
**   this list of conditions and the following disclaimer in the documentation 
**   and/or other materials provided with the distribution.
** * The name of the author may be used to endorse or promote products 
**   derived from this software without specific prior written permission.
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
** AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
** IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
** ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE 
** LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
** CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
** SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
** INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
** CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
** ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
** THE POSSIBILITY OF SUCH DAMAGE.
*/

/*
** Copyright (c) 2007 The Khronos Group Inc.
** 
** Permission is hereby granted, free of charge, to any person obtaining a
** copy of this software and/or associated documentation files (the
** "Materials"), to deal in the Materials without restriction, including
** without limitation the rights to use, copy, modify, merge, publish,
** distribute, sublicense, and/or sell copies of the Materials, and to
** permit persons to whom the Materials are furnished to do so, subject to
** the following conditions:
** 
** The above copyright notice and this permission notice shall be included
** in all copies or substantial portions of the Materials.
** 
** THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
** EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
** MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
** IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
** CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
** TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
** MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
*/

#ifndef __wglew_h__
#define __wglew_h__
#define __WGLEW_H__

#ifdef __wglext_h_
#error wglext.h included before wglew.h
#endif

#define __wglext_h_

#if !defined(WINAPI)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN 1
#  endif
#include <windows.h>
#  undef WIN32_LEAN_AND_MEAN
#endif

/*
 * GLEW_STATIC needs to be set when using the static version.
 * GLEW_BUILD is set when building the DLL version.
 */
#ifdef GLEW_STATIC
#  define GLEWAPI extern
#else
#  ifdef GLEW_BUILD
#    define GLEWAPI extern __declspec(dllexport)
#  else
#    define GLEWAPI extern __declspec(dllimport)
#  endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------- WGL_3DFX_multisample ------------------------- */

#ifndef WGL_3DFX_multisample
#define WGL_3DFX_multisample 1

#define WGL_SAMPLE_BUFFERS_3DFX 0x2060
#define WGL_SAMPLES_3DFX 0x2061

#define WGLEW_3DFX_multisample WGLEW_GET_VAR(__WGLEW_3DFX_multisample)

#endif /* WGL_3DFX_multisample */

/* ------------------------- WGL_3DL_stereo_control ------------------------ */

#ifndef WGL_3DL_stereo_control
#define WGL_3DL_stereo_control 1

#define WGL_STEREO_EMITTER_ENABLE_3DL 0x2055
#define WGL_STEREO_EMITTER_DISABLE_3DL 0x2056
#define WGL_STEREO_POLARITY_NORMAL_3DL 0x2057
#define WGL_STEREO_POLARITY_INVERT_3DL 0x2058

typedef BOOL (WINAPI * PFNWGLSETSTEREOEMITTERSTATE3DLPROC) (HDC hDC, UINT uState);

#define wglSetStereoEmitterState3DL WGLEW_GET_FUN(__wglewSetStereoEmitterState3DL)

#define WGLEW_3DL_stereo_control WGLEW_GET_VAR(__WGLEW_3DL_stereo_control)

#endif /* WGL_3DL_stereo_control */

/* ------------------------ WGL_AMD_gpu_association ------------------------ */

#ifndef WGL_AMD_gpu_association
#define WGL_AMD_gpu_association 1

#define WGL_GPU_VENDOR_AMD 0x1F00
#define WGL_GPU_RENDERER_STRING_AMD 0x1F01
#define WGL_GPU_OPENGL_VERSION_STRING_AMD 0x1F02
#define WGL_GPU_FASTEST_TARGET_GPUS_AMD 0x21A2
#define WGL_GPU_RAM_AMD 0x21A3
#define WGL_GPU_CLOCK_AMD 0x21A4
#define WGL_GPU_NUM_PIPES_AMD 0x21A5
#define WGL_GPU_NUM_SIMD_AMD 0x21A6
#define WGL_GPU_NUM_RB_AMD 0x21A7
#define WGL_GPU_NUM_SPI_AMD 0x21A8

typedef VOID (WINAPI * PFNWGLBLITCONTEXTFRAMEBUFFERAMDPROC) (HGLRC dstCtx, GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter);
typedef HGLRC (WINAPI * PFNWGLCREATEASSOCIATEDCONTEXTAMDPROC) (UINT id);
typedef HGLRC (WINAPI * PFNWGLCREATEASSOCIATEDCONTEXTATTRIBSAMDPROC) (UINT id, HGLRC hShareContext, const int* attribList);
typedef BOOL (WINAPI * PFNWGLDELETEASSOCIATEDCONTEXTAMDPROC) (HGLRC hglrc);
typedef UINT (WINAPI * PFNWGLGETCONTEXTGPUIDAMDPROC) (HGLRC hglrc);
typedef HGLRC (WINAPI * PFNWGLGETCURRENTASSOCIATEDCONTEXTAMDPROC) (void);
typedef UINT (WINAPI * PFNWGLGETGPUIDSAMDPROC) (UINT maxCount, UINT* ids);
typedef INT (WINAPI * PFNWGLGETGPUINFOAMDPROC) (UINT id, INT property, GLenum dataType, UINT size, void* data);
typedef BOOL (WINAPI * PFNWGLMAKEASSOCIATEDCONTEXTCURRENTAMDPROC) (HGLRC hglrc);

#define wglBlitContextFramebufferAMD WGLEW_GET_FUN(__wglewBlitContextFramebufferAMD)
#define wglCreateAssociatedContextAMD WGLEW_GET_FUN(__wglewCreateAssociatedContextAMD)
#define wglCreateAssociatedContextAttribsAMD WGLEW_GET_FUN(__wglewCreateAssociatedContextAttribsAMD)
#define wglDeleteAssociatedContextAMD WGLEW_GET_FUN(__wglewDeleteAssociatedContextAMD)
#define wglGetContextGPUIDAMD WGLEW_GET_FUN(__wglewGetContextGPUIDAMD)
#define wglGetCurrentAssociatedContextAMD WGLEW_GET_FUN(__wglewGetCurrentAssociatedContextAMD)
#define wglGetGPUIDsAMD WGLEW_GET_FUN(__wglewGetGPUIDsAMD)
#define wglGetGPUInfoAMD WGLEW_GET_FUN(__wglewGetGPUInfoAMD)
#define wglMakeAssociatedContextCurrentAMD WGLEW_GET_FUN(__wglewMakeAssociatedContextCurrentAMD)

#define WGLEW_AMD_gpu_association WGLEW_GET_VAR(__WGLEW_AMD_gpu_association)

#endif /* WGL_AMD_gpu_association */

/* ------------------------- WGL_ARB_buffer_region ------------------------- */

#ifndef WGL_ARB_buffer_region
#define WGL_ARB_buffer_region 1

#define WGL_FRONT_COLOR_BUFFER_BIT_ARB 0x00000001
#define WGL_BACK_COLOR_BUFFER_BIT_ARB 0x00000002
#define WGL_DEPTH_BUFFER_BIT_ARB 0x00000004
#define WGL_STENCIL_BUFFER_BIT_ARB 0x00000008

typedef HANDLE (WINAPI * PFNWGLCREATEBUFFERREGIONARBPROC) (HDC hDC, int iLayerPlane, UINT uType);
typedef VOID (WINAPI * PFNWGLDELETEBUFFERREGIONARBPROC) (HANDLE hRegion);
typedef BOOL (WINAPI * PFNWGLRESTOREBUFFERREGIONARBPROC) (HANDLE hRegion, int x, int y, int width, int height, int xSrc, int ySrc);
typedef BOOL (WINAPI * PFNWGLSAVEBUFFERREGIONARBPROC) (HANDLE hRegion, int x, int y, int width, int height);

#define wglCreateBufferRegionARB WGLEW_GET_FUN(__wglewCreateBufferRegionARB)
#define wglDeleteBufferRegionARB WGLEW_GET_FUN(__wglewDeleteBufferRegionARB)
#define wglRestoreBufferRegionARB WGLEW_GET_FUN(__wglewRestoreBufferRegionARB)
#define wglSaveBufferRegionARB WGLEW_GET_FUN(__wglewSaveBufferRegionARB)

#define WGLEW_ARB_buffer_region WGLEW_GET_VAR(__WGLEW_ARB_buffer_region)

#endif /* WGL_ARB_buffer_region */

/* ------------------------- WGL_ARB_create_context ------------------------ */

#ifndef WGL_ARB_create_context
#define WGL_ARB_create_context 1

#define WGL_CONTEXT_DEBUG_BIT_ARB 0x0001
#define WGL_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB 0x0002
#define WGL_CONTEXT_MAJOR_VERSION_ARB 0x2091
#define WGL_CONTEXT_MINOR_VERSION_ARB 0x2092
#define WGL_CONTEXT_LAYER_PLANE_ARB 0x2093
#define WGL_CONTEXT_FLAGS_ARB 0x2094

typedef HGLRC (WINAPI * PFNWGLCREATECONTEXTATTRIBSARBPROC) (HDC hDC, HGLRC hShareContext, const int* attribList);

#define wglCreateContextAttribsARB WGLEW_GET_FUN(__wglewCreateContextAttribsARB)

#define WGLEW_ARB_create_context WGLEW_GET_VAR(__WGLEW_ARB_create_context)

#endif /* WGL_ARB_create_context */

/* --------------------- WGL_ARB_create_context_profile -------------------- */

#ifndef WGL_ARB_create_context_profile
#define WGL_ARB_create_context_profile 1

#define WGL_CONTEXT_CORE_PROFILE_BIT_ARB 0x00000001
#define WGL_CONTEXT_COMPATIBILITY_PROFILE_BIT_ARB 0x00000002
#define WGL_CONTEXT_PROFILE_MASK_ARB 0x9126

#define WGLEW_ARB_create_context_profile WGLEW_GET_VAR(__WGLEW_ARB_create_context_profile)

#endif /* WGL_ARB_create_context_profile */

/* ------------------- WGL_ARB_create_context_robustness ------------------- */

#ifndef WGL_ARB_create_context_robustness
#define WGL_ARB_create_context_robustness 1

#define WGL_CONTEXT_ROBUST_ACCESS_BIT_ARB 0x00000004
#define WGL_LOSE_CONTEXT_ON_RESET_ARB 0x8252
#define WGL_CONTEXT_RESET_NOTIFICATION_STRATEGY_ARB 0x8256
#define WGL_NO_RESET_NOTIFICATION_ARB 0x8261

#define WGLEW_ARB_create_context_robustness WGLEW_GET_VAR(__WGLEW_ARB_create_context_robustness)

#endif /* WGL_ARB_create_context_robustness */

/* ----------------------- WGL_ARB_extensions_string ----------------------- */

#ifndef WGL_ARB_extensions_string
#define WGL_ARB_extensions_string 1

typedef const char* (WINAPI * PFNWGLGETEXTENSIONSSTRINGARBPROC) (HDC hdc);

#define wglGetExtensionsStringARB WGLEW_GET_FUN(__wglewGetExtensionsStringARB)

#define WGLEW_ARB_extensions_string WGLEW_GET_VAR(__WGLEW_ARB_extensions_string)

#endif /* WGL_ARB_extensions_string */

/* ------------------------ WGL_ARB_framebuffer_sRGB ----------------------- */

#ifndef WGL_ARB_framebuffer_sRGB
#define WGL_ARB_framebuffer_sRGB 1

#define WGL_FRAMEBUFFER_SRGB_CAPABLE_ARB 0x20A9

#define WGLEW_ARB_framebuffer_sRGB WGLEW_GET_VAR(__WGLEW_ARB_framebuffer_sRGB)

#endif /* WGL_ARB_framebuffer_sRGB */

/* ----------------------- WGL_ARB_make_current_read ----------------------- */

#ifndef WGL_ARB_make_current_read
#define WGL_ARB_make_current_read 1

#define ERROR_INVALID_PIXEL_TYPE_ARB 0x2043
#define ERROR_INCOMPATIBLE_DEVICE_CONTEXTS_ARB 0x2054

typedef HDC (WINAPI * PFNWGLGETCURRENTREADDCARBPROC) (VOID);
typedef BOOL (WINAPI * PFNWGLMAKECONTEXTCURRENTARBPROC) (HDC hDrawDC, HDC hReadDC, HGLRC hglrc);

#define wglGetCurrentReadDCARB WGLEW_GET_FUN(__wglewGetCurrentReadDCARB)
#define wglMakeContextCurrentARB WGLEW_GET_FUN(__wglewMakeContextCurrentARB)

#define WGLEW_ARB_make_current_read WGLEW_GET_VAR(__WGLEW_ARB_make_current_read)

#endif /* WGL_ARB_make_current_read */

/* -------------------------- WGL_ARB_multisample -------------------------- */

#ifndef WGL_ARB_multisample
#define WGL_ARB_multisample 1

#define WGL_SAMPLE_BUFFERS_ARB 0x2041
#define WGL_SAMPLES_ARB 0x2042

#define WGLEW_ARB_multisample WGLEW_GET_VAR(__WGLEW_ARB_multisample)

#endif /* WGL_ARB_multisample */

/* ---------------------------- WGL_ARB_pbuffer ---------------------------- */

#ifndef WGL_ARB_pbuffer
#define WGL_ARB_pbuffer 1

#define WGL_DRAW_TO_PBUFFER_ARB 0x202D
#define WGL_MAX_PBUFFER_PIXELS_ARB 0x202E
#define WGL_MAX_PBUFFER_WIDTH_ARB 0x202F
#define WGL_MAX_PBUFFER_HEIGHT_ARB 0x2030
#define WGL_PBUFFER_LARGEST_ARB 0x2033
#define WGL_PBUFFER_WIDTH_ARB 0x2034
#define WGL_PBUFFER_HEIGHT_ARB 0x2035
#define WGL_PBUFFER_LOST_ARB 0x2036

DECLARE_HANDLE(HPBUFFERARB);

typedef HPBUFFERARB (WINAPI * PFNWGLCREATEPBUFFERARBPROC) (HDC hDC, int iPixelFormat, int iWidth, int iHeight, const int* piAttribList);
typedef BOOL (WINAPI * PFNWGLDESTROYPBUFFERARBPROC) (HPBUFFERARB hPbuffer);
typedef HDC (WINAPI * PFNWGLGETPBUFFERDCARBPROC) (HPBUFFERARB hPbuffer);
typedef BOOL (WINAPI * PFNWGLQUERYPBUFFERARBPROC) (HPBUFFERARB hPbuffer, int iAttribute, int* piValue);
typedef int (WINAPI * PFNWGLRELEASEPBUFFERDCARBPROC) (HPBUFFERARB hPbuffer, HDC hDC);

#define wglCreatePbufferARB WGLEW_GET_FUN(__wglewCreatePbufferARB)
#define wglDestroyPbufferARB WGLEW_GET_FUN(__wglewDestroyPbufferARB)
#define wglGetPbufferDCARB WGLEW_GET_FUN(__wglewGetPbufferDCARB)
#define wglQueryPbufferARB WGLEW_GET_FUN(__wglewQueryPbufferARB)
#define wglReleasePbufferDCARB WGLEW_GET_FUN(__wglewReleasePbufferDCARB)

#define WGLEW_ARB_pbuffer WGLEW_GET_VAR(__WGLEW_ARB_pbuffer)

#endif /* WGL_ARB_pbuffer */

/* -------------------------- WGL_ARB_pixel_format ------------------------- */

#ifndef WGL_ARB_pixel_format
#define WGL_ARB_pixel_format 1

#define WGL_NUMBER_PIXEL_FORMATS_ARB 0x2000
#define WGL_DRAW_TO_WINDOW_ARB 0x2001
#define WGL_DRAW_TO_BITMAP_ARB 0x2002
#define WGL_ACCELERATION_ARB 0x2003
#define WGL_NEED_PALETTE_ARB 0x2004
#define WGL_NEED_SYSTEM_PALETTE_ARB 0x2005
#define WGL_SWAP_LAYER_BUFFERS_ARB 0x2006
#define WGL_SWAP_METHOD_ARB 0x2007
#define WGL_NUMBER_OVERLAYS_ARB 0x2008
#define WGL_NUMBER_UNDERLAYS_ARB 0x2009
#define WGL_TRANSPARENT_ARB 0x200A
#define WGL_SHARE_DEPTH_ARB 0x200C
#define WGL_SHARE_STENCIL_ARB 0x200D
#define WGL_SHARE_ACCUM_ARB 0x200E
#define WGL_SUPPORT_GDI_ARB 0x200F
#define WGL_SUPPORT_OPENGL_ARB 0x2010
#define WGL_DOUBLE_BUFFER_ARB 0x2011
#define WGL_STEREO_ARB 0x2012
#define WGL_PIXEL_TYPE_ARB 0x2013
#define WGL_COLOR_BITS_ARB 0x2014
#define WGL_RED_BITS_ARB 0x2015
#define WGL_RED_SHIFT_ARB 0x2016
#define WGL_GREEN_BITS_ARB 0x2017
#define WGL_GREEN_SHIFT_ARB 0x2018
#define WGL_BLUE_BITS_ARB 0x2019
#define WGL_BLUE_SHIFT_ARB 0x201A
#define WGL_ALPHA_BITS_ARB 0x201B
#define WGL_ALPHA_SHIFT_ARB 0x201C
#define WGL_ACCUM_BITS_ARB 0x201D
#define WGL_ACCUM_RED_BITS_ARB 0x201E
#define WGL_ACCUM_GREEN_BITS_ARB 0x201F
#define WGL_ACCUM_BLUE_BITS_ARB 0x2020
#define WGL_ACCUM_ALPHA_BITS_ARB 0x2021
#define WGL_DEPTH_BITS_ARB 0x2022
#define WGL_STENCIL_BITS_ARB 0x2023
#define WGL_AUX_BUFFERS_ARB 0x2024
#define WGL_NO_ACCELERATION_ARB 0x2025
#define WGL_GENERIC_ACCELERATION_ARB 0x2026
#define WGL_FULL_ACCELERATION_ARB 0x2027
#define WGL_SWAP_EXCHANGE_ARB 0x2028
#define WGL_SWAP_COPY_ARB 0x2029
#define WGL_SWAP_UNDEFINED_ARB 0x202A
#define WGL_TYPE_RGBA_ARB 0x202B
#define WGL_TYPE_COLORINDEX_ARB 0x202C
#define WGL_TRANSPARENT_RED_VALUE_ARB 0x2037
#define WGL_TRANSPARENT_GREEN_VALUE_ARB 0x2038
#define WGL_TRANSPARENT_BLUE_VALUE_ARB 0x2039
#define WGL_TRANSPARENT_ALPHA_VALUE_ARB 0x203A
#define WGL_TRANSPARENT_INDEX_VALUE_ARB 0x203B

typedef BOOL (WINAPI * PFNWGLCHOOSEPIXELFORMATARBPROC) (HDC hdc, const int* piAttribIList, const FLOAT *pfAttribFList, UINT nMaxFormats, int *piFormats, UINT *nNumFormats);
typedef BOOL (WINAPI * PFNWGLGETPIXELFORMATATTRIBFVARBPROC) (HDC hdc, int iPixelFormat, int iLayerPlane, UINT nAttributes, const int* piAttributes, FLOAT *pfValues);
typedef BOOL (WINAPI * PFNWGLGETPIXELFORMATATTRIBIVARBPROC) (HDC hdc, int iPixelFormat, int iLayerPlane, UINT nAttributes, const int* piAttributes, int *piValues);

#define wglChoosePixelFormatARB WGLEW_GET_FUN(__wglewChoosePixelFormatARB)
#define wglGetPixelFormatAttribfvARB WGLEW_GET_FUN(__wglewGetPixelFormatAttribfvARB)
#define wglGetPixelFormatAttribivARB WGLEW_GET_FUN(__wglewGetPixelFormatAttribivARB)

#define WGLEW_ARB_pixel_format WGLEW_GET_VAR(__WGLEW_ARB_pixel_format)

#endif /* WGL_ARB_pixel_format */

/* ----------------------- WGL_ARB_pixel_format_float ---------------------- */

#ifndef WGL_ARB_pixel_format_float
#define WGL_ARB_pixel_format_float 1

#define WGL_TYPE_RGBA_FLOAT_ARB 0x21A0

#define WGLEW_ARB_pixel_format_float WGLEW_GET_VAR(__WGLEW_ARB_pixel_format_float)

#endif /* WGL_ARB_pixel_format_float */

/* ------------------------- WGL_ARB_render_texture ------------------------ */

#ifndef WGL_ARB_render_texture
#define WGL_ARB_render_texture 1

#define WGL_BIND_TO_TEXTURE_RGB_ARB 0x2070
#define WGL_BIND_TO_TEXTURE_RGBA_ARB 0x2071
#define WGL_TEXTURE_FORMAT_ARB 0x2072
#define WGL_TEXTURE_TARGET_ARB 0x2073
#define WGL_MIPMAP_TEXTURE_ARB 0x2074
#define WGL_TEXTURE_RGB_ARB 0x2075
#define WGL_TEXTURE_RGBA_ARB 0x2076
#define WGL_NO_TEXTURE_ARB 0x2077
#define WGL_TEXTURE_CUBE_MAP_ARB 0x2078
#define WGL_TEXTURE_1D_ARB 0x2079
#define WGL_TEXTURE_2D_ARB 0x207A
#define WGL_MIPMAP_LEVEL_ARB 0x207B
#define WGL_CUBE_MAP_FACE_ARB 0x207C
#define WGL_TEXTURE_CUBE_MAP_POSITIVE_X_ARB 0x207D
#define WGL_TEXTURE_CUBE_MAP_NEGATIVE_X_ARB 0x207E
#define WGL_TEXTURE_CUBE_MAP_POSITIVE_Y_ARB 0x207F
#define WGL_TEXTURE_CUBE_MAP_NEGATIVE_Y_ARB 0x2080
#define WGL_TEXTURE_CUBE_MAP_POSITIVE_Z_ARB 0x2081
#define WGL_TEXTURE_CUBE_MAP_NEGATIVE_Z_ARB 0x2082
#define WGL_FRONT_LEFT_ARB 0x2083
#define WGL_FRONT_RIGHT_ARB 0x2084
#define WGL_BACK_LEFT_ARB 0x2085
#define WGL_BACK_RIGHT_ARB 0x2086
#define WGL_AUX0_ARB 0x2087
#define WGL_AUX1_ARB 0x2088
#define WGL_AUX2_ARB 0x2089
#define WGL_AUX3_ARB 0x208A
#define WGL_AUX4_ARB 0x208B
#define WGL_AUX5_ARB 0x208C
#define WGL_AUX6_ARB 0x208D
#define WGL_AUX7_ARB 0x208E
#define WGL_AUX8_ARB 0x208F
#define WGL_AUX9_ARB 0x2090

typedef BOOL (WINAPI * PFNWGLBINDTEXIMAGEARBPROC) (HPBUFFERARB hPbuffer, int iBuffer);
typedef BOOL (WINAPI * PFNWGLRELEASETEXIMAGEARBPROC) (HPBUFFERARB hPbuffer, int iBuffer);
typedef BOOL (WINAPI * PFNWGLSETPBUFFERATTRIBARBPROC) (HPBUFFERARB hPbuffer, const int* piAttribList);

#define wglBindTexImageARB WGLEW_GET_FUN(__wglewBindTexImageARB)
#define wglReleaseTexImageARB WGLEW_GET_FUN(__wglewReleaseTexImageARB)
#define wglSetPbufferAttribARB WGLEW_GET_FUN(__wglewSetPbufferAttribARB)

#define WGLEW_ARB_render_texture WGLEW_GET_VAR(__WGLEW_ARB_render_texture)

#endif /* WGL_ARB_render_texture */

/* ----------------------- WGL_ATI_pixel_format_float ---------------------- */

#ifndef WGL_ATI_pixel_format_float
#define WGL_ATI_pixel_format_float 1

#define WGL_TYPE_RGBA_FLOAT_ATI 0x21A0
#define GL_RGBA_FLOAT_MODE_ATI 0x8820
#define GL_COLOR_CLEAR_UNCLAMPED_VALUE_ATI 0x8835

#define WGLEW_ATI_pixel_format_float WGLEW_GET_VAR(__WGLEW_ATI_pixel_format_float)

#endif /* WGL_ATI_pixel_format_float */

/* -------------------- WGL_ATI_render_texture_rectangle ------------------- */

#ifndef WGL_ATI_render_texture_rectangle
#define WGL_ATI_render_texture_rectangle 1

#define WGL_TEXTURE_RECTANGLE_ATI 0x21A5

#define WGLEW_ATI_render_texture_rectangle WGLEW_GET_VAR(__WGLEW_ATI_render_texture_rectangle)

#endif /* WGL_ATI_render_texture_rectangle */

/* ------------------- WGL_EXT_create_context_es2_profile ------------------ */

#ifndef WGL_EXT_create_context_es2_profile
#define WGL_EXT_create_context_es2_profile 1

#define WGL_CONTEXT_ES2_PROFILE_BIT_EXT 0x00000004

#define WGLEW_EXT_create_context_es2_profile WGLEW_GET_VAR(__WGLEW_EXT_create_context_es2_profile)

#endif /* WGL_EXT_create_context_es2_profile */

/* -------------------------- WGL_EXT_depth_float -------------------------- */

#ifndef WGL_EXT_depth_float
#define WGL_EXT_depth_float 1

#define WGL_DEPTH_FLOAT_EXT 0x2040

#define WGLEW_EXT_depth_float WGLEW_GET_VAR(__WGLEW_EXT_depth_float)

#endif /* WGL_EXT_depth_float */

/* ---------------------- WGL_EXT_display_color_table ---------------------- */

#ifndef WGL_EXT_display_color_table
#define WGL_EXT_display_color_table 1

typedef GLboolean (WINAPI * PFNWGLBINDDISPLAYCOLORTABLEEXTPROC) (GLushort id);
typedef GLboolean (WINAPI * PFNWGLCREATEDISPLAYCOLORTABLEEXTPROC) (GLushort id);
typedef void (WINAPI * PFNWGLDESTROYDISPLAYCOLORTABLEEXTPROC) (GLushort id);
typedef GLboolean (WINAPI * PFNWGLLOADDISPLAYCOLORTABLEEXTPROC) (GLushort* table, GLuint length);

#define wglBindDisplayColorTableEXT WGLEW_GET_FUN(__wglewBindDisplayColorTableEXT)
#define wglCreateDisplayColorTableEXT WGLEW_GET_FUN(__wglewCreateDisplayColorTableEXT)
#define wglDestroyDisplayColorTableEXT WGLEW_GET_FUN(__wglewDestroyDisplayColorTableEXT)
#define wglLoadDisplayColorTableEXT WGLEW_GET_FUN(__wglewLoadDisplayColorTableEXT)

#define WGLEW_EXT_display_color_table WGLEW_GET_VAR(__WGLEW_EXT_display_color_table)

#endif /* WGL_EXT_display_color_table */

/* ----------------------- WGL_EXT_extensions_string ----------------------- */

#ifndef WGL_EXT_extensions_string
#define WGL_EXT_extensions_string 1

typedef const char* (WINAPI * PFNWGLGETEXTENSIONSSTRINGEXTPROC) (void);

#define wglGetExtensionsStringEXT WGLEW_GET_FUN(__wglewGetExtensionsStringEXT)

#define WGLEW_EXT_extensions_string WGLEW_GET_VAR(__WGLEW_EXT_extensions_string)

#endif /* WGL_EXT_extensions_string */

/* ------------------------ WGL_EXT_framebuffer_sRGB ----------------------- */

#ifndef WGL_EXT_framebuffer_sRGB
#define WGL_EXT_framebuffer_sRGB 1

#define WGL_FRAMEBUFFER_SRGB_CAPABLE_EXT 0x20A9

#define WGLEW_EXT_framebuffer_sRGB WGLEW_GET_VAR(__WGLEW_EXT_framebuffer_sRGB)

#endif /* WGL_EXT_framebuffer_sRGB */

/* ----------------------- WGL_EXT_make_current_read ----------------------- */

#ifndef WGL_EXT_make_current_read
#define WGL_EXT_make_current_read 1

#define ERROR_INVALID_PIXEL_TYPE_EXT 0x2043

typedef HDC (WINAPI * PFNWGLGETCURRENTREADDCEXTPROC) (VOID);
typedef BOOL (WINAPI * PFNWGLMAKECONTEXTCURRENTEXTPROC) (HDC hDrawDC, HDC hReadDC, HGLRC hglrc);

#define wglGetCurrentReadDCEXT WGLEW_GET_FUN(__wglewGetCurrentReadDCEXT)
#define wglMakeContextCurrentEXT WGLEW_GET_FUN(__wglewMakeContextCurrentEXT)

#define WGLEW_EXT_make_current_read WGLEW_GET_VAR(__WGLEW_EXT_make_current_read)

#endif /* WGL_EXT_make_current_read */

/* -------------------------- WGL_EXT_multisample -------------------------- */

#ifndef WGL_EXT_multisample
#define WGL_EXT_multisample 1

#define WGL_SAMPLE_BUFFERS_EXT 0x2041
#define WGL_SAMPLES_EXT 0x2042

#define WGLEW_EXT_multisample WGLEW_GET_VAR(__WGLEW_EXT_multisample)

#endif /* WGL_EXT_multisample */

/* ---------------------------- WGL_EXT_pbuffer ---------------------------- */

#ifndef WGL_EXT_pbuffer
#define WGL_EXT_pbuffer 1

#define WGL_DRAW_TO_PBUFFER_EXT 0x202D
#define WGL_MAX_PBUFFER_PIXELS_EXT 0x202E
#define WGL_MAX_PBUFFER_WIDTH_EXT 0x202F
#define WGL_MAX_PBUFFER_HEIGHT_EXT 0x2030
#define WGL_OPTIMAL_PBUFFER_WIDTH_EXT 0x2031
#define WGL_OPTIMAL_PBUFFER_HEIGHT_EXT 0x2032
#define WGL_PBUFFER_LARGEST_EXT 0x2033
#define WGL_PBUFFER_WIDTH_EXT 0x2034
#define WGL_PBUFFER_HEIGHT_EXT 0x2035

DECLARE_HANDLE(HPBUFFEREXT);

typedef HPBUFFEREXT (WINAPI * PFNWGLCREATEPBUFFEREXTPROC) (HDC hDC, int iPixelFormat, int iWidth, int iHeight, const int* piAttribList);
typedef BOOL (WINAPI * PFNWGLDESTROYPBUFFEREXTPROC) (HPBUFFEREXT hPbuffer);
typedef HDC (WINAPI * PFNWGLGETPBUFFERDCEXTPROC) (HPBUFFEREXT hPbuffer);
typedef BOOL (WINAPI * PFNWGLQUERYPBUFFEREXTPROC) (HPBUFFEREXT hPbuffer, int iAttribute, int* piValue);
typedef int (WINAPI * PFNWGLRELEASEPBUFFERDCEXTPROC) (HPBUFFEREXT hPbuffer, HDC hDC);

#define wglCreatePbufferEXT WGLEW_GET_FUN(__wglewCreatePbufferEXT)
#define wglDestroyPbufferEXT WGLEW_GET_FUN(__wglewDestroyPbufferEXT)
#define wglGetPbufferDCEXT WGLEW_GET_FUN(__wglewGetPbufferDCEXT)
#define wglQueryPbufferEXT WGLEW_GET_FUN(__wglewQueryPbufferEXT)
#define wglReleasePbufferDCEXT WGLEW_GET_FUN(__wglewReleasePbufferDCEXT)

#define WGLEW_EXT_pbuffer WGLEW_GET_VAR(__WGLEW_EXT_pbuffer)

#endif /* WGL_EXT_pbuffer */

/* -------------------------- WGL_EXT_pixel_format ------------------------- */

#ifndef WGL_EXT_pixel_format
#define WGL_EXT_pixel_format 1

#define WGL_NUMBER_PIXEL_FORMATS_EXT 0x2000
#define WGL_DRAW_TO_WINDOW_EXT 0x2001
#define WGL_DRAW_TO_BITMAP_EXT 0x2002
#define WGL_ACCELERATION_EXT 0x2003
#define WGL_NEED_PALETTE_EXT 0x2004
#define WGL_NEED_SYSTEM_PALETTE_EXT 0x2005
#define WGL_SWAP_LAYER_BUFFERS_EXT 0x2006
#define WGL_SWAP_METHOD_EXT 0x2007
#define WGL_NUMBER_OVERLAYS_EXT 0x2008
#define WGL_NUMBER_UNDERLAYS_EXT 0x2009
#define WGL_TRANSPARENT_EXT 0x200A
#define WGL_TRANSPARENT_VALUE_EXT 0x200B
#define WGL_SHARE_DEPTH_EXT 0x200C
#define WGL_SHARE_STENCIL_EXT 0x200D
#define WGL_SHARE_ACCUM_EXT 0x200E
#define WGL_SUPPORT_GDI_EXT 0x200F
#define WGL_SUPPORT_OPENGL_EXT 0x2010
#define WGL_DOUBLE_BUFFER_EXT 0x2011
#define WGL_STEREO_EXT 0x2012
#define WGL_PIXEL_TYPE_EXT 0x2013
#define WGL_COLOR_BITS_EXT 0x2014
#define WGL_RED_BITS_EXT 0x2015
#define WGL_RED_SHIFT_EXT 0x2016
#define WGL_GREEN_BITS_EXT 0x2017
#define WGL_GREEN_SHIFT_EXT 0x2018
#define WGL_BLUE_BITS_EXT 0x2019
#define WGL_BLUE_SHIFT_EXT 0x201A
#define WGL_ALPHA_BITS_EXT 0x201B
#define WGL_ALPHA_SHIFT_EXT 0x201C
#define WGL_ACCUM_BITS_EXT 0x201D
#define WGL_ACCUM_RED_BITS_EXT 0x201E
#define WGL_ACCUM_GREEN_BITS_EXT 0x201F
#define WGL_ACCUM_BLUE_BITS_EXT 0x2020
#define WGL_ACCUM_ALPHA_BITS_EXT 0x2021
#define WGL_DEPTH_BITS_EXT 0x2022
#define WGL_STENCIL_BITS_EXT 0x2023
#define WGL_AUX_BUFFERS_EXT 0x2024
#define WGL_NO_ACCELERATION_EXT 0x2025
#define WGL_GENERIC_ACCELERATION_EXT 0x2026
#define WGL_FULL_ACCELERATION_EXT 0x2027
#define WGL_SWAP_EXCHANGE_EXT 0x2028
#define WGL_SWAP_COPY_EXT 0x2029
#define WGL_SWAP_UNDEFINED_EXT 0x202A
#define WGL_TYPE_RGBA_EXT 0x202B
#define WGL_TYPE_COLORINDEX_EXT 0x202C

typedef BOOL (WINAPI * PFNWGLCHOOSEPIXELFORMATEXTPROC) (HDC hdc, const int* piAttribIList, const FLOAT *pfAttribFList, UINT nMaxFormats, int *piFormats, UINT *nNumFormats);
typedef BOOL (WINAPI * PFNWGLGETPIXELFORMATATTRIBFVEXTPROC) (HDC hdc, int iPixelFormat, int iLayerPlane, UINT nAttributes, int* piAttributes, FLOAT *pfValues);
typedef BOOL (WINAPI * PFNWGLGETPIXELFORMATATTRIBIVEXTPROC) (HDC hdc, int iPixelFormat, int iLayerPlane, UINT nAttributes, int* piAttributes, int *piValues);

#define wglChoosePixelFormatEXT WGLEW_GET_FUN(__wglewChoosePixelFormatEXT)
#define wglGetPixelFormatAttribfvEXT WGLEW_GET_FUN(__wglewGetPixelFormatAttribfvEXT)
#define wglGetPixelFormatAttribivEXT WGLEW_GET_FUN(__wglewGetPixelFormatAttribivEXT)

#define WGLEW_EXT_pixel_format WGLEW_GET_VAR(__WGLEW_EXT_pixel_format)

#endif /* WGL_EXT_pixel_format */

/* ------------------- WGL_EXT_pixel_format_packed_float ------------------- */

#ifndef WGL_EXT_pixel_format_packed_float
#define WGL_EXT_pixel_format_packed_float 1

#define WGL_TYPE_RGBA_UNSIGNED_FLOAT_EXT 0x20A8

#define WGLEW_EXT_pixel_format_packed_float WGLEW_GET_VAR(__WGLEW_EXT_pixel_format_packed_float)

#endif /* WGL_EXT_pixel_format_packed_float */

/* -------------------------- WGL_EXT_swap_control ------------------------- */

#ifndef WGL_EXT_swap_control
#define WGL_EXT_swap_control 1

typedef int (WINAPI * PFNWGLGETSWAPINTERVALEXTPROC) (void);
typedef BOOL (WINAPI * PFNWGLSWAPINTERVALEXTPROC) (int interval);

#define wglGetSwapIntervalEXT WGLEW_GET_FUN(__wglewGetSwapIntervalEXT)
#define wglSwapIntervalEXT WGLEW_GET_FUN(__wglewSwapIntervalEXT)

#define WGLEW_EXT_swap_control WGLEW_GET_VAR(__WGLEW_EXT_swap_control)

#endif /* WGL_EXT_swap_control */

/* --------------------- WGL_I3D_digital_video_control --------------------- */

#ifndef WGL_I3D_digital_video_control
#define WGL_I3D_digital_video_control 1

#define WGL_DIGITAL_VIDEO_CURSOR_ALPHA_FRAMEBUFFER_I3D 0x2050
#define WGL_DIGITAL_VIDEO_CURSOR_ALPHA_VALUE_I3D 0x2051
#define WGL_DIGITAL_VIDEO_CURSOR_INCLUDED_I3D 0x2052
#define WGL_DIGITAL_VIDEO_GAMMA_CORRECTED_I3D 0x2053

typedef BOOL (WINAPI * PFNWGLGETDIGITALVIDEOPARAMETERSI3DPROC) (HDC hDC, int iAttribute, int* piValue);
typedef BOOL (WINAPI * PFNWGLSETDIGITALVIDEOPARAMETERSI3DPROC) (HDC hDC, int iAttribute, const int* piValue);

#define wglGetDigitalVideoParametersI3D WGLEW_GET_FUN(__wglewGetDigitalVideoParametersI3D)
#define wglSetDigitalVideoParametersI3D WGLEW_GET_FUN(__wglewSetDigitalVideoParametersI3D)

#define WGLEW_I3D_digital_video_control WGLEW_GET_VAR(__WGLEW_I3D_digital_video_control)

#endif /* WGL_I3D_digital_video_control */

/* ----------------------------- WGL_I3D_gamma ----------------------------- */

#ifndef WGL_I3D_gamma
#define WGL_I3D_gamma 1

#define WGL_GAMMA_TABLE_SIZE_I3D 0x204E
#define WGL_GAMMA_EXCLUDE_DESKTOP_I3D 0x204F

typedef BOOL (WINAPI * PFNWGLGETGAMMATABLEI3DPROC) (HDC hDC, int iEntries, USHORT* puRed, USHORT *puGreen, USHORT *puBlue);
typedef BOOL (WINAPI * PFNWGLGETGAMMATABLEPARAMETERSI3DPROC) (HDC hDC, int iAttribute, int* piValue);
typedef BOOL (WINAPI * PFNWGLSETGAMMATABLEI3DPROC) (HDC hDC, int iEntries, const USHORT* puRed, const USHORT *puGreen, const USHORT *puBlue);
typedef BOOL (WINAPI * PFNWGLSETGAMMATABLEPARAMETERSI3DPROC) (HDC hDC, int iAttribute, const int* piValue);

#define wglGetGammaTableI3D WGLEW_GET_FUN(__wglewGetGammaTableI3D)
#define wglGetGammaTableParametersI3D WGLEW_GET_FUN(__wglewGetGammaTableParametersI3D)
#define wglSetGammaTableI3D WGLEW_GET_FUN(__wglewSetGammaTableI3D)
#define wglSetGammaTableParametersI3D WGLEW_GET_FUN(__wglewSetGammaTableParametersI3D)

#define WGLEW_I3D_gamma WGLEW_GET_VAR(__WGLEW_I3D_gamma)

#endif /* WGL_I3D_gamma */

/* ---------------------------- WGL_I3D_genlock ---------------------------- */

#ifndef WGL_I3D_genlock
#define WGL_I3D_genlock 1

#define WGL_GENLOCK_SOURCE_MULTIVIEW_I3D 0x2044
#define WGL_GENLOCK_SOURCE_EXTERNAL_SYNC_I3D 0x2045
#define WGL_GENLOCK_SOURCE_EXTERNAL_FIELD_I3D 0x2046
#define WGL_GENLOCK_SOURCE_EXTERNAL_TTL_I3D 0x2047
#define WGL_GENLOCK_SOURCE_DIGITAL_SYNC_I3D 0x2048
#define WGL_GENLOCK_SOURCE_DIGITAL_FIELD_I3D 0x2049
#define WGL_GENLOCK_SOURCE_EDGE_FALLING_I3D 0x204A
#define WGL_GENLOCK_SOURCE_EDGE_RISING_I3D 0x204B
#define WGL_GENLOCK_SOURCE_EDGE_BOTH_I3D 0x204C

typedef BOOL (WINAPI * PFNWGLDISABLEGENLOCKI3DPROC) (HDC hDC);
typedef BOOL (WINAPI * PFNWGLENABLEGENLOCKI3DPROC) (HDC hDC);
typedef BOOL (WINAPI * PFNWGLGENLOCKSAMPLERATEI3DPROC) (HDC hDC, UINT uRate);
typedef BOOL (WINAPI * PFNWGLGENLOCKSOURCEDELAYI3DPROC) (HDC hDC, UINT uDelay);
typedef BOOL (WINAPI * PFNWGLGENLOCKSOURCEEDGEI3DPROC) (HDC hDC, UINT uEdge);
typedef BOOL (WINAPI * PFNWGLGENLOCKSOURCEI3DPROC) (HDC hDC, UINT uSource);
typedef BOOL (WINAPI * PFNWGLGETGENLOCKSAMPLERATEI3DPROC) (HDC hDC, UINT* uRate);
typedef BOOL (WINAPI * PFNWGLGETGENLOCKSOURCEDELAYI3DPROC) (HDC hDC, UINT* uDelay);
typedef BOOL (WINAPI * PFNWGLGETGENLOCKSOURCEEDGEI3DPROC) (HDC hDC, UINT* uEdge);
typedef BOOL (WINAPI * PFNWGLGETGENLOCKSOURCEI3DPROC) (HDC hDC, UINT* uSource);
typedef BOOL (WINAPI * PFNWGLISENABLEDGENLOCKI3DPROC) (HDC hDC, BOOL* pFlag);
typedef BOOL (WINAPI * PFNWGLQUERYGENLOCKMAXSOURCEDELAYI3DPROC) (HDC hDC, UINT* uMaxLineDelay, UINT *uMaxPixelDelay);

#define wglDisableGenlockI3D WGLEW_GET_FUN(__wglewDisableGenlockI3D)
#define wglEnableGenlockI3D WGLEW_GET_FUN(__wglewEnableGenlockI3D)
#define wglGenlockSampleRateI3D WGLEW_GET_FUN(__wglewGenlockSampleRateI3D)
#define wglGenlockSourceDelayI3D WGLEW_GET_FUN(__wglewGenlockSourceDelayI3D)
#define wglGenlockSourceEdgeI3D WGLEW_GET_FUN(__wglewGenlockSourceEdgeI3D)
#define wglGenlockSourceI3D WGLEW_GET_FUN(__wglewGenlockSourceI3D)
#define wglGetGenlockSampleRateI3D WGLEW_GET_FUN(__wglewGetGenlockSampleRateI3D)
#define wglGetGenlockSourceDelayI3D WGLEW_GET_FUN(__wglewGetGenlockSourceDelayI3D)
#define wglGetGenlockSourceEdgeI3D WGLEW_GET_FUN(__wglewGetGenlockSourceEdgeI3D)
#define wglGetGenlockSourceI3D WGLEW_GET_FUN(__wglewGetGenlockSourceI3D)
#define wglIsEnabledGenlockI3D WGLEW_GET_FUN(__wglewIsEnabledGenlockI3D)
#define wglQueryGenlockMaxSourceDelayI3D WGLEW_GET_FUN(__wglewQueryGenlockMaxSourceDelayI3D)

#define WGLEW_I3D_genlock WGLEW_GET_VAR(__WGLEW_I3D_genlock)

#endif /* WGL_I3D_genlock */

/* -------------------------- WGL_I3D_image_buffer ------------------------- */

#ifndef WGL_I3D_image_buffer
#define WGL_I3D_image_buffer 1

#define WGL_IMAGE_BUFFER_MIN_ACCESS_I3D 0x00000001
#define WGL_IMAGE_BUFFER_LOCK_I3D 0x00000002

typedef BOOL (WINAPI * PFNWGLASSOCIATEIMAGEBUFFEREVENTSI3DPROC) (HDC hdc, HANDLE* pEvent, LPVOID *pAddress, DWORD *pSize, UINT count);
typedef LPVOID (WINAPI * PFNWGLCREATEIMAGEBUFFERI3DPROC) (HDC hDC, DWORD dwSize, UINT uFlags);
typedef BOOL (WINAPI * PFNWGLDESTROYIMAGEBUFFERI3DPROC) (HDC hDC, LPVOID pAddress);
typedef BOOL (WINAPI * PFNWGLRELEASEIMAGEBUFFEREVENTSI3DPROC) (HDC hdc, LPVOID* pAddress, UINT count);

#define wglAssociateImageBufferEventsI3D WGLEW_GET_FUN(__wglewAssociateImageBufferEventsI3D)
#define wglCreateImageBufferI3D WGLEW_GET_FUN(__wglewCreateImageBufferI3D)
#define wglDestroyImageBufferI3D WGLEW_GET_FUN(__wglewDestroyImageBufferI3D)
#define wglReleaseImageBufferEventsI3D WGLEW_GET_FUN(__wglewReleaseImageBufferEventsI3D)

#define WGLEW_I3D_image_buffer WGLEW_GET_VAR(__WGLEW_I3D_image_buffer)

#endif /* WGL_I3D_image_buffer */

/* ------------------------ WGL_I3D_swap_frame_lock ------------------------ */

#ifndef WGL_I3D_swap_frame_lock
#define WGL_I3D_swap_frame_lock 1

typedef BOOL (WINAPI * PFNWGLDISABLEFRAMELOCKI3DPROC) (VOID);
typedef BOOL (WINAPI * PFNWGLENABLEFRAMELOCKI3DPROC) (VOID);
typedef BOOL (WINAPI * PFNWGLISENABLEDFRAMELOCKI3DPROC) (BOOL* pFlag);
typedef BOOL (WINAPI * PFNWGLQUERYFRAMELOCKMASTERI3DPROC) (BOOL* pFlag);

#define wglDisableFrameLockI3D WGLEW_GET_FUN(__wglewDisableFrameLockI3D)
#define wglEnableFrameLockI3D WGLEW_GET_FUN(__wglewEnableFrameLockI3D)
#define wglIsEnabledFrameLockI3D WGLEW_GET_FUN(__wglewIsEnabledFrameLockI3D)
#define wglQueryFrameLockMasterI3D WGLEW_GET_FUN(__wglewQueryFrameLockMasterI3D)

#define WGLEW_I3D_swap_frame_lock WGLEW_GET_VAR(__WGLEW_I3D_swap_frame_lock)

#endif /* WGL_I3D_swap_frame_lock */

/* ------------------------ WGL_I3D_swap_frame_usage ----------------------- */

#ifndef WGL_I3D_swap_frame_usage
#define WGL_I3D_swap_frame_usage 1

typedef BOOL (WINAPI * PFNWGLBEGINFRAMETRACKINGI3DPROC) (void);
typedef BOOL (WINAPI * PFNWGLENDFRAMETRACKINGI3DPROC) (void);
typedef BOOL (WINAPI * PFNWGLGETFRAMEUSAGEI3DPROC) (float* pUsage);
typedef BOOL (WINAPI * PFNWGLQUERYFRAMETRACKINGI3DPROC) (DWORD* pFrameCount, DWORD *pMissedFrames, float *pLastMissedUsage);

#define wglBeginFrameTrackingI3D WGLEW_GET_FUN(__wglewBeginFrameTrackingI3D)
#define wglEndFrameTrackingI3D WGLEW_GET_FUN(__wglewEndFrameTrackingI3D)
#define wglGetFrameUsageI3D WGLEW_GET_FUN(__wglewGetFrameUsageI3D)
#define wglQueryFrameTrackingI3D WGLEW_GET_FUN(__wglewQueryFrameTrackingI3D)

#define WGLEW_I3D_swap_frame_usage WGLEW_GET_VAR(__WGLEW_I3D_swap_frame_usage)

#endif /* WGL_I3D_swap_frame_usage */

/* --------------------------- WGL_NV_copy_image --------------------------- */

#ifndef WGL_NV_copy_image
#define WGL_NV_copy_image 1

typedef BOOL (WINAPI * PFNWGLCOPYIMAGESUBDATANVPROC) (HGLRC hSrcRC, GLuint srcName, GLenum srcTarget, GLint srcLevel, GLint srcX, GLint srcY, GLint srcZ, HGLRC hDstRC, GLuint dstName, GLenum dstTarget, GLint dstLevel, GLint dstX, GLint dstY, GLint dstZ, GLsizei width, GLsizei height, GLsizei depth);

#define wglCopyImageSubDataNV WGLEW_GET_FUN(__wglewCopyImageSubDataNV)

#define WGLEW_NV_copy_image WGLEW_GET_VAR(__WGLEW_NV_copy_image)

#endif /* WGL_NV_copy_image */

/* -------------------------- WGL_NV_float_buffer -------------------------- */

#ifndef WGL_NV_float_buffer
#define WGL_NV_float_buffer 1

#define WGL_FLOAT_COMPONENTS_NV 0x20B0
#define WGL_BIND_TO_TEXTURE_RECTANGLE_FLOAT_R_NV 0x20B1
#define WGL_BIND_TO_TEXTURE_RECTANGLE_FLOAT_RG_NV 0x20B2
#define WGL_BIND_TO_TEXTURE_RECTANGLE_FLOAT_RGB_NV 0x20B3
#define WGL_BIND_TO_TEXTURE_RECTANGLE_FLOAT_RGBA_NV 0x20B4
#define WGL_TEXTURE_FLOAT_R_NV 0x20B5
#define WGL_TEXTURE_FLOAT_RG_NV 0x20B6
#define WGL_TEXTURE_FLOAT_RGB_NV 0x20B7
#define WGL_TEXTURE_FLOAT_RGBA_NV 0x20B8

#define WGLEW_NV_float_buffer WGLEW_GET_VAR(__WGLEW_NV_float_buffer)

#endif /* WGL_NV_float_buffer */

/* -------------------------- WGL_NV_gpu_affinity -------------------------- */

#ifndef WGL_NV_gpu_affinity
#define WGL_NV_gpu_affinity 1

#define WGL_ERROR_INCOMPATIBLE_AFFINITY_MASKS_NV 0x20D0
#define WGL_ERROR_MISSING_AFFINITY_MASK_NV 0x20D1

DECLARE_HANDLE(HGPUNV);
typedef struct _GPU_DEVICE {
  DWORD cb; 
  CHAR DeviceName[32]; 
  CHAR DeviceString[128]; 
  DWORD Flags; 
  RECT rcVirtualScreen; 
} GPU_DEVICE, *PGPU_DEVICE;

typedef HDC (WINAPI * PFNWGLCREATEAFFINITYDCNVPROC) (const HGPUNV *phGpuList);
typedef BOOL (WINAPI * PFNWGLDELETEDCNVPROC) (HDC hdc);
typedef BOOL (WINAPI * PFNWGLENUMGPUDEVICESNVPROC) (HGPUNV hGpu, UINT iDeviceIndex, PGPU_DEVICE lpGpuDevice);
typedef BOOL (WINAPI * PFNWGLENUMGPUSFROMAFFINITYDCNVPROC) (HDC hAffinityDC, UINT iGpuIndex, HGPUNV *hGpu);
typedef BOOL (WINAPI * PFNWGLENUMGPUSNVPROC) (UINT iGpuIndex, HGPUNV *phGpu);

#define wglCreateAffinityDCNV WGLEW_GET_FUN(__wglewCreateAffinityDCNV)
#define wglDeleteDCNV WGLEW_GET_FUN(__wglewDeleteDCNV)
#define wglEnumGpuDevicesNV WGLEW_GET_FUN(__wglewEnumGpuDevicesNV)
#define wglEnumGpusFromAffinityDCNV WGLEW_GET_FUN(__wglewEnumGpusFromAffinityDCNV)
#define wglEnumGpusNV WGLEW_GET_FUN(__wglewEnumGpusNV)

#define WGLEW_NV_gpu_affinity WGLEW_GET_VAR(__WGLEW_NV_gpu_affinity)

#endif /* WGL_NV_gpu_affinity */

/* ---------------------- WGL_NV_multisample_coverage ---------------------- */

#ifndef WGL_NV_multisample_coverage
#define WGL_NV_multisample_coverage 1

#define WGL_COVERAGE_SAMPLES_NV 0x2042
#define WGL_COLOR_SAMPLES_NV 0x20B9

#define WGLEW_NV_multisample_coverage WGLEW_GET_VAR(__WGLEW_NV_multisample_coverage)

#endif /* WGL_NV_multisample_coverage */

/* -------------------------- WGL_NV_present_video ------------------------- */

#ifndef WGL_NV_present_video
#define WGL_NV_present_video 1

#define WGL_NUM_VIDEO_SLOTS_NV 0x20F0

DECLARE_HANDLE(HVIDEOOUTPUTDEVICENV);

typedef BOOL (WINAPI * PFNWGLBINDVIDEODEVICENVPROC) (HDC hDc, unsigned int uVideoSlot, HVIDEOOUTPUTDEVICENV hVideoDevice, const int* piAttribList);
typedef int (WINAPI * PFNWGLENUMERATEVIDEODEVICESNVPROC) (HDC hDc, HVIDEOOUTPUTDEVICENV* phDeviceList);
typedef BOOL (WINAPI * PFNWGLQUERYCURRENTCONTEXTNVPROC) (int iAttribute, int* piValue);

#define wglBindVideoDeviceNV WGLEW_GET_FUN(__wglewBindVideoDeviceNV)
#define wglEnumerateVideoDevicesNV WGLEW_GET_FUN(__wglewEnumerateVideoDevicesNV)
#define wglQueryCurrentContextNV WGLEW_GET_FUN(__wglewQueryCurrentContextNV)

#define WGLEW_NV_present_video WGLEW_GET_VAR(__WGLEW_NV_present_video)

#endif /* WGL_NV_present_video */

/* ---------------------- WGL_NV_render_depth_texture ---------------------- */

#ifndef WGL_NV_render_depth_texture
#define WGL_NV_render_depth_texture 1

#define WGL_NO_TEXTURE_ARB 0x2077
#define WGL_BIND_TO_TEXTURE_DEPTH_NV 0x20A3
#define WGL_BIND_TO_TEXTURE_RECTANGLE_DEPTH_NV 0x20A4
#define WGL_DEPTH_TEXTURE_FORMAT_NV 0x20A5
#define WGL_TEXTURE_DEPTH_COMPONENT_NV 0x20A6
#define WGL_DEPTH_COMPONENT_NV 0x20A7

#define WGLEW_NV_render_depth_texture WGLEW_GET_VAR(__WGLEW_NV_render_depth_texture)

#endif /* WGL_NV_render_depth_texture */

/* -------------------- WGL_NV_render_texture_rectangle -------------------- */

#ifndef WGL_NV_render_texture_rectangle
#define WGL_NV_render_texture_rectangle 1

#define WGL_BIND_TO_TEXTURE_RECTANGLE_RGB_NV 0x20A0
#define WGL_BIND_TO_TEXTURE_RECTANGLE_RGBA_NV 0x20A1
#define WGL_TEXTURE_RECTANGLE_NV 0x20A2

#define WGLEW_NV_render_texture_rectangle WGLEW_GET_VAR(__WGLEW_NV_render_texture_rectangle)

#endif /* WGL_NV_render_texture_rectangle */

/* --------------------------- WGL_NV_swap_group --------------------------- */

#ifndef WGL_NV_swap_group
#define WGL_NV_swap_group 1

typedef BOOL (WINAPI * PFNWGLBINDSWAPBARRIERNVPROC) (GLuint group, GLuint barrier);
typedef BOOL (WINAPI * PFNWGLJOINSWAPGROUPNVPROC) (HDC hDC, GLuint group);
typedef BOOL (WINAPI * PFNWGLQUERYFRAMECOUNTNVPROC) (HDC hDC, GLuint* count);
typedef BOOL (WINAPI * PFNWGLQUERYMAXSWAPGROUPSNVPROC) (HDC hDC, GLuint* maxGroups, GLuint *maxBarriers);
typedef BOOL (WINAPI * PFNWGLQUERYSWAPGROUPNVPROC) (HDC hDC, GLuint* group, GLuint *barrier);
typedef BOOL (WINAPI * PFNWGLRESETFRAMECOUNTNVPROC) (HDC hDC);

#define wglBindSwapBarrierNV WGLEW_GET_FUN(__wglewBindSwapBarrierNV)
#define wglJoinSwapGroupNV WGLEW_GET_FUN(__wglewJoinSwapGroupNV)
#define wglQueryFrameCountNV WGLEW_GET_FUN(__wglewQueryFrameCountNV)
#define wglQueryMaxSwapGroupsNV WGLEW_GET_FUN(__wglewQueryMaxSwapGroupsNV)
#define wglQuerySwapGroupNV WGLEW_GET_FUN(__wglewQuerySwapGroupNV)
#define wglResetFrameCountNV WGLEW_GET_FUN(__wglewResetFrameCountNV)

#define WGLEW_NV_swap_group WGLEW_GET_VAR(__WGLEW_NV_swap_group)

#endif /* WGL_NV_swap_group */

/* ----------------------- WGL_NV_vertex_array_range ----------------------- */

#ifndef WGL_NV_vertex_array_range
#define WGL_NV_vertex_array_range 1

typedef void * (WINAPI * PFNWGLALLOCATEMEMORYNVPROC) (GLsizei size, GLfloat readFrequency, GLfloat writeFrequency, GLfloat priority);
typedef void (WINAPI * PFNWGLFREEMEMORYNVPROC) (void *pointer);

#define wglAllocateMemoryNV WGLEW_GET_FUN(__wglewAllocateMemoryNV)
#define wglFreeMemoryNV WGLEW_GET_FUN(__wglewFreeMemoryNV)

#define WGLEW_NV_vertex_array_range WGLEW_GET_VAR(__WGLEW_NV_vertex_array_range)

#endif /* WGL_NV_vertex_array_range */

/* -------------------------- WGL_NV_video_output -------------------------- */

#ifndef WGL_NV_video_output
#define WGL_NV_video_output 1

#define WGL_BIND_TO_VIDEO_RGB_NV 0x20C0
#define WGL_BIND_TO_VIDEO_RGBA_NV 0x20C1
#define WGL_BIND_TO_VIDEO_RGB_AND_DEPTH_NV 0x20C2
#define WGL_VIDEO_OUT_COLOR_NV 0x20C3
#define WGL_VIDEO_OUT_ALPHA_NV 0x20C4
#define WGL_VIDEO_OUT_DEPTH_NV 0x20C5
#define WGL_VIDEO_OUT_COLOR_AND_ALPHA_NV 0x20C6
#define WGL_VIDEO_OUT_COLOR_AND_DEPTH_NV 0x20C7
#define WGL_VIDEO_OUT_FRAME 0x20C8
#define WGL_VIDEO_OUT_FIELD_1 0x20C9
#define WGL_VIDEO_OUT_FIELD_2 0x20CA
#define WGL_VIDEO_OUT_STACKED_FIELDS_1_2 0x20CB
#define WGL_VIDEO_OUT_STACKED_FIELDS_2_1 0x20CC

DECLARE_HANDLE(HPVIDEODEV);

typedef BOOL (WINAPI * PFNWGLBINDVIDEOIMAGENVPROC) (HPVIDEODEV hVideoDevice, HPBUFFERARB hPbuffer, int iVideoBuffer);
typedef BOOL (WINAPI * PFNWGLGETVIDEODEVICENVPROC) (HDC hDC, int numDevices, HPVIDEODEV* hVideoDevice);
typedef BOOL (WINAPI * PFNWGLGETVIDEOINFONVPROC) (HPVIDEODEV hpVideoDevice, unsigned long* pulCounterOutputPbuffer, unsigned long *pulCounterOutputVideo);
typedef BOOL (WINAPI * PFNWGLRELEASEVIDEODEVICENVPROC) (HPVIDEODEV hVideoDevice);
typedef BOOL (WINAPI * PFNWGLRELEASEVIDEOIMAGENVPROC) (HPBUFFERARB hPbuffer, int iVideoBuffer);
typedef BOOL (WINAPI * PFNWGLSENDPBUFFERTOVIDEONVPROC) (HPBUFFERARB hPbuffer, int iBufferType, unsigned long* pulCounterPbuffer, BOOL bBlock);

#define wglBindVideoImageNV WGLEW_GET_FUN(__wglewBindVideoImageNV)
#define wglGetVideoDeviceNV WGLEW_GET_FUN(__wglewGetVideoDeviceNV)
#define wglGetVideoInfoNV WGLEW_GET_FUN(__wglewGetVideoInfoNV)
#define wglReleaseVideoDeviceNV WGLEW_GET_FUN(__wglewReleaseVideoDeviceNV)
#define wglReleaseVideoImageNV WGLEW_GET_FUN(__wglewReleaseVideoImageNV)
#define wglSendPbufferToVideoNV WGLEW_GET_FUN(__wglewSendPbufferToVideoNV)

#define WGLEW_NV_video_output WGLEW_GET_VAR(__WGLEW_NV_video_output)

#endif /* WGL_NV_video_output */

/* -------------------------- WGL_OML_sync_control ------------------------- */

#ifndef WGL_OML_sync_control
#define WGL_OML_sync_control 1

typedef BOOL (WINAPI * PFNWGLGETMSCRATEOMLPROC) (HDC hdc, INT32* numerator, INT32 *denominator);
typedef BOOL (WINAPI * PFNWGLGETSYNCVALUESOMLPROC) (HDC hdc, INT64* ust, INT64 *msc, INT64 *sbc);
typedef INT64 (WINAPI * PFNWGLSWAPBUFFERSMSCOMLPROC) (HDC hdc, INT64 target_msc, INT64 divisor, INT64 remainder);
typedef INT64 (WINAPI * PFNWGLSWAPLAYERBUFFERSMSCOMLPROC) (HDC hdc, INT fuPlanes, INT64 target_msc, INT64 divisor, INT64 remainder);
typedef BOOL (WINAPI * PFNWGLWAITFORMSCOMLPROC) (HDC hdc, INT64 target_msc, INT64 divisor, INT64 remainder, INT64* ust, INT64 *msc, INT64 *sbc);
typedef BOOL (WINAPI * PFNWGLWAITFORSBCOMLPROC) (HDC hdc, INT64 target_sbc, INT64* ust, INT64 *msc, INT64 *sbc);

#define wglGetMscRateOML WGLEW_GET_FUN(__wglewGetMscRateOML)
#define wglGetSyncValuesOML WGLEW_GET_FUN(__wglewGetSyncValuesOML)
#define wglSwapBuffersMscOML WGLEW_GET_FUN(__wglewSwapBuffersMscOML)
#define wglSwapLayerBuffersMscOML WGLEW_GET_FUN(__wglewSwapLayerBuffersMscOML)
#define wglWaitForMscOML WGLEW_GET_FUN(__wglewWaitForMscOML)
#define wglWaitForSbcOML WGLEW_GET_FUN(__wglewWaitForSbcOML)

#define WGLEW_OML_sync_control WGLEW_GET_VAR(__WGLEW_OML_sync_control)

#endif /* WGL_OML_sync_control */

/* ------------------------------------------------------------------------- */

#ifdef GLEW_MX
#define WGLEW_EXPORT
#else
#define WGLEW_EXPORT GLEWAPI
#endif /* GLEW_MX */

#ifdef GLEW_MX
struct WGLEWContextStruct
{
#endif /* GLEW_MX */

WGLEW_EXPORT PFNWGLSETSTEREOEMITTERSTATE3DLPROC __wglewSetStereoEmitterState3DL;

WGLEW_EXPORT PFNWGLBLITCONTEXTFRAMEBUFFERAMDPROC __wglewBlitContextFramebufferAMD;
WGLEW_EXPORT PFNWGLCREATEASSOCIATEDCONTEXTAMDPROC __wglewCreateAssociatedContextAMD;
WGLEW_EXPORT PFNWGLCREATEASSOCIATEDCONTEXTATTRIBSAMDPROC __wglewCreateAssociatedContextAttribsAMD;
WGLEW_EXPORT PFNWGLDELETEASSOCIATEDCONTEXTAMDPROC __wglewDeleteAssociatedContextAMD;
WGLEW_EXPORT PFNWGLGETCONTEXTGPUIDAMDPROC __wglewGetContextGPUIDAMD;
WGLEW_EXPORT PFNWGLGETCURRENTASSOCIATEDCONTEXTAMDPROC __wglewGetCurrentAssociatedContextAMD;
WGLEW_EXPORT PFNWGLGETGPUIDSAMDPROC __wglewGetGPUIDsAMD;
WGLEW_EXPORT PFNWGLGETGPUINFOAMDPROC __wglewGetGPUInfoAMD;
WGLEW_EXPORT PFNWGLMAKEASSOCIATEDCONTEXTCURRENTAMDPROC __wglewMakeAssociatedContextCurrentAMD;

WGLEW_EXPORT PFNWGLCREATEBUFFERREGIONARBPROC __wglewCreateBufferRegionARB;
WGLEW_EXPORT PFNWGLDELETEBUFFERREGIONARBPROC __wglewDeleteBufferRegionARB;
WGLEW_EXPORT PFNWGLRESTOREBUFFERREGIONARBPROC __wglewRestoreBufferRegionARB;
WGLEW_EXPORT PFNWGLSAVEBUFFERREGIONARBPROC __wglewSaveBufferRegionARB;

WGLEW_EXPORT PFNWGLCREATECONTEXTATTRIBSARBPROC __wglewCreateContextAttribsARB;

WGLEW_EXPORT PFNWGLGETEXTENSIONSSTRINGARBPROC __wglewGetExtensionsStringARB;

WGLEW_EXPORT PFNWGLGETCURRENTREADDCARBPROC __wglewGetCurrentReadDCARB;
WGLEW_EXPORT PFNWGLMAKECONTEXTCURRENTARBPROC __wglewMakeContextCurrentARB;

WGLEW_EXPORT PFNWGLCREATEPBUFFERARBPROC __wglewCreatePbufferARB;
WGLEW_EXPORT PFNWGLDESTROYPBUFFERARBPROC __wglewDestroyPbufferARB;
WGLEW_EXPORT PFNWGLGETPBUFFERDCARBPROC __wglewGetPbufferDCARB;
WGLEW_EXPORT PFNWGLQUERYPBUFFERARBPROC __wglewQueryPbufferARB;
WGLEW_EXPORT PFNWGLRELEASEPBUFFERDCARBPROC __wglewReleasePbufferDCARB;

WGLEW_EXPORT PFNWGLCHOOSEPIXELFORMATARBPROC __wglewChoosePixelFormatARB;
WGLEW_EXPORT PFNWGLGETPIXELFORMATATTRIBFVARBPROC __wglewGetPixelFormatAttribfvARB;
WGLEW_EXPORT PFNWGLGETPIXELFORMATATTRIBIVARBPROC __wglewGetPixelFormatAttribivARB;

WGLEW_EXPORT PFNWGLBINDTEXIMAGEARBPROC __wglewBindTexImageARB;
WGLEW_EXPORT PFNWGLRELEASETEXIMAGEARBPROC __wglewReleaseTexImageARB;
WGLEW_EXPORT PFNWGLSETPBUFFERATTRIBARBPROC __wglewSetPbufferAttribARB;

WGLEW_EXPORT PFNWGLBINDDISPLAYCOLORTABLEEXTPROC __wglewBindDisplayColorTableEXT;
WGLEW_EXPORT PFNWGLCREATEDISPLAYCOLORTABLEEXTPROC __wglewCreateDisplayColorTableEXT;
WGLEW_EXPORT PFNWGLDESTROYDISPLAYCOLORTABLEEXTPROC __wglewDestroyDisplayColorTableEXT;
WGLEW_EXPORT PFNWGLLOADDISPLAYCOLORTABLEEXTPROC __wglewLoadDisplayColorTableEXT;

WGLEW_EXPORT PFNWGLGETEXTENSIONSSTRINGEXTPROC __wglewGetExtensionsStringEXT;

WGLEW_EXPORT PFNWGLGETCURRENTREADDCEXTPROC __wglewGetCurrentReadDCEXT;
WGLEW_EXPORT PFNWGLMAKECONTEXTCURRENTEXTPROC __wglewMakeContextCurrentEXT;

WGLEW_EXPORT PFNWGLCREATEPBUFFEREXTPROC __wglewCreatePbufferEXT;
WGLEW_EXPORT PFNWGLDESTROYPBUFFEREXTPROC __wglewDestroyPbufferEXT;
WGLEW_EXPORT PFNWGLGETPBUFFERDCEXTPROC __wglewGetPbufferDCEXT;
WGLEW_EXPORT PFNWGLQUERYPBUFFEREXTPROC __wglewQueryPbufferEXT;
WGLEW_EXPORT PFNWGLRELEASEPBUFFERDCEXTPROC __wglewReleasePbufferDCEXT;

WGLEW_EXPORT PFNWGLCHOOSEPIXELFORMATEXTPROC __wglewChoosePixelFormatEXT;
WGLEW_EXPORT PFNWGLGETPIXELFORMATATTRIBFVEXTPROC __wglewGetPixelFormatAttribfvEXT;
WGLEW_EXPORT PFNWGLGETPIXELFORMATATTRIBIVEXTPROC __wglewGetPixelFormatAttribivEXT;

WGLEW_EXPORT PFNWGLGETSWAPINTERVALEXTPROC __wglewGetSwapIntervalEXT;
WGLEW_EXPORT PFNWGLSWAPINTERVALEXTPROC __wglewSwapIntervalEXT;

WGLEW_EXPORT PFNWGLGETDIGITALVIDEOPARAMETERSI3DPROC __wglewGetDigitalVideoParametersI3D;
WGLEW_EXPORT PFNWGLSETDIGITALVIDEOPARAMETERSI3DPROC __wglewSetDigitalVideoParametersI3D;

WGLEW_EXPORT PFNWGLGETGAMMATABLEI3DPROC __wglewGetGammaTableI3D;
WGLEW_EXPORT PFNWGLGETGAMMATABLEPARAMETERSI3DPROC __wglewGetGammaTableParametersI3D;
WGLEW_EXPORT PFNWGLSETGAMMATABLEI3DPROC __wglewSetGammaTableI3D;
WGLEW_EXPORT PFNWGLSETGAMMATABLEPARAMETERSI3DPROC __wglewSetGammaTableParametersI3D;

WGLEW_EXPORT PFNWGLDISABLEGENLOCKI3DPROC __wglewDisableGenlockI3D;
WGLEW_EXPORT PFNWGLENABLEGENLOCKI3DPROC __wglewEnableGenlockI3D;
WGLEW_EXPORT PFNWGLGENLOCKSAMPLERATEI3DPROC __wglewGenlockSampleRateI3D;
WGLEW_EXPORT PFNWGLGENLOCKSOURCEDELAYI3DPROC __wglewGenlockSourceDelayI3D;
WGLEW_EXPORT PFNWGLGENLOCKSOURCEEDGEI3DPROC __wglewGenlockSourceEdgeI3D;
WGLEW_EXPORT PFNWGLGENLOCKSOURCEI3DPROC __wglewGenlockSourceI3D;
WGLEW_EXPORT PFNWGLGETGENLOCKSAMPLERATEI3DPROC __wglewGetGenlockSampleRateI3D;
WGLEW_EXPORT PFNWGLGETGENLOCKSOURCEDELAYI3DPROC __wglewGetGenlockSourceDelayI3D;
WGLEW_EXPORT PFNWGLGETGENLOCKSOURCEEDGEI3DPROC __wglewGetGenlockSourceEdgeI3D;
WGLEW_EXPORT PFNWGLGETGENLOCKSOURCEI3DPROC __wglewGetGenlockSourceI3D;
WGLEW_EXPORT PFNWGLISENABLEDGENLOCKI3DPROC __wglewIsEnabledGenlockI3D;
WGLEW_EXPORT PFNWGLQUERYGENLOCKMAXSOURCEDELAYI3DPROC __wglewQueryGenlockMaxSourceDelayI3D;

WGLEW_EXPORT PFNWGLASSOCIATEIMAGEBUFFEREVENTSI3DPROC __wglewAssociateImageBufferEventsI3D;
WGLEW_EXPORT PFNWGLCREATEIMAGEBUFFERI3DPROC __wglewCreateImageBufferI3D;
WGLEW_EXPORT PFNWGLDESTROYIMAGEBUFFERI3DPROC __wglewDestroyImageBufferI3D;
WGLEW_EXPORT PFNWGLRELEASEIMAGEBUFFEREVENTSI3DPROC __wglewReleaseImageBufferEventsI3D;

WGLEW_EXPORT PFNWGLDISABLEFRAMELOCKI3DPROC __wglewDisableFrameLockI3D;
WGLEW_EXPORT PFNWGLENABLEFRAMELOCKI3DPROC __wglewEnableFrameLockI3D;
WGLEW_EXPORT PFNWGLISENABLEDFRAMELOCKI3DPROC __wglewIsEnabledFrameLockI3D;
WGLEW_EXPORT PFNWGLQUERYFRAMELOCKMASTERI3DPROC __wglewQueryFrameLockMasterI3D;

WGLEW_EXPORT PFNWGLBEGINFRAMETRACKINGI3DPROC __wglewBeginFrameTrackingI3D;
WGLEW_EXPORT PFNWGLENDFRAMETRACKINGI3DPROC __wglewEndFrameTrackingI3D;
WGLEW_EXPORT PFNWGLGETFRAMEUSAGEI3DPROC __wglewGetFrameUsageI3D;
WGLEW_EXPORT PFNWGLQUERYFRAMETRACKINGI3DPROC __wglewQueryFrameTrackingI3D;

WGLEW_EXPORT PFNWGLCOPYIMAGESUBDATANVPROC __wglewCopyImageSubDataNV;

WGLEW_EXPORT PFNWGLCREATEAFFINITYDCNVPROC __wglewCreateAffinityDCNV;
WGLEW_EXPORT PFNWGLDELETEDCNVPROC __wglewDeleteDCNV;
WGLEW_EXPORT PFNWGLENUMGPUDEVICESNVPROC __wglewEnumGpuDevicesNV;
WGLEW_EXPORT PFNWGLENUMGPUSFROMAFFINITYDCNVPROC __wglewEnumGpusFromAffinityDCNV;
WGLEW_EXPORT PFNWGLENUMGPUSNVPROC __wglewEnumGpusNV;

WGLEW_EXPORT PFNWGLBINDVIDEODEVICENVPROC __wglewBindVideoDeviceNV;
WGLEW_EXPORT PFNWGLENUMERATEVIDEODEVICESNVPROC __wglewEnumerateVideoDevicesNV;
WGLEW_EXPORT PFNWGLQUERYCURRENTCONTEXTNVPROC __wglewQueryCurrentContextNV;

WGLEW_EXPORT PFNWGLBINDSWAPBARRIERNVPROC __wglewBindSwapBarrierNV;
WGLEW_EXPORT PFNWGLJOINSWAPGROUPNVPROC __wglewJoinSwapGroupNV;
WGLEW_EXPORT PFNWGLQUERYFRAMECOUNTNVPROC __wglewQueryFrameCountNV;
WGLEW_EXPORT PFNWGLQUERYMAXSWAPGROUPSNVPROC __wglewQueryMaxSwapGroupsNV;
WGLEW_EXPORT PFNWGLQUERYSWAPGROUPNVPROC __wglewQuerySwapGroupNV;
WGLEW_EXPORT PFNWGLRESETFRAMECOUNTNVPROC __wglewResetFrameCountNV;

WGLEW_EXPORT PFNWGLALLOCATEMEMORYNVPROC __wglewAllocateMemoryNV;
WGLEW_EXPORT PFNWGLFREEMEMORYNVPROC __wglewFreeMemoryNV;

WGLEW_EXPORT PFNWGLBINDVIDEOIMAGENVPROC __wglewBindVideoImageNV;
WGLEW_EXPORT PFNWGLGETVIDEODEVICENVPROC __wglewGetVideoDeviceNV;
WGLEW_EXPORT PFNWGLGETVIDEOINFONVPROC __wglewGetVideoInfoNV;
WGLEW_EXPORT PFNWGLRELEASEVIDEODEVICENVPROC __wglewReleaseVideoDeviceNV;
WGLEW_EXPORT PFNWGLRELEASEVIDEOIMAGENVPROC __wglewReleaseVideoImageNV;
WGLEW_EXPORT PFNWGLSENDPBUFFERTOVIDEONVPROC __wglewSendPbufferToVideoNV;

WGLEW_EXPORT PFNWGLGETMSCRATEOMLPROC __wglewGetMscRateOML;
WGLEW_EXPORT PFNWGLGETSYNCVALUESOMLPROC __wglewGetSyncValuesOML;
WGLEW_EXPORT PFNWGLSWAPBUFFERSMSCOMLPROC __wglewSwapBuffersMscOML;
WGLEW_EXPORT PFNWGLSWAPLAYERBUFFERSMSCOMLPROC __wglewSwapLayerBuffersMscOML;
WGLEW_EXPORT PFNWGLWAITFORMSCOMLPROC __wglewWaitForMscOML;
WGLEW_EXPORT PFNWGLWAITFORSBCOMLPROC __wglewWaitForSbcOML;
WGLEW_EXPORT GLboolean __WGLEW_3DFX_multisample;
WGLEW_EXPORT GLboolean __WGLEW_3DL_stereo_control;
WGLEW_EXPORT GLboolean __WGLEW_AMD_gpu_association;
WGLEW_EXPORT GLboolean __WGLEW_ARB_buffer_region;
WGLEW_EXPORT GLboolean __WGLEW_ARB_create_context;
WGLEW_EXPORT GLboolean __WGLEW_ARB_create_context_profile;
WGLEW_EXPORT GLboolean __WGLEW_ARB_create_context_robustness;
WGLEW_EXPORT GLboolean __WGLEW_ARB_extensions_string;
WGLEW_EXPORT GLboolean __WGLEW_ARB_framebuffer_sRGB;
WGLEW_EXPORT GLboolean __WGLEW_ARB_make_current_read;
WGLEW_EXPORT GLboolean __WGLEW_ARB_multisample;
WGLEW_EXPORT GLboolean __WGLEW_ARB_pbuffer;
WGLEW_EXPORT GLboolean __WGLEW_ARB_pixel_format;
WGLEW_EXPORT GLboolean __WGLEW_ARB_pixel_format_float;
WGLEW_EXPORT GLboolean __WGLEW_ARB_render_texture;
WGLEW_EXPORT GLboolean __WGLEW_ATI_pixel_format_float;
WGLEW_EXPORT GLboolean __WGLEW_ATI_render_texture_rectangle;
WGLEW_EXPORT GLboolean __WGLEW_EXT_create_context_es2_profile;
WGLEW_EXPORT GLboolean __WGLEW_EXT_depth_float;
WGLEW_EXPORT GLboolean __WGLEW_EXT_display_color_table;
WGLEW_EXPORT GLboolean __WGLEW_EXT_extensions_string;
WGLEW_EXPORT GLboolean __WGLEW_EXT_framebuffer_sRGB;
WGLEW_EXPORT GLboolean __WGLEW_EXT_make_current_read;
WGLEW_EXPORT GLboolean __WGLEW_EXT_multisample;
WGLEW_EXPORT GLboolean __WGLEW_EXT_pbuffer;
WGLEW_EXPORT GLboolean __WGLEW_EXT_pixel_format;
WGLEW_EXPORT GLboolean __WGLEW_EXT_pixel_format_packed_float;
WGLEW_EXPORT GLboolean __WGLEW_EXT_swap_control;
WGLEW_EXPORT GLboolean __WGLEW_I3D_digital_video_control;
WGLEW_EXPORT GLboolean __WGLEW_I3D_gamma;
WGLEW_EXPORT GLboolean __WGLEW_I3D_genlock;
WGLEW_EXPORT GLboolean __WGLEW_I3D_image_buffer;
WGLEW_EXPORT GLboolean __WGLEW_I3D_swap_frame_lock;
WGLEW_EXPORT GLboolean __WGLEW_I3D_swap_frame_usage;
WGLEW_EXPORT GLboolean __WGLEW_NV_copy_image;
WGLEW_EXPORT GLboolean __WGLEW_NV_float_buffer;
WGLEW_EXPORT GLboolean __WGLEW_NV_gpu_affinity;
WGLEW_EXPORT GLboolean __WGLEW_NV_multisample_coverage;
WGLEW_EXPORT GLboolean __WGLEW_NV_present_video;
WGLEW_EXPORT GLboolean __WGLEW_NV_render_depth_texture;
WGLEW_EXPORT GLboolean __WGLEW_NV_render_texture_rectangle;
WGLEW_EXPORT GLboolean __WGLEW_NV_swap_group;
WGLEW_EXPORT GLboolean __WGLEW_NV_vertex_array_range;
WGLEW_EXPORT GLboolean __WGLEW_NV_video_output;
WGLEW_EXPORT GLboolean __WGLEW_OML_sync_control;

#ifdef GLEW_MX
}; /* WGLEWContextStruct */
#endif /* GLEW_MX */

/* ------------------------------------------------------------------------- */

#ifdef GLEW_MX

typedef struct WGLEWContextStruct WGLEWContext;
GLEWAPI GLenum wglewContextInit (WGLEWContext* ctx);
GLEWAPI GLboolean wglewContextIsSupported (WGLEWContext* ctx, const char* name);

#define wglewInit() wglewContextInit(wglewGetContext())
#define wglewIsSupported(x) wglewContextIsSupported(wglewGetContext(), x)

#define WGLEW_GET_VAR(x) (*(const GLboolean*)&(wglewGetContext()->x))
#define WGLEW_GET_FUN(x) wglewGetContext()->x

#else /* GLEW_MX */

#define WGLEW_GET_VAR(x) (*(const GLboolean*)&x)
#define WGLEW_GET_FUN(x) x

GLEWAPI GLboolean wglewIsSupported (const char* name);

#endif /* GLEW_MX */

GLEWAPI GLboolean wglewGetExtension (const char* name);

#ifdef __cplusplus
}
#endif

#undef GLEWAPI

#endif /* __wglew_h__ */
